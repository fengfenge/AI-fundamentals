# 拓扑感知调度——让调度器知道 NVLink 和 NUMA

默认 K8s 调度器处理 `nvidia.com/gpu: 4` 时，只要能找到 4 张空闲 GPU 就视为满足。它不管这 4 张 GPU 之间走 NVLink 还是 PCIe、在同一个 NUMA node 上还是跨 socket。结果是：训练 Pod 虽然启动成功了，但 NCCL 通信带宽可能比最优路径差一个数量级。

本文解释拓扑感知调度的三层：GPU 亲和性、NUMA 亲和性、跨节点亲和性。

---

## 一、GPU 亲和性——NVLink 域内 vs 跨 NVSwitch

NVSwitch 域关注 GPU-GPU 距离

同一个节点上 8 张 A100/H100 通过 NVSwitch 全互联。任意两张 GPU 都能达到 NVLink 最高带宽（H100: ~450 GB/s 理论单向带宽，NCCL all_reduce 实测 bus_bw ~316 GB/s；A100: ~600 GB/s 双向）。但如果 4 张 GPU 分别来自两个 NVSwitch 域（A100 有 2 个 NVSwitch，每 4 张 GPU 一组），跨域通信需要经过 PCIe 中转：

```text
NVSwitch 域 1 (GPU 0-3):         NVSwitch 域 2 (GPU 4-7):
  GPU 0 ←→ GPU 3: NVLink          GPU 4 ←→ GPU 7: NVLink
  GPU 0 ←→ GPU 4: PCIe P2P        ← 跨 NVSwitch 域，走 PCIe
                                    （带宽约 NVLink 的 1/10-1/5）
```

调度器的任务是：为 TP=4 的作业分配 4 张 GPU 时，优先选择**全部在同一个 NVSwitch 域内**的 GPU 组。

```
NVSwitch域内，GPU之间NVLink互联架构：带宽很高，比如 H100 级别每卡 NVLink 可达数百 GB/s，延迟很低。
GPU0 ── GPU1
 │  \  /  │
 │   NV   │
 │  /  \  │
GPU3 ── GPU2
```


TP = Tensor Parallelism，张量并行。
在分布式训练 / 推理里，TP=4 表示：
这个作业把同一个模型层、同一个张量计算，切分到 4 张 GPU 上协同完成。
这 4 张 GPU 组成一个 TP 组 / TP rank group。每一张卡只算一部分，然后通过集合通信把结果合并。

常见并行策略对比：
```
缩写	全称	                  含义	                通信频率
TP	Tensor Parallelism	  张量并行，切单层张量	    极高，每层都通信
PP	Pipeline Parallelism	流水线并行，切模型层	    中等，阶段间传激活
DP	Data Parallelism	    数据并行，复制模型切数据	较低，梯度同步
EP	Expert Parallelism	  专家并行，MoE 专家分布	   高，All-to-All
```

```
Y = X @ W
TP=4 时，可能把权重 W 按列切成 4 份:
W = [W0, W1, W2, W3]
4 张 GPU 分别计算：
Y0 = X @ W0
Y1 = X @ W1
Y2 = X @ W2
Y3 = X @ W3
再通过 AllReduce、AllGather 或 ReduceScatter 合并成完整结果
```

### 1.1 实现方式

NVIDIA GPU Operator 提供的 Topology-Aware Scheduler 基于 `nvidia-smi topo -m` 的输出构建拓扑图，在 Filter 和 Score 阶段做两件事：

- **Filter**：剔除不满足拓扑约束的节点。如果 Pod 指定了 `nvidia.com/gpu.topo.nvswitch=4`，则 Filter 阶段只保留定义了包含 4 张 GPU 的 NVSwitch 域的节点。
- **Score**：在满足 Filter 的节点中，给拓扑更优的节点更高分。同 NVSwitch 域 > 同 PCIe switch > 同 NUMA node。

### 1.2 配置示例

```yaml
# 使用 NVIDIA GPU Operator 的拓扑感知调度
apiVersion: v1
kind: Pod
metadata:
  name: training-tp4
spec:
  schedulerName: nvidia-topology-aware-scheduler
  containers:
    - resources:
        limits:
          nvidia.com/gpu: 4
  annotations:
    nvidia.com/gpu.topology.prefer: "nvswitch" # 偏好同 NVSwitch 域
    # nvidia.com/gpu.topology.require: "nvswitch" # 强制同 NVSwitch 域
```

### 1.3 代价

强制执行拓扑约束会降低 GPU 利用率——如果在某个节点上同 NVSwitch 域的 4 张 GPU 被占用，即使有其他 4 张空闲 GPU 分布在两个 NVSwitch 域中，这个 Pod 也不会被调度。对于训练任务这是正确的取舍（慢 10 倍的训练 ≈ 浪费），但对于推理任务（单 GPU，无跨 GPU 通信），拓扑约束没有意义。

---

## 二、NUMA 亲和性——GPU 离哪个 CPU 更近

NUMA 亲和性关注 CPU-GPU 距离

NUMA 是单台服务器内部的拓扑概念，不是多台服务器之间的概念。每个 NUMA 节点包含: CPU 核心 + 直连的本地内存 + 本地 PCIe 控制器。

多台服务器之间通过网络互联，那叫集群 / 分布式系统，不叫 NUMA。NUMA 里的“节点”是单机内部的 CPU + 内存分组，和集群里的“节点”不是一回事。

CPU 访问自己节点的内存快；访问另一个节点的内存慢，因为要走 CPU 之间的互连，比如 UPI、Infinity Fabric。

真正定义 NUMA 节点的是：哪些 CPU 核心和哪块本地内存、哪些 PCIe 设备离得更近。通常 node0 对应 socket0

GPU 通过 PCIe 连接到特定的 NUMA node。H2D 传输时，如果 CPU 线程和 GPU 不在同一个 NUMA node，数据需要跨 QPI/UPI 传输，延迟翻倍：

```
numactl -H
lscpu | grep -i numa
nvidia-smi topo -m #nvidia-smi topo -m 可以看 GPU 挂在哪个 NUMA 节点，也就是“GPU 离哪个 CPU 更近”
```

```text
2-socket 服务器 (Intel Xeon):
  socket 0 ──QPI── socket 1
    │               │
  NUMA 0         NUMA 1
    │               │
  GPU 0-3         GPU 4-7     ← GPU 物理上绑定到特定的 NUMA node
  CPU 0-23        CPU 24-47   ← CPU core 也绑定到 NUMA node

错误: GPU 0 (NUMA 0) + CPU 24 (NUMA 1) → H2D 跨 QPI，~2 倍延迟
数据要先跨 CPU 互连，再走 PCIe 到 GPU，所以延迟更高，带宽也可能下降。

正确: GPU 0 (NUMA 0) + CPU 0  (NUMA 0) → H2D 本地，最优延迟
CPU 0 在 NUMA 0
GPU 0 也在 NUMA 0
host 内存在 NUMA 0
NUMA 0 本地内存
        ↓ PCIe
GPU 0 显存

实际使用建议：
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 python train.py
```

H2D = Host to Device，意思是：从主机内存拷贝到 GPU 显存。
在 CUDA 里就是：
```
cuda
cudaMemcpy(dst_gpu, src_cpu, size, cudaMemcpyHostToDevice);
Host：CPU + 系统内存 RAM
Device：GPU + 显存 VRAM
H2D 方向：CPU 内存 → PCIe → GPU 显存
```

对应的还有：
```
缩写	含义	          方向
H2D	Host to Device	  主机内存 → GPU 显存
D2H	Device to Host	  GPU 显存 → 主机内存
D2D	Device to Device	GPU 显存 → GPU 显存
H2H	Host to Host	    主机内存 → 主机内存
```

在 PyTorch 里，这些都会触发 H2D：

```python
x = torch.randn(1024, 1024)   # CPU 内存
x = x.cuda()                  # H2D：CPU 内存 -> GPU 显存
```

调度器应在 Score 阶段为 CPU 和 GPU 在同一个 NUMA node 的分配赋予更高分。

### 2.1 实现方式

K8s 1.27+ 的 Topology Manager 配合 CPU Manager + Memory Manager 可以做 NUMA 对齐。GPU 的 NUMA 亲和性需要 Device Plugin 在 `Allocate()` 时上报拓扑信息：

```yaml
# Kubelet Topology Manager 策略
# /var/lib/kubelet/config.yaml
topologyManagerPolicy: single-numa-node # 所有资源在同一 NUMA node
# 可选: best-effort, restricted, single-numa-node
```

`single-numa-node` 策略会导致 Pod 被限制在一个 NUMA node 内——如果 GPU 在 NUMA 0 但所需的 CPU 核心不够，Pod 不会调度。这是最严格的策略，只在延迟极其敏感的场景（如 HPC 训练）中使用。

### 2.2 建议

| 场景                                | NUMA 策略                        |
| ----------------------------------- | -------------------------------- |
| 单 GPU 推理、H2D 只发生一次         | 不需要 NUMA 对齐                 |
| 多 GPU 训练，DataLoader 放在 CPU 端 | `best-effort` — 尽量对齐但不强制 |
| HPC 训练、GPU-direct RDMA 频繁传输  | `single-numa-node` — 强制执行    |

---

## 三、跨节点亲和性——什么时候才需要关心

Gang Scheduling（§02）解决了「跨节点 GPU 数量足够」的问题，但不解决「跨节点通信有多快」的问题。TP=8 如果跨了 2 个节点，NCCL 通信从 NVLink（μs 级）变成了 IB（10-20 μs 级）——延迟上升了 10-20 倍。

对于 TP（Tensor Parallelism），跨节点是必须避免的——TP 的通信频率极高，延迟的放大直接反映为训练吞吐的下降。调度器应优先将 TP 组内的 GPU 放在同一节点上。

对于 DP（Data Parallelism），跨节点是不可避免的——梯度同步本来就是节点间的操作，IB 带宽可接受。DP 的训练作业不需要拓扑约束。

```text
训练作业: 2 节点 × 4 GPU = 8 GPU 总量
  TP=4: 每个节点 4 GPU，节点内 NVLink 通信 → 拓扑约束: 同节点
  DP=2: 节点间 IB 梯度同步 → 拓扑约束: 无特殊要求
```

大多数深度学习框架（PyTorch DDP、DeepSpeed）在启动时允许用户指定 TP size 和节点的映射关系，调度器只需要确保 TP 组内的 GPU 在同一个节点上即可，不再需要更细粒度的 NVSwitch 域感知。

---

## 四、实测案例：NVLink 和 PCIe 到底差多少

以下数据来自同一台 8×H100 NVSwitch 服务器（详见 [NCCL 通信路径逐层压测](../03_nccl/06_nccl_path_benchmark.md)）：

| 通信路径                                 | 512MB all_reduce bus_bw | 调度含义                         |
| ---------------------------------------- | ----------------------- | -------------------------------- |
| NVLink GPU0↔GPU1（同 NVSwitch 域）       | **315.8 GB/s**          | 最优——TP 组内 GPU 的期望值       |
| NVLink GPU0↔GPU4（跨 NUMA，同 NVSwitch） | **316.3 GB/s**          | 同 NVSwitch 域内，NUMA 无影响    |
| P2P 禁用 GPU0↔GPU1（走 PCIe fallback）   | **23.8 GB/s**           | 比 NVLink 慢 13 倍——训练不可接受 |

关键发现：同 NVSwitch 域内跨 NUMA 不影响带宽（316.3 ≈ 315.8 GB/s），调度器不需要担心 NUMA 对 GPU-GPU 通信的影响。真正要避免的是跨 NVSwitch 域（PCIe P2P）或禁用 P2P（CPU fallback）的路径——它们比 NVLink 慢一个数量级。

---

## 五、三层拓扑感知总结

| 层级         | 调度器需要知道什么                      | 约束强度                 | 典型场景                  |
| ------------ | --------------------------------------- | ------------------------ | ------------------------- |
| GPU 亲和性   | 哪些 GPU 在同一个 NVSwitch 域内         | 强（TP 组必须同域）      | TP=4/8 训练               |
| NUMA 亲和性  | 哪些 CPU core 和 GPU 在同一个 NUMA node | 中（Score 偏好，不强制） | DataLoader 性能敏感的训练 |
| 跨节点亲和性 | TP 组不跨节点                           | 强（TP 组必须同节点）    | 任何 TP > 1 的训练        |

---

## 相关资源

- [NVIDIA GPU Operator Topology-Aware Scheduling](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-topology-aware-scheduler.html)
- [K8s Topology Manager](https://kubernetes.io/docs/tasks/administer-cluster/topology-manager/)
- [GPU 调度问题总览](01_gpu_scheduling_problem.md)
- [NCCL 通信路径逐层压测](../03_nccl/06_nccl_path_benchmark.md) — NVLink 和 PCIe 实测差距
