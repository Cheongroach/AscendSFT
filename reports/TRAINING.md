# 910B4 长度重计算：性能实测与采用边界

历史实验日期：2026-10-15。硬件为 **2×Ascend 910B4 64GB**，Torch/TorchNPU 2.7.1、CANN 8.3 RC1、Transformers 5.2.0、PEFT 0.18.1。模型 Qwen3.5-2B，LoRA r32/alpha64/dropout0.05，BF16 计算；micro batch2 × world2 × accumulation1，global batch4。cutoff4096、不 packing，动态 padding 至8的倍数。

实验保留完整词表与监督位置，不通过删样本、改 loss 归一化或改 global batch 提高吞吐。视觉塔仍加载，文本 collator 仍含 dummy 视觉占位路径，因此不能将其描述为已经删除视觉分支的纯文本实现。BF16 计算不意味着所有参数/优化器状态都是 BF16；该栈保存的 LoRA 参数为 FP32。

原始工作负载、样本、业务语义和数据 fingerprint 不公开。本材料是允许数值的脱敏摘录；公开分支中的新写补丁没有在这套历史工作负载上重新执行。

## 策略与门禁

baseline 始终启用梯度重计算；adaptive 在 **实际 padded length≤1536** 时关闭，超过时开启。常规片段各 rank 的所有 micro batch 均落在阈值以下。长尾恢复原 checkpoint 路径。

预设单步门禁：`|Δloss|≤0.001`、全部可训练参数梯度 relative L2≤2%、cosine≥0.999、裁剪后 Adam 参数更新 relative L2≤2%。参考和候选使用共同初值、同 seed、新建 AdamW；非零 LoRA B 初值覆盖 LoRA A 梯度。先检查 baseline 自重复。

| 候选 | 结果 | 采用 |
|---|---|---|
| adaptive | 有 dropout 的单步 loss 差2.98e-8；梯度、Adam 更新及状态差为0 | 进入有限吞吐实验 |
| CANN 原生 RMSNorm 替换 | 梯度 relative L2 5.57%，Adam 更新24.02% | 拒绝 |
| GDN 有限倍增级数 | 算子级检查通过；整模出现缺失或非有限梯度，原日志未区分二者 | 拒绝 |

RMSNorm 的整模替换与 [算法回移微基准](RMSNORM.md) 是不同实验，不能混写为“17/17 通过后完整训练采用”。

## 无 profiler 的 64-step 实测

相同 worker，每模式3次，交替复跑；每次64更新，205027有效 tokens。时间为两 rank 中较慢者的同步训练时间。包括训练正常输入处理、前反向、优化器与通信；不包括安装、模型下载和 profile 导出。

| 指标，三次中位数 | baseline | adaptive |
|---|---:|---:|
| 训练时间 | 111.421804 s | 86.793097 s |
| 有效吞吐 | 1840.10 tokens/s | 2362.25 tokens/s |
| 第11–64步稳态吞吐 | 1889.83 tokens/s | 2510.07 tokens/s |
| device step P50 / P95 | 1.613 / 2.436 s | 1.196 / 1.822 s |
| 峰值 allocated，每卡较大值 | 13.16 GiB | 36.24 GiB |
| 峰值 reserved，每卡较大值 | 46.54 GiB | 58.41 GiB |
| 完整训练子进程时间 | 147.04 s | 123.07 s |

`throughput_gain=2362.2500717/1840.0976522−1=28.3763%`；训练时间减少22.1040%，完整子进程减少16.30%。吞吐增幅与耗时降幅不能互换。稳态吞吐增幅32.82%也不等于完整任务增幅。

| 复跑 | baseline训练秒 | adaptive训练秒 |
|---|---:|---:|
| r0 | 111.387719 | 87.096452 |
| r1 | 113.363241 | 86.793097 |
| r2 | 111.421804 | 85.046680 |

显存为 allocator 统计，不是整卡 HBM 物理使用量。reserved 包含缓存块，不能全部解释为活跃激活。device event 计时包含队列等待，不是纯 kernel 时间。

## 长尾和训练漂移

实际源序列最大3146；门禁 padded length3152，长尾训练 padded length3080/3192。虽然 cutoff=4096，**未验证真实4096长度**。

长尾每模式单次8更新，两组都开启重计算并完成：有效吞吐2588.09/2574.27 tokens/s，allocated23.048/23.049 GiB。约−0.53%的差异不足以判断显著退化或加速。

64-step 三组成对运行的有效 LoRA 矩阵 `BA` relative L2 分别20.47%、7.97%、17.32%；baseline 自重复为11.90%、16.59%。正常性能运行未强制 deterministic algorithms，重复波动可观，但不能将所有跨模式差异都归于噪声。未预设64步权重非劣阈值，也未做完整收敛或独立效果验收。

因此结论是 **有限实验中有明确性能收益的重计算策略**，完整训练质量仍需独立验证。

## Trace 与跨硬件参考

每 rank 的2个 active steps：

| 计数 | baseline r0 / r1 | adaptive r0 / r1 |
|---|---:|---:|
| GDN前向 | 72 / 72 | 36 / 36 |
| GQA前向 | 24 / 24 | 12 / 12 |
| MLP前向 | 96 / 96 | 48 / 48 |
| host事件 | 1013134 / 1063192 | 809602 / 850624 |
| kernel记录，含copy与通信 | 201734 / 212591 | 159950 / 168876 |

前向次数减半与减少重计算一致；反向和优化器仍需执行，所以总训练不会快一倍。profiler 明显扰动运行，不用采集时间替代上述无 profiler 实测，也不将时间线无 kernel 区间直接等同 CPU 瓶颈。

相同有效 token 数的 H20 参考训练阶段为70.830831秒、2894.60 tokens/s；adaptive 910B4 约为其81.6%。硬件、软件版本和调度都不同，仅是系统组合参考，不是纯芯片性能或理论峰值利用率。

机器可读数值：[training-summary.json](../data/training-summary.json)、[numerical-gates.json](../data/numerical-gates.json)、[trace-summary.json](../data/trace-summary.json)。
