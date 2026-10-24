# AscendSFT

Ascend 上的 Qwen3.5 LoRA SFT：LLaMA-Factory 公开补丁、性能实验结论与工具方法。

| 分支 | 内容 |
|---|---|
| [code](https://github.com/Cheongroach/AscendSFT/tree/code) | 固定公开 LLaMA-Factory 来源的长度重计算补丁、应用工具、CPU 门禁测试、公开数据训练示例 |
| [performance](https://github.com/Cheongroach/AscendSFT/tree/performance) | 脱敏性能报告、数值摘要、RMSNorm 回移负结论、profiling 操作说明与独立分析工具 |

主要历史观察：在 2×Ascend 910B4 64GB 的同一 worker 上，三次 64-step 实验的中位有效吞吐从 **1840.10** 提升到 **2362.25 tokens/s（+28.38%）**，峰值 allocated 显存从 **13.16** 增至 **36.24 GiB/卡**。短样本关闭重计算，长样本恢复重计算，候选阈值为 padded length 1536。

这是未公开工作负载上的有限性能结果，不是完整训练质量验收，也不是公开演示数据或本次改写补丁的实测。GDN 有限级数与原生 RMSNorm 替换未通过整模门禁，均未作为训练优化采用；另一个 RMSNorm 算法回移通过算子检查，但未证明超过 CANN 强参照。

公开内容由允许字段重新编写和生成。仓库不含内部训练工程、业务数据/样本/提示词、数据 fingerprint、内部地址、账户/机器/任务标识、原始 trace、内部日志、checkpoint 或内部 Git 历史。公开上游的来源和版权归属保留。

代码遵循 Apache-2.0；数据与模型按各自公开来源的条款使用。详细证据范围见 [performance](https://github.com/Cheongroach/AscendSFT/tree/performance)。
