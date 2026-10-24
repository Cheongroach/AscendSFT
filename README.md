# AscendSFT · 公开性能材料

本分支仅包含脱敏报告、规范化数值摘录，以及发布时重新编写的通用采集/分析示例。原始数据、trace、内部日志与研究工程源码没有公开。

| 材料 | 内容 |
|---|---|
| [训练性能与门禁](reports/TRAINING.md) | +28.38% 的口径、显存代价、长尾与训练漂移 |
| [RMSNorm 回移](reports/RMSNORM.md) | 精度修订、17/17 算子门禁、强参照与拒绝接入 |
| [工具操作](reports/TOOLS.md) | 类似 Nsight Systems 的采集与查看、CSV 分析、跨 rank 通信判断 |
| [架构与优化判断](reports/ARCHITECTURE.md) | GDN/GQA、显存、算子融合与重计算的边界 |
| [公开数值](data/) | 训练、trace 计数、数值门禁与 RMSNorm 的 JSON 摘录 |
| [分析程序](tools/analyze_public.py) | 在 CPU 上复核收益、计数和区间并集 |
| [采集示例](tools/profile_public.py) | 独立随机张量 NPU 小模型；不导入历史研究代码 |
| [发布范围](RELEASE_SCOPE.md) | 脱敏方式、允许内容与验证范围 |

历史主结果为双卡 910B4 上 64-step 中位吞吐 **1840.10 → 2362.25 tokens/s**，以峰值 allocated **13.16 → 36.24 GiB/卡** 为代价。收益来自减少重计算，原 GDN 保留。单步数值门禁不证明完整训练轨迹等价。

这些数值摘录允许复核计算，不能代替原始 trace 或独立重现实验。工作负载没有公开；[code 分支](https://github.com/Cheongroach/AscendSFT/tree/code) 的补丁是在公开 LLaMA-Factory 上新写的示例，尚无 NPU 实测成绩。

在 CPU 环境运行：

```bash
python tools/analyze_public.py
```

自有合法 CSV 的区间分析：

```bash
python tools/analyze_public.py intervals own-kernel-details.csv.gz \
  --start-column 'Start Time(us)' --duration-column 'Duration(us)'
```

CSV 字段按自己的 CANN 导出版本调整；分析工具只打印聚合时间，不打印事件名、样本或路径列。
