# 公开数据来源

| 名称 | 公开位置 | 本仓库用途 |
|---|---|---|
| `alpaca_en_demo` | [LLaMA-Factory 演示文件](https://github.com/hiyouga/LLaMA-Factory/blob/c683e59edd8f893abc5e1403184a1d0644817d2a/data/alpaca_en_demo.json) | 验证公开训练入口；仅通过 dataset key 引用 |
| 数据注册 | [dataset_info.json](https://github.com/hiyouga/LLaMA-Factory/blob/c683e59edd8f893abc5e1403184a1d0644817d2a/data/dataset_info.json) | 使用相同固定上游版本的注册表 |
| Alpaca 原始项目 | [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) | 查阅数据生成方法及适用条款 |
| 模型 | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 标准公开模型标识 |

公开可访问不代表没有使用限制。请按各数据与模型来源的许可证、模型输出条款使用；本仓库 Apache-2.0 不替代数据许可证，也没有将演示数据声明为无条件商用。

本仓库不包含历史性能实验的数据样本、token ID、数据索引、源数据 fingerprint、业务 prompt、标签或 checkpoint。演示数据与历史性能工作负载不同，不能直接比较两者的绝对吞吐。
