# AscendSFT · LLaMA-Factory 公开代码

本分支提供固定公开上游版本上的文本 LoRA SFT 补丁和公开数据示例。补丁在发布时直接针对公开 LLaMA-Factory 编写；没有复制内部训练工程、作业启动器、自定义 NPU kernel、数据处理代码或内部 Git 历史。

上游：[hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，固定 commit [`c683e59edd8f893abc5e1403184a1d0644817d2a`](https://github.com/hiyouga/LLaMA-Factory/tree/c683e59edd8f893abc5e1403184a1d0644817d2a)。完整源文件 SHA256、公开来源与许可见 [upstream/source.json](upstream/source.json)。

补丁在 `CustomSeq2SeqTrainer.compute_loss()` 的训练前向之前读取 **实际 `input_ids` padded length**：

```text
padded_length <= ASCENDSFT_GC_THRESHOLD → 不重计算
padded_length >  ASCENDSFT_GC_THRESHOLD → 重计算
ASCENDSFT_GC_THRESHOLD=0 或未设置      → 完全沿用上游
```

策略复用上游已初始化的 checkpoint 函数和输入梯度处理，不重新实现 checkpoint、loss 或优化器。`1536` 是已有 2×910B4 64GB 实验的候选阈值，不是任意模型、batch、LoRA 或显存配置上的通用安全值。数据内容不会进入策略日志。

## 应用与检查

```bash
git clone --branch code --single-branch https://github.com/Cheongroach/AscendSFT.git AscendSFT-code
git clone https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory
git -C LLaMA-Factory switch --detach c683e59edd8f893abc5e1403184a1d0644817d2a
python AscendSFT-code/scripts/apply_overlay.py LLaMA-Factory
python -m unittest discover -s AscendSFT-code/tests -v
```

应用程序核验公开上游 commit、trainer 内容 hash 和 patch 可应用性。若目标文件已经修改或 overlay 已存在，会停止，不覆盖已有改动。

## Ascend 环境与公开数据演示

先按公开 [NPU 安装指南](https://github.com/hiyouga/LLaMA-Factory/tree/c683e59edd8f893abc5e1403184a1d0644817d2a/docker/docker-npu) 准备与设备/驱动匹配的 CANN、Torch 和 TorchNPU，安装固定上游的依赖。历史参考栈为 Torch/TorchNPU 2.7.1、CANN 8.3 RC1、Transformers 5.2.0、PEFT 0.18.1；本次未启动 NPU 训练验证此公开改写。避免在已有 NPU 环境中用不受约束的 `pip install` 替换 Torch。

在已准备好的环境里，从 `LLaMA-Factory` 目录执行；两个命令使用相同模型、数据、batch 和优化器。更改模式时使用新的 `output_dir`：

```bash
# baseline：上游重计算保持开启
ASCENDSFT_GC_THRESHOLD=0 ASCEND_RT_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 \
  llamafactory-cli train ../AscendSFT-code/examples/qwen35_2b_lora_public.yaml \
  output_dir=saves/ascendsft/baseline_public

# adaptive：按当前 micro batch 的长度切换
ASCENDSFT_GC_THRESHOLD=1536 ASCEND_RT_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 \
  llamafactory-cli train ../AscendSFT-code/examples/qwen35_2b_lora_public.yaml \
  output_dir=saves/ascendsft/adaptive_public
```

示例引用上游自带的 `alpaca_en_demo`，没有在本仓库复制样本。来源和使用边界见 [examples/PUBLIC_DATA.md](examples/PUBLIC_DATA.md)。短演示样本可能全部落在阈值以下，应另用合法公开长样本检查长尾与 OOM 行为。

## 验证范围

- 发布检查覆盖上游补丁应用、Python 语法、公开 YAML 字段和 CPU 测试。
- CPU 测试覆盖阈值边界、eval 不切换、初始化错误，以及带 dropout、非零 LoRA A/B 的 loss、全部可训练参数梯度、裁剪后 Adam 更新。
- 当前范围是单机文本 LoRA SFT / DDP、普通同步 forward/backward、上游 checkpoint 初始化。FSDP、DeepSpeed、编译图和真实多模态长度策略未验证。
- 不开启 packing；混合 GDN 模型需要单独确认 recurrent state 的样本边界处理。

历史 **28.4%** 是另一实验实现和未公开工作负载上的实测，**不是本补丁或 `alpaca_en_demo` 的实测成绩**。历史报告、数值拒绝结果和工具方法放在独立 [performance 分支](https://github.com/Cheongroach/AscendSFT/tree/performance)。
