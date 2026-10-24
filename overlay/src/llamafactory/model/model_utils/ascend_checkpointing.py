# Copyright 2026 AscendSFT contributors.
# Licensed under the Apache License, Version 2.0.
"""LLaMA-Factory 文本 SFT 的实验性长度重计算策略。"""

from typing import Any


class LengthAwareCheckpointing:
    """复用上游已经初始化的 checkpoint 函数，只改变模块开关。"""

    def __init__(self, threshold: int):
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = threshold
        self._modules = None

    def apply(self, model: Any, inputs: dict[str, Any]) -> bool | None:
        # eval 不改变训练策略，也不读取样本内容。
        if not model.training:
            return None
        input_ids = inputs.get("input_ids")
        if input_ids is None or input_ids.ndim != 2:
            raise ValueError("Length-aware checkpointing requires rank-2 input_ids")
        if self._modules is None:
            modules = [
                module
                for module in model.modules()
                if isinstance(getattr(module, "gradient_checkpointing", None), bool)
            ]
            if not modules or not any(module.gradient_checkpointing for module in modules):
                raise RuntimeError("Initialize upstream gradient checkpointing before using this policy")
            if any(not callable(getattr(module, "_gradient_checkpointing_func", None)) for module in modules):
                raise RuntimeError("All checkpoint modules must have an initialized checkpoint function")
            self._modules = modules
        enabled = input_ids.shape[-1] > self.threshold
        for module in self._modules:
            module.gradient_checkpointing = enabled
        return enabled
