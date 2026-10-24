"""CPU 门禁：长度边界、eval、初始化失败以及 LoRA 梯度和 Adam 更新。"""

import importlib.util
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint


source = (
    Path(__file__).resolve().parents[1]
    / "overlay/src/llamafactory/model/model_utils/ascend_checkpointing.py"
)
spec = importlib.util.spec_from_file_location("ascend_checkpointing", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
LengthAwareCheckpointing = module.LengthAwareCheckpointing


class LoRABlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(8, 8, bias=False).requires_grad_(False)
        self.a = torch.nn.Linear(8, 2, bias=False)
        self.b = torch.nn.Linear(2, 8, bias=False)
        self.dropout = torch.nn.Dropout(0.2)
        self.gradient_checkpointing = True
        self._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)

    def operation(self, hidden):
        return self.base(hidden) + self.b(self.a(self.dropout(hidden)))

    def forward(self, hidden):
        if self.training and self.gradient_checkpointing:
            return self._gradient_checkpointing_func(self.operation, hidden)
        return self.operation(hidden)


class PolicyTests(unittest.TestCase):
    def test_length_boundaries_and_eval(self):
        model = LoRABlock()
        policy = LengthAwareCheckpointing(1536)
        for length, expected in [(1535, False), (1536, False), (1537, True), (64, False)]:
            self.assertEqual(policy.apply(model, {"input_ids": torch.zeros(1, length)}), expected)
            self.assertEqual(model.gradient_checkpointing, expected)
        model.eval()
        self.assertIsNone(policy.apply(model, {}))
        self.assertFalse(model.gradient_checkpointing)

    def test_uninitialized_model_and_bad_input(self):
        with self.assertRaises(ValueError):
            LengthAwareCheckpointing(0)
        policy = LengthAwareCheckpointing(1536)
        model = LoRABlock()
        model.gradient_checkpointing = False
        with self.assertRaises(RuntimeError):
            policy.apply(model, {"input_ids": torch.zeros(1, 16)})
        with self.assertRaises(ValueError):
            policy.apply(model, {"input_ids": torch.zeros(16)})
        model.gradient_checkpointing = True
        model._gradient_checkpointing_func = None
        with self.assertRaises(RuntimeError):
            policy.apply(model, {"input_ids": torch.zeros(1, 16)})

    def test_loss_gradients_and_adam_update(self):
        torch.manual_seed(11)
        initial = LoRABlock().state_dict()
        hidden = torch.randn(2, 12, 8)
        target = torch.randn_like(hidden)
        results = []
        for adaptive in (False, True):
            model = LoRABlock()
            model.load_state_dict(initial)
            optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4
            )
            if adaptive:
                LengthAwareCheckpointing(1536).apply(model, {"input_ids": torch.zeros(2, 12)})
            torch.manual_seed(37)
            loss = (model(hidden) - target).square().mean()
            loss.backward()
            gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad}
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            updates = {
                name: p.detach() - initial[name] for name, p in model.named_parameters() if p.requires_grad
            }
            results.append((loss.detach(), gradients, updates))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        for component in (1, 2):
            for name in results[0][component]:
                torch.testing.assert_close(
                    results[0][component][name], results[1][component][name], rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
