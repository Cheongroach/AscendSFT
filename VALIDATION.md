# 发布验证，2026-10-24

- 固定公开trainer文件SHA256；在隔离目录核对 `git apply --check`，应用补丁后检查Python语法。公开源来自指定GitHub commit，未使用内部Git历史。
- 公开示例YAML字段与固定上游参数声明和Transformers训练参数接口核对；`qwen3_5_nothink`存在于固定上游template注册。
- `python -m unittest discover -s tests -v`：3/3测试通过，包含1535/1536/1537长度边界、eval、缺失初始化和输入错误。
- CPU随机小LoRA模型：相同初值和dropout RNG下，checkpoint开/关的loss、所有可训练梯度与裁剪后Adam更新逐元素相同。该检查不是Qwen3.5整模NPU验收。
- 逐文件白名单、内部路径/业务词/标识扫描以及公开来源许可证检查。

未重新启动NPU训练、未测试NPU完整模型或公开数据吞吐。历史28.4%不属于本补丁的发布验证成绩。
