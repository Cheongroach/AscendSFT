# 性能工具：操作、产物与分析

历史 Ascend 实验采用 `torch_npu.profiler` 和 CANN 导出；它们提供类似 Nsight Systems 的 CPU/device 时间线、算子和通信观察。不是在 NPU 上运行 `nsys`，也不将 trace 观察当作逐指令性能证明。本仓库只公开脱敏统计和新写通用示例。

| 工具 | 主要回答的问题 | 典型输入 / 输出 |
|---|---|---|
| `torch_npu.profiler` | host 发射、NPU kernel、拷贝和 HCCL 如何交错 | 训练若干 steps / trace JSON、operator和kernel CSV |
| CANN `msprof` | SDK侧采集、导出和通信/算子细节 | 自有应用或已采集目录 / CANN分析产物 |
| MindStudio Insight / Perfetto | 时间线查看、放大、关联事件 | 受支持 trace；不同版本可能需要导出转换 |
| `npu-smi info` | 设备可见性、功耗、温度与利用率采样 | 设备状态 / 短间隔状态记录 |
| 同步计时 + allocator API | 无 profiler 吞吐、allocated/reserved峰值 | 固定工作量 / wall time、GiB统计 |
| CPU离线分析 | 中位数、计数减少、区间并集与配对 | 脱敏JSON或自有合法CSV / 聚合JSON |

## 1. 最小采集

在自己已安装正确 CANN/TorchNPU 的 NPU 环境中执行：

```bash
python tools/profile_public.py --output profile-output --steps 6
```

示例使用随机张量的小 MLP，不导入历史研究代码。调度为wait2、warmup1、active2，共6更新；关键代码：

```python
with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    schedule=torch_npu.profiler.schedule(wait=2, warmup=1, active=2, repeat=1),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("profile-output"),
    record_shapes=False, profile_memory=False, with_stack=False,
) as recording:
    # 自己的 forward、loss.backward()、optimizer.step()
    recording.step()  # 每次更新推进一次
```

导出文件名/布局随 TorchNPU/CANN 版本变化，常见包含 `trace_view.json`、`operator_details.csv`、`kernel_details.csv`、`communication.json`。检查 active steps确实落在稳态窗口，且不是只采到加载、保存或编译。

先以低扰动配置采 CPU/NPU 时间线；只有需要定位调用栈、shape或显存时，再单独采启用对应选项的短窗口。profile时间与无profile时间分开，历史运行即使关闭这些选项仍有显著扰动。

## 2. CANN和时间线查看

用已安装版本的帮助确认采集参数，避免跨 SDK 版本照搬：

```bash
msprof --help
npu-smi info
```

`msprof`可包裹自有应用或分析已采集目录；若 `torch_npu.profiler` 已导出需要的CSV，无需为了“工具齐全”再次重跑。打开 MindStudio Insight支持的目录/trace，或在本地/受控 Perfetto环境加载兼容trace。查看时先定位active窗口，再选择一个完整forward/backward/update：

1. 从host操作到device事件关联，区分分配、copy、算子发射与等待。
2. 按stream查看kernel密度，观察长kernel与密集短kernel是否串行。
3. 对比baseline/adaptive同模块前向次数。两步中GDN72→36、GQA24→12，与关闭重计算一致。
4. 将通信与各rank前一段计算对齐，查看谁先进入同一collective、谁晚到。

没有原始trace的公开读者只能验证本仓库的计数摘录，不能自行验证事件关联或关键路径归因。

## 3. 无 profiler 吞吐与显存

使用同一数据、模型初值、global batch和更新数；先预热并排除下载/JIT。有效token计数来自CPU输入元数据，计时中避免为了计数额外做 `.item()` 同步：

```python
torch.npu.reset_peak_memory_stats()
torch.npu.synchronize()
t0 = time.perf_counter()
# 执行固定数量的更新，保留正常DDP和数据准备
torch.npu.synchronize()
seconds = time.perf_counter() - t0
allocated_gib = torch.npu.max_memory_allocated() / 2**30
reserved_gib = torch.npu.max_memory_reserved() / 2**30
```

多rank使用已完成同步训练的较慢rank时间；吞吐=`有效tokens/seconds`。完整应用wall time另报。计算 `median()` 使用run作为单位，不能把相关steps当独立复跑。

## 4. CSV区间与小算子密度

公开离线工具支持普通CSV或`.csv.gz`：

```bash
python tools/analyze_public.py intervals own-kernel-details.csv.gz \
  --start-column 'Start Time(us)' --duration-column 'Duration(us)'
```

核心方法是将每行转为 `[start,start+duration]`，排序合并重叠区间，再计算：

```text
span = last_end − first_start
covered = Σ merged_interval_duration
uncovered = span − covered
```

单纯累计duration会重复计算多stream重叠；uncovered还可能来自未采集事件、同步等待、不同stream和采集缺口，不直接等于CPU瓶颈。

判断小算子密度时，固定一个窗口，用明确的kernel类别筛选，报告窗口内计数、duration P50及总覆盖时间。历史示例在5ms窗口观察到约142条小算子、duration P50约10.21μs；它是局部窗口现象，不是整段训练每5ms都有相同密度的证明。

## 5. HCCL跨rank分析

按collective类型、group、序号、shape/字节数匹配；不能仅按行号或“第一个AllReduce”配对。确认跨rank时钟基准可比较后，计算两端进入时间偏差。历史长collective区间约252–283ms，晚到rank的到达偏斜可达约235ms，支持等待对端的解释。

`较早rank的collective duration − 到达偏斜`只是一种分解估计，不是纯网络时间；通信和计算可重叠，协议和profiler也会引入等待。未测独立通信基准和链路流量时，不宣称HCCL带宽提高或网络是唯一瓶颈。

## 6. 公开结果与重现实验

`python tools/analyze_public.py`会核验三次run的中位数、token/时间换算、+28.3763%吞吐和−22.1040%训练时间，以及trace计数减少比例。所用JSON为人工允许字段摘录，未包含hostname、PID、任务标识、源路径、样本或内部栈。

采集示例尚未在本次发布中运行NPU，只做Python语法检查。没有新增GPU/NPU训练任务，也没有发布真实原始trace或内部日志。
