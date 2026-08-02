# 工业园区储能规划数据集盘点与下载决策

> 生成时间：2026-08-01T11:46:19+00:00；原始数据只读，尚未做重采样或节点映射。

## 先看结论

- 本次案例清单已有 **64 个文件，250.51 MiB**；逐文件大小和 SHA256 全部复核，验收总状态：**通过**。
- 原来已有的 **15 个 AUS/P1U `metrics.csv`，0.82 MiB** 全部保留，但只是静态馈线指标，不能训练 CVAE/DFL。
- 数据源不是 Smart-DS 和 IEEE 二选一：**IEEE13 负责物理三相网络；Smart-DS GSO industrial 负责工业负荷/PV/天气时间形状**。
- 严谨场景名称应为：**IEEE13 4.16 kV 工业微电网馈线（含 0.48 kV 二次节点）**，不是纯低压居民系统。
- 下载与处理顺序已经固定：**原始数据下载并校验完成后，再在 `interim/processed` 中处理；绝不改写 `raw`**。

## 统一后的系统与数据边界

| 部分 | 本项目采用 | 为什么 |
|---|---|---|
| 物理电网 | IEEE13官方/EPRI OpenDSS | 保留三相不平衡相别、线路阻抗、变压器、调压器和电容器 |
| 工业负荷 | Smart-DS GSO/industrial，2016–2018 | 目标馈线100%工业；含真实P/Q、PF和15分钟建筑负荷形状 |
| PV与天气 | 同一GSO区域、同三年 | 避免负荷与天气跨地区；同时支持PV、PUE和极端热天 |
| 数据中心工作量 | Azure Functions 2019 | Smart-DS没有计算任务到达量；以公开工作负载构造归一化任务曲线 |
| 电网碳 | EIA-930 DUK | GSO/Triad位于Duke Energy Carolinas服务逻辑内；使用消费侧而非生产侧碳强度 |
| 购电费率 | OpenEI 2026 OPT-V Large Primary | 园区总进口约3–4 MW；匹配3 MW以上、600 V–44 kV主计量客户 |
| 停电极端 | 后续规则合成 | 正常历史中停电样本稀少；以4–12小时连续失网形成可行性场景 |

## 原来15个文件逐项判断

这些目录名虽然带 `timeseries`，但文件本身没有时间戳；每个都是96条馈线、80列静态汇总。结论统一为“保留作参考，不进模型”。

| 序号 | 文件 | 大小 | 内容/处理决定 |
|---|---|---|---|
| 1 | raw\smartds\2016\AUS\P1U\scenarios\base_timeseries\metrics.csv | 56,927 B | Smart-DS 2016 AUS/P1U 场景 base_timeseries 的96条馈线×80项静态汇总指标；无时间戳。 |
| 2 | raw\smartds\2016\AUS\P1U\scenarios\solar_extreme_batteries_high_timeseries\metrics.csv | 58,279 B | Smart-DS 2016 AUS/P1U 场景 solar_extreme_batteries_high_timeseries 的96条馈线×80项静态汇总指标；无时间戳。 |
| 3 | raw\smartds\2016\AUS\P1U\scenarios\solar_extreme_batteries_low_timeseries\metrics.csv | 58,172 B | Smart-DS 2016 AUS/P1U 场景 solar_extreme_batteries_low_timeseries 的96条馈线×80项静态汇总指标；无时间戳。 |
| 4 | raw\smartds\2016\AUS\P1U\scenarios\solar_extreme_batteries_none_timeseries\metrics.csv | 57,950 B | Smart-DS 2016 AUS/P1U 场景 solar_extreme_batteries_none_timeseries 的96条馈线×80项静态汇总指标；无时间戳。 |
| 5 | raw\smartds\2016\AUS\P1U\scenarios\solar_high_batteries_high_timeseries\metrics.csv | 57,929 B | Smart-DS 2016 AUS/P1U 场景 solar_high_batteries_high_timeseries 的96条馈线×80项静态汇总指标；无时间戳。 |
| 6 | raw\smartds\2016\AUS\P1U\scenarios\solar_high_batteries_low_timeseries\metrics.csv | 57,822 B | Smart-DS 2016 AUS/P1U 场景 solar_high_batteries_low_timeseries 的96条馈线×80项静态汇总指标；无时间戳。 |
| 7 | raw\smartds\2016\AUS\P1U\scenarios\solar_high_batteries_none_timeseries\metrics.csv | 57,600 B | Smart-DS 2016 AUS/P1U 场景 solar_high_batteries_none_timeseries 的96条馈线×80项静态汇总指标；无时间戳。 |
| 8 | raw\smartds\2016\AUS\P1U\scenarios\solar_low_batteries_high_timeseries\metrics.csv | 57,537 B | Smart-DS 2016 AUS/P1U 场景 solar_low_batteries_high_timeseries 的96条馈线×80项静态汇总指标；无时间戳。 |
| 9 | raw\smartds\2016\AUS\P1U\scenarios\solar_low_batteries_low_timeseries\metrics.csv | 57,430 B | Smart-DS 2016 AUS/P1U 场景 solar_low_batteries_low_timeseries 的96条馈线×80项静态汇总指标；无时间戳。 |
| 10 | raw\smartds\2016\AUS\P1U\scenarios\solar_low_batteries_none_timeseries\metrics.csv | 57,208 B | Smart-DS 2016 AUS/P1U 场景 solar_low_batteries_none_timeseries 的96条馈线×80项静态汇总指标；无时间戳。 |
| 11 | raw\smartds\2016\AUS\P1U\scenarios\solar_medium_batteries_high_timeseries\metrics.csv | 57,579 B | Smart-DS 2016 AUS/P1U 场景 solar_medium_batteries_high_timeseries 的96条馈线×80项静态汇总指标；无时间戳。 |
| 12 | raw\smartds\2016\AUS\P1U\scenarios\solar_medium_batteries_low_timeseries\metrics.csv | 57,472 B | Smart-DS 2016 AUS/P1U 场景 solar_medium_batteries_low_timeseries 的96条馈线×80项静态汇总指标；无时间戳。 |
| 13 | raw\smartds\2016\AUS\P1U\scenarios\solar_medium_batteries_none_timeseries\metrics.csv | 57,250 B | Smart-DS 2016 AUS/P1U 场景 solar_medium_batteries_none_timeseries 的96条馈线×80项静态汇总指标；无时间戳。 |
| 14 | raw\smartds\2016\AUS\P1U\scenarios\solar_none_batteries_high_timeseries\metrics.csv | 57,256 B | Smart-DS 2016 AUS/P1U 场景 solar_none_batteries_high_timeseries 的96条馈线×80项静态汇总指标；无时间戳。 |
| 15 | raw\smartds\2016\AUS\P1U\scenarios\solar_none_batteries_low_timeseries\metrics.csv | 57,149 B | Smart-DS 2016 AUS/P1U 场景 solar_none_batteries_low_timeseries 的96条馈线×80项静态汇总指标；无时间戳。 |

## 本次下载的数据

| 数据组 | 文件数 | 大小 | 用途 |
|---|---|---|---|
| Smart-DS工业负荷Parquet | 22 | 28.93 MiB | 工业P/Q/PF和分项负荷；处理后进入模型 |
| Smart-DS太阳能/天气CSV | 6 | 9.90 MiB | PV、温度/PUE、极端天气；处理后进入模型 |
| Smart-DS目标馈线静态文件 | 24 | 0.06 MiB | profile引用和工业馈线映射；不替代IEEE13 |
| Smart-DS目标馈线指标 | 3 | 0.05 MiB | 证明所选馈线与规模；不作为时序训练 |
| IEEE13 OpenDSS | 2 | 0.02 MiB | 最终三相物理网络 |
| IEEE13坐标 | 1 | 0.00 MiB | 绘图和拓扑核对 |
| IEEE官方说明 | 1 | 0.06 MiB | 参数出处 |
| Azure工作负载压缩包 | 1 | 136.35 MiB | 任务到达量 |
| EIA DUK碳工作簿 | 1 | 75.07 MiB | 小时消费侧碳强度 |
| OpenEI工业费率JSON | 1 | 0.00 MiB | 电量价和需量费 |
| 官方说明文件 | 2 | 0.07 MiB | 字段、许可和处理依据 |

所有64个案例文件都已逐行列在可用 Excel 打开的 `manifests/all_files_audit.csv`，包含：相对路径、行列数、时间范围、用途、下载网址、SHA256和验收结果。

`DUK.xlsx`已做字段级验收：`Published Hourly Data`为97177行×89列，目标是第89列`CO2 Emissions Intensity for Consumed Electricity`；2021–2023恰好26280个连续UTC小时。

### 22条工业负荷profile

| 年份 | profile文件 | 每文件行数 | 采样 |
|---|---|---|---|
| 2016 | com_12903.parquet, com_13058.parquet, com_13063.parquet, com_13490.parquet, com_13641.parquet, com_14591.parquet, com_15212.parquet | 35040 | 15 min |
| 2017 | com_12916.parquet, com_13490.parquet, com_14102.parquet, com_14206.parquet, com_14250.parquet, com_14427.parquet, com_14591.parquet, com_14612.parquet | 35040 | 15 min |
| 2018 | com_12794.parquet, com_12903.parquet, com_12998.parquet, com_13445.parquet, com_13636.parquet, com_13887.parquet, com_13965.parquet | 35040 | 15 min |

### 6条PV/天气profile

| 年份 | 文件 | 记录数 | 字段 |
|---|---|---|---|
| 2016 | GSO_36.0976_-80.0003_15_180_full.csv, GSO_36.0976_-80.0003_25_180_full.csv | 2 × 35040 | DNI/DHI/GHI、风速、温度、POA、1 MW PV |
| 2017 | GSO_36.0976_-80.0003_15_180_full.csv, GSO_36.0976_-80.0003_25_180_full.csv | 2 × 35040 | DNI/DHI/GHI、风速、温度、POA、1 MW PV |
| 2018 | GSO_36.0976_-80.0003_15_180_full.csv, GSO_36.0976_-80.0003_25_180_full.csv | 2 × 35040 | DNI/DHI/GHI、风速、温度、POA、1 MW PV |

## `.tex`模型需要什么，当前是否已具备

| 模型输入 | 原始来源 | 当前状态 | 后续处理 |
|---|---|---|---|
| 节点有功/无功负荷 | Smart-DS P/Q + IEEE13基准负荷 | 原始数据齐 | 归一化profile，按IEEE13节点-相基准缩放 |
| 节点PV最大可用功率 | Smart-DS 1 MW PV曲线 | 原始数据齐 | 缩放到634/675/680的0.25/0.65/0.75 MW |
| PUE | Smart-DS温度 | 可派生 | 明确温度-PUE公式后生成 |
| 工作量到达A | Azure Functions | 原始数据齐 | 分钟聚合到小时、归一化、缩放到0.60 MW数据中心 |
| 电网碳强度 | EIA-930 DUK | 原始数据齐 | 取2021–2023三个完整UTC年消费侧强度，再按日历映射 |
| 购电电价/需量费 | OpenEI OPT-V Large Primary | 参数源齐 | 按月/季节/时段展开；需量费单独建模 |
| 电网可用性 | 规则合成 | 无需下载 | 正常全1；极端窗口注入4–12 h连续0 |
| 三相拓扑 | IEEE13 OpenDSS | 原始数据齐 | 解析bus-phase、完整阻抗矩阵和Y/Δ连接 |
| 储能/机组成本与技术参数 | 案例假设/文献 | 不是数据集 | 后续建立参数表并做敏感性分析 |

## 数据量级已经统一

- 当前 `data` 原始与清单合计约 **251.33 MiB**（不含将来解压后的Azure中间文件）。
- Smart-DS每条profile每年是 **35040个15分钟点**；3年统一成 **26280个小时点** 后再切窗口。
- 120小时（5天）按1天步长滑窗，三年理论上最多约 **1083个候选窗口**；不是只下载三年中的5天。
- 优化器最终只吃春/夏/秋/冬/极端各一个120小时块，但CVAE/DFL必须先看到完整历史候选池。
- 建议预留 **2–3 GB**：Azure压缩包解压、小时主表、三相张量、滑窗缓存和训练/验证/测试文件都会扩大。

## 处理顺序（下一步照这个做）

1. **Raw已完成**：下载、逐文件验收、SHA256、来源保留；禁止直接修改。
2. **建立映射**：解析IEEE13和Smart-DS `Loads.dss/PVSystems.dss`，形成Smart-DS profile → IEEE13节点/相的映射表。
3. **统一时标**：保留15分钟副本；聚合小时功率用平均值，小时能量用积分；以UTC唯一索引并处理DST。
4. **构造小时主表**：P/Q/PV/温度/PUE/workload/碳/费率/availability全部对齐，但记录各变量并非联合实测。
5. **分割而非泄漏**：2016训练、2017验证、2018测试；归一化参数只用2016计算。
6. **切120小时候选窗**：保留季节、日期、正常/极端、发生权重等元数据。
7. **再改CVAE/DFL代码**：当前代码仍只生成toy数据，且是单相平衡模型，尚不会读取这些文件。

## 必须写进论文的方法限制

- Smart-DS 2016–2018的负荷/PV/天气彼此同地区，但Azure工作量和EIA碳不是同一现场的联合观测；它们是按日历条件组合的案例场景。
- Smart-DS官方说明明确将闰年2016的最后一天截掉，只保留365天。下载的2016负荷Parquet还存在一个末行时间戳跳变（`2016-12-30 23:45`直接到`2017-01-01 00:00`）；处理时按行序重建`2016-01-01 00:00`至`2016-12-30 23:45`的标准15分钟索引，并把这项修正写入manifest，不插补12月31日。
- EIA第89列`CO2 Emissions Intensity for Consumed Electricity`自2018-07-01起有值，单位为`lb/kWh`；换算公式为`tCO2/MWh = lb/kWh × 0.45359237`。2019和2020各缺一个DST附近的23小时段，因此最终选2021–2023：共26280小时、UTC无缺失/重复/跳变。不能声称这是与Smart-DS 2016–2018严格同步的联合历史。
- Azure Functions只有14天，不能冒充三年历史；应做有放回采样/扰动并保留来源日标签。
- 当前`.tex`与代码都是平衡LinDistFlow；如果坚持三相不平衡，数学式、schema和规划约束都必须同步改。
- 极端场景由历史联合尾部窗口加合成停电构造；其年发生次数设为0，只约束规划可行性。

## 可重复执行

```powershell
conda activate storage-dfl
cd E:\Vscode\ESS_planning\dfl-storage-planning
python scripts\download_case_data.py --include-workload --include-grid --workers 4
python scripts\audit_case_data.py
```

下载清单种类计数：`{'bus_coordinates': 1, 'data_center_workload': 1, 'documentation': 2, 'feeder_metrics': 3, 'grid_carbon_workbook': 1, 'industrial_tariff_json': 1, 'load_timeseries': 22, 'official_report': 1, 'opendss': 2, 'opendss_feeder': 24, 'solar_weather_timeseries': 6}`。
