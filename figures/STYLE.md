# STYLE.md — BCREA 论文绘图规范（2026-10-13 建）

> 本文件管 `code/figure/` 下全部绘图脚本。任何 agent 画正文或 SI 的图，先读本文件。
> 规矩来源：用户 2026-10-13 批准的整改方案 §7–§8；IEEE 图形尺寸与分辨率依据见文末。

## 一、定位与布图原则

1. **一图一科学问题。** 每张图只回答一个 Results 层面的问题；每个 panel 必须有独立的论证作用，
   与别的 panel 重复证据的 panel 删除、合并或降级到 SI。
2. **按论证布图，不按旧编号换皮。** 图的内容由正文论证链决定，不是把旧图重画一遍。
   表用于准确数字、图用于结构趋势，不做两套完整排行。
3. **正文／SI 分配。** 现象、主比较、机制边界、真实换案的结论性证据放正文；
   完整网格、全部收敛曲线、消融、覆盖数分布、异质性细节放 SI。
   结论改变的边界条件不得藏到 SI。
4. **示意图同样守规矩。** 流程图、时序图用同一套色板和字号地板，axis off，箭头线宽 1.0 pt。

## 二、尺寸与字号

1. **栏宽。** IEEE 双栏版式：单栏图宽 88.9 mm（3.5 in），跨栏图宽 182 mm（7.16 in）。
   设计稿宽与最终落版宽之比保持 0.9–1.1；禁止把双栏设计稿缩进单栏（旧 workflow 曾缩到 0.5 倍）。
2. **字号（落版后实测）。** 正文文字推荐 9–10 pt；本项目地板 **8 pt**，任何 glyph 不得低于 8 pt；
   panel 标签 **10 pt 加粗**。字号不够就重排版，不许缩字。
3. **字体。** 正文 Times New Roman（缺失回退 STIXGeneral / DejaVu Serif），数学字体 STIX，
   与 IEEEtran 观感对齐。PDF 必须嵌入字体（`pdf.fonttype=42`）。
4. **最小字号检查由 QA 脚本执行**（见第九节），不靠肉眼猜。

## 三、颜色与线型

1. **方法色全篇固定**，读者跨图不重学：

   | 方法（config `method`） | 显示名 | 颜色 | 线型 / 标记 |
   |---|---|---|---|
   | `nominal_kcea` | Nominal KCEA | 灰 `#999999` | 实线 / o |
   | `pessimistic_resolve` | Pessimistic re-solve | 朱红 `#D55E00` | 实线 / ^ |
   | `nsga2` | NSGA-II | 绿 `#009E73` | 实线 / v |
   | `sampled_worst_case` | Sampled worst-case | 紫 `#CC79A7` | 实线 / P |
   | `bcrea_band` | BCREA (band) | 蓝 `#0072B2` | **实线** / s |
   | `bcrea_strict` | BCREA (strict) | 蓝 `#0072B2` | **虚线** / D |

2. **band 与 strict 同色族、不同线型**——它们是同一方法的两个变体，用线型区分；
   nominal 用灰；其他基线用固定可辨认颜色，不用浅灰到看不清。
3. **双编码。** 关键区分同时用颜色 + 线型／标记，保证黑白打印和色盲读者可分辨（Okabe-Ito 色板）。
   每张图交付前做灰度检查。
4. **规模色独立一套**（24/48/100 目标：浅蓝→蓝→黑），不与方法色混用。

## 四、线宽、标记与坐标

1. 线宽层级：数据线 1.2 pt；强调（本文方法、主结论）1.8 pt；参考线 0.8 pt 配虚线。
2. 标记大小 3.5 pt；坐标轴线宽 0.6 pt；去上／右边框；仅 y 向细网格（0.4 pt, alpha 0.35）。
3. 图例无框；能直接标注的系列优先直接标注，减少视线往返。
4. 单位、有效位数全篇统一；对数轴上的零值不得悄悄改成正数，要么换轴要么明确说明。

## 五、面板对齐

1. 可比较的 panel 最终 **plot-area 必须对齐**，容差 1.5 pt；不等宽布局必须有明确理由。
2. panel 标签 `(a)/(b)/…` 加粗 10 pt，置于轴区左上外侧，用 `style.panel_tag()` 统一。
3. 每个多面板图导出前必须跑对齐审计（`audit_panel_alignment.py`），不过不交付。

## 六、统计呈现

1. 误差条／区间定义全篇一致并在图注写明（中位数 + IQR，或均值 + 95% CI），
   **不得把 min–max 画成置信区间**。
2. 样本量 n 及其单位（实例 / 种子）在图注或轴旁写清；可比较 panel 用同一不确定性定义。
3. 配对比较（同实例同种子）优先画配对差，不只画边际分布重叠。

## 七、数据溯源（铁律）

1. **所有性能图只读 `codes/results/` 下本轮锁定的新批次**（v3 批次），禁止读旧批次冒充正式结果。
   示意图（fig1、fig3）不读实验数据，是唯一例外。
2. 每张图脚本开头用 `style.require_batch()` 声明并核对批次存在，缺失时明确报错退出，不得画假数据。
3. 逐 run 唯一源表：脚本只认 `.npz` 里的 `config`，跑前核对 `method` 标签与预算字段；
   路径、批次号只出现在代码里，不进论文正文、图注、SI。
4. 抖动、抽样等随机展示规则用固定种子，保证重跑图一致。
5. 图与表同源：表由 `codes/` 分析脚本生成，图从同一批次读取，不手抄数字。

## 八、可重现性

1. **一图一脚本**：每张正文／SI 成品图对应 `figures/figX_<name>.py`，从项目根运行：
   `py -3.11 figures/figX_<name>.py`。无开发机硬编码路径，一律用 `style.ROOT` 相对寻址。
2. 公共样式在 `style.py`；公共只读数据层在 `io.py`：`load_pool / load_scan / load_reopt /
   load_analysis` 四个读取函数、六方法／30 实例／10 种子的枚举常量、实例短码解析，
   并把项目根加进 `sys.path` 以复用 `core.robust_metrics`（`equiv_set /
   greedy_rc_select / robust_coverage`）。两者都不产图。`io.py` 与标准库 `io` 同名，普通 `import io`
   永远拿到标准库，必须按路径加载：`io = style.load_io()`。
3. 正确解释器是 `py -3.11`（系统默认 `python` 是 Inkscape 捆绑版，缺 scipy，禁止用它跑图）。
4. 最终 PDF 写入 `paper/figures/`（进 tex）；PNG 预览与 QA 产物写入 `figures/build/`，
   QA 产物不进 `paper/figures/`。
5. 图号与文件名按最终编译顺序一致；SI 图以 `figS<n>_` 前缀。

## 九、导出规格与 QA 流程

1. 正文图一律矢量 PDF，字体嵌入（Type 42）。确需光栅时：彩色／灰度 >300 dpi，黑白线图 >600 dpi。
   PNG 只作预览。
2. 每张图交付前按顺序过 QA（脚本在 `code/figure/`，QA 产物进 `build/`）：
   源验证（`validate_figure.py --project-style`，四条项目豁免见 `build/QA_NOTES.md`）
   → 数据一致核对 → 字号／尺寸 → 多面板对齐
   （`audit_panel_alignment.py`，容差 1.5 pt）→ PDF 文本碰撞（`audit_figure_collisions.py`）
   → PDF 字号地板（`audit_pdf_text.py`，≥5 pt 硬底线、≥8 pt 项目线）→ 灰度 → 最终尺寸逐 panel 视觉检查。
3. **每次改图重跑全部 QA**，不是只重跑改动那张的预览。

## 十、依据

- IEEE 图形尺寸与分辨率：<https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/>
- 字体参考（Proceedings of the IEEE 建议 9–10 pt）：<https://proceedingsoftheieee.ieee.org/resources/guidelines-for-figures-and-tables/>
  8 pt 是本项目约定地板，不冒充 TSMC 官方硬要求。
