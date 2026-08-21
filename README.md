<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Jan-Jianhao-Liu/Jan-Jianhao-Liu.github.io/assets/waifu_agenteval_banner.jpg" width="100%" alt="AgentEval · 黑盒边界测试的观测审判者" />
</p>

---

# AgentEval 复现工程（Independent Re-implementation）

> **English**: [README.en.md](README.en.md)
>
> **Independent re-implementation of "Mining Workflow Graphs for Black-Box Boundary Testing of Conversational LLM Agents"**（独立复现，方法思想归原作者）。
>
> 本仓库是论文方法的**独立复现与实验验证**，非官方代码。所有模块均为自研实现，被测环境集成开源 [τ³-bench (tau2-bench)](https://github.com/sierra-research/tau2-bench)。

## 原论文

- **标题**：Mining Workflow Graphs for Black-Box Boundary Testing of Conversational LLM Agents
- **作者**：Liting Lin, Boxi Yu, Yuzhong Zhang, Lionel Briand, David-Paul Niland, Emir Muñoz
- **arXiv**：2607.06873（cs.SE）
- **摘要**：AgentEval 通过挖掘对话工作流图（conversational workflow graph）对会话式 LLM 智能体做黑盒边界测试——先与智能体交互挖掘行为图，再沿图结构枚举守卫（guard）与前置条件作为测试目标，回放对话路径至边界后注入扰动，仅凭对话轮次判定通过/失败。在四个 τ³-bench 智能体上覆盖 23~38 个独立边界。

## 本复现验证的结论

- **真实 τ³-bench 四域**（airline / retail / telecom / banking_knowledge）完整复现：full_method 独立边界 **45~87**（论文区间 23~38）、重复率 **≤0.083**（论文 0.26）、覆盖召回率 **0.98~1.00**（论文 0.97）
- **三组消融跨域趋势一致**：full_method（按图遍历）显著优于 prompt-only / graph-context 基线（独立边界 45~50 vs 1~3），证实"图结构引导边界发现"是方法核心
- **特权审计器价值量化**：LLM 裁判误报率高达 0.93，基于业务规则库 + 独立大模型（GLM-5.2）的审计器可将其抑制到接近真值（20% 人工基准，133 条样本）

## 实验配置（真实环境）

| 角色 | 模型 |
| --- | --- |
| 被测智能体（τ³-bench LLMAgent） | deepseek-v4-flash（LiteLLM 驱动） |
| 框架模块（抽象/打分/生成/裁判） | deepseek-chat |
| 特权审计器（独立复核） | GLM-5.2（智谱） |

全部温度 0；探索会话 R_disc=12；用例上限 N_b=50/90。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置大模型密钥（.env，勿提交）
# 复制以下模板并填入实际 key：
#   DEEPSEEK_API_KEY=sk-xxx
#   GLM_API_KEY=xxx

# 3. 部署被测环境（τ³-bench）
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench && pip install -e . && pip install audioop-lts  # Python 3.13 需要
cp ../agenteval-repro/.env .env   # tau2 读取同一密钥

# 4. 探索采集（每域 12 条会话）
python -m agenteval.cli --config config/exp_tau.yaml --task smoke

# 5. 消融实验（prompt_only / graph_context / full_method）
python -m agenteval.cli --config config/exp_tau.yaml --task experiment

# 6. 20% 人工标注基准
python scripts/build_annotation.py --envs tau tau_retail tau_telecom tau_banking
```

四域配置：`config/exp_tau*.yaml`（airline/retail/telecom/banking_knowledge）。

## 目录结构

```
src/agenteval/
├── cli.py            # 命令行入口（smoke/experiment）
├── config.py         # YAML 配置加载（含 env:VAR 密钥展开）
├── domain/           # 会话轨迹数据模型
├── drivers/          # L1 交互驱动（会话/探索/τ³-bench 适配器）
├── abstraction/      # L2 事件抽象 + DFG 构建（bge-m3 语义规范化）
├── boundary/         # L3 边界枚举/打分/扰动生成
├── evaluate/         # L4 裁判 + 特权审计器（规则库 + LLM 复核）
├── experiment/       # 消融流水线 + 指标（DBSCAN 聚类）
├── llm/              # LLM 网关（OpenAI 兼容 / 缓存 / 熔断 / 兜底）
├── storage/          # 持久化
└── trace/            # 全链路 Trace
docs/                 # 周报 / 复现报告 / 白话版方案 / 人工标注与复核
data/                 # 实验数据（sessions/trace 等运行时数据不提交）
```

## 许可与致谢

- 本仓库代码：**MIT License**
- 被测环境 τ³-bench：来自 [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)（遵循其 LICENSE）
- 方法归属：Liting Lin et al.（2026），arXiv:2607.06873
- 本仓库为独立复现，与原论文作者无隶属关系；发现问题欢迎提 Issue

## 安全说明

API 密钥仅经 `.env` 注入（已被 `.gitignore` 排除），本仓库不含任何真实密钥；如发现泄漏请立即作废对应 key。