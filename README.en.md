<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Jan-Jianhao-Liu/Jan-Jianhao-Liu.github.io/assets/waifu_agenteval_banner_v2.jpg" width="100%" alt="AgentEval · 黑盒边界测试的观测审判者" />
</p>

---

# AgentEval Reproduction Project (Independent Re-implementation)

> **Independent re-implementation of "Mining Workflow Graphs for Black-Box Boundary Testing of Conversational LLM Agents"** — all modules are self-implemented; the method belongs to the original authors.
>
> The agent under test runs on the open-source [τ³-bench (tau2-bench)](https://github.com/sierra-research/tau2-bench).

## Original Paper

- **Title**: Mining Workflow Graphs for Black-Box Boundary Testing of Conversational LLM Agents
- **Authors**: Liting Lin, Boxi Yu, Yuzhong Zhang, Lionel Briand, David-Paul Niland, Emir Muñoz
- **arXiv**: 2607.06873 (cs.SE)
- **Abstract**: AgentEval is a black-box testing framework that mines a *conversational workflow graph* from interactions with an LLM agent, then uses the graph structure to enumerate guards and prerequisites as test targets, replays the conversation path to each boundary, and applies a perturbation — judging pass/fail from conversation turns only. On four τ³-bench agents it covers 23–38 distinct boundaries.

## Findings Reproduced

- **Four real τ³-bench domains** (airline / retail / telecom / banking_knowledge): full_method distinct boundaries **45–87** (paper: 23–38), duplicate rate **≤0.083** (paper: 0.26), coverage recall **0.98–1.00** (paper: 0.97)
- **Ablation trend consistent across domains**: full_method (graph traversal) far outperforms prompt-only / graph-context baselines (45–50 vs 1–3 distinct boundaries), confirming graph-guided boundary discovery as the core mechanism
- **Privileged auditor value quantified**: the LLM judge has a false-alarm rate of 0.93; a rule-based auditor with an independent LLM (GLM-5.2) suppresses it close to ground truth (20% human-annotation baseline, 133 samples)

## Experiment Setup (Real Environment)

| Role | Model |
| --- | --- |
| Agent under test (τ³-bench LLMAgent) | deepseek-v4-flash (via LiteLLM) |
| Framework modules (abstraction/scoring/generation/judge) | deepseek-chat |
| Privileged auditor (independent review) | GLM-5.2 (Zhipu) |

Temperature 0 everywhere; exploration R_disc=12; test-case budget N_b=50/90.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure LLM API keys (.env, do NOT commit)
#   DEEPSEEK_API_KEY=sk-xxx
#   GLM_API_KEY=xxx

# 3. Deploy the environment under test (τ³-bench)
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench && pip install -e . && pip install audioop-lts   # required on Python 3.13
cp ../agenteval-repro/.env .env

# 4. Exploration (12 sessions per domain)
python -m agenteval.cli --config config/exp_tau.yaml --task smoke

# 5. Ablation experiments (prompt_only / graph_context / full_method)
python -m agenteval.cli --config config/exp_tau.yaml --task experiment

# 6. 20% human-annotation baseline
python scripts/build_annotation.py --envs tau tau_retail tau_telecom tau_banking
```

Per-domain configs: `config/exp_tau*.yaml`.

## Repository Layout

```
src/agenteval/
├── cli.py            # CLI entry (smoke / experiment)
├── config.py         # YAML config with env:VAR secret expansion
├── domain/           # conversation trace models
├── drivers/          # L1 interaction drivers (session / explorer / τ³-bench adapter)
├── abstraction/      # L2 event abstraction + DFG construction (bge-m3 semantic merge)
├── boundary/         # L3 boundary enumeration / scoring / perturbation
├── evaluate/         # L4 judge + privileged auditor (rule base + LLM review)
├── experiment/       # ablation pipeline + metrics (DBSCAN clustering)
├── llm/              # LLM gateway (OpenAI-compatible / cache / circuit breaker / fallback)
├── storage/          # persistence
└── trace/            # end-to-end tracing
docs/                 # weekly report / experiment report / annotation & review artifacts
data/                 # experimental results (runtime sessions/traces not committed)
```

## License & Acknowledgments

- This repository: **MIT License**
- Environment under test: [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) (under its own LICENSE)
- Method attribution: Liting Lin et al. (2026), arXiv:2607.06873
- This is an independent re-implementation, not affiliated with the original authors. Issues and PRs are welcome.

## Security

API keys are injected via `.env` only (excluded by `.gitignore`). This repository contains no real keys; if you suspect a leak, revoke the key immediately.