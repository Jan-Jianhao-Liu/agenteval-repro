"""配置加载器：PyYAML + Pydantic 强校验，dev/exp/baseline 三环境一键切换。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


def load_dotenv(path: str | Path = ".env") -> None:
    """加载 .env（KEY=VALUE），不覆盖已有环境变量；幂等可重复调用。"""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k and k not in os.environ:
            os.environ[k] = v


def resolve_env(value: str) -> str:
    """展开 'env:VAR' 形式的环境变量引用；非该形式原样返回。"""
    if value.startswith("env:") and len(value) > 4:
        return os.environ.get(value[4:], "")
    return value


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str = "ollama_mock"          # ollama_mock | t3bench
    domain: str = "airline"            # τ³-bench 域：airline | retail | telecom | banking_knowledge
    model: str = "qwen3.5:4b"          # ollama 模型名 或 LiteLLM 模型名（如 deepseek/deepseek-v4-flash）
    api_base: str = "http://127.0.0.1:11434"
    num_ctx: int = 8192
    temperature: float = 0.7
    system_prompt_file: str = ""       # 业务域 system prompt 文本路径（相对 config 同级根）


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str = "ollama"           # ollama | openai_compat
    model: str = "my-qwen4b-no-think"
    api_base: str = "http://127.0.0.1:11434"
    base_url: str = ""                 # OpenAI 兼容 base url（openai_compat 用，如 https://api.deepseek.com/v1）
    api_key: str = ""                  # 支持 env:VAR 形式从 .env/环境变量展开
    temperature: float = 0.0
    timeout_sec: int = 120
    max_retries: int = 3
    retry_backoff: list[float] = Field(default_factory=lambda: [1.0, 3.0, 5.0])
    circuit_fail_threshold: int = 5
    circuit_cooldown_sec: int = 60
    json_fix_retries: int = 2


class SessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_rounds: int = 3
    max_turns: int = 8
    concurrency: int = 2
    reset_on_error: bool = True
    explorer: str = "simple"     # simple | llm
    objectives: list[str] = Field(default_factory=list)  # 探索目标清单；空则用 CLI 默认


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    dir: str = "data/cache"


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_dir: str = "data/dev"
    trace_dir: str = "data/dev/trace"


class DbscanConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    eps: float = 0.7
    min_samples: int = 1


class TestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    boundary_limit: int = 50
    scorer_threshold: float = 0.4      # 边界打分筛选阈值（越低候选越多）
    judge_verdicts: list[str] = Field(default_factory=lambda: ["pass", "fail", "inconclusive"])
    fault_types: list[str] = Field(default_factory=lambda: [
        "none", "skip_confirmation", "missing_identity",
        "missing_eligibility", "illegal_param_passed",
    ])


class TraceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: str = "DEBUG"
    save_input_output: bool = True


class AppConfig(BaseModel):
    """全局配置根模型：新增配置段仅需扩展字段，兼容旧文件（extra=ignore）。"""

    model_config = ConfigDict(extra="ignore")

    env_name: str = "dev"
    exp_mode: str = "full_method"      # prompt_only | graph_context | full_method
    taxonomy_file: str = ""            # 规范标签集（smallset json）路径；空 = 不约束（自由标签）
    agent: AgentConfig = Field(default_factory=AgentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    auditor_llm: LLMConfig = Field(default_factory=LLMConfig)  # 审计器独立 LLM（如 GLM-5.2）；model 为空则复用主 LLM
    session: SessionConfig = Field(default_factory=SessionConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    dbscan: DbscanConfig = Field(default_factory=DbscanConfig)
    test: TestConfig = Field(default_factory=TestConfig)
    trace: TraceConfig = Field(default_factory=TraceConfig)

    @field_validator("exp_mode")
    @classmethod
    def _valid_exp_mode(cls, v: str) -> str:
        if v not in {"prompt_only", "graph_context", "full_method"}:
            raise ValueError(f"非法 exp_mode: {v}")
        return v

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        """从 yaml 文件加载并强校验（自动加载 .env 并展开 env:VAR 引用）。"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        load_dotenv(path.parent.parent / ".env")
        load_dotenv(path.parent / ".env")
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls.model_validate(raw)
        cfg.llm.api_key = resolve_env(cfg.llm.api_key)
        cfg.llm.base_url = resolve_env(cfg.llm.base_url)
        if not cfg.llm.base_url and cfg.llm.provider == "openai_compat":
            cfg.llm.base_url = cfg.llm.api_base  # 兼容旧字段
        cfg.auditor_llm.api_key = resolve_env(cfg.auditor_llm.api_key)
        cfg.auditor_llm.base_url = resolve_env(cfg.auditor_llm.base_url)
        if not cfg.auditor_llm.base_url and cfg.auditor_llm.provider == "openai_compat":
            cfg.auditor_llm.base_url = cfg.auditor_llm.api_base
        # 相对路径统一锚定到配置文件所在目录，保证任意 cwd 可运行
        base = path.parent
        for sub in (cfg.storage.data_dir, cfg.storage.trace_dir, cfg.cache.dir):
            p = Path(sub)
            if not p.is_absolute():
                _resolve_relative(cfg, base)
                break
        return cfg


def _resolve_relative(cfg: AppConfig, base: Path) -> None:
    """将配置中相对路径锚定到 config/ 同级（即项目根）。"""
    root = base.parent if base.name == "config" else base
    cfg.storage.data_dir = str(Path(cfg.storage.data_dir) if Path(cfg.storage.data_dir).is_absolute() else root / cfg.storage.data_dir)
    cfg.storage.trace_dir = str(Path(cfg.storage.trace_dir) if Path(cfg.storage.trace_dir).is_absolute() else root / cfg.storage.trace_dir)
    cfg.cache.dir = str(Path(cfg.cache.dir) if Path(cfg.cache.dir).is_absolute() else root / cfg.cache.dir)


def load_config(env: str = "dev", path: Optional[str] = None) -> AppConfig:
    """便捷入口：按环境名或显式路径加载。"""
    if path:
        return AppConfig.load(path)
    here = Path(__file__).resolve().parent.parent.parent
    return AppConfig.load(here / "config" / f"{env}.yaml")
