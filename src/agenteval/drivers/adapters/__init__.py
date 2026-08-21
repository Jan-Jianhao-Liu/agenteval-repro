"""Agent 适配器包。"""

from .base import AgentAdapter
from .ollama_agent import AIRLINE_SYSTEM_PROMPT, OllamaAgent
from .t3bench import T3BenchAdapter

__all__ = ["AIRLINE_SYSTEM_PROMPT", "AgentAdapter", "OllamaAgent", "T3BenchAdapter"]
