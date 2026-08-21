"""纯逻辑单元测试：实体校验、JSON 提取、缓存、熔断、DFG 构建（不依赖 LLM/网络）。"""

import asyncio
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from agenteval.abstraction.dfg_builder import DFGBuilder  # noqa: E402
from agenteval.domain import DialogueEvent, SessionTrace, WorkflowGraph  # noqa: E402
from agenteval.llm.cache import LLMCache  # noqa: E402
from agenteval.llm.circuit import CircuitBreaker  # noqa: E402
from agenteval.llm.gateway import _extract_json  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def test_entities():
    print("[1] 领域实体校验")
    ev = DialogueEvent(turn_id=0, user_utterance="你好", agent_response="您好")
    trace = SessionTrace(domain="airline")
    trace.add_event(ev)
    check("DialogueEvent 缺标签字段默认 None", ev.user_action is None)
    check("SessionTrace 追加事件", trace.turn_count == 1)

    try:
        DialogueEvent(turn_id=0, user_utterance="", agent_response="x")
        check("空 user_utterance 应被拦截", False)
    except Exception:
        check("空 user_utterance 应被拦截", True)

    try:
        DialogueEvent(turn_id=0, user_utterance="a", agent_response="b", extra_field=1)
        check("extra_forbid 拦截多余字段", False)
    except Exception:
        check("extra_forbid 拦截多余字段", True)


def test_workflow_graph():
    print("[2] WorkflowGraph <-> NetworkX")
    g = WorkflowGraph(method="full_method")
    g.nodes = [{"id": "a", "label": "a", "count": 2}, {"id": "b", "label": "b", "count": 1}]
    g.edges = [{"source": "a", "target": "b", "count": 2}]
    nxg = g.to_networkx()
    check("NetworkX 节点数", nxg.number_of_nodes() == 2)
    check("NetworkX 边数", nxg.number_of_edges() == 1)
    back = WorkflowGraph.from_networkx(nxg)
    check("回转换节点数", back.node_count == 2)
    check("图转 Prompt 文本", "a -> b" in g.to_prompt_text())


def test_json_extract():
    print("[3] JSON 提取（纯 JSON / 围栏 / 前后缀废话）")
    check("纯 JSON", _extract_json('{"a": 1}') == {"a": 1})
    check("markdown 围栏", _extract_json('```json\n{"a": 1}\n```') == {"a": 1})
    check("前后缀废话", _extract_json('好的，结果如下：{"a": 1} 就是这样') == {"a": 1})
    check("非 JSON 返回 None", _extract_json("完全不是 JSON") is None)
    check("空串返回 None", _extract_json("") is None)
    check("数组 JSON 返回 None(仅接受 dict)", _extract_json("[1,2]") is None)


def test_cache():
    print("[4] LLM 缓存命中/隔离")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        c = LLMCache(d)
        check("未命中返回 None", c.get("m", "fp", "in") is None)
        c.put("m", "fp", "in", {"v": 1})
        check("命中返回值", c.get("m", "fp", "in") == {"v": 1})
        check("不同输入不串", c.get("m", "fp", "other") is None)
        c2 = LLMCache(d)  # 重启进程模拟
        check("磁盘持久化跨实例命中", c2.get("m", "fp", "in") == {"v": 1})


def test_circuit():
    print("[5] 熔断器状态机")
    cb = CircuitBreaker(fail_threshold=3, cooldown_sec=60)
    cb.record_failure()
    cb.record_failure()
    check("阈值内 CLOSED", not cb.is_open)
    cb.record_failure()
    check("达阈值 OPEN", cb.is_open)
    cb.record_success()
    check("成功恢复 CLOSED", not cb.is_open)


def test_dfg():
    print("[6] DFG 构建")
    t1 = SessionTrace(domain="airline")
    t1.events = [
        DialogueEvent(turn_id=0, user_utterance="u0", agent_response="a0", user_action="ask_cancel", agent_activity="require_order", semantic_key="ask_cancel"),
        DialogueEvent(turn_id=1, user_utterance="u1", agent_response="a1", user_action="provide_order", agent_activity="verify_order", semantic_key="provide_order"),
    ]
    t2 = SessionTrace(domain="airline")
    t2.events = [
        DialogueEvent(turn_id=0, user_utterance="u0", agent_response="a0", user_action="ask_cancel", agent_activity="require_order", semantic_key="ask_cancel"),
        DialogueEvent(turn_id=1, user_utterance="u1", agent_response="a1", user_action="provide_order", agent_activity="verify_order", semantic_key="provide_order"),
        DialogueEvent(turn_id=2, user_utterance="u2", agent_response="a2", user_action="confirm_cancel", agent_activity="deny_cancel", semantic_key="confirm_cancel"),
    ]
    dfg = DFGBuilder().build([t1, t2])
    check("节点合并计数", dfg.node_count == 3)
    # DiGraph 为简单图：重复边合并为一条，频次累加到 count
    check("唯一边计数", dfg.edge_count == 2)
    repeated = [e for e in dfg.edges if e["source"] == "ask_cancel" and e["target"] == "provide_order"]
    check("重复边频次累加", len(repeated) == 1 and repeated[0]["count"] == 2)


async def test_simple_explorer():
    print("[7] SimpleExplorer 轮转")
    from agenteval.drivers import SimpleExplorer
    ex = SimpleExplorer()
    t = SessionTrace(domain="airline")
    u0 = await ex(t, "目标A", 0)
    u1 = await ex(t, "目标A", 1)
    check("首轮发目标", u0 == "目标A")
    check("后续轮跟进", "退票" in u1)


if __name__ == "__main__":
    test_entities()
    test_workflow_graph()
    test_json_extract()
    test_cache()
    test_circuit()
    test_dfg()
    asyncio.run(test_simple_explorer())
    print(f"\n===== 单元测试: {PASS} 通过 / {FAIL} 失败 =====")
    sys.exit(1 if FAIL else 0)
