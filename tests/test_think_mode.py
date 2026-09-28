import asyncio
import json

import httpx
from fastapi.testclient import TestClient

from core import converter
from core.responses_adapter import ResponsesStreamConverter, responses_request_to_chat


def _chunk(delta, finish=None, usage=None):
    obj = {"id": "chatcmpl-1", "model": "glm-5.3",
           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage:
        obj["usage"] = usage
    return obj


REASONING_THEN_BODY = [
    _chunk({"role": "assistant", "reasoning_content": "let me "}),
    _chunk({"reasoning_content": "think"}),
    _chunk({"content": "po"}),
    _chunk({"content": "ng"}),
    _chunk({}, finish="stop", usage={"prompt_tokens": 1, "completion_tokens": 5, "total_tokens": 6,
                                     "completion_tokens_details": {"reasoning_tokens": 3}}),
]


def _backend(chunks, delay=0.0):
    class SSE(httpx.AsyncByteStream):
        async def __aiter__(self):
            for c in chunks:
                if delay:
                    await asyncio.sleep(delay)
                yield f"data: {json.dumps(c)}\n\n".encode()
            yield b"data: [DONE]\n\n"
    return SSE()


def _client(monkeypatch, chunks, delay=0.0, env=None):
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda req: httpx.Response(200, stream=_backend(chunks, delay)))
    monkeypatch.setattr(converter.httpx, "AsyncClient", lambda *a, **kw: real_client(transport=transport))
    monkeypatch.setattr(converter, "_cred", lambda: type("C", (), {"get_headers": lambda self: {}})())
    monkeypatch.setattr(converter, "_check_auth", lambda *a: None)
    monkeypatch.delenv("WB_THINK_MODE", raising=False)
    monkeypatch.delenv("WB_THINK_TIMEOUT", raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    return TestClient(converter.app)


def _stream(client, headers=None):
    r = client.post("/v1/chat/completions", headers=headers or {},
                    json={"model": "glm-5.3", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    data = [l[6:] for l in r.text.splitlines() if l.startswith("data: ")]
    objs = [json.loads(d) for d in data if d != "[DONE]"]
    content = "".join((c.get("delta") or {}).get("content") or "" for o in objs for c in o.get("choices", []))
    reasoning = "".join((c.get("delta") or {}).get("reasoning_content") or "" for o in objs for c in o.get("choices", []))
    return data, content, reasoning, objs


# ── _ThinkRewriter ──

def test_rewriter_tag_wraps_reasoning_and_closes_before_body():
    rw = converter._ThinkRewriter("tag")
    out = [rw.transform(json.loads(json.dumps(c))) for c in REASONING_THEN_BODY]
    text = "".join((o["choices"][0]["delta"].get("content") or "") for o in out if o)
    assert text == "<think>\nlet me think\n</think>\n\npong"
    assert all("reasoning_content" not in o["choices"][0]["delta"] for o in out if o)


def test_rewriter_tag_closes_think_before_tool_call():
    rw = converter._ThinkRewriter("tag")
    rw.transform(_chunk({"reasoning_content": "plan"}))
    out = rw.transform(_chunk({"tool_calls": [{"index": 0, "function": {"name": "shell"}}]}))
    assert out["choices"][0]["delta"]["content"] == converter.THINK_CLOSE
    assert rw.close({}) == b""


def test_rewriter_off_drops_reasoning_only_chunks():
    rw = converter._ThinkRewriter("off")
    assert rw.transform(_chunk({"reasoning_content": "x"})) is None
    assert rw.transform(_chunk({"content": "y"}))["choices"][0]["delta"] == {"content": "y"}


def test_think_mode_resolution(monkeypatch):
    monkeypatch.delenv("WB_THINK_MODE", raising=False)
    assert converter._think_mode() == "native"
    monkeypatch.setenv("WB_THINK_MODE", "TAG")
    assert converter._think_mode() == "tag"
    monkeypatch.setenv("WB_THINK_MODE", "bogus")
    assert converter._think_mode() == "native"


# ── /v1/chat/completions 流式 ──

def test_native_mode_passes_reasoning_through_unchanged(monkeypatch):
    data, content, reasoning, _ = _stream(_client(monkeypatch, REASONING_THEN_BODY))
    assert content == "pong" and reasoning == "let me think"
    assert data.count("[DONE]") == 1


def test_tag_mode_via_env(monkeypatch):
    data, content, reasoning, _ = _stream(_client(monkeypatch, REASONING_THEN_BODY, env={"WB_THINK_MODE": "tag"}))
    assert content == "<think>\nlet me think\n</think>\n\npong"
    assert reasoning == ""
    assert data.count("[DONE]") == 1 and data[-1] == "[DONE]"


def test_header_overrides_env(monkeypatch):
    c = _client(monkeypatch, REASONING_THEN_BODY, env={"WB_THINK_MODE": "tag"})
    _, content, reasoning, _ = _stream(c, headers={"X-Think-Mode": "off"})
    assert content == "pong" and reasoning == ""


def test_tag_mode_closes_think_when_stream_ends_mid_reasoning(monkeypatch):
    chunks = [_chunk({"reasoning_content": "still thinking"})]
    _, content, _, _ = _stream(_client(monkeypatch, chunks, env={"WB_THINK_MODE": "tag"}))
    assert content == "<think>\nstill thinking\n</think>\n\n"


def test_think_budget_aborts_with_note(monkeypatch):
    chunks = [_chunk({"reasoning_content": "hmm "})] * 20 + [_chunk({"content": "late body"})]
    c = _client(monkeypatch, chunks, delay=0.02, env={"WB_THINK_TIMEOUT": "0.1", "WB_THINK_MODE": "tag"})
    data, content, _, objs = _stream(c)
    assert "late body" not in content
    assert "思考已超过" in content and content.count("</think>") == 1
    assert objs[-1]["choices"][0]["finish_reason"] == "length"
    assert data[-1] == "[DONE]" and data.count("[DONE]") == 1


def test_think_budget_not_triggered_once_body_started(monkeypatch):
    chunks = [_chunk({"content": "a"})] + [_chunk({"content": "b"})] * 10
    c = _client(monkeypatch, chunks, delay=0.02, env={"WB_THINK_TIMEOUT": "0.05"})
    _, content, _, _ = _stream(c)
    assert content == "a" + "b" * 10


# ── /v1/chat/completions 非流式 ──

def _nonstream(client, headers=None):
    r = client.post("/v1/chat/completions", headers=headers or {},
                    json={"model": "glm-5.3", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    return r.json()["choices"][0]


def test_nonstream_native_keeps_reasoning_field(monkeypatch):
    choice = _nonstream(_client(monkeypatch, REASONING_THEN_BODY))
    assert choice["message"]["content"] == "pong"
    assert choice["message"]["reasoning_content"] == "let me think"


def test_nonstream_tag_mode(monkeypatch):
    choice = _nonstream(_client(monkeypatch, REASONING_THEN_BODY), headers={"X-Think-Mode": "tag"})
    assert choice["message"]["content"] == "<think>\nlet me think\n</think>\n\npong"
    assert "reasoning_content" not in choice["message"]


# ── Responses ──

def test_responses_maps_reasoning_effort():
    chat = responses_request_to_chat({"model": "kimi-k3", "input": "hi", "reasoning": {"effort": "low"}})
    assert chat["reasoning_effort"] == "low"


def test_responses_start_emitted_once_and_reasoning_tokens():
    conv = ResponsesStreamConverter(model="kimi-k3")
    first = conv.start()
    assert "response.created" in first and "response.in_progress" in first
    assert conv.start() == ""
    later = "".join(conv.feed_line("data: " + json.dumps(c)) for c in REASONING_THEN_BODY)
    assert "response.created" not in later
    assert conv.get_nonstream_response()["usage"]["output_tokens_details"]["reasoning_tokens"] == 3
