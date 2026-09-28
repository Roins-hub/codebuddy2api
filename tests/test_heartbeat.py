import asyncio

from core import converter


async def _slow_stream(delays, log=None):
    try:
        for i, delay in enumerate(delays):
            await asyncio.sleep(delay)
            yield f"chunk{i}".encode()
    finally:
        if log is not None:
            log.append("closed")


def _collect(stream):
    async def run():
        return [c async for c in stream]
    return asyncio.run(run())


def test_heartbeat_inserted_while_upstream_is_silent():
    out = _collect(converter._with_heartbeat(_slow_stream([0.25, 0]), b"HB", interval=0.1))
    assert out[-2:] == [b"chunk0", b"chunk1"]
    assert out[:-2] and set(out[:-2]) == {b"HB"}


def test_no_heartbeat_when_upstream_is_fast():
    out = _collect(converter._with_heartbeat(_slow_stream([0, 0, 0]), b"HB", interval=1))
    assert out == [b"chunk0", b"chunk1", b"chunk2"]


def test_heartbeat_disabled_with_non_positive_interval():
    out = _collect(converter._with_heartbeat(_slow_stream([0.15]), b"HB", interval=0))
    assert out == [b"chunk0"]


def test_client_disconnect_closes_upstream():
    log = []

    async def run():
        wrapped = converter._with_heartbeat(_slow_stream([0, 5], log), b"HB", interval=0.05)
        first = await wrapped.__anext__()
        assert first == b"chunk0"
        assert await wrapped.__anext__() == b"HB"  # upstream still waiting
        await wrapped.aclose()  # client went away

    asyncio.run(run())
    assert log == ["closed"]


def test_heartbeat_interval_from_env(monkeypatch):
    monkeypatch.setenv("WB_HEARTBEAT_SECONDS", "42")
    assert converter._heartbeat_seconds() == 42
    monkeypatch.setenv("WB_HEARTBEAT_SECONDS", "oops")
    assert converter._heartbeat_seconds() == 15


def test_anthropic_ping_is_valid_sse_event():
    assert converter.ANTHROPIC_PING.startswith(b"event: ping\n")
    assert converter.SSE_KEEPALIVE.startswith(b":")


# ── 端到端：慢速上游（先长时间只给思考片段）经真实路由转换后，客户端能持续收到心跳 ──

import json

import httpx
from fastapi.testclient import TestClient


class _FakeCred:
    def get_headers(self):
        return {}


def _slow_backend_stream():
    chunks = [{"choices": [{"index": 0, "delta": {"reasoning_content": "thinking "}}]}] * 3
    chunks.append({"choices": [{"index": 0, "delta": {"content": "pong"}}]})
    chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    class SlowSSE(httpx.AsyncByteStream):
        async def __aiter__(self):
            for c in chunks:
                await asyncio.sleep(0.15)
                yield f"data: {json.dumps(c)}\n\n".encode()
            yield b"data: [DONE]\n\n"

    return SlowSSE()


def _client(monkeypatch):
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda req: httpx.Response(200, stream=_slow_backend_stream()))
    monkeypatch.setattr(converter.httpx, "AsyncClient",
                        lambda *a, **kw: real_client(transport=transport))
    monkeypatch.setattr(converter, "_cred", lambda: _FakeCred())
    monkeypatch.setattr(converter, "_check_auth", lambda *a: None)
    monkeypatch.setenv("WB_HEARTBEAT_SECONDS", "0.1")
    return TestClient(converter.app)


def test_messages_stream_sends_ping_during_thinking(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/v1/messages", json={"model": "kimi-k3", "max_tokens": 100, "stream": True,
                                     "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert 'event: ping' in r.text
    text = "".join(json.loads(l[6:])["delta"].get("text", "") for l in r.text.splitlines()
                   if l.startswith("data: ") and '"content_block_delta"' in l)
    assert text == "pong"
    assert "message_stop" in r.text


def test_responses_stream_sends_keepalive_during_thinking(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/v1/responses", json={"model": "kimi-k3", "stream": True, "input": "hi"})
    assert r.status_code == 200
    assert ": keep-alive" in r.text
    assert "response.completed" in r.text
    deltas = [json.loads(l[6:]) for l in r.text.splitlines()
              if l.startswith("data: ") and "output_text.delta" in l]
    assert "".join(d["delta"] for d in deltas) == "pong"
