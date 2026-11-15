#!/usr/bin/env python3
"""Free-model aggregator for make.fun competition free-model-aggregator-02.

Strategy: epsilon-greedy multi-armed bandit over the top-rated eligible
models, with rating-ordered failover.

Each free model is one arm of the bandit. An arm's prior value is its
house rating (rating_milli / 2000), held with PRIOR_PULLS pseudo-pulls,
and every gateway call updates it: a served prompt banks a reward equal
to the serving model's rating, a refusal banks 0, so a model that refuses
loses value for the rest of the trial and the bandit drifts away from it.
For each prompt the router takes the eligible arms (tools / vision /
JSON / context, from the brief's table), then with probability EPSILON
explores -- a seeded uniform pick among the top EXPLORE_TOP_K eligible
arms by current value -- and otherwise exploits the best-valued arm.
Failover walks the remaining eligible arms in current-value order: a
5xx is retried once on the same model (transient), a 429 or a second
failure moves to the next arm. Every refusal is reported on the response
in the x-makefun-upstream-failures header so the failover column can see
recovered retries.

All choices are functions of the brief, the request, the seeded RNG
(MAKEFUN_TRIAL_SEED) and the gateway answers seen so far: no wall clock
and no unseeded randomness, so a replay leaves the same state export
byte for byte. The RNG is drawn exactly once per prompt (one uniform
draw, plus one choice draw only when exploring), in prompt order, under
one lock, so live and replay draws agree.
"""

import http.client
import json
import os
import random
import socket
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GATEWAY_TIMEOUT = 10.0
MAX_ATTEMPTS_PER_PROMPT = 3
DEFAULT_MAX_TOKENS = 512
MAX_TOKENS_CAP = 8192

# Bandit parameters.
EPSILON = 0.10          # exploration probability per prompt
PRIOR_PULLS = 4         # pseudo-pulls anchoring each arm to its rating
EXPLORE_TOP_K = 3       # exploration picks uniformly among the top-K arms


# --------------------------------------------------------------------------
# Brief: the model table the trial hands us in MAKEFUN_PROMPT_FILE.
# --------------------------------------------------------------------------

def load_models(path):
    """Parse the brief's models/v1 table. Returns a list of model dicts."""
    models = []
    if not path or not os.path.exists(path):
        return models
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            schema = obj.get("schema", "")
            if not isinstance(schema, str) or not schema.endswith("/models/v1"):
                continue
            for raw in obj.get("models", []):
                if not raw.get("free", True):
                    continue
                models.append({
                    "id": raw["id"],
                    "rating_milli": int(raw.get("rating_milli", 0)),
                    "context_length": raw.get("context_length"),
                    "tools": bool(raw.get("tools", False)),
                    "vision": bool(raw.get("vision", False)),
                    "json": bool(raw.get("json", False)),
                })
    return models


def build_catalog(models):
    """The catalog document, exactly the brief's free models, once each."""
    return {
        "providers": [{
            "id": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "key_env": "OPENROUTER_API_KEY",
            "wire": "openai",
            "models": [
                {
                    "id": m["id"],
                    "free": True,
                    "context_length": m.get("context_length"),
                }
                for m in models
            ],
        }]
    }


def catalog_json(models):
    return json.dumps(build_catalog(models), separators=(",", ":"))


# --------------------------------------------------------------------------
# Requirements: what a prompt needs, derived from its own request body.
# --------------------------------------------------------------------------

def estimate_tokens(obj):
    """Rough token estimate (chars / 4), deterministic, for context checks."""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return max(1, len(text) // 4)


def requirements_of(body, wire):
    """Derive what a prompt needs from its own request body."""
    req = {"tools": False, "vision": False, "json": False}
    if wire == "openai":
        fmt = body.get("response_format") or {}
        if fmt.get("type") in ("json_object", "json_schema"):
            req["json"] = True
        if body.get("tools"):
            req["tools"] = True
        for msg in body.get("messages") or []:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        req["vision"] = True
    else:
        if body.get("tools"):
            req["tools"] = True
        for msg in body.get("messages") or []:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        req["vision"] = True
    req["min_context"] = estimate_tokens(body)
    return req


def declared_tool_names(body, wire):
    names = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        if wire == "openai":
            fn = tool.get("function") or {}
            if isinstance(fn.get("name"), str):
                names.append(fn["name"])
        elif isinstance(tool.get("name"), str):
            names.append(tool["name"])
    return names


# --------------------------------------------------------------------------
# Epsilon-greedy bandit over the eligible arms.
# --------------------------------------------------------------------------

class Bandit:
    """One arm per model; value starts at the house rating, moves with
    observed rewards (served prompt -> rating reward, refusal -> 0)."""

    def __init__(self, models, seed):
        self.by_id = {m["id"]: m for m in models}
        self.pulls = {m["id"]: 0 for m in models}
        self.reward_sum = {m["id"]: 0.0 for m in models}
        self.rng = random.Random(seed or "")

    def prior(self, model):
        return model["rating_milli"] / 2000.0

    def value(self, model):
        mid = model["id"]
        pulls = self.pulls[mid]
        return (
            PRIOR_PULLS * self.prior(model) + self.reward_sum[mid]
        ) / (PRIOR_PULLS + pulls)

    def record(self, model, reward):
        mid = model["id"]
        self.pulls[mid] += 1
        self.reward_sum[mid] += reward

    def eligible(self, req):
        """Models meeting the prompt's requirements, in brief order."""
        out = []
        for model in self.by_id.values():
            if req["tools"] and not model["tools"]:
                continue
            if req["vision"] and not model["vision"]:
                continue
            if req["json"] and not model["json"]:
                continue
            ctx = model.get("context_length")
            if ctx is not None and ctx < req["min_context"]:
                continue
            out.append(model)
        return out

    def order(self, models):
        """Current-value order: value desc, rating desc, id asc."""
        return sorted(
            models,
            key=lambda m: (-self.value(m), -m["rating_milli"], m["id"]),
        )

    def pick_first(self, req):
        """Epsilon-greedy pick plus the failover chain behind it.

        Returns (first_model, chain): chain is every eligible arm in
        current-value order with the picked arm moved to the front.
        The RNG is drawn exactly once per call (plus one choice draw
        only when the exploration branch is taken).
        """
        ranked = self.order(self.eligible(req))
        if not ranked:
            return None, []
        if self.rng.random() < EPSILON:
            pool = ranked[:EXPLORE_TOP_K]
            first = self.rng.choice(pool)
        else:
            first = ranked[0]
        chain = [first] + [m for m in ranked if m["id"] != first["id"]]
        return first, chain


# --------------------------------------------------------------------------
# Gateway client: every free model is dialled on Chat Completions.
# --------------------------------------------------------------------------

class GatewayFailure(Exception):
    """A model call that did not return a usable answer."""

    def __init__(self, status, code, model):
        super().__init__(code)
        self.status = status
        self.code = code
        self.model = model


def error_code(raw):
    """Pull the provider's error code out of a refusal body, if any."""
    try:
        obj = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return None
    err = obj.get("error") if isinstance(obj, dict) else None
    if isinstance(err, dict):
        for key in ("code", "type"):
            val = err.get(key)
            if isinstance(val, str) and val:
                return val
    return None


def gateway_call(model_id, body, stream):
    """One Chat Completions call. Returns (status, raw_bytes)."""
    base = os.environ.get("OPENAI_BASE_URL", "")
    key = os.environ.get("OPENAI_API_KEY", "")
    parsed = urllib.parse.urlparse(base)
    use_https = parsed.scheme == "https"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if use_https else 80)
    path = (parsed.path or "").rstrip("/") + "/chat/completions"
    payload = dict(body)
    payload["model"] = model_id
    payload["stream"] = stream
    if stream:
        payload["stream_options"] = {"include_usage": True}
    data = json.dumps(payload).encode("utf-8")
    conn_cls = http.client.HTTPSConnection if use_https else http.client.HTTPConnection
    conn = conn_cls(host, port, timeout=GATEWAY_TIMEOUT)
    try:
        conn.request(
            "POST",
            path,
            body=data,
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            },
        )
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Wire translation: Anthropic-shaped requests become Chat Completions
# bodies; provider answers are translated back into the client's wire.
# --------------------------------------------------------------------------

def anthropic_content_to_openai(content):
    """An Anthropic content value -> an OpenAI content value."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "image":
            src = block.get("source") or {}
            if src.get("type") == "base64":
                url = "data:%s;base64,%s" % (
                    src.get("media_type", "image/png"),
                    src.get("data", ""),
                )
                parts.append({"type": "image_url", "image_url": {"url": url}})
            elif src.get("type") == "url":
                parts.append(
                    {"type": "image_url", "image_url": {"url": src.get("url", "")}}
                )
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    return parts


def to_openai_body(body, wire):
    """Translate a client request body into a Chat Completions body."""
    if wire == "openai":
        out = {}
        out["messages"] = body.get("messages") or []
        out["max_tokens"] = body.get("max_tokens") or body.get(
            "max_completion_tokens") or DEFAULT_MAX_TOKENS
        if out["max_tokens"] > MAX_TOKENS_CAP:
            out["max_tokens"] = MAX_TOKENS_CAP
        if "temperature" in body:
            out["temperature"] = body["temperature"]
        if body.get("response_format"):
            out["response_format"] = body["response_format"]
        if body.get("tools"):
            out["tools"] = body["tools"]
        if body.get("tool_choice") is not None:
            out["tool_choice"] = body["tool_choice"]
        return out
    out = {}
    messages = []
    system = body.get("system")
    if system is not None:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text = "".join(
                b.get("text", "")
                for b in system
                if isinstance(b, dict) and b.get("type") == "text"
            )
            if text:
                messages.append({"role": "system", "content": text})
    for msg in body.get("messages") or []:
        messages.append({
            "role": msg.get("role", "user"),
            "content": anthropic_content_to_openai(msg.get("content")),
        })
    out["messages"] = messages
    out["max_tokens"] = body.get("max_tokens") or DEFAULT_MAX_TOKENS
    if out["max_tokens"] > MAX_TOKENS_CAP:
        out["max_tokens"] = MAX_TOKENS_CAP
    if "temperature" in body:
        out["temperature"] = body["temperature"]
    tools = body.get("tools") or []
    if tools:
        out["tools"] = [{
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {},
            },
        } for t in tools if isinstance(t, dict)]
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        kind = choice.get("type")
        if kind == "auto":
            out["tool_choice"] = "auto"
        elif kind == "any":
            out["tool_choice"] = "required"
        elif kind == "none":
            out["tool_choice"] = "none"
        elif kind == "tool":
            out["tool_choice"] = {
                "type": "function",
                "function": {"name": choice.get("name", "")},
            }
    return out


OPENAI_TO_ANTHROPIC_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
}


def tool_calls_of(message, declared):
    """Tool calls of a Chat Completions message, kept to declared names."""
    calls = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        if declared and name not in declared:
            continue
        args = fn.get("arguments") or "{}"
        try:
            parsed_args = json.loads(args) if isinstance(args, str) else args
        except ValueError:
            parsed_args = {}
        if not isinstance(parsed_args, dict):
            parsed_args = {}
        calls.append({
            "id": call.get("id") or "toolu_0",
            "name": name,
            "input": parsed_args,
        })
    return calls


def synth_usage(request_body, text):
    """Usage when the provider sent none: input estimate, output >= 1."""
    return {
        "prompt_tokens": estimate_tokens(request_body),
        "completion_tokens": max(1, estimate_tokens(text)),
        "total_tokens": estimate_tokens(request_body) + max(1, estimate_tokens(text)),
    }


def shape_openai_answer(raw, dialed_id, request_body, declared):
    """Rewrite a provider Chat Completions body for the OpenAI client."""
    try:
        value = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        value = {}
    if not isinstance(value, dict):
        value = {}
    value["model"] = dialed_id
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices:
        choices = [{"index": 0, "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop"}]
        value["choices"] = choices
    choice = choices[0] if isinstance(choices[0], dict) else {}
    choices[0] = choice
    message = choice.get("message")
    if not isinstance(message, dict):
        message = {"role": "assistant", "content": ""}
        choice["message"] = message
    message["role"] = "assistant"
    if not isinstance(choice.get("finish_reason"), str):
        choice["finish_reason"] = "stop"
    if message.get("tool_calls"):
        kept = tool_calls_of(message, declared)
        if kept:
            message["tool_calls"] = [{
                "id": c["id"],
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": json.dumps(c["input"], separators=(",", ":")),
                },
            } for c in kept]
        else:
            message.pop("tool_calls", None)
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    usage = value.get("usage")
    if not isinstance(usage, dict) or "completion_tokens" not in usage:
        value["usage"] = synth_usage(request_body, text)
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def shape_anthropic_answer(raw, dialed_id, request_body, declared, msg_id):
    """Translate a provider Chat Completions body into a Messages body."""
    try:
        value = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        value = {}
    if not isinstance(value, dict):
        value = {}
    choices = value.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(choice, dict):
        choice = {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    calls = tool_calls_of(message, declared)
    blocks = []
    if text:
        blocks.append({"type": "text", "text": text})
    for call in calls:
        blocks.append({
            "type": "tool_use",
            "id": call["id"],
            "name": call["name"],
            "input": call["input"],
        })
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    finish = choice.get("finish_reason")
    if calls:
        stop = "tool_use"
    elif isinstance(finish, str):
        stop = OPENAI_TO_ANTHROPIC_STOP.get(finish, "end_turn")
    else:
        stop = "end_turn"
    usage = value.get("usage")
    if isinstance(usage, dict) and "completion_tokens" in usage:
        out_tokens = usage["completion_tokens"]
        in_tokens = usage.get("prompt_tokens", estimate_tokens(request_body))
    else:
        synth = synth_usage(request_body, text)
        in_tokens = synth["prompt_tokens"]
        out_tokens = synth["completion_tokens"]
    answer = {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": blocks,
        "model": dialed_id,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {
            "input_tokens": in_tokens,
            "output_tokens": max(1, out_tokens),
        },
    }
    return json.dumps(answer, separators=(",", ":")).encode("utf-8")


def parse_sse_frames(raw):
    """Split gateway SSE bytes into a list of parsed chunk dicts."""
    chunks = []
    text = raw.decode("utf-8", "replace")
    for frame in text.split("\n\n"):
        payload = None
        for line in frame.split("\n"):
            if line.startswith("data:"):
                payload = line[5:].strip()
        if payload is None or payload == "[DONE]":
            continue
        try:
            chunks.append(json.loads(payload))
        except ValueError:
            continue
    return chunks


def collect_stream(raw):
    """Fold gateway SSE chunks into (text, tool_calls, finish, usage)."""
    text_parts = []
    calls = {}
    order = []
    finish = None
    usage = None
    saw_finish = False
    for chunk in parse_sse_frames(raw):
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or {}
        if isinstance(delta.get("content"), str):
            text_parts.append(delta["content"])
        for call in delta.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            idx = call.get("index", 0)
            slot = calls.setdefault(idx, {"id": "", "name": "", "args": ""})
            if idx not in order:
                order.append(idx)
            fn = call.get("function") or {}
            if isinstance(call.get("id"), str) and call["id"]:
                slot["id"] = call["id"]
            if isinstance(fn.get("name"), str) and fn["name"]:
                slot["name"] = fn["name"]
            if isinstance(fn.get("arguments"), str):
                slot["args"] += fn["arguments"]
        if isinstance(choice.get("finish_reason"), str):
            finish = choice["finish_reason"]
            saw_finish = True
    tool_calls = []
    for idx in order:
        slot = calls[idx]
        if not slot["name"]:
            continue
        try:
            parsed_args = json.loads(slot["args"]) if slot["args"] else {}
        except ValueError:
            parsed_args = {}
        if not isinstance(parsed_args, dict):
            parsed_args = {}
        tool_calls.append({
            "id": slot["id"] or "toolu_0",
            "name": slot["name"],
            "input": parsed_args,
        })
    return {
        "text": "".join(text_parts),
        "tool_calls": tool_calls,
        "finish": finish,
        "usage": usage,
        "saw_finish": saw_finish,
    }


def emit_openai_stream(collected, dialed_id, request_body):
    """Re-emit buffered gateway chunks as an OpenAI SSE body."""
    frames = []
    text = collected["text"]
    # One content chunk, then the finish chunk, then usage, then [DONE].
    frames.append({
        "id": "chatcmpl_0",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": dialed_id,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text},
                     "finish_reason": None}],
    })
    if collected["tool_calls"]:
        delta_calls = [{
            "index": i,
            "id": call["id"],
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": json.dumps(call["input"], separators=(",", ":")),
            },
        } for i, call in enumerate(collected["tool_calls"])]
        frames.append({
            "id": "chatcmpl_0",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": dialed_id,
            "choices": [{"index": 0, "delta": {"tool_calls": delta_calls},
                         "finish_reason": None}],
        })
    finish = collected["finish"] or "stop"
    frames.append({
        "id": "chatcmpl_0",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": dialed_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
    })
    usage = collected["usage"]
    if isinstance(usage, dict) and "completion_tokens" in usage:
        out_tokens = usage["completion_tokens"]
        in_tokens = usage.get("prompt_tokens", estimate_tokens(request_body))
    else:
        synth = synth_usage(request_body, text)
        in_tokens = synth["prompt_tokens"]
        out_tokens = synth["completion_tokens"]
    frames.append({
        "id": "chatcmpl_0",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": dialed_id,
        "choices": [],
        "usage": {
            "prompt_tokens": in_tokens,
            "completion_tokens": max(1, out_tokens),
            "total_tokens": in_tokens + max(1, out_tokens),
        },
    })
    out = "".join(
        "data: %s\n\n" % json.dumps(f, separators=(",", ":"))
        for f in frames
    )
    return (out + "data: [DONE]\n\n").encode("utf-8")


def emit_anthropic_stream(collected, dialed_id, request_body, declared, msg_id):
    """Re-emit buffered gateway chunks as an Anthropic SSE body."""
    text = collected["text"]
    calls = [
        c for c in collected["tool_calls"]
        if not declared or c["name"] in declared
    ]
    usage = collected["usage"]
    if isinstance(usage, dict) and "completion_tokens" in usage:
        out_tokens = usage["completion_tokens"]
        in_tokens = usage.get("prompt_tokens", estimate_tokens(request_body))
    else:
        synth = synth_usage(request_body, text)
        in_tokens = synth["prompt_tokens"]
        out_tokens = synth["completion_tokens"]
    if calls:
        stop = "tool_use"
    elif collected["finish"]:
        stop = OPENAI_TO_ANTHROPIC_STOP.get(collected["finish"], "end_turn")
    else:
        stop = "end_turn"
    events = []
    events.append(("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": dialed_id,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": in_tokens, "output_tokens": 0},
        },
    }))
    index = 0
    if text or not calls:
        events.append(("content_block_start", {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        }))
        events.append(("content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        }))
        events.append(("content_block_stop", {
            "type": "content_block_stop",
            "index": index,
        }))
        index += 1
    for call in calls:
        events.append(("content_block_start", {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": call["id"],
                "name": call["name"],
                "input": {},
            },
        }))
        events.append(("content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps(call["input"], separators=(",", ":")),
            },
        }))
        events.append(("content_block_stop", {
            "type": "content_block_stop",
            "index": index,
        }))
        index += 1
    events.append(("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"output_tokens": max(1, out_tokens)},
    }))
    events.append(("message_stop", {"type": "message_stop"}))
    out = "".join(
        "event: %s\ndata: %s\n\n" % (name, json.dumps(data, separators=(",", ":")))
        for name, data in events
    )
    return out.encode("utf-8")


# --------------------------------------------------------------------------
# Serving one prompt: bandit pick + value-ordered failover chain.
# --------------------------------------------------------------------------

class Router:
    def __init__(self, models, seed):
        self.bandit = Bandit(models, seed)
        self.models = models
        self.lock = threading.Lock()
        self.msg_counter = 0

    def next_msg_id(self):
        self.msg_counter += 1
        return "msg_%06d" % self.msg_counter

    def serve(self, wire, body):
        """Serve one client request. Returns (status, content_type, payload, failures)."""
        with self.lock:
            return self._serve_locked(wire, body)

    def _serve_locked(self, wire, body):
        stream = body.get("stream") is True
        req = requirements_of(body, wire)
        first, chain = self.bandit.pick_first(req)
        if not chain:
            # No model meets the requirements: try every model in
            # value order rather than giving up the prompt outright.
            chain = self.bandit.order(self.models)
            first = chain[0] if chain else None
        if first is None:
            payload = json.dumps({
                "error": {
                    "message": "no free model available",
                    "type": "upstream_error",
                    "code": "no_model_available",
                },
            }, separators=(",", ":")).encode("utf-8")
            return 502, "application/json", payload, []
        declared = declared_tool_names(body, wire)
        gw_body = to_openai_body(body, wire)
        failures = []
        attempts = 0
        index = 0
        retried_same = False
        while index < len(chain) and attempts < MAX_ATTEMPTS_PER_PROMPT:
            model = chain[index]
            attempts += 1
            try:
                status, raw = gateway_call(model["id"], gw_body, stream)
            except (OSError, socket.timeout, http.client.HTTPException):
                failures.append({
                    "status": 504,
                    "code": "timeout",
                    "model": model["id"],
                })
                self.bandit.record(model, 0.0)
                index += 1
                retried_same = False
                continue
            if status == 429 or 500 <= status < 600:
                code = error_code(raw)
                if code is None:
                    code = "rate_limited" if status == 429 else "upstream_error"
                failures.append({
                    "status": status,
                    "code": code,
                    "model": model["id"],
                })
                self.bandit.record(model, 0.0)
                if status == 429:
                    index += 1
                    retried_same = False
                elif not retried_same:
                    retried_same = True
                else:
                    index += 1
                    retried_same = False
                continue
            if status < 200 or status >= 300:
                code = error_code(raw) or "request_error"
                failures.append({
                    "status": status,
                    "code": code,
                    "model": model["id"],
                })
                self.bandit.record(model, 0.0)
                index += 1
                retried_same = False
                continue
            # Success: bank the rating reward and shape the answer.
            self.bandit.record(model, self.bandit.prior(model))
            if stream:
                collected = collect_stream(raw)
                collected["tool_calls"] = [
                    c for c in collected["tool_calls"]
                    if not declared or c["name"] in declared
                ]
                if wire == "openai":
                    payload = emit_openai_stream(collected, model["id"], body)
                else:
                    payload = emit_anthropic_stream(
                        collected, model["id"], body, declared,
                        self.next_msg_id())
                return 200, "text/event-stream", payload, failures
            if wire == "openai":
                payload = shape_openai_answer(raw, model["id"], body, declared)
            else:
                payload = shape_anthropic_answer(
                    raw, model["id"], body, declared, self.next_msg_id())
            return 200, "application/json", payload, failures
        # Every model in the chain refused: give up, with the trail.
        payload = json.dumps({
            "error": {
                "message": "all eligible models refused",
                "type": "upstream_error",
                "code": "all_models_refused",
            },
        }, separators=(",", ":")).encode("utf-8")
        return 502, "application/json", payload, failures


# --------------------------------------------------------------------------
# HTTP server: both client wires, both model-list shapes.
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    router = None
    models = []
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, ctype, payload, failures=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        if failures:
            trail = json.dumps(failures[:8], separators=(",", ":"))
            self.send_header("x-makefun-upstream-failures", trail)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if urllib.parse.urlparse(self.path).path != "/v1/models":
            self._send(404, "application/json",
                       b'{"error":{"message":"not found","type":"not_found"}}')
            return
        ids = [m["id"] for m in self.models]
        if "x-api-key" in self.headers:
            data = [{
                "id": mid,
                "type": "model",
                "display_name": mid,
                "created_at": "2026-11-15T03:30:00Z",
            } for mid in ids]
            payload = json.dumps({
                "data": data,
                "has_more": False,
                "object": "list",
            }, separators=(",", ":")).encode("utf-8")
        else:
            data = [{
                "id": mid,
                "object": "model",
                "created": 0,
                "owned_by": "openrouter",
            } for mid in ids]
            payload = json.dumps({
                "object": "list",
                "data": data,
            }, separators=(",", ":")).encode("utf-8")
        self._send(200, "application/json", payload)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/v1/chat/completions":
            wire = "openai"
        elif path == "/v1/messages":
            wire = "anthropic"
        else:
            self._send(404, "application/json",
                       b'{"error":{"message":"not found","type":"not_found"}}')
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        status, ctype, payload, failures = self.router.serve(wire, body)
        self._send(status, ctype, payload, failures)


# --------------------------------------------------------------------------
# Main: --list, register, serve, poll /work until done.
# --------------------------------------------------------------------------

def http_json(method, url, payload=None, timeout=10.0):
    parsed = urllib.parse.urlparse(url)
    use_https = parsed.scheme == "https"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if use_https else 80)
    conn_cls = http.client.HTTPSConnection if use_https else http.client.HTTPConnection
    conn = conn_cls(host, port, timeout=timeout)
    try:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        conn.request(method, parsed.path or "/", body=data, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def pick_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def main(argv):
    models = load_models(os.environ.get("MAKEFUN_PROMPT_FILE", ""))
    if "--list" in argv:
        if models:
            sys.stdout.write(catalog_json(models) + "\n")
        else:
            sys.stdout.write('{"providers":[]}\n')
        return 0
    if not models:
        sys.stderr.write("free-model-aggregator-02-02: no models in brief\n")
        return 1
    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    if not env_url:
        sys.stderr.write("free-model-aggregator-02-02: MAKEFUN_ENV_URL unset\n")
        return 1
    port = pick_port()
    catalog = build_catalog(models)
    # Start serving BEFORE registering: the env verifies GET /v1/models
    # the moment register lands, and a not-yet-listening server loses
    # the first (OpenAI) check.
    router = Router(models, os.environ.get("MAKEFUN_TRIAL_SEED", ""))
    Handler.router = router
    Handler.models = models
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    sys.stderr.write("listening on %d\n" % port)
    status, raw = http_json(
        "POST",
        env_url.rstrip("/") + "/register",
        {"protocol_version": 1, "port": port, "catalog": catalog},
    )
    if status != 200:
        sys.stderr.write("register failed: %s %s\n" % (status, raw[:200]))
        server.shutdown()
        server.server_close()
        return 1
    try:
        while True:
            try:
                status, raw = http_json(
                    "GET", env_url.rstrip("/") + "/work", timeout=5.0)
                if status == 200:
                    work = json.loads(raw.decode("utf-8", "replace"))
                    if isinstance(work, dict) and work.get("done") is True:
                        break
            except (OSError, ValueError, http.client.HTTPException):
                pass
            time.sleep(0.5)
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
