import json
import os
import pathlib
import threading
from datetime import datetime, timezone


_LOCK = threading.Lock()
_LOG_PATH = None
_SUMMARY_PATH = None
_CONTEXT = {}
_SUMMARY = {
    "total_events": 0,
    "live_calls": 0,
    "cache_hits": 0,
    "success": 0,
    "error": 0,
    "total_duration_sec": 0.0,
    "stages": {},
    "models": {},
    "token_usage": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    },
}


def configure_llm_logging(output_dir, **context):
    global _LOG_PATH, _SUMMARY_PATH, _CONTEXT
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = output_path / "llm_calls.jsonl"
    _SUMMARY_PATH = output_path / "llm_summary.json"
    _CONTEXT = dict(context)
    flush_llm_summary()


def set_llm_context(**context):
    _CONTEXT.update({k: v for k, v in context.items() if v is not None})


def get_llm_provider(model_name=None):
    if os.getenv("USE_OPENROUTER", "0") == "1":
        return "openrouter"
    if model_name == "llama3.1-8b":
        return "local_openai"
    if model_name and str(model_name).startswith("deepseek"):
        return "deepseek"
    if model_name and str(model_name).startswith("gpt"):
        return "openai"
    return "unknown"


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    return str(value)


def _usage_dict(completion=None, token_usage=None):
    usage = token_usage
    if usage is None and completion is not None:
        usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) if not isinstance(usage, dict) else usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) if not isinstance(usage, dict) else usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) if not isinstance(usage, dict) else usage.get("total_tokens", 0) or 0),
    }


def log_llm_call(
    stage,
    model,
    prompt=None,
    messages=None,
    response=None,
    duration_sec=0.0,
    success=True,
    error_message=None,
    cached=False,
    provider=None,
    api_model=None,
    completion=None,
    token_usage=None,
    **extra,
):
    if _LOG_PATH is None:
        return

    usage = _usage_dict(completion=completion, token_usage=token_usage)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **_CONTEXT,
        "stage": stage,
        "provider": provider or get_llm_provider(model),
        "model": model,
        "api_model": api_model,
        "cached": cached,
        "prompt": prompt,
        "messages": messages,
        "response_text": response,
        "duration_sec": round(float(duration_sec or 0.0), 6),
        "token_usage": usage,
        "success": bool(success),
        "error_message": error_message,
        **extra,
    }

    with _LOCK:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(event), ensure_ascii=False) + "\n")
        _update_summary(event)
        flush_llm_summary()


def _update_summary(event):
    _SUMMARY["total_events"] += 1
    if event["cached"]:
        _SUMMARY["cache_hits"] += 1
    else:
        _SUMMARY["live_calls"] += 1
    if event["success"]:
        _SUMMARY["success"] += 1
    else:
        _SUMMARY["error"] += 1
    _SUMMARY["total_duration_sec"] += float(event.get("duration_sec") or 0.0)

    stage = event.get("stage") or "unknown"
    _SUMMARY["stages"].setdefault(stage, {"events": 0, "live_calls": 0, "cache_hits": 0, "errors": 0, "duration_sec": 0.0})
    _SUMMARY["stages"][stage]["events"] += 1
    _SUMMARY["stages"][stage]["live_calls"] += 0 if event["cached"] else 1
    _SUMMARY["stages"][stage]["cache_hits"] += 1 if event["cached"] else 0
    _SUMMARY["stages"][stage]["errors"] += 0 if event["success"] else 1
    _SUMMARY["stages"][stage]["duration_sec"] += float(event.get("duration_sec") or 0.0)

    model = event.get("api_model") or event.get("model") or "unknown"
    _SUMMARY["models"].setdefault(model, {"events": 0, "live_calls": 0, "cache_hits": 0})
    _SUMMARY["models"][model]["events"] += 1
    _SUMMARY["models"][model]["live_calls"] += 0 if event["cached"] else 1
    _SUMMARY["models"][model]["cache_hits"] += 1 if event["cached"] else 0

    usage = event.get("token_usage") or {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        _SUMMARY["token_usage"][key] += int(usage.get(key) or 0)


def flush_llm_summary(**extra):
    if _SUMMARY_PATH is None:
        return
    summary = {
        **_CONTEXT,
        **_SUMMARY,
        "total_duration_sec": round(_SUMMARY["total_duration_sec"], 6),
        **extra,
    }
    with open(_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2, ensure_ascii=False)
