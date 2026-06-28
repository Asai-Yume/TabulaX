import os
import pathlib

import openai


BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute()


def read_secret(env_name: str, file_name: str) -> str:
    val = os.getenv(env_name, "").strip()
    if val:
        return val

    path = BASE_PATH / file_name
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace").strip()

    return ""


def get_openrouter_model_name(model_name: str | None) -> str:
    override = os.getenv("OPENROUTER_MODEL", "").strip()
    if override:
        return override

    model_name = model_name or "gpt-4o-2024-05-13"

    if model_name.startswith("gpt-4o-mini"):
        return "openai/gpt-4o-mini"
    if model_name.startswith("gpt-4o"):
        return "openai/gpt-4o"

    return "openai/gpt-4o-mini"


def get_client_and_model(model_name: str | None):
    model_name = model_name or "gpt-4o-2024-05-13"

    if os.getenv("USE_OPENROUTER", "0") == "1":
        api_key = read_secret("OPENROUTER_API_KEY", "openrouter.key")
        if not api_key:
            raise ValueError(
                "USE_OPENROUTER=1 but no OpenRouter key was found. "
                "Set OPENROUTER_API_KEY or create openrouter.key at the TabulaX repo root."
            )

        return openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-OpenRouter-Title": "TabulaX replication",
            },
            max_retries=6,
            timeout=120.0,
        ), get_openrouter_model_name(model_name)

    if model_name.startswith("gpt"):
        api_key = read_secret("OPENAI_API_KEY", "openai.key")
        if not api_key:
            raise ValueError(
                "No OpenAI API key was found. Set OPENAI_API_KEY, create openai.key, "
                "or set USE_OPENROUTER=1."
            )

        return openai.OpenAI(
            api_key=api_key,
            max_retries=6,
            timeout=120.0,
        ), model_name

    if model_name == "llama3.1-8b":
        return openai.OpenAI(
            api_key="None",
            base_url="http://localhost:8000/v1",
            max_retries=6,
            timeout=120.0,
        ), "meta-llama/Llama-3.1-8B-Instruct"

    if model_name.startswith("deepseek"):
        api_key = read_secret("DEEPSEEK_API_KEY", "deepseek.key")
        if not api_key:
            raise ValueError(
                "No DeepSeek API key was found. Set DEEPSEEK_API_KEY or create deepseek.key."
            )

        return openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            max_retries=6,
            timeout=120.0,
        ), model_name

    raise NotImplementedError(f"Model {model_name} not implemented")


def is_openrouter_gpt5_model(api_model_name: str | None) -> bool:
    return (
        os.getenv("USE_OPENROUTER", "0") == "1"
        and api_model_name is not None
        and "gpt-5" in api_model_name.lower()
    )


def chat_completion_kwargs(api_model_name: str, messages: list[dict], max_tokens: int = 1000) -> dict:
    kwargs = {
        "model": api_model_name,
        "messages": messages,
    }

    if is_openrouter_gpt5_model(api_model_name):
        # GPT-5 models may spend output budget on hidden reasoning.
        # Use an env override so we can tune without editing code.
        kwargs["max_tokens"] = int(os.getenv("OPENROUTER_MAX_TOKENS", str(max(4096, max_tokens))))
        kwargs["extra_body"] = {
            "reasoning": {
                "effort": os.getenv("OPENROUTER_REASONING_EFFORT", "low"),
                "exclude": True,
            }
        }
    else:
        kwargs["temperature"] = 0.0000001
        kwargs["seed"] = 12345
        kwargs["max_tokens"] = max_tokens

    return kwargs


def extract_response_text(completion) -> str | None:
    if completion is None or not getattr(completion, "choices", None):
        return None

    message = completion.choices[0].message
    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in ("text", "output_text"):
                    parts.append(item.get("text", ""))
                elif "text" in item:
                    parts.append(item["text"])
        joined = "".join(parts).strip()
        return joined or None

    return None


def require_response_text(respond, stage: str, api_model_name: str, completion=None) -> str:
    if respond is not None and str(respond).strip():
        return respond

    finish_reason = None
    try:
        finish_reason = completion.choices[0].finish_reason
    except Exception:
        pass

    raise RuntimeError(
        f"{stage} returned no visible text. "
        f"api_model={api_model_name!r}, finish_reason={finish_reason!r}. "
        "For GPT-5-mini, increase OPENROUTER_MAX_TOKENS or lower reasoning effort."
    )