import os
import pathlib
import pickle
import re
import time

import openai
from llm_logging import get_llm_provider, log_llm_call

BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute()
CODE_BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.absolute()
CODE_PROMPT_CACHE_PATH = BASE_PATH / "cache/str_code_prompts"
# @TODO: Caching ignores temp and other running params

with open(BASE_PATH / 'openai.key', 'r') as f:
    API_KEY = f.read()

with open(BASE_PATH / 'openrouter.key', 'r') as f:
    OPENROUTER_API_KEY = f.read().strip()


def get_openrouter_model_name(model_name):
    override = os.getenv("OPENROUTER_MODEL", "").strip()
    if override:
        return override

    if model_name.startswith("gpt-4o-mini"):
        return "openai/gpt-4o-mini"
    if model_name.startswith("gpt-4o"):
        return "openai/gpt-4o"

    return "openai/gpt-4o-mini"


def get_gpt_client_and_model(model_name):
    if os.getenv("USE_OPENROUTER", "0") == "1":
        return openai.OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-OpenRouter-Title": "TabulaX replication",
            },
        ), get_openrouter_model_name(model_name)

    return openai.OpenAI(api_key=API_KEY), model_name    


def is_openrouter_gpt5_model(api_model_name):
    return (
        os.getenv("USE_OPENROUTER", "0") == "1"
        and api_model_name is not None
        and "gpt-5" in api_model_name.lower()
    )


def chat_completion_kwargs(api_model_name, messages):
    kwargs = {
        "model": api_model_name,
        "messages": messages,
    }

    if is_openrouter_gpt5_model(api_model_name):
        kwargs["max_tokens"] = int(os.getenv("OPENROUTER_MAX_TOKENS", "4096"))
        kwargs["extra_body"] = {
            "reasoning": {
                "effort": os.getenv("OPENROUTER_REASONING_EFFORT", "low"),
                "exclude": True,
            }
        }
    else:
        kwargs["temperature"] = 0.0000001
        kwargs["seed"] = 12345
        kwargs["max_tokens"] = 1000

    return kwargs


def extract_response_text(completion):
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


with open(BASE_PATH / 'deepseek.key', 'r') as f:
    DEEPSEEK_API_KEY = f.read()





def get_code(examples, model_name, prompt_version):
    cache_file = CODE_PROMPT_CACHE_PATH / f"{model_name}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fp:
            cache_dict = pickle.load(fp)
    else:
        cache_dict = {}

    mdl = model_name
    if mdl.startswith("gpt-4o-"):
        mdl = "gpt-4o"
    if mdl == "llama3.1-8b":
        mdl = "llama3"
    if mdl.startswith("deepseek"):
        mdl = "deepseek"

    with open(CODE_BASE_PATH / f"transformers/prompts/{mdl}/str_code_prompt_{prompt_version}.txt") as f:
        pmpt = f.read()

    str_examp = ""

    for exp in examples:
        str_examp += f"Input: \"{exp[0]}\"\nExpected Output:\"{exp[1]}\"\n***\n"

    prompt = pmpt.format(examples=str_examp)
    messages = [{"role": "user", "content": prompt}]


    if prompt in cache_dict:
        completion = cache_dict[prompt]
        respond = extract_response_text(completion)
        api_model_name = get_openrouter_model_name(model_name) if get_llm_provider(model_name) == "openrouter" else model_name
        log_llm_call(
            "string_transformer",
            model_name,
            prompt=prompt,
            messages=messages,
            response=respond,
            cached=True,
            provider=get_llm_provider(model_name),
            api_model=api_model_name,
            completion=completion,
        )
        # print("Hit cache")
    else:
        api_model_name = model_name
        provider = get_llm_provider(model_name)
        if model_name.startswith("gpt"):
            client, api_model_name = get_gpt_client_and_model(model_name)
        elif model_name == "llama3.1-8b":
            client = openai.OpenAI(
                api_key="None",
                base_url="http://localhost:8000/v1",
            )
            api_model_name = "meta-llama/Llama-3.1-8B-Instruct"
        elif model_name.startswith("deepseek"):
            client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")

        started = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                **chat_completion_kwargs(api_model_name, messages)
            )   
        except Exception as exc:
            log_llm_call(
                "string_transformer",
                model_name,
                prompt=prompt,
                messages=messages,
                duration_sec=time.perf_counter() - started,
                success=False,
                error_message=repr(exc),
                cached=False,
                provider=provider,
                api_model=api_model_name,
            )
            raise
        respond = extract_response_text(completion)
        duration_sec = time.perf_counter() - started
        log_llm_call(
            "string_transformer",
            model_name,
            prompt=prompt,
            messages=messages,
            response=respond,
            duration_sec=duration_sec,
            cached=False,
            provider=provider,
            api_model=api_model_name,
            completion=completion,
        )
        
        if respond is None or not str(respond).strip():
            finish_reason = None
            try:
                finish_reason = completion.choices[0].finish_reason
            except Exception:
                pass

            raise RuntimeError(
                "String transformer returned no visible text. "
                f"api_model={api_model_name!r}, finish_reason={finish_reason!r}. "
                "For GPT-5-mini, increase OPENROUTER_MAX_TOKENS or lower reasoning effort."
            )
        
        cache_dict[prompt] = completion

        with open(cache_file, 'wb') as fp:
            pickle.dump(cache_dict, fp)

    if respond is None or not str(respond).strip():
        raise RuntimeError(
            "String transformer response is empty after cache/live extraction. "
            "Clear cache\\str_code_prompts and rerun."
        )

    return respond, {
        'prompt': prompt,
        'respond': respond,
    }




def get_string_function(examples, model_name, prompt_version):
    # (bridge, target, src)
    exmps = [(exp[2], exp[1]) for exp in examples]
    func, detail = get_code(exmps, model_name, prompt_version)


    return detail, func

