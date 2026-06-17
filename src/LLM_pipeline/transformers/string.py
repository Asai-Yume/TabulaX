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
        respond = completion.choices[0].message.content
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
                model=api_model_name,
                messages=messages,
                temperature=0.0000001,
                seed=12345,
                max_tokens=1000,
                # frequency_penalty=0.0
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
        respond = completion.choices[0].message.content
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
        cache_dict[prompt] = completion

        with open(cache_file, 'wb') as fp:
            pickle.dump(cache_dict, fp)


    return respond, {
        'prompt': prompt,
        'respond': respond,
    }




def get_string_function(examples, model_name, prompt_version):
    # (bridge, target, src)
    exmps = [(exp[2], exp[1]) for exp in examples]
    func, detail = get_code(exmps, model_name, prompt_version)


    return detail, func

