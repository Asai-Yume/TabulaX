import os
import pathlib
import pickle
import re
import time

from llm_logging import get_llm_provider, log_llm_call
from transformers.llm_common import (
    chat_completion_kwargs,
    extract_response_text,
    get_client_and_model,
    get_openrouter_model_name,
    require_response_text,
)

BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute()
CODE_BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.absolute()
REL_PROMPT_CACHE_PATH = BASE_PATH / "cache/alg_rel_prompts"
CODE_PROMPT_CACHE_PATH = BASE_PATH / "cache/alg_code_prompts"
# @TODO: Caching ignores temp and other running params


def _prompt_model_dir(model_name):
    mdl = model_name
    if mdl.startswith("gpt-4o-"):
        mdl = "gpt-4o"
    if mdl == "llama3.1-8b":
        mdl = "llama3"
    if mdl.startswith("deepseek"):
        mdl = "deepseek"
    return mdl


def get_relation(examples, model_name, prompt_version):
    REL_PROMPT_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    cache_file = REL_PROMPT_CACHE_PATH / f"{model_name}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fp:
            cache_dict = pickle.load(fp)
    else:
        cache_dict = {}

    mdl = _prompt_model_dir(model_name)

    with open(CODE_BASE_PATH / f"transformers/prompts/{mdl}/alg_rel_prompt_{prompt_version}.txt") as f:
        pmpt = f.read()

    str_examp = ""
    for exp in examples:
        str_examp += f"(\"{exp[0]}\" -> \"{exp[1]}\"),"

    prompt = pmpt.format(examples=str_examp)
    messages = [{"role": "user", "content": prompt}]
    provider = get_llm_provider(model_name)
    api_model_name = get_openrouter_model_name(model_name) if provider == "openrouter" else model_name

    if prompt in cache_dict:
        completion = cache_dict[prompt]
        respond = extract_response_text(completion)
        log_llm_call(
            "algorithmic_relation",
            model_name,
            prompt=prompt,
            messages=messages,
            response=respond,
            cached=True,
            provider=provider,
            api_model=api_model_name,
            completion=completion,
        )
    else:
        client, api_model_name = get_client_and_model(model_name)
        started = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                **chat_completion_kwargs(api_model_name, messages, max_tokens=100)
            )
        except Exception as exc:
            log_llm_call(
                "algorithmic_relation",
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
            "algorithmic_relation",
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

        respond = require_response_text(respond, "algorithmic_relation", api_model_name, completion)
        cache_dict[prompt] = completion

    with open(cache_file, 'wb') as fp:
        pickle.dump(cache_dict, fp)

    respond = require_response_text(respond, "algorithmic_relation", api_model_name, completion)
    out = respond.replace("Relationship:", "").strip()
    return out


def get_code(examples, relation, model_name, prompt_version):
    CODE_PROMPT_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    cache_file = CODE_PROMPT_CACHE_PATH / f"{model_name}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fp:
            cache_dict = pickle.load(fp)
    else:
        cache_dict = {}

    mdl = _prompt_model_dir(model_name)

    with open(CODE_BASE_PATH / f"transformers/prompts/{mdl}/alg_code_prompt_{prompt_version}.txt") as f:
        pmpt = f.read()

    str_examp = ""
    for exp in examples:
        str_examp += f"Input: \"{exp[0]}\"\nExpected Output:\"{exp[1]}\"\n***\n"

    prompt = pmpt.format(examples=str_examp, relation=relation)
    messages = [{"role": "user", "content": prompt}]
    provider = get_llm_provider(model_name)
    api_model_name = get_openrouter_model_name(model_name) if provider == "openrouter" else model_name

    if prompt in cache_dict:
        completion = cache_dict[prompt]
        respond = extract_response_text(completion)
        log_llm_call(
            "algorithmic_transformer",
            model_name,
            prompt=prompt,
            messages=messages,
            response=respond,
            cached=True,
            provider=provider,
            api_model=api_model_name,
            completion=completion,
        )
    else:
        client, api_model_name = get_client_and_model(model_name)
        started = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                **chat_completion_kwargs(api_model_name, messages, max_tokens=1000)
            )
        except Exception as exc:
            log_llm_call(
                "algorithmic_transformer",
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
            "algorithmic_transformer",
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

        respond = require_response_text(respond, "algorithmic_transformer", api_model_name, completion)
        cache_dict[prompt] = completion

    with open(cache_file, 'wb') as fp:
        pickle.dump(cache_dict, fp)

    respond = require_response_text(respond, "algorithmic_transformer", api_model_name, completion)

    return respond, {
        'relationship': relation,
        'prompt': prompt,
        'respond': respond,
    }


def get_algorithmic_function(examples, model_name, prompt_version):
    exmps = [(exp[0], exp[1]) for exp in examples]
    relation = get_relation(exmps, model_name, prompt_version)
    func, detail = get_code(exmps, relation, model_name, prompt_version)

    return detail, func