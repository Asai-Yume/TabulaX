import os
import pathlib
import pickle
import time

from llm_logging import get_llm_provider, log_llm_call
from transformers.llm_common import (
    chat_completion_kwargs,
    extract_response_text,
    get_client_and_model,
    get_openrouter_model_name,
    require_response_text,
)


USE_TQDM = False

BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute()
CODE_BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.absolute()
BRIDGE_PROMPT_CACHE_PATH = BASE_PATH / "cache/basic_bridge_prompts"
# @TODO: Caching ignores temp and other running params

bridge_cache_dict = None
bridge_cache_file = None


def _prompt_model_dir(model_name):
    mdl = model_name
    if mdl.startswith("gpt-4o-"):
        mdl = "gpt-4o"
    if mdl == "llama3.1-8b":
        mdl = "llama3"
    if mdl.startswith("deepseek"):
        mdl = "deepseek"
    return mdl


def predict_bridge_value(examples, src, model_name, prompt_version, sleep=-1):

    mdl = _prompt_model_dir(model_name)

    with open(CODE_BASE_PATH / f"transformers/prompts/{mdl}/basic_bridge_prompt_{prompt_version}.txt") as f:
        pmpt = f.read()

    str_examp = ""
    for exp in examples:
        str_examp += f"{exp[0]} -> {exp[1]}\n"

    prompt = pmpt.format(examples=str_examp, src_value=src)
    messages = [{"role": "user", "content": prompt}]
    provider = get_llm_provider(model_name)
    api_model_name = get_openrouter_model_name(model_name) if provider == "openrouter" else model_name

    if prompt in bridge_cache_dict:
        completion = bridge_cache_dict[prompt]
        respond = extract_response_text(completion)
        log_llm_call(
            "basic_predict_bridge",
            model_name,
            prompt=prompt,
            messages=messages,
            response=respond,
            cached=True,
            provider=provider,
            api_model=api_model_name,
            completion=completion,
            src_value=src,
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
                "basic_predict_bridge",
                model_name,
                prompt=prompt,
                messages=messages,
                duration_sec=time.perf_counter() - started,
                success=False,
                error_message=repr(exc),
                cached=False,
                provider=provider,
                api_model=api_model_name,
                src_value=src,
            )
            raise

        respond = extract_response_text(completion)
        duration_sec = time.perf_counter() - started
        log_llm_call(
            "basic_predict_bridge",
            model_name,
            prompt=prompt,
            messages=messages,
            response=respond,
            duration_sec=duration_sec,
            cached=False,
            provider=provider,
            api_model=api_model_name,
            completion=completion,
            src_value=src,
        )

        respond = require_response_text(respond, "basic_predict_bridge", api_model_name, completion)
        bridge_cache_dict[prompt] = completion

        with open(bridge_cache_file, 'wb') as fp:
            pickle.dump(bridge_cache_dict, fp)

        if sleep > 0:
            time.sleep(sleep)

    respond = require_response_text(respond, "basic_predict_bridge", api_model_name, completion)
    out = respond.strip()

    return out


def get_basic_values(examples, test, params):

    model_name = params["model_name"]
    prompt_version = params["prompt_version"]

    global bridge_cache_dict, bridge_cache_file

    BRIDGE_PROMPT_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    bridge_cache_file = BRIDGE_PROMPT_CACHE_PATH / f"{model_name}.pkl"
    if os.path.exists(bridge_cache_file):
        with open(bridge_cache_file, 'rb') as fp:
            bridge_cache_dict = pickle.load(fp)
    else:
        bridge_cache_dict = {}

    # (bridge, target, src)
    exmps = [(exp[2], exp[1]) for exp in examples]

    test_new = []
    try:
        if not USE_TQDM:
            raise ImportError
        import tqdm
        itr = tqdm.tqdm(test)
        print()
        time.sleep(0.1)
    except Exception:
        itr = test

    for exp in itr:
        bridge = predict_bridge_value(exmps, exp[2], model_name, prompt_version)
        test_new.append((bridge, exp[1], exp[2]))

    return test_new, {

    }