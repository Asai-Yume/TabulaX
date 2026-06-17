import os
import pathlib
import pickle
import time

import openai
from llm_logging import get_llm_provider, log_llm_call


USE_TQDM = False

BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute()
CODE_BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.absolute()
REL_PROMPT_CACHE_PATH = BASE_PATH / "cache/gen_rel_prompts"
BRIDGE_PROMPT_CACHE_PATH = BASE_PATH / "cache/gen_bridge_prompts"
# @TODO: Caching ignores temp and other running params


def _read_key_file(path: pathlib.Path) -> str:
    """Read an API key file if it exists; otherwise return an empty string."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


API_KEY = os.getenv("OPENAI_API_KEY", "").strip() or _read_key_file(BASE_PATH / "openai.key")
OPENROUTER_API_KEY = (
    os.getenv("OPENROUTER_API_KEY", "").strip()
    or _read_key_file(BASE_PATH / "openrouter.key")
)


def get_openrouter_model_name(model_name: str) -> str:
    """
    Map TabulaX model names to OpenRouter model names.

    Keep MODEL_NAME as a TabulaX-style name such as
    gpt-4o-mini-2024-07-18 so prompt folders still resolve correctly.
    Use OPENROUTER_MODEL for the actual remote model ID.
    """
    override = os.getenv("OPENROUTER_MODEL", "").strip()
    if override:
        return override

    if model_name.startswith("gpt-4o-mini"):
        return "openai/gpt-4o-mini"
    if model_name.startswith("gpt-4o"):
        return "openai/gpt-4o"

    return "openai/gpt-4o-mini"


def get_gpt_client_and_model(model_name: str):
    """
    Return an OpenAI-compatible client and the actual API model name.

    Supports:
      - USE_OPENROUTER=1 with OPENROUTER_MODEL=openai/gpt-4o-mini
      - direct OpenAI API with OPENAI_API_KEY/openai.key
      - local llama3.1-8b server
    """
    if os.getenv("USE_OPENROUTER", "0") == "1":
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "USE_OPENROUTER=1 but no OpenRouter key was found. "
                "Set OPENROUTER_API_KEY or create openrouter.key at the TabulaX repo root."
            )
        return openai.OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-OpenRouter-Title": "TabulaX replication",
            },
            max_retries=6,
            timeout=120.0,
        ), get_openrouter_model_name(model_name)

    if model_name.startswith("gpt"):
        if not API_KEY:
            raise ValueError(
                "No OpenAI API key was found. Set OPENAI_API_KEY, create openai.key, "
                "or set USE_OPENROUTER=1."
            )
        return openai.OpenAI(
            api_key=API_KEY,
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

    raise NotImplementedError(f"Model {model_name} not implemented")


bridge_cache_dict = None
bridge_cache_file = None


def get_relation(examples, model_name, prompt_version):
    REL_PROMPT_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    cache_file = REL_PROMPT_CACHE_PATH / f"{model_name}.pkl"
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

    with open(CODE_BASE_PATH / f"transformers/prompts/{mdl}/gen_rel_prompt_{prompt_version}.txt") as f:
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
        respond = completion.choices[0].message.content
        log_llm_call(
            "general_relation",
            model_name,
            prompt=prompt,
            messages=messages,
            response=respond,
            cached=True,
            provider=provider,
            api_model=api_model_name,
            completion=completion,
        )
        # print("Hit cache")
    else:
        client, api_model_name = get_gpt_client_and_model(model_name)

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
                "general_relation",
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
            "general_relation",
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

    out = respond.replace("Relationship:", "").strip()
    return out


def predict_bridge_value(examples, src, relation_array, model_name, prompt_version, sleep=-1):

    mdl = model_name
    if mdl.startswith("gpt-4o-"):
        mdl = "gpt-4o"
    if mdl == "llama3.1-8b":
        mdl = "llama3"

    with open(CODE_BASE_PATH / f"transformers/prompts/{mdl}/gen_bridge_prompt_{prompt_version}.txt") as f:
        pmpt = f.read()

    str_examp = ""

    for exp in examples:
        str_examp += f"{exp[0]} -> {exp[1]}\n"

    prompt = pmpt.format(examples=str_examp, src_type=relation_array[0], target_type=relation_array[1], src_value=src)
    messages = [{"role": "user", "content": prompt}]
    provider = get_llm_provider(model_name)
    api_model_name = get_openrouter_model_name(model_name) if provider == "openrouter" else model_name
    # print(f"=========\n{prompt}\n***")
    if prompt in bridge_cache_dict:
        completion = bridge_cache_dict[prompt]
        respond = completion.choices[0].message.content
        log_llm_call(
            "general_predict_bridge",
            model_name,
            prompt=prompt,
            messages=messages,
            response=respond,
            cached=True,
            provider=provider,
            api_model=api_model_name,
            completion=completion,
            src_value=src,
            src_type=relation_array[0],
            target_type=relation_array[1],
        )
        # print("Hit cache")
    else:
        client, api_model_name = get_gpt_client_and_model(model_name)

        started = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model=api_model_name,
                messages=messages,
                temperature=0.0000001,
                seed=12345,
                max_tokens=100,
                # frequency_penalty=0.0
            )
        except Exception as exc:
            log_llm_call(
                "general_predict_bridge",
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
                src_type=relation_array[0],
                target_type=relation_array[1],
            )
            raise
        respond = completion.choices[0].message.content
        duration_sec = time.perf_counter() - started
        log_llm_call(
            "general_predict_bridge",
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
            src_type=relation_array[0],
            target_type=relation_array[1],
        )
        bridge_cache_dict[prompt] = completion

        with open(bridge_cache_file, 'wb') as fp:
            pickle.dump(bridge_cache_dict, fp)

        if sleep > 0:
            time.sleep(sleep)

    out = respond.strip()
    if model_name == "llama3.1-8b":
        out = out.split("-> ")[-1].strip()
    # print(out)

    return out


def get_bridge_values(examples, test, params):

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
    relation = get_relation(exmps, model_name, prompt_version)
    # tmp = relation.split(" to ")
    tmp = relation.split("\"),")[-1].strip().split(" to ")
    if len(tmp) != 2:
        import sys
        print(f" **** Relation {relation} is not valid", file=sys.stderr)
        tmp = ["Unknown", "Unknown"]
        # raise ValueError(f"Relation {relation} is not valid")

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
        bridge = predict_bridge_value(exmps, exp[2], tmp, model_name, prompt_version)
        test_new.append((bridge, exp[1], exp[2]))

    return test_new, {
        'relation': relation,
    }
