import json
import os
import pathlib
import pickle
import re
import time

import openai
from llm_logging import get_llm_provider, log_llm_call

ALLOWED_CLASSES = ("String", "General", "Numbers", "Algorithmic")

BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute()
CODE_BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.absolute()
DFX_CLASSES_PATH = str(CODE_BASE_PATH / "classifier" / "DFX_classes.csv")
TDE_CLASSES_PATH = str(CODE_BASE_PATH / "classifier" / "TDE_classes.csv")

ALL_CLASSES_JSON = str(BASE_PATH / "data/Classes/gpt_classified.json")
PROMPT_CACHE_PATH = BASE_PATH / "cache/classifier_prompts"

DFX_CLASSES = {}
TDE_CLASSES = {}


def get_gold_label(tbl_name, ds_path):
    ds_name = pathlib.Path(ds_path).name

    if ds_name in ("AutoJoin", "FlashFill"):
        return "String"

    elif ds_name == "DataXFormer":
        if len(DFX_CLASSES) < 80:
            with open(DFX_CLASSES_PATH, 'r') as f:
                lines = f.readlines()
            rows = [line.strip().split(',') for line in lines]
            for row in rows:
                assert len(row) == 2
                assert row[1] in ALLOWED_CLASSES
                DFX_CLASSES[row[0]] = row[1]

        return DFX_CLASSES[tbl_name]

    elif ds_name == "All_TDE":
        if len(TDE_CLASSES) < 229:
            with open(TDE_CLASSES_PATH, 'r') as f:
                lines = f.readlines()
            rows = [line.strip().split(',') for line in lines]
            for row in rows:
                assert row[1] in ALLOWED_CLASSES
                TDE_CLASSES[row[0]] = row[1]

        return TDE_CLASSES[tbl_name]

    else:
        raise NotImplementedError(f"The {ds_name} dataset has no golden labels")


def _read_secret(env_name: str, file_name: str) -> str:
    """Read an API key from an environment variable, falling back to a repo-local key file."""
    val = os.getenv(env_name, "").strip()
    if val:
        return val

    path = BASE_PATH / file_name
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace").strip()

    raise ValueError(
        f"Missing {env_name}. Set it in CMD or create {path}."
    )


def _openrouter_model_name(model_name: str | None) -> str:
    override = os.getenv("OPENROUTER_MODEL", "").strip()
    if override:
        return override

    model_name = model_name or "gpt-4o-2024-05-13"
    if "gpt-4o-mini" in model_name:
        return "openai/gpt-4o-mini"
    if "gpt-4o" in model_name:
        return "openai/gpt-4o"
    return model_name


def _get_client_and_model(model_name: str | None):
    """Return an OpenAI-compatible client plus the model name to send to the API."""
    model_name = model_name or "gpt-4o-2024-05-13"

    if os.getenv("USE_OPENROUTER", "0") == "1":
        return openai.OpenAI(
            api_key=_read_secret("OPENROUTER_API_KEY", "openrouter.key"),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-OpenRouter-Title": "TabulaX replication",
            },
        ), _openrouter_model_name(model_name)

    return openai.OpenAI(api_key=_read_secret("OPENAI_API_KEY", "openai.key")), model_name


def _prompt_model_dir(model_name: str | None) -> str:
    """Map API model names to the prompt directory used by the TabulaX repo."""
    model_name = model_name or "gpt-4o-2024-05-13"
    if "gpt-4o" in model_name:
        return "gpt-4o"
    # Fallback to the strongest/default prompt template.
    return "gpt-4o"


def _cache_file_name(model_name: str | None, prompt_version: str) -> pathlib.Path:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name or "default")
    return PROMPT_CACHE_PATH / f"{safe_model}_{prompt_version}.pkl"


def _load_json_cache() -> dict:
    path = pathlib.Path(ALL_CLASSES_JSON)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def _save_json_cache(labels_dict: dict) -> None:
    path = pathlib.Path(ALL_CLASSES_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels_dict, f, indent=2, ensure_ascii=False)


def _predict_class_from_examples(
    examples,
    model_name: str | None = None,
    prompt_version: str = "v002",
    tbl_name: str | None = None,
    ds_name: str | None = None,
) -> str:
    """
    Classify a table live from source->target example pairs.

    examples can contain 2-tuples (src, target) or 3-tuples (src, target, raw_src).
    """
    PROMPT_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_file_name(model_name, prompt_version)

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as fp:
            cache_dict = pickle.load(fp)
    else:
        cache_dict = {}

    prompt_dir = _prompt_model_dir(model_name)
    prompt_path = CODE_BASE_PATH / f"classifier/prompts/{prompt_dir}/class_prompt_{prompt_version}.txt"
    if not prompt_path.exists() and prompt_version != "v001":
        # Some checkouts only have v001.
        prompt_version = "v001"
        prompt_path = CODE_BASE_PATH / f"classifier/prompts/{prompt_dir}/class_prompt_{prompt_version}.txt"

    with open(prompt_path, "r", encoding="utf-8", errors="replace") as f:
        prompt_template = f.read()

    str_examp = ""
    for exp in examples:
        str_examp += f"(\"{exp[0]}\" -> \"{exp[1]}\"),"

    prompt = prompt_template.format(examples=str_examp)
    messages = [{"role": "user", "content": prompt}]
    provider = get_llm_provider(model_name)
    api_model_name = _openrouter_model_name(model_name) if provider == "openrouter" else model_name

    if prompt in cache_dict:
        completion = cache_dict[prompt]
        respond = completion.choices[0].message.content
        log_llm_call(
            "classifier",
            model_name,
            prompt=prompt,
            messages=messages,
            response=respond,
            cached=True,
            provider=provider,
            api_model=api_model_name,
            completion=completion,
            table=tbl_name,
            dataset=ds_name,
        )
    else:
        client, api_model_name = _get_client_and_model(model_name)
        started = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model=api_model_name,
                messages=messages,
                temperature=0.0000001,
                seed=12345,
                max_tokens=100,
            )
        except Exception as exc:
            log_llm_call(
                "classifier",
                model_name,
                prompt=prompt,
                messages=messages,
                duration_sec=time.perf_counter() - started,
                success=False,
                error_message=repr(exc),
                cached=False,
                provider=provider,
                api_model=api_model_name,
                table=tbl_name,
                dataset=ds_name,
            )
            raise
        respond = completion.choices[0].message.content
        duration_sec = time.perf_counter() - started
        log_llm_call(
            "classifier",
            model_name,
            prompt=prompt,
            messages=messages,
            response=respond,
            duration_sec=duration_sec,
            cached=False,
            provider=provider,
            api_model=api_model_name,
            completion=completion,
            table=tbl_name,
            dataset=ds_name,
        )
        cache_dict[prompt] = completion

        with open(cache_file, "wb") as fp:
            pickle.dump(cache_dict, fp)

    out = re.split(r"\s+", respond.replace("Class:", "").strip())[0]
    if out in ALLOWED_CLASSES:
        return out

    raise ValueError(f"Invalid classifier output: {out!r}; full response={respond!r}")


def get_gpt_label(
    tbl_name,
    ds_path,
    train_examples=None,
    model_name: str | None = None,
    prompt_version: str = "v002",
):
    """
    Return the GPT-predicted transformation class.

    Behavior:
      1. If data/Classes/gpt_classified.json exists and contains tbl_name, use it.
      2. Otherwise, if train_examples are provided, classify live using the current
         TabulaX examples, then cache the prediction in gpt_classified.json.
      3. Otherwise, fail with an actionable message.
    """
    labels_dict = _load_json_cache()

    if tbl_name in labels_dict:
        prediction = labels_dict[tbl_name]["predicted_value"]
        log_llm_call(
            "classifier_lookup",
            model_name,
            response=prediction,
            cached=True,
            provider=get_llm_provider(model_name),
            table=tbl_name,
            dataset=pathlib.Path(ds_path).name,
        )
        return prediction

    ds_name = pathlib.Path(ds_path).name
    full_name = f"{ds_name}/{tbl_name}"
    for entry in labels_dict.values():
        if isinstance(entry, dict) and entry.get("full_name") == full_name:
            prediction = entry["predicted_value"]
            log_llm_call(
                "classifier_lookup",
                model_name,
                response=prediction,
                cached=True,
                provider=get_llm_provider(model_name),
                table=tbl_name,
                dataset=ds_name,
            )
            return prediction

    if train_examples is None:
        raise FileNotFoundError(
            f"No cached GPT class found for {tbl_name!r} in {ALL_CLASSES_JSON}, "
            "and no train_examples were provided for live classification."
        )

    prediction = _predict_class_from_examples(
        train_examples,
        model_name=model_name,
        prompt_version=prompt_version,
        tbl_name=tbl_name,
        ds_name=ds_name,
    )

    labels_dict[tbl_name] = {
        "golden_value": None,
        "predicted_value": prediction,
        "full_name": full_name,
    }
    _save_json_cache(labels_dict)

    print(f"[TabulaX classifier] {tbl_name} -> {prediction} (live; cached to {ALL_CLASSES_JSON})")
    return prediction
