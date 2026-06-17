import os
import pandas as pd


def load_manifest_examples(pair_id: str, example_size: int):
    """
    Load fixed source->target examples from a CSV manifest.

    Expected columns:
      pair_id, seed, source_value, target_value
    Optional:
      dataset, method, score, is_gt, q
    """
    manifest_path = os.getenv("TABULAX_EXAMPLE_MANIFEST")
    if not manifest_path:
        return None

    seed = int(os.getenv("TABULAX_EXAMPLE_SEED", "0"))

    df = pd.read_csv(manifest_path, dtype=str)

    if "pair_id" in df.columns:
        df = df[df["pair_id"].astype(str) == str(pair_id)]

    if "seed" in df.columns:
        df = df[df["seed"].astype(str) == str(seed)]

    if df.empty:
        raise ValueError(
            f"No examples found in manifest={manifest_path!r} "
            f"for pair_id={pair_id!r}, seed={seed}"
        )

    df = df.head(example_size)

    examples = []
    for _, row in df.iterrows():
        source = str(row["source_value"])
        target = str(row["target_value"])
        examples.append((source, target, source))

    print(
        f"[TabulaX manifest] using {len(examples)} fixed examples "
        f"from {manifest_path} for pair_id={pair_id}, seed={seed}"
    )

    for i, (s, t, _) in enumerate(examples):
        print(f"  example {i}: {s!r} -> {t!r}")

    return examples


def apply_manifest_split(table: dict, pair_id: str, example_size: int) -> dict:
    """
    Replace TabulaX's random train examples with fixed manifest examples.

    TabulaX table format:
      table['train'] = [(source, target, raw_source), ...]
      table['test']  = [(source, target, raw_source), ...]
    """
    manifest_examples = load_manifest_examples(pair_id, example_size)
    if manifest_examples is None:
        return table

    all_rows = list(table.get("train", [])) + list(table.get("test", []))

    manifest_pairs = {
        (str(source), str(target))
        for source, target, _ in manifest_examples
    }

    new_test = []
    for row in all_rows:
        source = str(row[0])
        target = str(row[1])
        if (source, target) not in manifest_pairs:
            new_test.append(row)

    table["train"] = manifest_examples
    table["test"] = new_test

    print(
        f"[TabulaX manifest] train={len(table['train'])}, "
        f"test={len(table['test'])}"
    )

    return table