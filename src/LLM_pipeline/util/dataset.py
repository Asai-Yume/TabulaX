import csv
import os
import random

RANDOM_SEED = 12345


def _dataset_dirs(ds_path, tbl_names=None):
    """
    Return (table_id, table_dir) pairs.

    Original TabulaX layout:
        ds_path/
          table_a/
            rows.txt
            ground truth.csv

    Also supports pointing ds_path directly at one table directory:
        ds_path/
          rows.txt
          ground truth.csv
    """
    tbl_names = tbl_names or []

    if os.path.exists(os.path.join(ds_path, "rows.txt")) and \
       os.path.exists(os.path.join(ds_path, "ground truth.csv")):
        table_id = os.path.basename(os.path.normpath(ds_path))
        if len(tbl_names) == 0 or table_id in tbl_names:
            return [(table_id, ds_path)]
        return []

    dirs = [
        dI for dI in os.listdir(ds_path)
        if os.path.isdir(os.path.join(ds_path, dI))
    ]

    result = []
    for dir_name in dirs:
        if len(tbl_names) > 0 and dir_name not in tbl_names:
            continue
        result.append((dir_name, os.path.join(ds_path, dir_name)))

    return result


def get_pairs_from_files(ds_path, tbl_names=None):
    assert os.path.isdir(ds_path)

    res = {}
    res['inputs'] = {}

    for dir_name, ds_dir in _dataset_dirs(ds_path, tbl_names):
        # assert os.path.exists(ds_dir + "/source.csv")
        # assert os.path.exists(ds_dir + "/target.csv")
        assert os.path.exists(os.path.join(ds_dir, "rows.txt"))
        assert os.path.exists(os.path.join(ds_dir, "ground truth.csv"))

        src_col, target_col = "", ""

        with open(os.path.join(ds_dir, "rows.txt"), encoding="utf-8", errors="replace") as f:
            l = f.readline().strip().split(':')
            src_col = l[0]
            target_col = l[1]
            direction = f.readline().strip()

        pairs = []

        with open(os.path.join(ds_dir, "ground truth.csv"), newline='', encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            titles = next(reader)

            if not "source-" + src_col in titles:
                print(ds_dir)

            assert "source-" + src_col in titles
            assert "target-" + target_col in titles

            src_idx = titles.index("source-" + src_col)
            target_idx = titles.index("target-" + target_col)

            if direction.lower() == "target":
                src_idx, target_idx = target_idx, src_idx

            for items in reader:
                pairs.append((items[src_idx], items[target_idx]))

        res['inputs'][dir_name] = pairs

    return res


def _load_manifest_examples(table_id, example_size):
    """
    Load fixed source->target examples from TABULAX_EXAMPLE_MANIFEST.

    Expected CSV columns:
        source_value,target_value

    Recommended CSV columns:
        dataset,pair_id,seed,example_idx,source_value,target_value,method,score,is_gt,q

    Filtering:
        - If pair_id exists, keep rows where pair_id == table_id.
        - If seed exists, keep rows where seed == TABULAX_EXAMPLE_SEED.
    """
    manifest_path = os.getenv("TABULAX_EXAMPLE_MANIFEST")
    if not manifest_path:
        return None

    manifest_seed = os.getenv("TABULAX_EXAMPLE_SEED", "0")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"TABULAX_EXAMPLE_MANIFEST points to a missing file: {manifest_path}"
        )

    examples = []
    with open(manifest_path, newline='', encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)

        required = {"source_value", "target_value"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest {manifest_path} is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            if "pair_id" in row and str(row["pair_id"]) != str(table_id):
                continue

            if "seed" in row and str(row["seed"]) != str(manifest_seed):
                continue

            examples.append((str(row["source_value"]), str(row["target_value"])))

            if len(examples) >= example_size:
                break

    if not examples:
        raise ValueError(
            f"No manifest examples found for table_id={table_id!r}, "
            f"seed={manifest_seed!r}, manifest={manifest_path!r}"
        )

    print(
        f"[TabulaX manifest] table={table_id!r} seed={manifest_seed} "
        f"using {len(examples)} fixed examples from {manifest_path}"
    )
    for i, (src, tgt) in enumerate(examples):
        print(f"  example {i}: {src!r} -> {tgt!r}")

    return examples


def _split_with_manifest(table_id, rows, example_size):
    manifest_examples = _load_manifest_examples(table_id, example_size)
    if manifest_examples is None:
        return None

    row_set = set(rows)
    missing = [pair for pair in manifest_examples if pair not in row_set]
    if missing:
        preview = missing[:5]
        raise ValueError(
            f"{len(missing)} manifest examples were not found in table {table_id!r}. "
            f"First missing examples: {preview}. "
            f"This usually means the manifest uses the wrong source->target direction, "
            f"or the values differ from ground truth.csv after TabulaX direction handling."
        )

    manifest_pairs = set(manifest_examples)
    test_rows = [row for row in rows if row not in manifest_pairs]

    print(
        f"[TabulaX manifest] table={table_id!r} train={len(manifest_examples)} "
        f"test={len(test_rows)}"
    )

    return {
        'train': manifest_examples,
        'test': test_rows,
    }


def sample_data(ds_path, example_size, example_size_type="fixed"):
    pairs = get_pairs_from_files(ds_path, [])

    tables = dict()

    for table, rows in pairs['inputs'].items():
        # print(f"working on {table}")
        random.seed(RANDOM_SEED)
        random.shuffle(rows)
        # print(rows[1])

        manifest_split = _split_with_manifest(table, rows, example_size)
        if manifest_split is not None:
            tables[table] = manifest_split
            continue

        if example_size_type == "fixed":
            train_size = min(example_size, len(rows) - 1)
        else:
            raise NotImplementedError
            train_size = max(2, len(rows) * example_size)

        tables[table] = {
            'train': rows[:train_size],
            'test': rows[train_size:],
        }

    return tables
