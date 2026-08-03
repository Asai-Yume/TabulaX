from pathlib import Path
import csv

WT_ROOT = Path("data/wt")

EXPECTED = {
    "AJ-beatles-songs": ("Title", "Title"),
    "AJ-california-govs-1": ("Governor's Name", "Name"),
    "AJ-california-govs-2": ("Governor", "Name"),
    "AJ-chinese-provinces": ("Province", "Name"),
    "AJ-christmas-songs-1": ("Song", "column 2"),
    "AJ-christmas-songs-2": ("Title", "column 2"),
    "AJ-duke-cs-profs": ("column 0", "Name"),
    "AJ-fruits-1": ("FRUITS (raw)", "FRUIT CARB CHART"),
    "AJ-fruits-2": ("FRUITS (raw)", "Per single fruit or the portion stated"),
    "AJ-fsu-name-to-username": ("full name", "Username"),
    "AJ-k12-name-to-email": ("Name", "email"),
    "AJ-new-york-govs-1": ("Name", "Name (Tenure)"),
    "AJ-new-york-govs-2": ("Governor", "Name (Tenure)"),
    "AJ-new-york-govs-3": ("Governor's Name", "Name (Tenure)"),
    "AJ-new-york-govs-4": ("Governor", "Name (Tenure)"),
    "AJ-new-york-govs-5": ("Governor", "Governor"),
    "AJ-park-to-state-1": ("Name", "State"),
    "AJ-park-to-state-2": ("National Park", "State"),
    "AJ-sharif-email-to-url": ("Email", "Web Address"),
    "AJ-sharif-username-to-email": ("Username", "Email"),
    "AJ-texas-govs-1": ("Governor", "Governor's Name"),
    "AJ-texas-govs-2": ("Governor", "Governor's Name"),
    "AJ-uk-prime-ministers": ("name", "Column_0"),
    "AJ-us-cities": ("United States Cities", "City"),
    "AJ-us-presidents-1": ("President", "Name"),
    "AJ-us-presidents-2": ("President", "President"),
    "AJ-us-presidents-3": ("Presidents", "President"),
    "AJ-us-presidents-4": ("President", "Presidents"),
    "AJ-us-presidents-5": ("Name", "Presidents"),
    "AJ-us-presidents-6": ("Name", "President"),
    "AJ-vegetables": ("Vegetable", "Vegetable - Type"),
    "names200": ("fullname", "username"),
}

def read_lines(path):
    return path.read_text(encoding="utf-8", errors="replace").splitlines()

problems = []

for dataset, (src_col, tgt_col) in EXPECTED.items():
    folder = WT_ROOT / dataset
    rows_path = folder / "rows.txt"
    gt_path = folder / "ground truth.csv"

    if not folder.exists():
        problems.append((dataset, "MISSING_FOLDER", str(folder)))
        continue

    if not rows_path.exists():
        problems.append((dataset, "MISSING_ROWS_TXT", str(rows_path)))
    else:
        lines = [x.strip() for x in read_lines(rows_path) if x.strip()]
        expected_first = f"{src_col}:{tgt_col}"

        if len(lines) < 1:
            problems.append((dataset, "EMPTY_ROWS_TXT", str(rows_path)))
        elif lines[0] != expected_first:
            problems.append((dataset, "ROWS_MAPPING_MISMATCH", f"got={lines[0]!r}, expected={expected_first!r}"))

        if len(lines) < 2:
            problems.append((dataset, "ROWS_SECOND_LINE_MISSING", "expected 'source'"))
        elif lines[1].lower() != "source":
            problems.append((dataset, "ROWS_SECOND_LINE_NOT_SOURCE", f"got={lines[1]!r}, expected='source'"))

    if not gt_path.exists():
        problems.append((dataset, "MISSING_GROUND_TRUTH", str(gt_path)))
    else:
        try:
            with gt_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, [])
                rows = list(reader)

            expected_header = [f"source-{src_col}", f"target-{tgt_col}"]

            if header[:2] != expected_header:
                problems.append((dataset, "GT_HEADER_MISMATCH", f"got={header[:2]!r}, expected={expected_header!r}"))

            if len(rows) == 0:
                problems.append((dataset, "GT_NO_ROW_PAIRS", str(gt_path)))

        except Exception as e:
            problems.append((dataset, "GT_READ_ERROR", repr(e)))

print("dataset,problem,detail")
for dataset, problem, detail in problems:
    print(f"{dataset},{problem},{detail}")

if not problems:
    print("All WT rows.txt and ground truth.csv files match the WT mapping doc.")