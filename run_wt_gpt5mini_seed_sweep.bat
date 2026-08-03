@echo off
setlocal EnableExtensions

cd /d "C:\Users\Kai\Desktop\Files\New York University\VIDA\repo\TabulaX"

set "WT_ROOT=data\wt"

set "USE_OPENROUTER=1"
set "OPENROUTER_MODEL=openai/gpt-5-mini"
set "OPENROUTER_MAX_TOKENS=8192"
set "OPENROUTER_REASONING_EFFORT=low"
set "TABULAX_EMPTY_RESPONSE_RETRIES=2"

set "TABULAX_MATCHING_TYPES=edit_dist,exact"
set "TABULAX_MATCHING_TYPE="
set "TABULAX_EXAMPLE_MANIFEST="

for %%S in (0 1 2 3 4) do (
  for /D %%D in ("%WT_ROOT%\*") do (
    if exist "%%D\rows.txt" if exist "%%D\ground truth.csv" (
      python -c "import pathlib,sys; p=pathlib.Path(r'outputs\wt_%%~nxD_fixed10_bothmatch_gpt5mini_sweep_seed%%S\_res.csv'); lines=p.read_text(encoding='utf-8', errors='replace').splitlines() if p.exists() else []; complete=(len(lines)>=3) or (len(lines)>=2 and any(',num_dist,Numbers,' in line for line in lines[1:])); sys.exit(0 if complete else 1)"

      if errorlevel 1 (
        echo ===== Running seed %%S dataset %%~nxD =====

        echo {}> data\Classes\gpt_classified.json

        rmdir /s /q cache\classifier_prompts 2>nul
        mkdir cache\classifier_prompts

        rmdir /s /q cache\str_code_prompts 2>nul
        mkdir cache\str_code_prompts

        rmdir /s /q cache\gen_rel_prompts 2>nul
        mkdir cache\gen_rel_prompts

        rmdir /s /q cache\gen_bridge_prompts 2>nul
        mkdir cache\gen_bridge_prompts

        rmdir /s /q cache\alg_code_prompts 2>nul
        mkdir cache\alg_code_prompts

        rmdir /s /q cache\alg_rel_prompts 2>nul
        mkdir cache\alg_rel_prompts

        rmdir /s /q cache\basic_bridge_prompts 2>nul
        mkdir cache\basic_bridge_prompts

        set "TABULAX_EXAMPLE_SEED=%%S"
        set "TABULAX_DS_PATH=%CD%\%%D"
        set "TABULAX_DS_NAME=wt_%%~nxD_fixed10_bothmatch_gpt5mini_sweep_seed%%S"
        set "TABULAX_OUTPUT_DIR=%CD%\outputs\wt_%%~nxD_fixed10_bothmatch_gpt5mini_sweep_seed%%S"

        python src\LLM_pipeline\run_pipeline.py
      ) else (
        echo Skipping completed seed %%S dataset %%~nxD
      )
    ) else (
      echo Skipping %%~nxD because rows.txt or ground truth.csv is missing
    )
  )
)

echo ===== WT seed sweep finished =====