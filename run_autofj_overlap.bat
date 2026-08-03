@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "C:\Users\Kai\Desktop\Files\New York University\VIDA\repo\TabulaX"

rem ============================================================
rem GPT-5-mini through OpenRouter
rem ============================================================
set "USE_OPENROUTER=1"
set "OPENROUTER_MODEL=openai/gpt-5-mini"
set "OPENROUTER_MAX_TOKENS=8192"
set "OPENROUTER_REASONING_EFFORT=low"
set "TABULAX_EMPTY_RESPONSE_RETRIES=2"

rem ============================================================
rem Experiment configuration
rem ============================================================
set "TABULAX_MATCHING_TYPE=exact"
set "TABULAX_MATCHING_TYPES=exact"

rem Use TabulaX's own randomly sampled GT examples.
set "TABULAX_EXAMPLE_MANIFEST="

rem Use max(1, ceil(0.25 * gt_size)), while dataset.py leaves
rem at least one positive GT pair for testing.
set "TABULAX_EXAMPLE_SIZE_TYPE=fraction"
set "TABULAX_EXAMPLE_FRACTION=0.25"
set "TABULAX_EXAMPLE_SIZE="

set "DATA_ROOT=data\autofj_overlap"
set "OUTPUT_ROOT=outputs"
set "FAILURE_LOG=%CD%\outputs\autofj_overlap_pct25_exact_gpt5mini_failures.txt"

if not exist "%OUTPUT_ROOT%" mkdir "%OUTPUT_ROOT%"

echo AutoFJ overlap sweep started: %DATE% %TIME%> "%FAILURE_LOG%"
echo.>> "%FAILURE_LOG%"

rem ============================================================
rem Run all overlap folders for seeds 0-4.
rem Folder names are expected to end in:
rem   _f025, _f050, _f075, or _f100
rem ============================================================
for %%S in (0 1 2 3 4) do (
  for /D %%D in (
    "%DATA_ROOT%\*_f025"
    "%DATA_ROOT%\*_f050"
    "%DATA_ROOT%\*_f075"
    "%DATA_ROOT%\*_f100"
  ) do (
    set "PAIR_NAME=%%~nxD"
    set "RUN_NAME=autofj_overlap_!PAIR_NAME!_pct25_exact_gpt5mini_sweep_seed%%S"
    set "RUN_OUTPUT=%OUTPUT_ROOT%\!RUN_NAME!"

    rem A successful exact-only result has:
    rem   header + at least one exact row
    rem Numeric datasets may instead contain one num_dist row.
    python -c "import pathlib,sys; p=pathlib.Path(r'!RUN_OUTPUT!\_res.csv'); lines=p.read_text(encoding='utf-8', errors='replace').splitlines() if p.exists() else []; ok=len(lines)>=2 and any(tag in line for line in lines[1:] for tag in (',exact,', ',num_dist,')); sys.exit(0 if ok else 1)"

    if errorlevel 1 (
      if not exist "%%~fD\rows.txt" (
        echo SKIPPING !PAIR_NAME! seed %%S: missing rows.txt
        echo SKIPPED !PAIR_NAME! seed %%S: missing rows.txt>> "%FAILURE_LOG%"
      ) else if not exist "%%~fD\ground truth.csv" (
        echo SKIPPING !PAIR_NAME! seed %%S: missing ground truth.csv
        echo SKIPPED !PAIR_NAME! seed %%S: missing ground truth.csv>> "%FAILURE_LOG%"
      ) else (
        echo.
        echo ============================================================
        echo Running seed %%S dataset !PAIR_NAME!
        echo Input:  %%~fD
        echo Output: !RUN_OUTPUT!
        echo ============================================================

        rem Clear LLM prompt/classifier caches before every actual run.
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

        rem Do not delete cache\edit_distance\ed.pkl.
        set "TABULAX_EXAMPLE_SEED=%%S"
        set "TABULAX_DS_PATH=%%~fD"
        set "TABULAX_DS_NAME=!RUN_NAME!"
        set "TABULAX_OUTPUT_DIR=%CD%\!RUN_OUTPUT!"

        python src\LLM_pipeline\run_pipeline.py

        if errorlevel 1 (
          echo FAILED seed %%S dataset !PAIR_NAME!
          echo FAILED seed %%S dataset !PAIR_NAME!>> "%FAILURE_LOG%"
        ) else (
          rem Validate that the run produced a usable result row.
          python -c "import pathlib,sys; p=pathlib.Path(r'!RUN_OUTPUT!\_res.csv'); lines=p.read_text(encoding='utf-8', errors='replace').splitlines() if p.exists() else []; ok=len(lines)>=2 and any(tag in line for line in lines[1:] for tag in (',exact,', ',num_dist,')); sys.exit(0 if ok else 1)"

          if errorlevel 1 (
            echo INCOMPLETE seed %%S dataset !PAIR_NAME!
            echo INCOMPLETE seed %%S dataset !PAIR_NAME!>> "%FAILURE_LOG%"
          ) else (
            echo COMPLETED seed %%S dataset !PAIR_NAME!
          )
        )
      )
    ) else (
      echo Skipping completed seed %%S dataset !PAIR_NAME!
    )
  )
)

echo.>> "%FAILURE_LOG%"
echo AutoFJ overlap sweep finished: %DATE% %TIME%>> "%FAILURE_LOG%"

echo.
echo ============================================================
echo Sweep finished.
echo Failure log: %FAILURE_LOG%
echo ============================================================

endlocal
