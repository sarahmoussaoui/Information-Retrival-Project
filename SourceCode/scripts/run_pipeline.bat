@echo off
REM Complete pipeline: download MEDLINE and run preprocessing/indexing scripts

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set BASE_DIR=%SCRIPT_DIR:~0,-1%
set DATA_DIR=%BASE_DIR%\data
set RAW_DIR=%DATA_DIR%\raw

echo.
echo ==========================================
echo IR Pipeline: Download ^& Build
echo ==========================================
echo.

REM Step 1: Download MEDLINE if not already present
if exist "%RAW_DIR%\MED.ALL" (
    echo [OK] MEDLINE already present
) else (
    echo [*] Step 1: Downloading MEDLINE collection...
    call "%SCRIPT_DIR%downloadingToProcessing\download_medline.bat"
    if errorlevel 1 (
        echo [ERROR] Failed to download MEDLINE
        exit /b 1
    )
)

REM Step 2: Parse documents, queries, and relevance judgments
echo.
echo [*] Step 2: Parsing MEDLINE and preprocessing...
python "%SCRIPT_DIR%downloadingToProcessing\parse_preprocess.py"
if errorlevel 1 (
    echo [ERROR] Failed to parse/preprocess
    exit /b 1
)

REM Step 3: Compute TF-IDF statistics
echo.
echo [*] Step 3: Computing TF-IDF statistics...
python "%SCRIPT_DIR%downloadingToProcessing\build_tf_idf_stats.py"
if errorlevel 1 (
    echo [ERROR] Failed to compute TF-IDF stats
    exit /b 1
)

REM Step 4: Build document-term matrix and inverted index
echo.
echo [*] Step 4: Building indexes...
python "%SCRIPT_DIR%downloadingToProcessing\build_index.py"
if errorlevel 1 (
    echo [ERROR] Failed to build indexes
    exit /b 1
)

echo.
echo ==========================================
echo [OK] Pipeline completed successfully!
echo ==========================================
echo.
echo Outputs saved to: data\processed\
echo   parse_preprocess\          - parsed docs, queries, qrels
echo   build_tf_idf_stats\   - tf/idf statistics
echo   build_index\          - doc-term matrix, vocab, inverted index
echo.
echo ==========================================
echo.
pause
