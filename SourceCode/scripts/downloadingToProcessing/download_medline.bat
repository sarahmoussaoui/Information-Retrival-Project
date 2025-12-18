@echo off
REM Download MED collection files from Glasgow IR resources

setlocal enabledelayedexpansion

REM Configuration
set MED_URL=https://ir.dcs.gla.ac.uk/resources/test_collections/medl/med.tar.gz
set SCRIPT_DIR=%~dp0
set DATA_DIR=%SCRIPT_DIR%..\..\data\raw

echo.
echo ==================================================
echo Downloading MEDLINE Test Collection
echo ==================================================
echo.

REM Create data directory if it doesn't exist
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"

REM Download the collection using PowerShell (more reliable on Windows)
echo Downloading from: %MED_URL%
powershell -NoProfile -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%MED_URL%' -OutFile '%DATA_DIR%\med.tar.gz' -UseBasicParsing } catch { Write-Host 'Download failed'; exit 1 }"

if errorlevel 1 (
    echo [ERROR] Failed to download file
    exit /b 1
)

REM Extract the archive using tar (available in Windows 10+)
echo Extracting archive...
cd /d "%DATA_DIR%"
tar -xzf med.tar.gz

if errorlevel 1 (
    echo [ERROR] Failed to extract archive
    exit /b 1
)

REM Clean up
echo Cleaning up...
del med.tar.gz

echo.
echo ==================================================
echo [OK] Download complete!
echo Files available in: %DATA_DIR%
echo ==================================================
echo.
