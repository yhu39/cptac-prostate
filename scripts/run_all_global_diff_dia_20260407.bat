@echo off
setlocal

rem Run all DIA global differential analyses plus the summary workflow.
rem This batch file is intended for archival/reuse and does not require package installation.

set "PYTHON_EXE=C:\Users\yhu39\AppData\Local\anaconda3\envs\prostate\python.exe"
set "REPO_DIR=C:\Users\yhu39\Documents\lab\cptac-prostate"
set "RUN_DIR=E:\lab\cptac-prostate\runs\20260407_global_diff_msfragger_dia"
set "PYTHONPATH=%REPO_DIR%\src"

if not exist "%PYTHON_EXE%" (
    echo ERROR: Python executable not found: %PYTHON_EXE%
    exit /b 1
)

if not exist "%REPO_DIR%\src\cptac_prostate\cli.py" (
    echo ERROR: Repo CLI not found under: %REPO_DIR%
    exit /b 1
)

if not exist "%RUN_DIR%" (
    echo ERROR: Run directory not found: %RUN_DIR%
    exit /b 1
)

call :run_config "config_tumor_vs_normal.ini"
call :run_config "config_G2_vs_normal.ini"
call :run_config "config_G3_vs_normal.ini"
call :run_config "config_G4_vs_normal.ini"
call :run_config "config_G5_vs_normal.ini"
call :run_config "config_diff_summary.ini"

echo.
echo All DIA global_diff jobs completed successfully.
exit /b 0

:run_config
set "CONFIG_NAME=%~1"
set "CONFIG_PATH=%RUN_DIR%\%CONFIG_NAME%"

if not exist "%CONFIG_PATH%" (
    echo ERROR: Config file not found: %CONFIG_PATH%
    exit /b 1
)

echo.
echo ==================================================
echo Running %CONFIG_NAME%
echo ==================================================
pushd "%REPO_DIR%"
"%PYTHON_EXE%" -m cptac_prostate.cli --config "%CONFIG_PATH%"
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" (
    echo ERROR: %CONFIG_NAME% failed with exit code %EXIT_CODE%
    exit /b %EXIT_CODE%
)

exit /b 0
