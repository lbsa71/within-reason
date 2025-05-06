@echo off
echo Phi-4-Mini-Reasoning Benchmarking Tool
echo =====================================

:: Set environment variable to disable symlinks warning
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

:: Set custom cache directory (optional)
set HF_HOME=%~dp0cache

:: Activate the virtual environment
call phi4_benchmark_env\Scripts\activate.bat

:: Parse command line arguments
set DEVICE=cuda
set PROMPT_SET=all
set VISUALIZE=
set NUM_RUNS=3
set MAX_TOKENS=512
set MODEL=microsoft/Phi-4-mini-reasoning
set USE_LOCAL=
set CACHE_MODEL=--cache_model

:parse_args
if "%~1"=="" goto check_cuda
if /i "%~1"=="--cpu" set DEVICE=cpu& shift & goto parse_args
if /i "%~1"=="--cuda" set DEVICE=cuda& shift & goto parse_args
if /i "%~1"=="--short" set PROMPT_SET=short& shift & goto parse_args
if /i "%~1"=="--medium" set PROMPT_SET=medium& shift & goto parse_args
if /i "%~1"=="--long" set PROMPT_SET=long& shift & goto parse_args
if /i "%~1"=="--reasoning" set PROMPT_SET=reasoning& shift & goto parse_args
if /i "%~1"=="--creative" set PROMPT_SET=creative& shift & goto parse_args
if /i "%~1"=="--all" set PROMPT_SET=all& shift & goto parse_args
if /i "%~1"=="--visualize" set VISUALIZE=--visualize& shift & goto parse_args
if /i "%~1"=="--runs" set NUM_RUNS=%~2& shift & shift & goto parse_args
if /i "%~1"=="--max-tokens" set MAX_TOKENS=%~2& shift & shift & goto parse_args
if /i "%~1"=="--model" set MODEL=%~2& shift & shift & goto parse_args
if /i "%~1"=="--use-local" set USE_LOCAL=--use_local_model& shift & goto parse_args
if /i "%~1"=="--no-cache" set CACHE_MODEL=& shift & goto parse_args
if /i "%~1"=="--cache" set CACHE_MODEL=--cache_model& shift & goto parse_args
shift
goto parse_args

:check_cuda
:: Check packages first
call :check_packages

:: If CUDA is requested, let the Python script handle the CUDA check
:: This ensures the same environment is used for checking and running
if /i "%DEVICE%"=="cuda" (
    echo Checking CUDA availability in Python environment...
    python -c "import torch; import sys; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'None'); sys.exit(0 if torch.cuda.is_available() else 1)"
    
    if %ERRORLEVEL% neq 0 (
        echo ERROR: CUDA is not available in the current Python environment.
        echo Your PyTorch installation does not have CUDA support.
        echo.
        echo Would you like to:
        echo 1. Install PyTorch with CUDA support
        echo 2. Run on CPU instead
        echo 3. Exit
        set /p choice="Enter choice (1-3): "
        
        if "%choice%"=="1" (
            echo Installing PyTorch with CUDA support...
            pip uninstall -y torch
            pip install torch --index-url https://download.pytorch.org/whl/cu121
            goto check_cuda
        ) else if "%choice%"=="2" (
            echo Running on CPU instead...
            set DEVICE=cpu
        ) else (
            echo Exiting...
            exit /b 1
        )
    )
)

goto run

:check_packages
:: Install required packages if not already installed
pip show transformers >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installing required packages...
    pip install transformers accelerate matplotlib numpy
    
    :: Install the appropriate PyTorch version
    if /i "%DEVICE%"=="cuda" (
        echo Installing PyTorch with CUDA support...
        pip install torch --index-url https://download.pytorch.org/whl/cu121
    ) else (
        echo Installing PyTorch for CPU...
        pip install torch
    )
)

:: Create results directory if it doesn't exist
if not exist results mkdir results

:: Create models directory if it doesn't exist
if not exist models mkdir models
exit /b 0

:run
echo Running benchmark with the following configuration:
echo - Model: %MODEL%
echo - Device: %DEVICE%
echo - Prompt set: %PROMPT_SET%
echo - Number of runs: %NUM_RUNS%
echo - Max tokens: %MAX_TOKENS%
if defined USE_LOCAL echo - Using locally cached model: Yes
if defined CACHE_MODEL echo - Cache model for future runs: Yes
if defined VISUALIZE echo - Visualize: Yes
if not defined VISUALIZE echo - Visualize: No

:: Run the benchmark directly with Python, not through run_benchmarks.py
:: This ensures the same Python environment is used for the entire process
python benchmark.py --model "%MODEL%" --device %DEVICE% --prompt_types %PROMPT_SET% --num_runs %NUM_RUNS% --max_new_tokens %MAX_TOKENS% %USE_LOCAL% %CACHE_MODEL%

:: Check if there are any results files
echo.
if exist results\*.json (
    echo Benchmark complete! Results saved in the results directory:
    dir /b results\benchmark_*.json
    echo.
    echo Generated answers saved in:
    dir /b results\answers_*.json
    
    if defined VISUALIZE (
        for %%f in (results\benchmark_*.json) do (
            echo Visualizing results from %%f...
            python visualize_results.py --results_file "%%f"
        )
    )
) else (
    echo No results files were created. Check for errors above.
)

:: Deactivate the virtual environment
call deactivate
