@echo off
echo Phi-4-Mini-Reasoning Benchmarking Tool
echo =====================================

:: Activate the virtual environment
call phi4_benchmark_env\Scripts\activate.bat

:: Parse command line arguments
set DEVICE=cuda
set PROMPT_SET=all
set VISUALIZE=
set NUM_RUNS=3
set MAX_TOKENS=512

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
shift
goto parse_args

:check_cuda
:: Check if CUDA is available when requested
if /i "%DEVICE%"=="cuda" (
    echo Checking CUDA availability...
    python -c "import torch; import sys; cuda_available = torch.cuda.is_available(); print(f'CUDA available: {cuda_available}'); sys.exit(0 if cuda_available else 1)"
    if %ERRORLEVEL% neq 0 (
        echo ERROR: CUDA device requested but CUDA is not available.
        echo Your PyTorch installation does not have CUDA support.
        echo.
        echo To fix this, reinstall PyTorch with CUDA support:
        echo pip uninstall -y torch
        echo pip install torch --index-url https://download.pytorch.org/whl/cu121
        echo.
        echo Or run the benchmark on CPU instead:
        echo .\run_benchmark.bat --cpu
        exit /b 1
    )
    
    :: Only try to get device name if CUDA is available
    python -c "import torch; print(f'CUDA is available. Using device: {torch.cuda.get_device_name(0)}')"
) else (
    echo Using CPU for inference.
)

:run
echo Running benchmark with the following configuration:
echo - Device: %DEVICE%
echo - Prompt set: %PROMPT_SET%
echo - Number of runs: %NUM_RUNS%
echo - Max tokens: %MAX_TOKENS%
if defined VISUALIZE echo - Visualize: Yes
if not defined VISUALIZE echo - Visualize: No

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

:: Run the benchmark
python run_benchmarks.py --device %DEVICE% --prompt_sets %PROMPT_SET% --num_runs %NUM_RUNS% --max_new_tokens %MAX_TOKENS% %VISUALIZE%

echo.
echo Benchmark complete!
echo Results are saved in the results directory.

:: Deactivate the virtual environment
call deactivate
