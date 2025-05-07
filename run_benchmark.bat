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
set QUANTIZE=0
set OPTIMIZE_GPU=
set BATCH_SIZE=
set DISABLE_KV_CACHE=
set DISABLE_MIXED_PRECISION=

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
if /i "%~1"=="--4bit" set QUANTIZE=4& shift & goto parse_args
if /i "%~1"=="--8bit" set QUANTIZE=8& shift & goto parse_args
if /i "%~1"=="--optimize-gpu" set OPTIMIZE_GPU=--optimize_gpu& shift & goto parse_args
if /i "%~1"=="--batch" set BATCH_SIZE=--batch_size %~2& shift & shift & goto parse_args
if /i "%~1"=="--disable-kv-cache" set DISABLE_KV_CACHE=--disable_kv_cache& shift & goto parse_args
if /i "%~1"=="--disable-mixed-precision" set DISABLE_MIXED_PRECISION=--disable_mixed_precision& shift & goto parse_args
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

:: Check if quantization is requested and install bitsandbytes if needed
if %QUANTIZE% GTR 0 (
    pip show bitsandbytes >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo Installing bitsandbytes for quantization...
        pip install bitsandbytes
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
if %QUANTIZE% EQU 4 echo - Using 4-bit quantization
if %QUANTIZE% EQU 8 echo - Using 8-bit quantization
if defined VISUALIZE echo - Visualize: Yes
if not defined VISUALIZE echo - Visualize: No
if defined OPTIMIZE_GPU echo - GPU optimization: Enabled
if defined BATCH_SIZE echo - Batch size: %BATCH_SIZE:~13%
if defined DISABLE_KV_CACHE echo - KV cache: Disabled
if defined DISABLE_MIXED_PRECISION echo - Mixed precision: Disabled

:: Run the benchmark directly with Python, not through run_benchmarks.py
:: This ensures the same Python environment is used for the entire process
set TIMESTAMP=%DATE:~-4%%DATE:~-7,2%%DATE:~-10,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set MODEL_NAME=%MODEL:/=_%
set MODEL_NAME=%MODEL_NAME:\=_%
set RESULT_FILE=results\benchmark_%MODEL_NAME%_%PROMPT_SET%_%TIMESTAMP%.json
set ANSWER_FILE=results\answers_%MODEL_NAME%_%PROMPT_SET%_%TIMESTAMP%.json

python benchmark.py --model "%MODEL%" --device %DEVICE% --prompt_types %PROMPT_SET% --num_runs %NUM_RUNS% --max_new_tokens %MAX_TOKENS% %USE_LOCAL% %CACHE_MODEL% --quantize %QUANTIZE% %OPTIMIZE_GPU% %BATCH_SIZE% %DISABLE_KV_CACHE% %DISABLE_MIXED_PRECISION%

:: Check if the expected result file was created
echo.
if exist %RESULT_FILE% (
    echo Benchmark complete! Results saved to:
    echo %RESULT_FILE%
    echo.
    echo Generated answers saved to:
    echo %ANSWER_FILE%
    
    if defined VISUALIZE (
        echo Visualizing results...
        python visualize_results.py --results_file "%RESULT_FILE%"
    )
) else (
    echo No results were created for this benchmark run. Check for errors above.
)

:: Deactivate the virtual environment
call deactivate
