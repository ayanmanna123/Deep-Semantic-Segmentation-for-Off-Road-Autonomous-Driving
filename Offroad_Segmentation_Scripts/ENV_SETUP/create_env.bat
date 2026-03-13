@echo off

:: Check if conda is available in the system
where conda >nul 2>nul
IF ERRORLEVEL 1 (
    IF EXIST "C:\ProgramData\anaconda3\condabin\conda.bat" (
        set "CONDA_PATH=C:\ProgramData\anaconda3\condabin\conda.bat"
    ) ELSE (
        echo "Conda is not found in your system. Please install Miniconda or Anaconda first."
        exit /b 1
    )
) ELSE (
    set "CONDA_PATH=conda"
)

:: Create the environment
echo Creating the Conda environment 'EDU' with Python 3.10...
call %CONDA_PATH% create --name EDU python=3.10 -y



