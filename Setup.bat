@echo off
setlocal

rem Define the path to your Python installation
set PYTHON_EXECUTABLE=C:\Path\to\python.exe

rem Define the path to your project and the virtual environment name
set PROJECT_PATH=C:\Path\to\your\project
set VENV_NAME=venv

rem Create the virtual environment
%PYTHON_EXECUTABLE% -m venv "%PROJECT_PATH%\%VENV_NAME%"

rem Activate the virtual environment
call "%PROJECT_PATH%\%VENV_NAME%\Scripts\activate.bat"

rem Install dependencies from requirements.txt
pip install -r "%PROJECT_PATH%\requirements.txt"

echo Virtual environment setup complete.
exit /b
