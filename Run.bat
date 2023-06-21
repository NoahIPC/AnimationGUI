@echo off
setlocal

rem Define the path to your project and the virtual environment name
set PROJECT_PATH=C:\Path\to\your\project
set VENV_NAME=venv

rem Activate the virtual environment
call "%PROJECT_PATH%\%VENV_NAME%\Scripts\activate.bat"

rem Run the application.py file
python "%PROJECT_PATH%\application.py"

exit /b
