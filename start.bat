@echo off

REM Get the current directory of the batch script
set current_dir=%~dp0

REM Navigate to the directory where setup.py is located
cd /d "%current_dir%"

REM Run the Python setup script and ensure it runs to completion
python setup.py

REM Keep the window open after the setup script completes
echo Setup complete. The server should be running at http://127.0.0.1:5000
pause