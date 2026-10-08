@echo off
setlocal
if not defined NF_PYTHON set "NF_PYTHON=python"
"%NF_PYTHON%" -I "%~dp0n3_power.py" %*
exit /b %errorlevel%
