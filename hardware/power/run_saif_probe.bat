@echo off
call "%~dp0run_n3_power.bat" probe %*
exit /b %errorlevel%
