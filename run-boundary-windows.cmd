@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_boundary_windows.ps1" %*
if errorlevel 1 pause
