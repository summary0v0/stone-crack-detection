@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0crackformer_ii\start_gui.ps1"
if errorlevel 1 pause
