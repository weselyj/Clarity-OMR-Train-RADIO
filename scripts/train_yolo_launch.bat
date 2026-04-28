@echo off
REM Usage:  train_yolo_launch.bat 8m   -- launches YOLOv8m baseline
REM         train_yolo_launch.bat 26m  -- launches YOLO26m
REM
REM Spawns WMI-detached training that survives this script closing.
REM The powershell launcher sets User-scope env vars and confirms the process is alive.

if "%1"=="8m" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_yolo8m_baseline_launch.ps1"
) else if "%1"=="26m" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_yolo26m_launch.ps1"
) else (
    echo Usage: %0 ^<8m^|26m^>
    exit /b 1
)
pause
