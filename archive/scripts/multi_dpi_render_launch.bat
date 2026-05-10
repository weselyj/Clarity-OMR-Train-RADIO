@echo off
REM Double-clickable launcher for the multi-DPI render.
REM Spawns the WMI-detached process and exits. Render survives this script closing.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0multi_dpi_render_launch.ps1"
pause
