# scripts/multi_dpi_render_launch.ps1
# Fire-and-forget launcher for the multi-DPI render. Survives SSH session crashes
# because the inner script runs in a fully detached process spawned via WMI.

$repo  = "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO"
$inner = "$repo\scripts\multi_dpi_render_inner.ps1"

$proc = Invoke-WmiMethod -Class Win32_Process -Name Create `
    -ArgumentList "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$inner`""

Write-Host "Launched multi-DPI render. PID: $($proc.ProcessId)"
Write-Host "Log: $repo\logs\multi_dpi_render.log"
Write-Host ""
Write-Host "Tail the log with:"
Write-Host "  ssh 10.10.1.29 'powershell -Command \"Get-Content $repo\logs\multi_dpi_render.log -Tail 30\"'"
