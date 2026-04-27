# Inner script — runs detached via WMI Create. Outer script returns
# immediately; this script survives SSH session close.
$ErrorActionPreference = "Continue"
$repo = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
$data = Join-Path $repo "data"
$log  = Join-Path $repo "logs\restore_data.log"

function Log($msg) {
    $stamp = Get-Date -Format "HH:mm:ss"
    "$stamp $msg" | Out-File -FilePath $log -Encoding utf8 -Append
}

function Ensure-Dir($p) {
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
}

function Download-If-Missing($url, $dest) {
    if (Test-Path $dest) {
        $size = (Get-Item $dest).Length
        if ($size -gt 1024) {
            Log "[skip ] download already present ($([Math]::Round($size/1MB,1)) MB): $dest"
            return
        }
    }
    Log "[dl   ] $url -> $dest"
    & curl.exe -L --fail --show-error -o $dest $url 2>&1 | Out-File -FilePath $log -Encoding utf8 -Append
    if ($LASTEXITCODE -ne 0) {
        Log "[fail ] curl exit $LASTEXITCODE for $url"
        throw "curl failed"
    }
    $size = (Get-Item $dest).Length
    Log "[done ] downloaded ($([Math]::Round($size/1MB,1)) MB): $dest"
}

function Extract-Tgz($archive, $into, $sentinel) {
    if (Test-Path $sentinel) {
        Log "[skip ] extract already done (sentinel exists): $sentinel"
        return
    }
    Log "[xtrct] $archive -> $into"
    Ensure-Dir $into
    Push-Location $into
    try {
        & tar.exe -xzf $archive 2>&1 | Out-File -FilePath $log -Encoding utf8 -Append
        if ($LASTEXITCODE -ne 0) { throw "tar failed for $archive" }
    } finally {
        Pop-Location
    }
    Log "[done ] extracted into $into"
}

Ensure-Dir (Split-Path $log)
Log "=== restore_data_inner started PID=$PID ==="

try {
    Ensure-Dir $data

    # 1. PrIMuS
    Ensure-Dir "$data\primus"
    Download-If-Missing "https://grfia.dlsi.ua.es/primus/packages/primusCalvoRizoAppliedSciences2018.tgz" `
                       "$data\primus\primusCalvoRizoAppliedSciences2018.tgz"
    Extract-Tgz "$data\primus\primusCalvoRizoAppliedSciences2018.tgz" "$data\primus" "$data\primus\package_aa"

    # 2. GrandStaff
    Ensure-Dir "$data\grandstaff"
    Download-If-Missing "https://grfia.dlsi.ua.es/musicdocs/grandstaff.tgz" `
                       "$data\grandstaff\grandstaff.tgz"
    Extract-Tgz "$data\grandstaff\grandstaff.tgz" "$data\grandstaff" "$data\grandstaff\scarlatti-d"

    # 3. Camera-PrIMuS
    Ensure-Dir "$data\camera_primus"
    Download-If-Missing "https://grfia.dlsi.ua.es/primus/packages/CameraPrIMuS.tgz" `
                       "$data\camera_primus\CameraPrIMuS.tgz"
    Extract-Tgz "$data\camera_primus\CameraPrIMuS.tgz" "$data\camera_primus" "$data\camera_primus\package_aa"

    # 4. OpenScore Lieder
    $ldir = Join-Path $data "openscore_lieder"
    if (Test-Path "$ldir\.git") {
        Log "[skip ] openscore_lieder already cloned"
    } else {
        Log "[clone] openscore_lieder"
        & git.exe clone https://github.com/OpenScore/Lieder.git $ldir 2>&1 | Out-File -FilePath $log -Encoding utf8 -Append
    }
    Push-Location $ldir
    try {
        & git.exe checkout 6b2dc542ce2e8aa4b78c8ee62103b210efc07015 2>&1 | Out-File -FilePath $log -Encoding utf8 -Append
    } finally { Pop-Location }

    Log ""
    Log "=== summary ==="
    Get-ChildItem $data -Directory | ForEach-Object {
        $size = (Get-ChildItem $_.FullName -Recurse -ErrorAction SilentlyContinue -Force | Measure-Object -Property Length -Sum).Sum
        if ($size) {
            Log ("{0,-22} {1,8:F2} GB" -f $_.Name, ($size/1GB))
        } else {
            Log ("{0,-22} (empty)" -f $_.Name)
        }
    }
    Log "=== restore_data_inner DONE ==="
} catch {
    Log "[exception] $_"
    Log "=== restore_data_inner FAILED ==="
}
