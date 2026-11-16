# A persistent per-user monitor also notices probe failures after the resume event.
[CmdletBinding(SupportsShouldProcess = $true)]
param([Parameter(Mandatory = $true)][string]$Distro,
      [Parameter(Mandatory = $true)][string]$LoopDir,
      [switch]$AllowWslRestart, [switch]$KeepAwake, [switch]$SignalOnly)
$ErrorActionPreference = 'Stop'
if ($Distro -notmatch '^[A-Za-z0-9_.-]+$' -or $LoopDir -notmatch '^/[A-Za-z0-9_./-]+$' -or $LoopDir.Contains('/../')) { throw 'Invalid distro or Linux loop path.' }
if (-not $PSCmdlet.ShouldProcess($Distro, 'Start WSL and request a health check')) { return }
# This is the same marker written by the Python resume-check CLI. It needs no
# guessed repository or virtualenv path on the Windows side.
& wsl.exe -d $Distro --exec mkdir -p -- "$LoopDir/control"
if ($LASTEXITCODE -ne 0) { throw 'WSL startup failed.' }
& wsl.exe -d $Distro --exec touch -- "$LoopDir/control/resume-check"
if ($LASTEXITCODE -ne 0) { throw 'Cannot write resume marker.' }
if ($SignalOnly) { return }
$hash = [Security.Cryptography.SHA256]::Create()
$id = ([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes("$Distro|$LoopDir")))).Replace('-', '').Substring(0, 12)
$hash.Dispose()
$mutex = New-Object Threading.Mutex($false, "Local\HasamiAutoloop-$id")
$owned = $false
$keeper = $null
$awakeProcess = $null
try {
    try { $owned = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $owned = $true }
    if (-not $owned) { return }
    if ($KeepAwake) {
        $awakeArgs = '-NoProfile -NonInteractive -WindowStyle Hidden -File "' + (Join-Path $PSScriptRoot 'keep_awake.ps1') + '" -Distro "' + $Distro + '" -LoopDir "' + $LoopDir + '"'
        $awakeProcess = Start-Process powershell.exe -ArgumentList $awakeArgs -WindowStyle Hidden -PassThru
    }
    while ($true) {
        if ($null -eq $keeper -or $keeper.HasExited) {
            $keeper = Start-Process wsl.exe -ArgumentList @('-d', $Distro, '--exec', 'sleep', 'infinity') -WindowStyle Hidden -PassThru
        }
        if ($AllowWslRestart) {
            & wsl.exe -d $Distro --exec test -f "$LoopDir/control/needs_wsl_restart"
            if ($LASTEXITCODE -eq 0 -and $PSCmdlet.ShouldProcess('ALL WSL distributions', 'Shutdown WSL after failed health probes')) {
                # Consume the marker before shutdown to avoid an immediate restart loop.
                & wsl.exe -d $Distro --exec rm -f -- "$LoopDir/control/needs_wsl_restart"
                if ($LASTEXITCODE -ne 0) { throw 'Cannot consume restart marker.' }
                & wsl.exe --shutdown
                if ($LASTEXITCODE -ne 0) { throw 'WSL shutdown failed.' }
                Start-Sleep -Seconds 30
                $keeper = Start-Process wsl.exe -ArgumentList @('-d', $Distro, '--exec', 'sleep', 'infinity') -WindowStyle Hidden -PassThru
            }
        }
        Start-Sleep -Seconds 30
    }
} finally {
    if ($null -ne $keeper -and -not $keeper.HasExited) { $keeper.Kill() }
    if ($null -ne $awakeProcess -and -not $awakeProcess.HasExited) { $awakeProcess.Kill() }
    if ($owned) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
