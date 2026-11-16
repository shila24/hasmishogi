# PowerShell 5.1; only prevent sleep while a recent loop heartbeat is running.
[CmdletBinding(SupportsShouldProcess = $true)]
param([Parameter(Mandatory = $true)][string]$Distro,
      [Parameter(Mandatory = $true)][string]$LoopDir)
$ErrorActionPreference = 'Stop'
if ($Distro -notmatch '^[A-Za-z0-9_.-]+$' -or $LoopDir -notmatch '^/[A-Za-z0-9_./-]+$' -or $LoopDir.Contains('/../')) { throw 'Invalid path or distro.' }
if (-not $PSCmdlet.ShouldProcess('Windows power management', 'Hold system awake while the loop runs')) { return }
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class HasamiPower {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
$continuous = [uint32]2147483648
try {
    while ($true) {
        $running = $false
        try {
            $raw = & wsl.exe -d $Distro --exec cat -- "$LoopDir/status.json" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $state = ($raw -join "`n") | ConvertFrom-Json
                $now = ([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())
                $running = $state.status -eq 'running' -and ($now - $state.last_heartbeat) -lt 180
            }
        } catch { $running = $false }
        $flags = if ($running) { $continuous -bor [uint32]1 } else { $continuous }
        [void][HasamiPower]::SetThreadExecutionState($flags)
        Start-Sleep -Seconds 30
    }
} finally { [void][HasamiPower]::SetThreadExecutionState($continuous) }
