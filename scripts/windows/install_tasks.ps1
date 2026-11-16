# PowerShell 5.1. Install per-user tasks; no network or external modules.
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)][string]$Distro,
    [Parameter(Mandatory = $true)][string]$LoopDir,
    [switch]$AllowWslRestart,
    [switch]$KeepAwake
)
$ErrorActionPreference = 'Stop'
if ($Distro -notmatch '^[A-Za-z0-9_.-]+$' -or $LoopDir -notmatch '^/[A-Za-z0-9_./-]+$' -or $LoopDir.Contains('/../')) {
    throw 'Use a simple distro name and an absolute Linux loop path without spaces or traversal.'
}
$bytes = [Text.Encoding]::UTF8.GetBytes("$Distro|$LoopDir")
$hash = [Security.Cryptography.SHA256]::Create()
$id = ([BitConverter]::ToString($hash.ComputeHash($bytes))).Replace('-', '').Substring(0, 12)
$hash.Dispose()
$name = "HasamiAutoloop-$id"
$root = Join-Path $env:LOCALAPPDATA $name
$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$runtime = Join-Path $root 'resume.ps1'
$awake = Join-Path $root 'keep_awake.ps1'
$flag = if ($AllowWslRestart) { ' -AllowWslRestart' } else { '' }
$awakeFlag = if ($KeepAwake) { ' -KeepAwake' } else { '' }
if ($PSCmdlet.ShouldProcess($root, 'Install task runtime scripts')) {
    New-Item -ItemType Directory -Force -Path $root | Out-Null
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'resume.ps1') -Destination $runtime -Force
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'keep_awake.ps1') -Destination $awake -Force
}
function Escape-Xml([string]$value) { return [Security.SecurityElement]::Escape($value) }
foreach ($kind in @('Logon', 'Resume')) {
    $trigger = if ($kind -eq 'Logon') {
        '<LogonTrigger><Enabled>true</Enabled><UserId>' + (Escape-Xml $user) + '</UserId></LogonTrigger>'
    } else {
        '<EventTrigger><Enabled>true</Enabled><Subscription>' + (Escape-Xml '<QueryList><Query Id="0" Path="System"><Select Path="System">*[System[Provider[@Name=''Microsoft-Windows-Power-Troubleshooter''] and EventID=1]]</Select></Query></QueryList>') + '</Subscription></EventTrigger>'
    }
    $arguments = '-NoProfile -NonInteractive -WindowStyle Hidden -File "' + $runtime + '" -Distro "' + $Distro + '" -LoopDir "' + $LoopDir + '"' + $flag + $awakeFlag
    if ($kind -eq 'Resume') { $arguments += ' -SignalOnly' }
    $xml = @"
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers>$trigger</Triggers>
  <Principals><Principal id="Owner"><UserId>$(Escape-Xml $user)</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries><ExecutionTimeLimit>PT0S</ExecutionTimeLimit><StartWhenAvailable>true</StartWhenAvailable></Settings>
  <Actions Context="Owner"><Exec><Command>powershell.exe</Command><Arguments>$(Escape-Xml $arguments)</Arguments></Exec></Actions>
</Task>
"@
    if ($PSCmdlet.ShouldProcess("$name-$kind", 'Register per-user scheduled task')) {
        try { Register-ScheduledTask -TaskName "$name-$kind" -Xml $xml -Force | Out-Null }
        catch { throw "Task registration failed. Windows policy may require administrator approval, especially for System event triggers. $($_.Exception.Message)" }
    }
}
Write-Output "Tasks: $name-Logon / $name-Resume. WSL shutdown permission: $AllowWslRestart. Windows/WSL execution is unverified."
