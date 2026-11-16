[CmdletBinding(SupportsShouldProcess = $true)]
param([Parameter(Mandatory = $true)][string]$Distro,
      [Parameter(Mandatory = $true)][string]$LoopDir)
$ErrorActionPreference = 'Stop'
$hash = [Security.Cryptography.SHA256]::Create()
$id = ([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes("$Distro|$LoopDir")))).Replace('-', '').Substring(0, 12)
$hash.Dispose()
foreach ($kind in @('Logon', 'Resume')) {
    $name = "HasamiAutoloop-$id-$kind"
    if ($PSCmdlet.ShouldProcess($name, 'Stop and unregister task (preserve all training data)')) {
        if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
            Stop-ScheduledTask -TaskName $name
            Unregister-ScheduledTask -TaskName $name -Confirm:$false
        }
    }
}
Write-Output 'Tasks removed. Runtime scripts and Linux training data were preserved. Stop any keep-awake PowerShell child if Windows left it running.'
