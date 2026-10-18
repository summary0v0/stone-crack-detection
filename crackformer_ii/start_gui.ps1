param([string]$ImagePath, [ValidateSet('auto','cpu','cuda')][string]$Device = 'auto')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
    throw 'Project Python was not found. Run setup_env.ps1 in the project root first.'
}
$taskGui = Join-Path $PSScriptRoot 'gui.py'
# Start-Process expects quoted paths in its argument string on Windows.
$taskArguments = '"' + $taskGui + '" --device ' + $Device
if ($ImagePath) {
    $taskResolvedImage = (Resolve-Path -LiteralPath $ImagePath).Path
    $taskArguments += ' --image "' + $taskResolvedImage + '"'
}
Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden
