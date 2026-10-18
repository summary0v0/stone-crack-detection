param(
    [string]$PackageIndex = 'https://pypi.org/simple'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$taskRuntimeVersion = '3.10.22'
$taskUvVersion = '0.13.1'
$taskUv = Join-Path $PSScriptRoot '.tools\bootstrap\bin\uv.exe'
$taskRuntimeRoot = Join-Path $PSScriptRoot '.python'
$taskBasePython = Join-Path $taskRuntimeRoot "cpython-$taskRuntimeVersion-windows-x86_64-none\python.exe"
$taskVenv = Join-Path $PSScriptRoot '.venv'
$taskPython = Join-Path $taskVenv 'Scripts\python.exe'
$taskCache = Join-Path $PSScriptRoot '.tools\pip-cache'

if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
    throw 'This setup targets 64-bit x86 Windows.'
}

# Bootstrap a project-local Python without changing PATH or Windows registration.
if (-not (Test-Path -LiteralPath $taskBasePython)) {
    if (-not (Test-Path -LiteralPath $taskUv)) {
        $taskUvDir = Split-Path -Parent $taskUv
        $taskArchive = Join-Path $PSScriptRoot '.tools\uv-windows.zip'
        New-Item -ItemType Directory -Force $taskUvDir | Out-Null
        Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/astral-sh/uv/releases/download/$taskUvVersion/uv-x86_64-pc-windows-msvc.zip" -OutFile $taskArchive
        Expand-Archive -LiteralPath $taskArchive -DestinationPath $taskUvDir -Force
    }
    & $taskUv python install $taskRuntimeVersion --install-dir $taskRuntimeRoot --no-bin --no-registry --no-config --cache-dir (Join-Path $PSScriptRoot '.tools\uv-cache')
    if ($LASTEXITCODE -ne 0) { throw 'Project-local Python installation failed.' }
}

if (-not (Test-Path -LiteralPath $taskPython)) {
    & $taskBasePython -m venv $taskVenv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
& $taskPython -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 is required"; assert sys.prefix != sys.base_prefix, "A virtual environment is required"'
if ($LASTEXITCODE -ne 0) { throw 'The existing .venv is incompatible. Select a Python 3.10 virtual environment.' }

& $taskPython -m pip --isolated install pip==26.2.1 --index-url $PackageIndex --cache-dir $taskCache --timeout 45 --retries 3 --progress-bar off --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment pip installation failed.' }

# Install CUDA wheels explicitly so a CPU-only package is never selected.
& $taskPython -m pip --isolated install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --no-deps --index-url https://download.pytorch.org/whl/cu118 --cache-dir $taskCache --timeout 45 --retries 3 --progress-bar off --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }

$taskRequirements = Join-Path $PSScriptRoot 'requirements.txt'
$taskLock = Join-Path $PSScriptRoot 'requirements-lock.txt'
$taskConstraints = @()
if (Test-Path -LiteralPath $taskLock) { $taskConstraints = @('-c', $taskLock) }
& $taskPython -m pip --isolated install -r $taskRequirements @taskConstraints --index-url $PackageIndex --cache-dir $taskCache --timeout 45 --retries 3 --prefer-binary --progress-bar off --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed.' }
& $taskPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
$taskReports = Join-Path $PSScriptRoot '.agent\reports'
New-Item -ItemType Directory -Force $taskReports | Out-Null
& $taskPython (Join-Path $PSScriptRoot 'check_environment.py') --output (Join-Path $taskReports 'local-environment.json')
if ($LASTEXITCODE -ne 0) { throw 'Runtime verification failed.' }
Write-Host "Ready: $taskPython"
