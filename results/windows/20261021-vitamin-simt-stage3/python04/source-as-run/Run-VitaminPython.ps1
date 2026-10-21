#requires -Version 7.0
# 可选原 Python wallclock 路线；数据与原生 CUDA events 归档分开。
[CmdletBinding()]
param(
    [string]$Python = 'D:\Anaconda\Anaconda\envs\pytorch\python.exe',
    [string]$VitaminDir,
    [string]$OutputDir,
    [ValidateSet(1024,2048,4096)][int[]]$Sizes = @(1024,2048,4096)
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
& "$PSScriptRoot\Initialize-Toolchain.ps1"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
if (!$VitaminDir) { $VitaminDir = Join-Path (Split-Path $repo) 'vitamin-cuda' }
$VitaminDir = (Resolve-Path -LiteralPath $VitaminDir).Path
if (!(Test-Path -LiteralPath $Python)) { throw "Missing existing Python: $Python" }
if (!$OutputDir) {
    $stamp = [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow,
        'China Standard Time').ToString('yyyyMMdd-HHmmss-fff')
    $OutputDir = Join-Path $repo "results\windows\vitamin-python-$stamp"
} elseif (![IO.Path]::IsPathRooted($OutputDir)) { $OutputDir = Join-Path $repo $OutputDir }
if (Test-Path -LiteralPath $OutputDir) { throw "Refusing to overwrite $OutputDir" }
New-Item -ItemType Directory -Path $OutputDir | Out-Null
$changes = [ordered]@{
    TORCH_CUDA_ARCH_LIST = '8.9'
    TORCH_EXTENSIONS_DIR = (Join-Path $repo 'build-windows-vitamin-python')
    PYTHONDONTWRITEBYTECODE = '1'
    # PyTorch 2.5 hashes source via open() without encoding; upstream CU files are UTF-8.
    PYTHONUTF8 = '1'
    PYTHONIOENCODING = 'utf-8'
    VSLANG = '1033'
    MAX_JOBS = '2'
    DISTUTILS_USE_SDK = '1'
    NVIDIA_TF32_OVERRIDE = '0'
}
$previous = @{}
foreach ($key in $changes.Keys) {
    $previous[$key] = [Environment]::GetEnvironmentVariable($key,'Process')
}
try {
    foreach ($key in $changes.Keys) {
        [Environment]::SetEnvironmentVariable($key,$changes[$key],'Process')
    }
    $arguments = @('-B','-u',(Join-Path $repo 'scripts\vitamin_python_probe.py'),
        '--vitamin',$VitaminDir,'--output-dir',$OutputDir,'--sizes') + @($Sizes | ForEach-Object { "$_" })
    [ordered]@{
        executable=$Python; arguments=$arguments; process_environment=$changes
        original_process_environment=$previous
        cl=(Get-Command cl.exe -ErrorAction Stop).Source
        nvcc=(Get-Command nvcc.exe -ErrorAction Stop).Source
        ninja=(Get-Command ninja.exe -ErrorAction Stop).Source
        script_sha256=(Get-FileHash (Join-Path $repo 'scripts\vitamin_python_probe.py') -Algorithm SHA256).Hash
        scope='Optional upstream single-batch wallclock protocol; never merge with native event samples.'
    } | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 (Join-Path $OutputDir 'command.json')
    & $Python @arguments 2>&1 | Tee-Object -FilePath (Join-Path $OutputDir 'complete-console.log')
    $code = $LASTEXITCODE
    [pscustomobject]@{command='vitamin_python_probe';exit_code=$code} |
        Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $OutputDir 'exit-codes.csv')
    if ($code -ne 0) {
        throw "Python route returned $code. Logs and partial results are preserved; no environment installation or repair was attempted."
    }
    'PASS: optional upstream wallclock experiment completed; see python-result.json and validation CSV.' |
        Set-Content -Encoding utf8 (Join-Path $OutputDir 'PYTHON_PASS.txt')
} catch {
    $_.Exception.Message | Set-Content -Encoding utf8 (Join-Path $OutputDir 'PYTHON_INCOMPLETE.txt')
    throw
} finally {
    foreach ($key in $changes.Keys) {
        [Environment]::SetEnvironmentVariable($key,$previous[$key],'Process')
    }
}
