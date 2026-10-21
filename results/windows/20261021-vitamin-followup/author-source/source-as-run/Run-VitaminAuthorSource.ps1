#requires -Version 7.0
# 直接调用作者未改的test_4096；只执行SIMT段，原固定顺序和wallclock协议。
[CmdletBinding()]
param(
    [string]$Python = 'D:\Anaconda\Anaconda\envs\pytorch\python.exe',
    [string]$VitaminDir,
    [string]$OutputDir,
    [ValidateRange(1,100)][int]$Batches = 7
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
    $OutputDir = Join-Path $repo "results\windows\vitamin-author-source-$stamp"
} elseif (![IO.Path]::IsPathRooted($OutputDir)) { $OutputDir = Join-Path $repo $OutputDir }
if (Test-Path -LiteralPath $OutputDir) { throw "Refusing to overwrite $OutputDir" }
New-Item -ItemType Directory -Path $OutputDir | Out-Null
$snapshot = Join-Path $OutputDir 'source-as-run'
New-Item -ItemType Directory -Path $snapshot | Out-Null
$sourceFiles = @(
    (Join-Path $repo 'scripts\vitamin_author_source.py'),
    (Join-Path $repo 'scripts\vitamin_python_probe.py'),
    $PSCommandPath,
    (Join-Path $PSScriptRoot 'Initialize-Toolchain.ps1')
)
$sourceRecords = foreach ($file in $sourceFiles) {
    $destination = Join-Path $snapshot (Split-Path $file -Leaf)
    Copy-Item -LiteralPath $file -Destination $destination
    [pscustomobject]@{path=$file;snapshot=$destination;sha256=(Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash}
}
$changes = [ordered]@{
    TORCH_CUDA_ARCH_LIST = '8.9'
    TORCH_EXTENSIONS_DIR = (Join-Path $repo 'build-windows-vitamin-python')
    PYTHONDONTWRITEBYTECODE = '1'
    PYTHONUTF8 = '1'
    PYTHONIOENCODING = 'utf-8'
    VSLANG = '1033'
    MAX_JOBS = '2'
    DISTUTILS_USE_SDK = '1'
    NVIDIA_TF32_OVERRIDE = '0'
}
$previous = @{}
foreach ($key in $changes.Keys) { $previous[$key] = [Environment]::GetEnvironmentVariable($key,'Process') }
try {
    foreach ($key in $changes.Keys) { [Environment]::SetEnvironmentVariable($key,$changes[$key],'Process') }
    $arguments = @('-B','-u',(Join-Path $repo 'scripts\vitamin_author_source.py'),
        '--vitamin',$VitaminDir,'--output-dir',$OutputDir,'--batches',"$Batches")
    [ordered]@{
        executable=$Python; arguments=$arguments; process_environment=$changes
        original_process_environment=$previous; sources=$sourceRecords
        cl=(Get-Command cl.exe -ErrorAction Stop).Source
        nvcc=(Get-Command nvcc.exe -ErrorAction Stop).Source
        ninja=(Get-Command ninja.exe -ErrorAction Stop).Source
        scope='Original test_4096 SIMT portion, stopped before TF32. Seven original wallclock batches; fixed order.'
    } | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 (Join-Path $OutputDir 'command.json')
    & $Python @arguments 2>&1 | Tee-Object -FilePath (Join-Path $OutputDir 'complete-console.log')
    $code = $LASTEXITCODE
    [pscustomobject]@{command='vitamin_author_source';exit_code=$code} |
        Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $OutputDir 'exit-codes.csv')
    if ($code -ne 0) { throw "Author-source route returned $code; raw and partial evidence is preserved." }
    'PASS: original test_4096 SIMT portion completed in all requested batches; TF32 boundary intercepted.' |
        Set-Content -Encoding utf8 (Join-Path $OutputDir 'AUTHOR_SOURCE_PASS.txt')
} catch {
    $_.Exception.Message | Set-Content -Encoding utf8 (Join-Path $OutputDir 'AUTHOR_SOURCE_INCOMPLETE.txt')
    throw
} finally {
    foreach ($key in $changes.Keys) { [Environment]::SetEnvironmentVariable($key,$previous[$key],'Process') }
}
