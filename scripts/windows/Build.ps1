#requires -Version 7.0
[CmdletBinding()]
param(
    [string]$BuildDir = 'build-windows-sm89',
    [ValidatePattern('^[0-9]+$')][string]$Architecture = '89',
    [int]$Jobs = 2,
    [switch]$WithExperiment
)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\Initialize-Toolchain.ps1"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
if (![IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
$log = Join-Path $BuildDir ("build-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
$experiment = if ($WithExperiment) { 'ON' } else { 'OFF' }
$configureArgs = @('-S', $repo, '-B', $BuildDir, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_TRY_COMPILE_CONFIGURATION=Release',
    "-DCMAKE_CUDA_ARCHITECTURES=$Architecture",
    "-DCMAKE_CUDA_COMPILER=$env:CUDACXX", "-DCMAKE_CUDA_HOST_COMPILER=$env:CUDAHOSTCXX",
    "-DCMAKE_CXX_COMPILER=$env:CUDAHOSTCXX", "-DCUDAToolkit_ROOT=$env:CUDA_PATH",
    '-DSGEMM_BUILD_K30_AUDIT=ON', "-DSGEMM_BUILD_WINDOWS_BENCHMARK=$experiment")
& cmake @configureArgs 2>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed; see $log" }
# 先运行最小 CUDA 验证；失败就不继续大程序构建。
& cmake --build $BuildDir --target cuda_smoke --parallel $Jobs --verbose 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { throw "CUDA smoke build failed; see $log" }
& (Join-Path $BuildDir 'cuda_smoke.exe') 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { throw "CUDA smoke run failed; see $log" }
& cmake --build $BuildDir --parallel $Jobs --verbose 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { throw "Build failed; see $log" }
$commands = Get-Content -LiteralPath (Join-Path $BuildDir 'compile_commands.json') -Raw | ConvertFrom-Json
$cudaCommands = @($commands | Where-Object { $_.file -match '\.cu$' })
if (!$cudaCommands.Count) { throw 'No CUDA compile commands were exported.' }
foreach ($command in $cudaCommands) {
    if ($command.command -notmatch "arch=compute_$Architecture,code=\[compute_$Architecture,sm_$Architecture\]") {
        throw "Unexpected CUDA architecture in actual command: $($command.file)"
    }
}
Write-Host "Verified sm_$Architecture in all $($cudaCommands.Count) CUDA compile commands."
Write-Host "Build complete: $BuildDir"
Write-Host "Compile command evidence: $BuildDir\compile_commands.json"
