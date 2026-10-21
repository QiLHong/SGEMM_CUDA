#requires -Version 7.0
# 两个独立 Release 构建目录，原始 fast_math 与严格 FP32 不共用对象文件。
[CmdletBinding()]
param(
    [ValidateSet('A','B','Both')][string]$Configuration = 'Both',
    [string]$BuildRoot = 'build-windows-vitamin-sm89',
    [string]$VitaminRepo = '..\vitamin-cuda',
    [string]$Python = 'D:\Anaconda\Anaconda\python.exe',
    [ValidateRange(1,16)][int]$Jobs = 2
)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\Initialize-Toolchain.ps1"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
if (![IO.Path]::IsPathRooted($BuildRoot)) { $BuildRoot = Join-Path $repo $BuildRoot }
if (![IO.Path]::IsPathRooted($VitaminRepo)) { $VitaminRepo = Join-Path $repo $VitaminRepo }
& $Python "$repo\scripts\extract_vitamin_simt.py" --vitamin $VitaminRepo `
    --output "$repo\src\kernels\vendor\vitamin_sgemm.cuh" `
    --manifest "$repo\src\kernels\vendor\provenance.json" --verify
if ($LASTEXITCODE) { throw 'Upstream device extraction verification failed.' }
$configs = if ($Configuration -eq 'Both') { @('A','B') } else { @($Configuration) }
foreach ($config in $configs) {
    $build = "$BuildRoot-$config"
    New-Item -ItemType Directory -Force -Path $build | Out-Null
    $log = Join-Path $build ("build-" + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '.log')
    $original = if ($config -eq 'A') { 'ON' } else { 'OFF' }
    $argsConfigure = @('-S',$repo,'-B',$build,'-G','Ninja',
        '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_TRY_COMPILE_CONFIGURATION=Release',
        '-DCMAKE_CUDA_ARCHITECTURES=89',"-DCMAKE_CUDA_COMPILER=$env:CUDACXX",
        "-DCMAKE_CUDA_HOST_COMPILER=$env:CUDAHOSTCXX","-DCMAKE_CXX_COMPILER=$env:CUDAHOSTCXX",
        "-DCUDAToolkit_ROOT=$env:CUDA_PATH",'-DSGEMM_BUILD_VITAMIN_BENCHMARK=ON',
        "-DSGEMM_VITAMIN_ORIGINAL_CONFIG=$original")
    & cmake @argsConfigure 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE) { throw "CMake configure failed: $config" }
    & cmake --build $build --target cuda_smoke windows_vitamin_benchmark --parallel $Jobs --verbose 2>&1 |
        Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE) { throw "Build failed: $config" }
    & "$build\cuda_smoke.exe" 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE) { throw "CUDA smoke failed: $config" }
    $entries = @(Get-Content "$build\compile_commands.json" -Raw | ConvertFrom-Json |
        Where-Object file -Match '\.cu$')
    if (!$entries.Count -or @($entries | Where-Object command -NotMatch 'arch=compute_89,code=\[compute_89,sm_89\]').Count) {
        throw 'CUDA compile command missing sm_89.'
    }
    $benchCommand = @($entries | Where-Object file -Match 'windows_vitamin_benchmark\.cu$')
    if ($benchCommand.Count -ne 1) { throw 'Missing unique benchmark compile command.' }
    $fast = $benchCommand[0].command -match '--use_fast_math'
    if ($fast -ne ($config -eq 'A')) { throw 'Fast math configuration mismatch.' }
    if ($benchCommand[0].command -match 'allow-unsupported-compiler') { throw 'Unsupported compiler bypass forbidden.' }
    [ordered]@{ configuration=$config; architecture='sm_89'; use_fast_math=$fast
        compile_command=$benchCommand[0].command
        binary_sha256=(Get-FileHash "$build\windows_vitamin_benchmark.exe" -Algorithm SHA256).Hash
    } | ConvertTo-Json -Depth 3 | Set-Content -Encoding utf8 "$build\verified-build.json"
}
