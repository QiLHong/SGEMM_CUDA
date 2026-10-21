#requires -Version 7.0
# 单进程按轮交替 K10 / PAD8 / K30 V2 / strict cuBLAS；结果始终写入新目录。
[CmdletBinding()]
param(
    [string]$BuildDir = 'build-windows-sm89',
    [string]$OutputDir,
    [ValidateRange(5,100)][int]$Rounds = 7,
    [ValidateRange(1,10000)][int]$Repeats = 30,
    [ValidateRange(1,1000)][int]$Warmup = 10,
    [switch]$SkipCorrectnessMatrix
)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\Initialize-Toolchain.ps1"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
if (![IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
if (!$OutputDir) {
    $stamp = [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow, 'China Standard Time').ToString('yyyyMMdd-HHmmss')
    $OutputDir = Join-Path $repo "results\windows\$stamp"
} elseif (![IO.Path]::IsPathRooted($OutputDir)) { $OutputDir = Join-Path $repo $OutputDir }
if (Test-Path -LiteralPath $OutputDir) { throw "Refusing to overwrite existing results: $OutputDir" }
$bench = Join-Path $BuildDir 'windows_benchmark.exe'
foreach ($exe in @($bench, (Join-Path $BuildDir 'sgemm.exe'), (Join-Path $BuildDir 'k30_audit.exe'))) {
    if (!(Test-Path -LiteralPath $exe)) { throw "Missing $exe. Run Build.ps1 -WithExperiment first." }
}
New-Item -ItemType Directory -Path $OutputDir | Out-Null
$log = Join-Path $OutputDir 'run.log'
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    ("COMMAND: {0} {1}" -f $Program, ($Arguments -join ' ')) | Tee-Object -FilePath $log -Append
    & $Program @Arguments 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Program; results retained at $OutputDir" }
}
$envNames = @('NVIDIA_TF32_OVERRIDE','DEVICE','SGEMM_SEED','SGEMM_ALPHA','SGEMM_BETA',
    'SGEMM_RESET_C_EACH_REPEAT','SGEMM_REPEAT','SGEMM_WARMUP','SGEMM_BENCH_SIZE')
$previous = @{}
foreach ($name in $envNames) { $previous[$name] = [Environment]::GetEnvironmentVariable($name,'Process') }
try {
    # 仅本进程防止 TF32 override；代码还明确指定 PEDANTIC math/compute 模式。
    $env:NVIDIA_TF32_OVERRIDE = '0'
    $env:DEVICE = '0'
    $env:SGEMM_SEED = '20261020'
    $env:SGEMM_ALPHA = '0.5'
    $env:SGEMM_RESET_C_EACH_REPEAT = '1'
    $env:SGEMM_REPEAT = '1'
    $env:SGEMM_WARMUP = '1'
    Remove-Item Env:SGEMM_BENCH_SIZE -ErrorAction SilentlyContinue
    $metadata = [ordered]@{
        time_china = [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow,'China Standard Time').ToString('o')
        os = [Environment]::OSVersion.VersionString
        branch = (& git -C $repo branch --show-current)
        base_commit = (& git -C $repo rev-parse HEAD)
        nvcc = ((& nvcc --version) -join "`n")
        cl_path = (Get-Command cl.exe).Source
        cl_version = (Get-Item (Get-Command cl.exe).Source).VersionInfo.FileVersion
        cmake = ((& cmake --version) -join "`n")
        ninja = (& ninja --version)
        windows_sdk = "$env:WindowsSdkDir $env:WindowsSDKVersion"
        architecture = 'See compile_commands.json; this run requires the selected build'
        rounds = $Rounds; repeats = $Repeats; warmup = $Warmup
        seed = 20261020; alpha = 0.5; beta = 3
        timing = 'CUDA events per GEMM; initial C restored by D2D before start event; copy excluded'
        accuracy = 'absolute error <=0.01 and no NaN/Inf; strict cuBLAS plus CPU double sample'
        NVIDIA_TF32_OVERRIDE = $env:NVIDIA_TF32_OVERRIDE
        original_NVIDIA_TF32_OVERRIDE = $previous['NVIDIA_TF32_OVERRIDE']
        CUDA_VISIBLE_DEVICES = $env:CUDA_VISIBLE_DEVICES
        CUDA_MODULE_LOADING = $env:CUDA_MODULE_LOADING
        correctness_matrix = !$SkipCorrectnessMatrix
    }
    $metadata | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $OutputDir 'environment.json')
    & nvidia-smi | Set-Content -Encoding utf8 (Join-Path $OutputDir 'nvidia-smi-before.txt')
    & git -C $repo status --short | Set-Content -Encoding utf8 (Join-Path $OutputDir 'git-status.txt')
    & git -C $repo diff --binary | Set-Content -Encoding utf8 (Join-Path $OutputDir 'tracked-changes.patch')
    Copy-Item -LiteralPath (Join-Path $BuildDir 'compile_commands.json') -Destination $OutputDir
    Copy-Item -LiteralPath (Join-Path $BuildDir 'CMakeCache.txt') -Destination $OutputDir
    Get-ChildItem -LiteralPath $BuildDir -Filter 'build-*.log' | Copy-Item -Destination $OutputDir
    $sourceFiles = Get-ChildItem -Path "$repo\src","$repo\scripts" -Recurse -File |
        Where-Object Extension -In '.cu','.cuh','.h','.ps1'
    $hashFiles = @($sourceFiles.FullName) + @("$repo\CMakeLists.txt","$repo\sgemm.cu","$repo\simplest_kernel.cu","$repo\cuBLAS_sgemm.cu") +
        @(Get-ChildItem -LiteralPath $BuildDir -Filter '*.exe' | ForEach-Object FullName)
    $hashFiles | ForEach-Object { Get-FileHash -LiteralPath $_ -Algorithm SHA256 } |
        Select-Object Path,Hash | Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $OutputDir 'sha256.csv')
    # 包括新增、尚未提交的源文件，避免仅保存 git diff 遗漏实验程序。
    Compress-Archive -Path "$repo\src","$repo\scripts","$repo\CMakeLists.txt",
        "$repo\sgemm.cu","$repo\simplest_kernel.cu","$repo\cuBLAS_sgemm.cu","$repo\LICENSE" `
        -DestinationPath (Join-Path $OutputDir 'source-snapshot.zip')
    Invoke-Checked (Join-Path $BuildDir 'cuda_smoke.exe') @()
    if (!$SkipCorrectnessMatrix) {
        foreach ($beta in 0,1,3) {
            $env:SGEMM_BETA = "$beta"
            # 原教学入口也实跑，保留原错误阈值和输出协议。
            Invoke-Checked (Join-Path $BuildDir 'sgemm.exe') @('10')
            foreach ($n in 128,256,512,1024,2048,4096) {
                Invoke-Checked (Join-Path $BuildDir 'k30_audit.exe') @('v2',"$n","$beta",'1','1')
                Invoke-Checked $bench @("$n","$beta",'1','1','1',(Join-Path $OutputDir "correctness-$n-beta$beta.csv"),'all')
            }
        }
    }
    foreach ($n in 1024,2048,4096) {
        & nvidia-smi --query-gpu=timestamp,name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used --format=csv |
            Add-Content -Encoding utf8 (Join-Path $OutputDir 'telemetry.csv')
        Invoke-Checked $bench @("$n",'3',"$Rounds","$Repeats","$Warmup",(Join-Path $OutputDir "timings-$n.csv"),'all')
    }
    & nvidia-smi | Set-Content -Encoding utf8 (Join-Path $OutputDir 'nvidia-smi-after.txt')
    & "$PSScriptRoot\Summarize-Results.ps1" -ResultDir $OutputDir
    'PASS: correctness (if enabled) and timing commands exited zero.' |
        Set-Content -Encoding utf8 (Join-Path $OutputDir 'PASS.txt')
    Write-Host "Results: $OutputDir"
} finally {
    foreach ($name in $envNames) { [Environment]::SetEnvironmentVariable($name,$previous[$name],'Process') }
}
