#requires -Version 7.0
# 第二阶段：复用 windows_benchmark 的数值/计时协议，选择原有 K10、K30 V1/V2。
[CmdletBinding()]
param(
    [string]$BuildDir = 'build-windows-k30-sync-sm89',
    [string]$OutputDir,
    [ValidateRange(7,100)][int]$Rounds = 7,
    [ValidateRange(30,10000)][int]$Repeats = 30,
    [ValidateRange(10,1000)][int]$Warmup = 10,
    [switch]$SkipCorrectnessMatrix
)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\Initialize-Toolchain.ps1"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
if (![IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
if (!$OutputDir) {
    $stamp = [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow,'China Standard Time').ToString('yyyyMMdd-HHmmss')
    $OutputDir = Join-Path $repo "results\windows\$stamp-k30-sync"
} elseif (![IO.Path]::IsPathRooted($OutputDir)) { $OutputDir = Join-Path $repo $OutputDir }
if (Test-Path -LiteralPath $OutputDir) { throw "Refusing to overwrite existing results: $OutputDir" }
$bench = Join-Path $BuildDir 'windows_benchmark.exe'
foreach ($path in @($bench, "$BuildDir\cuda_smoke.exe", "$BuildDir\k30_audit.exe", "$BuildDir\compile_commands.json")) {
    if (!(Test-Path -LiteralPath $path)) { throw "Missing $path. Run Build.ps1 -WithExperiment first." }
}
New-Item -ItemType Directory -Path $OutputDir | Out-Null
$log = Join-Path $OutputDir 'run.log'
function Invoke-Checked {
    param([string]$Program,[string[]]$Arguments)
    ("COMMAND: {0} {1}" -f $Program, ($Arguments -join ' ')) | Tee-Object -FilePath $log -Append
    & $Program @Arguments 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Program; evidence retained at $OutputDir" }
}
$originalTf32 = [Environment]::GetEnvironmentVariable('NVIDIA_TF32_OVERRIDE','Process')
$monitor = $null
try {
    $env:NVIDIA_TF32_OVERRIDE = '0'
    $commands = Get-Content "$BuildDir\compile_commands.json" -Raw | ConvertFrom-Json
    $cudaCommands = @($commands | Where-Object file -Match '\.cu$')
    if (!$cudaCommands.Count -or @($cudaCommands | Where-Object command -NotMatch 'arch=compute_89,code=\[compute_89,sm_89\]').Count) {
        throw 'This stage requires actual sm_89 compile commands.'
    }
    $metadata = [ordered]@{
        time_china = [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow,'China Standard Time').ToString('o')
        utc_start_unix_ms = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        host_time_zone = [TimeZoneInfo]::Local.Id
        host_utc_offset_minutes = [TimeZoneInfo]::Local.GetUtcOffset([DateTime]::Now).TotalMinutes
        os = [Environment]::OSVersion.VersionString
        branch = (& git -C $repo branch --show-current)
        base_commit = (& git -C $repo rev-parse HEAD)
        nvcc = ((& nvcc --version) -join "`n")
        cl_path = (Get-Command cl.exe).Source
        cl_version = (Get-Item (Get-Command cl.exe).Source).VersionInfo.FileVersion
        cmake = ((& cmake --version) -join "`n")
        ninja = (& ninja --version)
        windows_sdk = "$env:WindowsSdkDir $env:WindowsSDKVersion"
        architecture = 'sm_89; verified all CUDA entries in compile_commands.json'
        build_dir = $BuildDir
        implementations = @('k10_pad4','k30_v1','k30_v2','cublas_strict_fp32')
        rounds = $Rounds; repeats = $Repeats; warmup = $Warmup
        seed = 20261020; alpha = 0.5; beta = 3
        timing = 'CUDA event for each GEMM; C0 D2D before start event in same stream; reset excluded'
        validation = 'full output before timings and after each round; abs <=0.01, NaN/Inf fails; CPU double reference check'
        telemetry = 'one external nvidia-smi process, 200 ms interval during timing sweep only; timestamps are host local time; GPU clocks not locked'
        round_window = 'UTC epoch ms before warmup to after output validation; not GEMM-only interval'
        NVIDIA_TF32_OVERRIDE = $env:NVIDIA_TF32_OVERRIDE
        original_NVIDIA_TF32_OVERRIDE = $originalTf32
        CUDA_VISIBLE_DEVICES = $env:CUDA_VISIBLE_DEVICES
        CUDA_MODULE_LOADING = $env:CUDA_MODULE_LOADING
        correctness_matrix = !$SkipCorrectnessMatrix
    }
    $metadata | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 "$OutputDir\environment.json"
    & nvidia-smi | Set-Content -Encoding utf8 "$OutputDir\nvidia-smi-before.txt"
    & git -C $repo status --short | Set-Content -Encoding utf8 "$OutputDir\git-status.txt"
    & git -C $repo diff --binary | Set-Content -Encoding utf8 "$OutputDir\tracked-changes.patch"
    Copy-Item -LiteralPath "$BuildDir\compile_commands.json","$BuildDir\CMakeCache.txt" -Destination $OutputDir
    Get-ChildItem -LiteralPath $BuildDir -Filter 'build-*.log' | Copy-Item -Destination $OutputDir
    $sourceFiles = Get-ChildItem -Path "$repo\src","$repo\scripts" -Recurse -File |
        Where-Object Extension -In '.cu','.cuh','.h','.ps1','.py'
    $hashFiles = @($sourceFiles.FullName) + @("$repo\CMakeLists.txt","$repo\sgemm.cu","$repo\simplest_kernel.cu","$repo\cuBLAS_sgemm.cu") +
        @(Get-ChildItem -LiteralPath $BuildDir -Filter '*.exe' | ForEach-Object FullName)
    $hashFiles | ForEach-Object { Get-FileHash -LiteralPath $_ -Algorithm SHA256 } |
        Select-Object Path,Hash | Export-Csv -NoTypeInformation -Encoding utf8 "$OutputDir\sha256.csv"
    Compress-Archive -Path "$repo\src","$repo\scripts","$repo\CMakeLists.txt","$repo\sgemm.cu",
        "$repo\simplest_kernel.cu","$repo\cuBLAS_sgemm.cu","$repo\LICENSE" -DestinationPath "$OutputDir\source-snapshot.zip"
    Invoke-Checked "$BuildDir\cuda_smoke.exe" @()
    if (!$SkipCorrectnessMatrix) {
        foreach ($beta in 0,1,3) {
            foreach ($n in 128,256,512,1024,2048,4096) {
                # 同一组框架比较全部四种实现；原独立入口也逐一验证 V1/V2。
                Invoke-Checked "$BuildDir\k30_audit.exe" @('v1',"$n","$beta",'1','1')
                Invoke-Checked "$BuildDir\k30_audit.exe" @('v2',"$n","$beta",'1','1')
                Invoke-Checked $bench @("$n","$beta",'1','1','1',"$OutputDir\correctness-$n-beta$beta.csv",'sync')
            }
        }
    }
    $telemetryArgs = @('--query-gpu=timestamp,index,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used',
        '--format=csv','--loop-ms=200')
    # 只创建本次任务自己的监测进程；不改变时钟、电源、驱动或性能计数器权限。
    $monitor = Start-Process -FilePath (Get-Command nvidia-smi.exe).Source -ArgumentList $telemetryArgs `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput "$OutputDir\telemetry.csv" `
        -RedirectStandardError "$OutputDir\telemetry-stderr.txt"
    foreach ($n in 1024,2048,4096) {
        Invoke-Checked $bench @("$n",'3',"$Rounds","$Repeats","$Warmup","$OutputDir\timings-$n.csv",'sync')
    }
    if ($monitor.HasExited) { throw "GPU telemetry exited unexpectedly ($($monitor.ExitCode)); inspect telemetry-stderr.txt." }
    Stop-Process -Id $monitor.Id -ErrorAction Stop
    $monitor.WaitForExit()
    $monitor = $null
    if (!(Get-Content "$OutputDir\telemetry.csv" -TotalCount 2 | Select-Object -Skip 1)) { throw 'No telemetry samples collected.' }
    & nvidia-smi | Set-Content -Encoding utf8 "$OutputDir\nvidia-smi-after.txt"
    & "$PSScriptRoot\Summarize-Results.ps1" -ResultDir $OutputDir
    & "$PSScriptRoot\Summarize-Telemetry.ps1" -ResultDir $OutputDir
    'PASS: correctness (if enabled), CUDA event measurement, telemetry and summaries completed.' |
        Set-Content -Encoding utf8 "$OutputDir\PASS.txt"
    Write-Host "Results: $OutputDir"
} finally {
    if ($null -ne $monitor -and !$monitor.HasExited) { Stop-Process -Id $monitor.Id -ErrorAction SilentlyContinue }
    [Environment]::SetEnvironmentVariable('NVIDIA_TF32_OVERRIDE',$originalTf32,'Process')
}
