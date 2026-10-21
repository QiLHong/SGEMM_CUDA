#requires -Version 7.0
# 原始数学配置A和严格配置B使用独立可执行文件；共同C=AB与逐次C0恢复。
[CmdletBinding()]
param(
    [string]$BuildRoot = 'build-windows-vitamin-sm89',
    [string]$OutputDir,
    [ValidateRange(7,100)][int]$Rounds = 7,
    [ValidateRange(30,10000)][int]$Repeats = 30,
    [ValidateRange(10,1000)][int]$Warmup = 10
)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\Initialize-Toolchain.ps1"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
if (![IO.Path]::IsPathRooted($BuildRoot)) { $BuildRoot = Join-Path $repo $BuildRoot }
if (!$OutputDir) { $OutputDir = "results\windows\$((Get-Date).ToString('yyyyMMdd-HHmmss'))-vitamin-simt" }
if (![IO.Path]::IsPathRooted($OutputDir)) { $OutputDir = Join-Path $repo $OutputDir }
if (Test-Path -LiteralPath $OutputDir) { throw "Refusing to overwrite: $OutputDir" }
foreach ($config in 'A','B') {
    foreach ($name in 'cuda_smoke.exe','windows_vitamin_benchmark.exe','verified-build.json','compile_commands.json') {
        if (!(Test-Path -LiteralPath "$BuildRoot-$config\$name")) { throw "Missing $BuildRoot-$config\$name" }
    }
    $verified = Get-Content -LiteralPath "$BuildRoot-$config\verified-build.json" -Raw | ConvertFrom-Json
    if ($verified.configuration -ne $config -or
        $verified.binary_sha256 -ne (Get-FileHash -LiteralPath "$BuildRoot-$config\windows_vitamin_benchmark.exe" -Algorithm SHA256).Hash) {
        throw "Build manifest/binary mismatch for configuration $config. Rebuild before measuring."
    }
}
New-Item -ItemType Directory -Path $OutputDir | Out-Null
$log = "$OutputDir\run.log"
function Invoke-Checked {
    param([string]$Program,[string[]]$Arguments)
    "COMMAND: $Program $($Arguments -join ' ')" | Tee-Object -FilePath $log -Append
    & $Program @Arguments 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE) { throw "Command failed ($LASTEXITCODE); evidence retained: $OutputDir" }
}
$oldTf32 = [Environment]::GetEnvironmentVariable('NVIDIA_TF32_OVERRIDE','Process')
$monitor = $null
try {
    # Both comparisons exclude TF32. A still uses upstream COMPUTE_32F ordinary cuBLAS.
    $env:NVIDIA_TF32_OVERRIDE = '0'
    [ordered]@{
        utc_start=[DateTimeOffset]::UtcNow.ToString('o'); time_zone=[TimeZoneInfo]::Local.Id
        host_utc_offset_minutes=[TimeZoneInfo]::Local.GetUtcOffset([DateTime]::Now).TotalMinutes
        alpha=1; beta=0; seed=20261020; rounds=$Rounds; repeats=$Repeats; warmup=$Warmup
        semantic='C=AB; exact same generated A/B/C0 bytes for all implementations and configs'
        reset='cudaMemcpyAsync D2D C0 -> C before every invocation, same stream, outside CUDA events'
        reset_cache_caveat='Identical protocol, but C0 copy changes caches/traffic; timings describe this protocol'
        configurations=@{A='upstream fast_math; ordinary COMPUTE_32F / DEFAULT fresh cuBLAS handle'
                          B='no fast_math; normal FP32 FMA; PEDANTIC math and COMPUTE_32F_PEDANTIC'}
        validation='Full output vs independent strict cuBLAS; original allclose (candidate as second operand) and abs<=.01 separate; CPU double full128 or256fixedsamples'
        python_entry='This script runs native CUDA only. Any optional PyTorch wallclock reproduction is archived separately.'
        NVIDIA_TF32_OVERRIDE='0'; original_NVIDIA_TF32_OVERRIDE=$oldTf32
        CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES; CUDA_MODULE_LOADING=$env:CUDA_MODULE_LOADING
        sgmm_commit=(& git -C $repo rev-parse HEAD)
        vitamin_commit=(& git -C "$repo\..\vitamin-cuda" rev-parse HEAD)
        GPU='See nvidia-smi-before/after and periodic telemetry; clocks not locked'
        sanitizer='Not completed; unverified'
    } | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 "$OutputDir\environment.json"
    & nvidia-smi | Set-Content -Encoding utf8 "$OutputDir\nvidia-smi-before.txt"
    foreach ($config in 'A','B') {
        $dest = "$OutputDir\$config"
        New-Item -ItemType Directory -Path $dest | Out-Null
        Copy-Item -LiteralPath "$BuildRoot-$config\verified-build.json","$BuildRoot-$config\compile_commands.json",
            "$BuildRoot-$config\CMakeCache.txt" -Destination $dest
        Get-ChildItem -LiteralPath "$BuildRoot-$config" -Filter 'build-*.log' | Copy-Item -Destination $dest
        Invoke-Checked "$BuildRoot-$config\cuda_smoke.exe" @()
        foreach ($n in 128,256,512) {
            Invoke-Checked "$BuildRoot-$config\windows_vitamin_benchmark.exe" `
                @("$n",'1','1','1',"$dest\correctness-$n.csv",'all')
        }
        # Negative host-domain tests: require rejection before opening output / invoking CUDA.
        $reject = foreach ($n in 0,16,127,129,255,4100) {
            $out = "$dest\rejected-$n.csv"
            $text = & "$BuildRoot-$config\windows_vitamin_benchmark.exe" "$n" 1 1 0 $out all 2>&1
            $code = $LASTEXITCODE
            [pscustomobject]@{n=$n;exit_code=$code;created_output=(Test-Path -LiteralPath $out);message=($text -join "`n")}
            if ($code -eq 0 -or (Test-Path -LiteralPath $out)) { throw "Unsafe-shape rejection failed for $n" }
        }
        $reject | Export-Csv -NoTypeInformation -Encoding utf8 "$dest\rejected-shapes.csv"
    }
    $monitor = Start-Process -FilePath (Get-Command nvidia-smi.exe).Source `
        -ArgumentList @('--query-gpu=timestamp,index,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used','--format=csv','--loop-ms=200') `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput "$OutputDir\telemetry.csv" `
        -RedirectStandardError "$OutputDir\telemetry-stderr.txt"
    # Reverse A/B sequence for the middle size; within each process implementations rotate each round.
    foreach ($n in 1024,2048,4096) {
        $configs = if ($n -eq 2048) { @('B','A') } else { @('A','B') }
        foreach ($config in $configs) {
            Invoke-Checked "$BuildRoot-$config\windows_vitamin_benchmark.exe" `
                @("$n","$Rounds","$Repeats","$Warmup","$OutputDir\$config\timings-$n.csv",'all')
        }
    }
    if ($monitor.HasExited) { throw 'Telemetry process exited unexpectedly.' }
    Stop-Process -Id $monitor.Id
    $monitor.WaitForExit()
    $monitor = $null
    & nvidia-smi | Set-Content -Encoding utf8 "$OutputDir\nvidia-smi-after.txt"
    'COMPLETE: raw events retained. Separate allclose and absolute-error decisions in correctness sidecars.' |
        Set-Content -Encoding utf8 "$OutputDir\RUN_COMPLETE.txt"
} finally {
    if ($null -ne $monitor -and !$monitor.HasExited) { Stop-Process -Id $monitor.Id -ErrorAction SilentlyContinue }
    [Environment]::SetEnvironmentVariable('NVIDIA_TF32_OVERRIDE',$oldTf32,'Process')
}
