#requires -Version 7.0
# 每个配对轮单独启动两个binary，逐轮反转先后；所有GPU工作串行。
[CmdletBinding()]
param(
    [string]$BuildDir = 'build-windows-dbf-register-sm89',
    [string]$OutputDir,
    [ValidateRange(7,100)][int]$Rounds = 7,
    [ValidateRange(30,10000)][int]$Repeats = 30,
    [ValidateRange(10,1000)][int]$Warmup = 10,
    [ValidateSet(1024,2048,4096)][int[]]$Sizes = @(1024,2048,4096),
    [string]$Python = 'D:\Anaconda\Anaconda\python.exe'
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
& "$PSScriptRoot\Initialize-Toolchain.ps1"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
if (![IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
if (!$OutputDir) { $OutputDir = "results\windows\20261021-vitamin-followup\register-limit\run-$(Get-Date -Format 'yyyyMMdd-HHmmss')" }
if (![IO.Path]::IsPathRooted($OutputDir)) { $OutputDir = Join-Path $repo $OutputDir }
if (Test-Path -LiteralPath $OutputDir) { throw "Refusing to overwrite: $OutputDir" }
$verified = Get-Content -LiteralPath "$BuildDir\verified-build.json" -Raw | ConvertFrom-Json
if ($verified.precision -ne 'strict_fp32_no_fast_math' -or $verified.architecture -ne 'sm_89') {
    throw 'Unexpected build configuration.'
}
foreach ($variant in $verified.variants) {
    if ($variant.binary_sha256 -ne (Get-FileHash -LiteralPath $variant.executable -Algorithm SHA256).Hash) {
        throw "Binary mismatch: $($variant.variant)"
    }
}
foreach ($source in $verified.source_sha256) {
    if ($source.sha256 -ne (Get-FileHash -LiteralPath $source.path -Algorithm SHA256).Hash) {
        throw "Build source changed: $($source.path)"
    }
}
New-Item -ItemType Directory -Path $OutputDir | Out-Null
Copy-Item -LiteralPath "$BuildDir\verified-build.json" -Destination $OutputDir
$invocations = [System.Collections.Generic.List[object]]::new()
$log = "$OutputDir\run.log"
function Invoke-DbfRun {
    param([string]$Variant,[int]$Size,[int]$Pair,[int]$Order,[string]$Phase,[int]$Count,[int]$Warm)
    $program = "$BuildDir\windows_dbf_${Variant}_benchmark.exe"
    $filename = "$Phase-$Size-pair$Pair-$Variant.csv"
    $csv = Join-Path $OutputDir $filename
    $arguments = @("$Size",'1',"$Count","$Warm",$csv,'all')
    $start = [DateTimeOffset]::UtcNow
    "COMMAND: $program $($arguments -join ' ')" | Tee-Object -FilePath $log -Append
    & $program @arguments 2>&1 | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    $end = [DateTimeOffset]::UtcNow
    $invocations.Add([pscustomobject][ordered]@{
        phase=$Phase;n=$Size;pair=$Pair;order=$Order;variant=$Variant
        event_file=$filename;repeats=$Count;warmup=$Warm;exit_code=$code
        process_start_unix_ms=$start.ToUnixTimeMilliseconds()
        process_end_unix_ms=$end.ToUnixTimeMilliseconds()
    })
    $invocations | Export-Csv -LiteralPath "$OutputDir\invocations.csv" -NoTypeInformation -Encoding utf8
    if ($code) { throw "DBF $Variant n=$Size pair=$Pair failed ($code); evidence retained." }
}
$oldTf32 = [Environment]::GetEnvironmentVariable('NVIDIA_TF32_OVERRIDE','Process')
$monitor = $null
try {
    $env:NVIDIA_TF32_OVERRIDE = '0'
    [ordered]@{
        created_utc=[DateTimeOffset]::UtcNow.ToString('o')
        time_zone=[TimeZoneInfo]::Local.Id
        host_utc_offset_minutes=[TimeZoneInfo]::Local.GetUtcOffset([DateTime]::Now).TotalMinutes
        sizes=$Sizes;pairs=$Rounds;repeats=$Repeats;warmup=$Warmup
        alpha=1;beta=0;seed=20261020;precision='strict_fp32_no_fast_math'
        factor='default register allocation versus --maxrregcount=128'
        pairing='Odd pairs default then maxr128; even pairs maxr128 then default'
        process_boundary='Each variant/pair is a fresh process with its own strict reference and preflight; events exclude startup, allocation and validation'
        timing='D2D C0 reset before every invocation; same default stream; CUDA events enclose only DBF'
        correctness='Every preflight and final timed output vs strict reference and CPU double; original allclose and abs<=.01 separate'
        NVIDIA_TF32_OVERRIDE='0';original_NVIDIA_TF32_OVERRIDE=$oldTf32
        clocks='not locked';telemetry_period_ms=200
        compiled_evidence=$verified.evidence_dir
        build_script_gpu_execution=$verified.gpu_execution_performed
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$OutputDir\environment.json" -Encoding utf8
    & nvidia-smi | Set-Content -LiteralPath "$OutputDir\nvidia-smi-before.txt" -Encoding utf8
    foreach ($variant in 'default','maxr128') {
        Invoke-DbfRun -Variant $variant -Size 128 -Pair 0 -Order 0 -Phase 'correctness' -Count 1 -Warm 1
    }
    $monitor = Start-Process -FilePath (Get-Command nvidia-smi.exe).Source `
        -ArgumentList @('--query-gpu=timestamp,index,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used','--format=csv','--loop-ms=200') `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput "$OutputDir\telemetry.csv" `
        -RedirectStandardError "$OutputDir\telemetry-stderr.txt"
    foreach ($n in $Sizes) {
        for ($pair = 1; $pair -le $Rounds; ++$pair) {
            $variants = if ($pair % 2) { @('default','maxr128') } else { @('maxr128','default') }
            for ($position = 0; $position -lt 2; ++$position) {
                Invoke-DbfRun -Variant $variants[$position] -Size $n -Pair $pair -Order ($position+1) `
                    -Phase 'timings' -Count $Repeats -Warm $Warmup
            }
        }
    }
    if ($monitor.HasExited) { throw 'Telemetry monitor exited unexpectedly.' }
    Stop-Process -Id $monitor.Id
    $monitor.WaitForExit()
    $monitor = $null
    & nvidia-smi | Set-Content -LiteralPath "$OutputDir\nvidia-smi-after.txt" -Encoding utf8
    'COMPLETE: unprofiled CUDA events retained; original allclose independently reported.' |
        Set-Content -LiteralPath "$OutputDir\RUN_COMPLETE.txt" -Encoding utf8
    & $Python "$repo\scripts\summarize_dbf_register_results.py" --result-dir $OutputDir
    if ($LASTEXITCODE) { throw 'DBF CPU summary verification failed; raw evidence retained.' }
} finally {
    if ($null -ne $monitor -and !$monitor.HasExited) {
        Stop-Process -Id $monitor.Id -ErrorAction SilentlyContinue
    }
    [Environment]::SetEnvironmentVariable('NVIDIA_TF32_OVERRIDE',$oldTf32,'Process')
}
