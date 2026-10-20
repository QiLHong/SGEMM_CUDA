#requires -Version 7.0
[CmdletBinding()]
param([string]$BuildDir='build-windows-tensor-grid-sm89',[string]$OutputDir,
      [ValidateRange(7,100)][int]$Rounds=7,[ValidateRange(30,10000)][int]$Repeats=30,
      [ValidateRange(10,1000)][int]$Warmup=10,[ValidateSet('all','grid_w1','grid_w2','grid_w4')][string]$Selection='all', [int[]]$Sizes=@(1024,2048,4096))
$ErrorActionPreference='Stop'
& "$PSScriptRoot\Initialize-Toolchain.ps1"
$repo=(Resolve-Path "$PSScriptRoot\..\..").Path
if (![IO.Path]::IsPathRooted($BuildDir)) { $BuildDir=Join-Path $repo $BuildDir }
if (!$OutputDir) { $OutputDir="results\windows\20261021-tensorcore-grid\run-$(Get-Date -Format yyyyMMdd-HHmmss)" }
if (![IO.Path]::IsPathRooted($OutputDir)) { $OutputDir=Join-Path $repo $OutputDir }
if (Test-Path -LiteralPath $OutputDir) { throw "Refusing overwrite: $OutputDir" }
$verified=Get-Content "$BuildDir\verified-build.json" -Raw | ConvertFrom-Json
$exe=Join-Path $BuildDir 'windows_tensor_grid_benchmark.exe'
if ($verified.binary_sha256 -ne (Get-FileHash $exe -Algorithm SHA256).Hash -or
    $verified.source_sha256 -ne (Get-FileHash "$repo\experiments\tensorcore-grid\benchmark.cu" -Algorithm SHA256).Hash) {
    throw 'Binary/source mismatch: rebuild first'
}
New-Item -ItemType Directory -Path $OutputDir | Out-Null
Copy-Item -LiteralPath "$BuildDir\verified-build.json",$PSCommandPath -Destination $OutputDir
foreach ($dep in (Get-Content "$BuildDir\dependencies.json" -Raw | ConvertFrom-Json)) {
    if ((Get-FileHash -LiteralPath $dep.path -Algorithm SHA256).Hash -ne $dep.sha256) { throw "Changed dependency: $($dep.path)" }
}
$old=[Environment]::GetEnvironmentVariable('NVIDIA_TF32_OVERRIDE','Process')
$monitor=$null
function Run-Checked([string[]]$Arguments) {
    "COMMAND $exe $($Arguments -join ' ')" | Tee-Object "$OutputDir\run.log" -Append
    & $exe @Arguments 2>&1 | Tee-Object "$OutputDir\run.log" -Append
    if ($LASTEXITCODE) { throw "Experiment failed: $LASTEXITCODE; evidence retained" }
}
try {
    # 只影响本进程及子进程；strict handle仍显式PEDANTIC。
    [Environment]::SetEnvironmentVariable('NVIDIA_TF32_OVERRIDE',$null,'Process')
    [ordered]@{
        utc_start=[DateTimeOffset]::UtcNow.ToString('o');gpu='RTX 4090 D only; no RTX 3080 measurements'
        original_tf32_override=$old;active_tf32_override=$null;rounds=$Rounds;repeats=$Repeats;warmup=$Warmup
        alpha=1;beta=0;seed=20261020;precision_groups=@('TF32 input / FP32 accumulator+output','FP16 input+output / FP32 accumulator')
        fp32_reference='PEDANTIC FP32 checked separately; not timed in this experiment'
        conversion='GPU explicit conversion before timing; all implementations within a precision group share exact input bytes'
        timing='CUDA events per invocation; C0 zero restore D2D same stream before every warmup/timed invocation, outside events'
        cache_caveat='C0 reset changes caches; FP16 copies half as many bytes as FP32; cross-precision speed is not a single-factor algorithm claim'
        selection=$Selection
        order='Scan: rotate/reverse five implementations per precision. Confirmation: baseline/candidate adjacent and swap; cuBLAS rotates outside pair.'
        thresholds='Original abs<=.01 and SIMT atol=rtol=.001 reported independently; original TF32 .1 and HGEMM .016 also reported'
        cpu_double='Full128x1024 outputs for K32 and K64; 256 fixed positions per large square; FP16 reference rounded to FP16 output'
        sanitizer='Not completed: unverified';profiler='Separate from fair event table';clocks='Not locked; WDDM'
        sgmm_commit=(& git -C $repo rev-parse HEAD);vitamin_commit=(& git -C "$repo\..\vitamin-cuda" rev-parse HEAD)
    } | ConvertTo-Json -Depth 5 | Set-Content "$OutputDir\environment.json" -Encoding utf8
    & nvidia-smi | Set-Content "$OutputDir\nvidia-smi-before.txt"
    foreach ($k in 32,64) { Run-Checked @('128','1024',"$k",'1','1','1',"$OutputDir\small-$k.csv",'all') }
    $rejections=foreach ($shape in @(@(128,128,32),@(127,1024,32),@(128,1152,32),@(128,1024,31),@(0,1024,32))) {
        $out="$OutputDir\reject-$($shape -join '-').csv"
        $text=& $exe @shape 1 1 0 $out all 2>&1
        $exitCode=$LASTEXITCODE
        if ($exitCode -eq 0 -or (Test-Path -LiteralPath $out)) { throw 'Shape rejection failed' }
        [pscustomobject]@{shape=($shape -join 'x');exit_code=$exitCode;message=($text -join "`n")}
    }
    $rejections | Export-Csv "$OutputDir\shape-rejections.csv" -NoTypeInformation
    $monitor=Start-Process -FilePath (Get-Command nvidia-smi.exe).Source `
        -ArgumentList @('--query-gpu=timestamp,index,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used','--format=csv','--loop-ms=200') `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput "$OutputDir\telemetry.csv" -RedirectStandardError "$OutputDir\telemetry-stderr.txt"
    
    
    foreach ($n in $sizes) {
        Run-Checked @("$n","$n","$n","$Rounds","$Repeats","$Warmup","$OutputDir\timings-$n.csv",$selection)
    }
    if ($monitor.HasExited) { throw 'Telemetry exited unexpectedly' }
    Stop-Process -Id $monitor.Id
    $monitor.WaitForExit();$monitor=$null
    & nvidia-smi | Set-Content "$OutputDir\nvidia-smi-after.txt"
    'COMPLETE; original FP32 precision failures remain visible in checks CSV' | Set-Content "$OutputDir\RUN_COMPLETE.txt"
} finally {
    if ($null -ne $monitor -and !$monitor.HasExited) { Stop-Process -Id $monitor.Id -ErrorAction SilentlyContinue }
    [Environment]::SetEnvironmentVariable('NVIDIA_TF32_OVERRIDE',$old,'Process')
}

