#requires -Version 7.0
[CmdletBinding()]
param(
    [string]$BuildDir='build-windows-tensor-bn64-robustness-sm89',
    [Parameter(Mandatory)][string]$OutputDir,
    [switch]$PrepareOnly,
    [switch]$ExecutePrepared,
    [string]$SmokeEvidenceDir
)
$ErrorActionPreference='Stop'
if ($PrepareOnly -and $ExecutePrepared) { throw 'Choose PrepareOnly or ExecutePrepared' }
& "$PSScriptRoot\Initialize-Toolchain.ps1"
$repo=(Resolve-Path "$PSScriptRoot\..\..").Path
if (![IO.Path]::IsPathRooted($BuildDir)) { $BuildDir=Join-Path $repo $BuildDir }
if (![IO.Path]::IsPathRooted($OutputDir)) { $OutputDir=Join-Path $repo $OutputDir }
$exe=Join-Path $BuildDir 'windows_tensor_bn64_robustness_benchmark.exe'
$verified=Get-Content "$BuildDir\verified-build.json" -Raw | ConvertFrom-Json
$binaryHash=(Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
if ($verified.binary_sha256 -ne $binaryHash -or $verified.source_sha256 -ne
    (Get-FileHash "$repo\experiments\tensorcore-bn64-robustness\benchmark.cu" -Algorithm SHA256).Hash) {
    throw 'Binary/source mismatch: rebuild before running'
}
foreach ($dep in (Get-Content "$BuildDir\dependencies.json" -Raw | ConvertFrom-Json)) {
    if ((Get-FileHash -LiteralPath $dep.path -Algorithm SHA256).Hash -ne $dep.sha256) { throw "Changed dependency: $($dep.path)" }
}
$planFile=Join-Path $OutputDir 'plan.csv'
if (!$ExecutePrepared) {
    if (Test-Path -LiteralPath $OutputDir) { throw "Refusing overwrite: $OutputDir" }
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
    $names=@('vitamin_tf32_dbf','tf32_bn64_bk32','cublas_tf32')
    $permutations=@(@(0,1,2),@(0,2,1),@(1,0,2),@(1,2,0),@(2,0,1),@(2,1,0))
    $sizes=@(1024,2048,3072,4096);$seeds=@(20261020,42,12345)
    $lines=[System.Collections.Generic.List[string]]::new()
    $lines.Add('batch,case_index,m,n,k,seed,round,order,implementation')
    foreach ($batch in 1..3) {
        $case=0
        for($si=0;$si -lt 4;$si++) { for($vi=0;$vi -lt 3;$vi++) {
            $n=$sizes[($si+$batch-1)%4];$seed=$seeds[($vi+$batch-1)%3]
            for($round=0;$round -lt 7;$round++) {
                $pi=if($round -eq 6){$batch-1}else{($round+$case+$batch-1)%6}
                for($order=0;$order -lt 3;$order++) {
                    $name=$names[$permutations[$pi][$order]]
                    $lines.Add("$batch,$case,$n,$n,$n,$seed,$round,$order,$name")
                }
            }
            $case++
        }}
    }
    [IO.File]::WriteAllLines($planFile,$lines,[Text.UTF8Encoding]::new($false))
    Copy-Item -LiteralPath "$BuildDir\verified-build.json",$PSCommandPath -Destination $OutputDir
    [ordered]@{
        utc_prepared=[DateTimeOffset]::UtcNow.ToString('o');gpu='NVIDIA GeForce RTX 4090 D; sm_89; Windows WDDM'
        binary_sha256=$binaryHash;plan_sha256=(Get-FileHash -LiteralPath $planFile -Algorithm SHA256).Hash
        source_sha256=$verified.source_sha256;build_dir=$BuildDir
        batches=3;cases_per_batch=12;sizes=$sizes;seeds=$seeds;implementations=$names
        rounds=7;repeats=30;warmup=10;expected_formal_events=22680;alpha=1;beta=0
        independent_unit='Three consecutive independent benchmark processes / CUDA contexts; not cross-day or cross-reboot validation'
        input='Preserved xorshift generator; identical distribution and range; actual original and explicit TF32 bytes fingerprinted before/after'
        input_hash='FNV1a64, non-cryptographic, over GPU-read-back little-endian float32 bytes'
        input_sharing='Each implementation receives the same ta/tb allocation in a case; not merely matching seeds'
        order='All six permutations once per case; seventh permutation changes across batches. Size and seed loops rotate across batches.'
        timing='CUDA events per single GEMM; C0 zero D2D restore on same default stream before every warmup and timed call, outside events'
        cache_caveat='Identical 4-byte-per-output C0 restore can alter cache residency; checks between groups also affect cache before next warmup'
        precision='Explicit TF32 represented input; FP32 accumulation/output; original FP32 precision failures retained'
        conversion='Preserved GPU __float_to_tf32 preparation (+ unused FP16 side product), outside GEMM events; not end-to-end timing'
        cublas='CUBLAS_COMPUTE_32F_FAST_TF32; CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION; CUBLAS_ATOMICS_NOT_ALLOWED; CUBLAS_GEMM_DEFAULT; actual version in each case meta'
        reference='PEDANTIC FP32 on original and TF32 represented input; TF32 cuBLAS; original vitamin TF32; CPU double on actual TF32 input'
        thresholds='abs<=0.01; SIMT abs(got-ref)<=0.001+0.001*abs(got); native TF32 <=0.1+0.1*abs(got), separately reported'
        cpu_double='Full 131072 outputs on 128x1024 K32/64/96; 256 fixed unique positions incl four corners on formal squares, index seed20261020 independent of input seed'
        telemetry='nvidia-smi every200ms; no locked clocks or changed power limit; not per-event constant-frequency evidence'
        profiler='Separate after fair benchmark, excluded from formal event table'
        sgmm_commit=(& git -C $repo rev-parse HEAD);vitamin_commit=(& git -C "$repo\..\vitamin-cuda" rev-parse HEAD)
    } | ConvertTo-Json -Depth 6 | Set-Content "$OutputDir\environment.json" -Encoding utf8
    'Plan frozen before any smoke or formal CUDA call from this script.' | Set-Content "$OutputDir\PLAN_PREPARED.txt"
    if($PrepareOnly) { Write-Host "Prepared 756 scheduled groups / 22680 formal events: $planFile"; return }
} else {
    if (!(Test-Path -LiteralPath "$OutputDir\PLAN_PREPARED.txt")) { throw 'Prepared plan marker missing' }
    if (Test-Path -LiteralPath "$OutputDir\RUN_STARTED.txt") { throw 'Run already started; preserve partial data and prepare a new directory' }
}
$meta=Get-Content "$OutputDir\environment.json" -Raw | ConvertFrom-Json
if ($meta.plan_sha256 -ne (Get-FileHash $planFile -Algorithm SHA256).Hash -or $meta.binary_sha256 -ne $binaryHash) {
    throw 'Prepared plan or binary changed'
}
$old=[Environment]::GetEnvironmentVariable('NVIDIA_TF32_OVERRIDE','Process')
$monitor=$null
$records=[System.Collections.Generic.List[object]]::new()
try {
    [Environment]::SetEnvironmentVariable('NVIDIA_TF32_OVERRIDE',$null,'Process')
    [DateTimeOffset]::UtcNow.ToString('o') | Set-Content "$OutputDir\RUN_STARTED.txt"
    [ordered]@{inherited_NVIDIA_TF32_OVERRIDE=$old;effective_NVIDIA_TF32_OVERRIDE=$null;scope='This process and children only'} |
        ConvertTo-Json | Set-Content "$OutputDir\process-environment.json"
    & nvidia-smi | Set-Content "$OutputDir\nvidia-smi-before.txt"
    if ($SmokeEvidenceDir) {
        if (![IO.Path]::IsPathRooted($SmokeEvidenceDir)) { $SmokeEvidenceDir=Join-Path $repo $SmokeEvidenceDir }
        if (!(Test-Path -LiteralPath "$SmokeEvidenceDir\SMOKE_COMPLETE.txt")) { throw 'External smoke evidence is incomplete' }
        [ordered]@{path=$SmokeEvidenceDir;marker_sha256=(Get-FileHash "$SmokeEvidenceDir\SMOKE_COMPLETE.txt" -Algorithm SHA256).Hash;
            note='External smoke supplied explicitly by operator; binary association must also be retained in smoke launcher log'} |
            ConvertTo-Json | Set-Content "$OutputDir\smoke-reference.json"
    } else {
        & $exe --smoke "$OutputDir\smoke" 2>&1 | Tee-Object "$OutputDir\smoke.log"
        if ($LASTEXITCODE) { throw 'Small rectangular smoke / full CPU double check failed' }
    }
    $rejections=foreach ($shape in @(@(128,128,32),@(127,1024,32),@(128,1152,32),@(128,1024,31),@(128,1024,48),@(0,1024,32))) {
        $out="$OutputDir\reject-$($shape -join '-').csv"
        $text=& $exe --case @shape 20261020 1 1 0 $out 2>&1
        $exitCode=$LASTEXITCODE
        if ($exitCode -eq 0 -or (Test-Path -LiteralPath $out)) { throw 'Shape rejection failed' }
        [pscustomobject]@{shape=($shape -join 'x');exit_code=$exitCode;message=($text -join "`n")}
    }
    $rejections | Export-Csv "$OutputDir\shape-rejections.csv" -NoTypeInformation
    foreach ($batch in 1..3) {
        $args=@('--batch',"$batch",$planFile,"$OutputDir\batch-$batch")
        $begin=[DateTimeOffset]::UtcNow.ToString('o')
        & nvidia-smi | Set-Content "$OutputDir\nvidia-smi-before-batch-$batch.txt"
        $monitor=Start-Process -FilePath (Get-Command nvidia-smi.exe).Source -WindowStyle Hidden -PassThru `
            -ArgumentList @('--query-gpu=timestamp,index,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used','--format=csv','--loop-ms=200') `
            -RedirectStandardOutput "$OutputDir\telemetry-batch-$batch.csv" -RedirectStandardError "$OutputDir\telemetry-batch-$batch.stderr.txt"
        Write-Host "Launching independent batch $batch of 3 (12 cases, 7560 events)"
        & $exe @args 2>&1 | Tee-Object "$OutputDir\batch-$batch.log"
        $exitCode=$LASTEXITCODE
        $records.Add([pscustomobject]@{batch=$batch;executable=$exe;binary_sha256=$binaryHash;arguments=$args;
            utc_start=$begin;utc_end=[DateTimeOffset]::UtcNow.ToString('o');exit_code=$exitCode})
        $records | ConvertTo-Json -Depth 5 | Set-Content "$OutputDir\execution.json"
        $telemetryEnded=$monitor.HasExited
        if (!$telemetryEnded) { Stop-Process -Id $monitor.Id; $monitor.WaitForExit() }
        $monitor=$null
        & nvidia-smi | Set-Content "$OutputDir\nvidia-smi-after-batch-$batch.txt"
        if ($exitCode) { throw "Batch $batch failed with exit $exitCode; raw partial files retained" }
        if ($telemetryEnded) { "Telemetry ended early in batch $batch; missing intervals are unverified" | Add-Content "$OutputDir\TELEMETRY_MISSING.txt" }
        if (!(Test-Path -LiteralPath "$OutputDir\batch-$batch\BATCH_COMPLETE.txt")) { throw "Batch $batch completion marker missing" }
    }
    & nvidia-smi | Set-Content "$OutputDir\nvidia-smi-after.txt"
    'COMPLETE: 3 independent process batches, 22680 formal CUDA events; original FP32 failures remain in checks.' |
        Set-Content "$OutputDir\RUN_COMPLETE.txt"
} catch {
    $_.Exception.Message | Set-Content "$OutputDir\RUN_INCOMPLETE.txt"
    throw
} finally {
    if ($null -ne $monitor -and !$monitor.HasExited) { Stop-Process -Id $monitor.Id -ErrorAction SilentlyContinue }
    [Environment]::SetEnvironmentVariable('NVIDIA_TF32_OVERRIDE',$old,'Process')
}
