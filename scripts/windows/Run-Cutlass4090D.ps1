#requires -Version 7.0
[CmdletBinding()]
param([string]$BuildDir='build-windows-cutlass-4090d-sm89',
      [Parameter(Mandatory)][string]$OutputDir,[switch]$PrepareOnly,[switch]$ExecutePrepared)
$ErrorActionPreference='Stop'
if($PrepareOnly -and $ExecutePrepared){throw 'Choose PrepareOnly or ExecutePrepared'}
& "$PSScriptRoot\Initialize-Toolchain.ps1"
$repo=(Resolve-Path "$PSScriptRoot\..\..").Path
foreach($name in @('BuildDir','OutputDir')){
 $value=Get-Variable $name -ValueOnly
 if(![IO.Path]::IsPathRooted($value)){Set-Variable $name (Join-Path $repo $value)}
}
$exe=Join-Path $BuildDir 'windows_cutlass_4090d_benchmark.exe'
$verified=Get-Content "$BuildDir\verified-build.json" -Raw | ConvertFrom-Json
$binaryHash=(Get-FileHash -LiteralPath $exe).Hash
if($verified.binary_sha256 -ne $binaryHash -or $verified.source_sha256 -ne (Get-FileHash "$repo\experiments\cutlass-rtx4090d\benchmark.cu").Hash){throw 'Binary/source mismatch; rebuild'}
foreach($dep in (Get-Content "$BuildDir\dependencies.json" -Raw | ConvertFrom-Json)){
 if((Get-FileHash -LiteralPath $dep.path).Hash -ne $dep.sha256){throw "Changed dependency: $($dep.path)"}
}
$groups=[ordered]@{
 tf32=@('vitamin_tf32_dbf','tf32_bn64_bk32','cutlass_tf32_128x128','cutlass_tf32_128x64_bk16','cublas_tf32')
 fp32=@('k10_fp32','k30_fp32','cutlass_simt_256x128','cublas_strict_fp32')
}
if(!$ExecutePrepared){
 if(Test-Path -LiteralPath $OutputDir){throw "Refusing overwrite: $OutputDir"}
 New-Item -ItemType Directory -Path $OutputDir | Out-Null
 $sizes=@(1024,2048,3072,4096);$seeds=@(20261020,42,12345)
 $plans=[ordered]@{}
 foreach($group in $groups.Keys){
  $names=$groups[$group];$count=$names.Count
  $lines=[Collections.Generic.List[string]]::new()
  $lines.Add('batch,case_index,m,n,k,seed,round,order,implementation')
  foreach($batch in 1..3){
   $case=0
   for($si=0;$si -lt 4;$si++){for($vi=0;$vi -lt 3;$vi++){
    $n=$sizes[($si+$batch-1)%4];$seed=$seeds[($vi+$batch-1)%3]
    for($round=0;$round -lt 7;$round++){
     $offset=([int][Math]::Floor($round/2)+$case+$batch-1)%$count
     for($order=0;$order -lt $count;$order++){
      $idx=if(($round+$batch)%2 -eq 0){($offset+$order)%$count}else{($offset-$order+$count)%$count}
      $lines.Add("$batch,$case,$n,$n,$n,$seed,$round,$order,$($names[$idx])")
     }
    }
    $case++
   }}
  }
  $path=Join-Path $OutputDir "plan-$group.csv"
  [IO.File]::WriteAllLines($path,$lines,[Text.UTF8Encoding]::new($false))
  $plans[$group]=[ordered]@{path=$path;sha256=(Get-FileHash -LiteralPath $path).Hash;groups=$lines.Count-1;events=($lines.Count-1)*30}
 }
 Copy-Item -LiteralPath "$BuildDir\verified-build.json",$PSCommandPath -Destination $OutputDir
 [ordered]@{
  utc_prepared=[DateTimeOffset]::UtcNow.ToString('o');binary_sha256=$binaryHash;plans=$plans
  batches=3;seeds=$seeds;sizes=$sizes;groups=$groups;rounds=7;warmup=10;repeats=30;expected_formal_events=68040
  alpha=1;beta=0;cutlass_commit=$verified.cutlass_commit;sgemm_commit=(& git -C $repo rev-parse HEAD)
  order='Rotating offset and alternating direction by round/batch/case; size and seed orders rotate per batch; group process order alternates per batch.'
  timing='CUDA events around public launch APIs, C0 zero D2D restored on same stream outside every event; initialization/workspace allocation outside events'
  input='Same represented TF32 allocations for all TF32 implementations; same original FP32 allocations for all FP32 implementations; FNV1a64 before/after'
  cache_caveat='C0 restore writes output-sized bytes before every launch and can affect L2/cache residency equally by policy'
  cutlass='Official device::Gemm run; fixed published unit-test TF32 configurations, one historical SIMT configuration; split_k=1; not library-wide autotuning'
  accuracy='Preserved abs<=0.01; SIMT eps0.001 and TF32 eps0.1 allclose each separately recorded; CPU double full small shapes and 256 large outputs'
  resources='Windows WDDM, dynamic clocks; telemetry200ms; no power/clock changes'
  profiler='Separate launch stream/session; profiler time excluded from fair performance table'
 } | ConvertTo-Json -Depth 8 | Set-Content "$OutputDir\environment.json"
 'Plan frozen before smoke or formal GPU runs.' | Set-Content "$OutputDir\PLAN_PREPARED.txt"
 if($PrepareOnly){Write-Host "Prepared 68040 events: $OutputDir";return}
}else{
 if(!(Test-Path "$OutputDir\PLAN_PREPARED.txt") -or (Test-Path "$OutputDir\RUN_STARTED.txt")){throw 'Invalid prepared run or already started'}
}
$meta=Get-Content "$OutputDir\environment.json" -Raw | ConvertFrom-Json
if($meta.binary_sha256 -ne $binaryHash){throw 'Prepared binary changed'}
foreach($group in $groups.Keys){
 if($meta.plans.$group.sha256 -ne (Get-FileHash "$OutputDir\plan-$group.csv").Hash){throw 'Plan changed'}
}
$lockDir=Join-Path $repo 'build-windows-gpu-lock'
New-Item -ItemType Directory -Path $lockDir -Force | Out-Null
$gpuLock=[IO.File]::Open((Join-Path $lockDir 'measurement.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
$old=[Environment]::GetEnvironmentVariable('NVIDIA_TF32_OVERRIDE','Process')
$monitor=$null;$records=[Collections.Generic.List[object]]::new()
try{
 [Environment]::SetEnvironmentVariable('NVIDIA_TF32_OVERRIDE',$null,'Process')
 [DateTimeOffset]::UtcNow.ToString('o') | Set-Content "$OutputDir\RUN_STARTED.txt"
 [ordered]@{inherited_NVIDIA_TF32_OVERRIDE=$old;effective_NVIDIA_TF32_OVERRIDE=$null;scope='process and children only'} | ConvertTo-Json | Set-Content "$OutputDir\process-environment.json"
 & nvidia-smi | Set-Content "$OutputDir\nvidia-smi-before.txt"
 & $exe --smoke "$OutputDir\smoke" 2>&1 | Tee-Object "$OutputDir\smoke.log"
 if($LASTEXITCODE){throw 'Small-shape correctness failed'}
 $rejections=foreach($group in $groups.Keys){foreach($shape in @(@(127,1024,32),@(128,1024,31),@(0,1024,32))){
  $output="$OutputDir\reject-$group-$($shape -join '-').csv"
  $msg=& $exe --case $group @shape 20261020 1 1 0 $output 2>&1
  $exitCode=$LASTEXITCODE
  if($exitCode -eq 0 -or (Test-Path -LiteralPath $output)){throw 'Invalid-shape rejection failed'}
  [pscustomobject]@{group=$group;shape=($shape -join 'x');exit_code=$exitCode;message=($msg -join "`n")}
 }}
 $rejections | Export-Csv "$OutputDir\shape-rejections.csv" -NoTypeInformation
 foreach($batch in 1..3){
  $groupOrder=if($batch%2 -eq 1){@('tf32','fp32')}else{@('fp32','tf32')}
  foreach($group in $groupOrder){
   $label="$group-batch-$batch";$begin=[DateTimeOffset]::UtcNow.ToString('o')
   $arguments=@('--batch',$group,"$batch","$OutputDir\plan-$group.csv","$OutputDir\$label")
   $monitor=Start-Process -FilePath (Get-Command nvidia-smi.exe).Source -WindowStyle Hidden -PassThru `
    -ArgumentList @('--query-gpu=timestamp,index,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used','--format=csv','--loop-ms=200') `
    -RedirectStandardOutput "$OutputDir\telemetry-$label.csv" -RedirectStandardError "$OutputDir\telemetry-$label.stderr.txt"
   Write-Host "Starting $label (12 cases; no profiler)"
   & $exe @arguments 2>&1 | Tee-Object "$OutputDir\$label.log"
   $exitCode=$LASTEXITCODE
   $records.Add([pscustomobject]@{group=$group;batch=$batch;executable=$exe;binary_sha256=$binaryHash;arguments=$arguments;utc_start=$begin;utc_end=[DateTimeOffset]::UtcNow.ToString('o');exit_code=$exitCode})
   $records | ConvertTo-Json -Depth 5 | Set-Content "$OutputDir\execution.json"
   $ended=$monitor.HasExited
   if(!$ended){Stop-Process -Id $monitor.Id;$monitor.WaitForExit()}
   $monitor=$null
   if($ended){"Telemetry ended early: $label" | Add-Content "$OutputDir\TELEMETRY_MISSING.txt"}
   if($exitCode){throw "$label failed; partial files retained"}
   if(!(Test-Path "$OutputDir\$label\BATCH_COMPLETE.txt")){throw 'Batch completion marker missing'}
  }
 }
 & nvidia-smi | Set-Content "$OutputDir\nvidia-smi-after.txt"
 'COMPLETE: 6 independent process batches; 68040 formal events; two precision groups.' | Set-Content "$OutputDir\RUN_COMPLETE.txt"
}catch{ $_.Exception.Message | Set-Content "$OutputDir\RUN_INCOMPLETE.txt";throw }
finally{
 if($null -ne $monitor -and !$monitor.HasExited){Stop-Process -Id $monitor.Id -ErrorAction SilentlyContinue}
 [Environment]::SetEnvironmentVariable('NVIDIA_TF32_OVERRIDE',$old,'Process')
 $gpuLock.Dispose()
}
