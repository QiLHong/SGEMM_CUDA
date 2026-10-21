#requires -Version 7.0
[CmdletBinding()]
param([string]$BuildDir='build-windows-cutlass-4090d-sm89-final03',
 [Parameter(Mandatory)][string]$ResumeFrom,[Parameter(Mandatory)][string]$OutputDir)
$ErrorActionPreference='Stop'
& "$PSScriptRoot\Initialize-Toolchain.ps1"
$repo=(Resolve-Path "$PSScriptRoot\..\..").Path
foreach($name in @('BuildDir','ResumeFrom','OutputDir')){
 $value=Get-Variable $name -ValueOnly
 if(![IO.Path]::IsPathRooted($value)){Set-Variable $name (Join-Path $repo $value)}
}
if(Test-Path -LiteralPath $OutputDir){throw 'Recovery output must be a new directory'}
if(Get-Process -Name windows_cutlass_4090d_benchmark -ErrorAction SilentlyContinue){throw 'A comparison process is already active'}
$exe=Join-Path $BuildDir 'windows_cutlass_4090d_benchmark.exe'
$verified=Get-Content "$BuildDir\verified-build.json" -Raw | ConvertFrom-Json
$meta=Get-Content "$ResumeFrom\environment.json" -Raw | ConvertFrom-Json
$hash=(Get-FileHash $exe).Hash
if($hash -ne $verified.binary_sha256 -or $hash -ne $meta.binary_sha256){throw 'Recovery binary differs'}
foreach($dep in (Get-Content "$BuildDir\dependencies.json" -Raw | ConvertFrom-Json)){
 if((Get-FileHash -LiteralPath $dep.path).Hash -ne $dep.sha256){throw "Changed dependency $($dep.path)"}
}
foreach($g in @('tf32','fp32')){if((Get-FileHash "$ResumeFrom\plan-$g.csv").Hash -ne $meta.plans.$g.sha256){throw 'Original plan changed'}}
if(!(Test-Path "$ResumeFrom\smoke\SMOKE_COMPLETE.txt")){throw 'Original smoke was incomplete'}
New-Item -ItemType Directory -Path $OutputDir | Out-Null
foreach($file in @('environment.json','plan-tf32.csv','plan-fp32.csv','verified-build.json','PLAN_PREPARED.txt','shape-rejections.csv','smoke.log')){
 Copy-Item -LiteralPath "$ResumeFrom\$file" -Destination $OutputDir
}
Copy-Item -LiteralPath "$ResumeFrom\smoke" -Destination $OutputDir -Recurse
Copy-Item -LiteralPath $PSCommandPath -Destination $OutputDir
$oldRecords=@(Get-Content "$ResumeFrom\execution.json" -Raw | ConvertFrom-Json)
$records=[Collections.Generic.List[object]]::new();$copied=[Collections.Generic.List[object]]::new()
foreach($batch in 1..3){foreach($g in @('tf32','fp32')){
 $label="$g-batch-$batch";$folder=Join-Path $ResumeFrom $label
 if(!(Test-Path "$folder\BATCH_COMPLETE.txt")){continue}
 $record=@($oldRecords | Where-Object {$_.group -eq $g -and $_.batch -eq $batch -and $_.exit_code -eq 0})
 $process=@(Import-Csv "$folder\process.csv")
 if($record.Count -ne 1 -or $process.Count -ne 1 -or $process[0].status -ne 'COMPLETE' -or [int]$process[0].cases -ne 12){throw "Incomplete process evidence $label"}
 Copy-Item -LiteralPath $folder -Destination $OutputDir -Recurse
 foreach($file in @("$label.log","telemetry-$label.csv","telemetry-$label.stderr.txt")){
  Copy-Item -LiteralPath "$ResumeFrom\$file" -Destination $OutputDir
 }
 foreach($file in (Get-ChildItem -LiteralPath $folder -File -Recurse)){
  $relative=[IO.Path]::GetRelativePath($ResumeFrom,$file.FullName)
  $sha=(Get-FileHash -LiteralPath $file.FullName).Hash
  if((Get-FileHash -LiteralPath (Join-Path $OutputDir $relative)).Hash -ne $sha){throw 'Recovery copy changed bytes'}
  $copied.Add([pscustomobject]@{path=$relative;sha256=$sha;bytes=$file.Length})
 }
 $records.Add($record[0])
}}
$copied | Export-Csv "$OutputDir\reused-complete-batch-sha256.csv" -NoTypeInformation
[ordered]@{resume_from=$ResumeFrom;utc=[DateTimeOffset]::UtcNow.ToString('o');reused_complete_batches=$records.Count;binary_sha256=$hash;policy='Only completed processes with exit0 and completion marker are copied; partial batches remain in original directory; original plans and bytes unchanged'} | ConvertTo-Json | Set-Content "$OutputDir\recovery.json"
$monitor=$null;$old=[Environment]::GetEnvironmentVariable('NVIDIA_TF32_OVERRIDE','Process')
try{
 [Environment]::SetEnvironmentVariable('NVIDIA_TF32_OVERRIDE',$null,'Process')
 [DateTimeOffset]::UtcNow.ToString('o') | Set-Content "$OutputDir\RUN_STARTED.txt"
 [ordered]@{inherited_NVIDIA_TF32_OVERRIDE=$old;effective_NVIDIA_TF32_OVERRIDE=$null;scope='recovery process and children only'} | ConvertTo-Json | Set-Content "$OutputDir\process-environment.json"
 & nvidia-smi | Set-Content "$OutputDir\nvidia-smi-before.txt"
 foreach($batch in 1..3){
  $groupOrder=if($batch%2 -eq 1){@('tf32','fp32')}else{@('fp32','tf32')}
  foreach($g in $groupOrder){
   $label="$g-batch-$batch"
   if(Test-Path "$OutputDir\$label\BATCH_COMPLETE.txt"){Write-Host "Reused complete $label";continue}
   $arguments=@('--batch',$g,"$batch","$OutputDir\plan-$g.csv","$OutputDir\$label")
   $begin=[DateTimeOffset]::UtcNow.ToString('o')
   $monitor=Start-Process -FilePath (Get-Command nvidia-smi.exe).Source -WindowStyle Hidden -PassThru `
    -ArgumentList @('--query-gpu=timestamp,index,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used','--format=csv','--loop-ms=200') `
    -RedirectStandardOutput "$OutputDir\telemetry-$label.csv" -RedirectStandardError "$OutputDir\telemetry-$label.stderr.txt"
   Write-Host "Starting $label"
   & $exe @arguments 2>&1 | Tee-Object "$OutputDir\$label.log"
   $exitCode=$LASTEXITCODE
   $records.Add([pscustomobject]@{group=$g;batch=$batch;executable=$exe;binary_sha256=$hash;arguments=$arguments;utc_start=$begin;utc_end=[DateTimeOffset]::UtcNow.ToString('o');exit_code=$exitCode})
   $records | ConvertTo-Json -Depth 5 | Set-Content "$OutputDir\execution.json"
   if(!$monitor.HasExited){Stop-Process -Id $monitor.Id;$monitor.WaitForExit()}else{"Telemetry ended early $label" | Add-Content "$OutputDir\TELEMETRY_MISSING.txt"}
   $monitor=$null
   if($exitCode -or !(Test-Path "$OutputDir\$label\BATCH_COMPLETE.txt")){throw "Incomplete $label; all evidence retained"}
  }
 }
 & nvidia-smi | Set-Content "$OutputDir\nvidia-smi-after.txt"
 'COMPLETE: 6 nonoverlapping process batches; 68040 events. Recovered completed batches are documented in recovery.json.' | Set-Content "$OutputDir\RUN_COMPLETE.txt"
}catch{ $_.Exception.Message | Set-Content "$OutputDir\RUN_INCOMPLETE.txt";throw }
finally{
 if($null -ne $monitor -and !$monitor.HasExited){Stop-Process -Id $monitor.Id -ErrorAction SilentlyContinue}
 [Environment]::SetEnvironmentVariable('NVIDIA_TF32_OVERRIDE',$old,'Process')
}
