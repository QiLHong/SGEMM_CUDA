#requires -Version 7.0
[CmdletBinding()]
param([string]$BuildDir='build-windows-tensor-bn64-robustness-sm89',
      [Parameter(Mandatory)][string]$OutputDir,[int]$Seed=20261020,
      [int[]]$Sizes=@(2048,3072,4096),[int]$TimeoutSeconds=180,
      [string]$NcuPath='C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.1\target\windows-desktop-win7-x64\ncu.exe')
$ErrorActionPreference='Stop'
& "$PSScriptRoot\Initialize-Toolchain.ps1"
$repo=(Resolve-Path "$PSScriptRoot\..\..").Path
if (![IO.Path]::IsPathRooted($BuildDir)) { $BuildDir=Join-Path $repo $BuildDir }
if (![IO.Path]::IsPathRooted($OutputDir)) { $OutputDir=Join-Path $repo $OutputDir }
if (Test-Path -LiteralPath $OutputDir) { throw 'Refusing to overwrite profile evidence' }
$exe=Join-Path $BuildDir 'windows_tensor_bn64_robustness_benchmark.exe'
$manifest=Get-Content "$BuildDir\verified-build.json" -Raw | ConvertFrom-Json
if ($manifest.binary_sha256 -ne (Get-FileHash $exe).Hash) { throw 'Binary hash mismatch' }
New-Item -ItemType Directory -Path $OutputDir | Out-Null
Copy-Item -LiteralPath $PSCommandPath,"$BuildDir\verified-build.json" -Destination $OutputDir
$metrics=@(
 'sm__cycles_active.avg.pct_of_peak_sustained_elapsed',
 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active',
 'sm__throughput.avg.pct_of_peak_sustained_active',
 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
 'sm__warps_active.avg.pct_of_peak_sustained_active',
 'smsp__warps_eligible.avg.per_cycle_active','smsp__issue_active.avg.pct_of_peak_sustained_active',
 'smsp__warps_issue_stalled_barrier.avg.per_cycle_active',
 'smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active',
 'smsp__warps_issue_stalled_short_scoreboard.avg.per_cycle_active',
 'lts__t_sectors_op_read.sum','lts__t_sector_hit_rate.pct',
 'dram__bytes_read.sum','dram__bytes_write.sum',
 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
 'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
 'gpu__time_duration.sum','launch__registers_per_thread','launch__shared_mem_per_block',
 'launch__shared_mem_config_size','launch__grid_size','launch__waves_per_multiprocessor',
 'launch__occupancy_limit_registers','launch__occupancy_limit_shared_mem')
$commands=[System.Collections.Generic.List[object]]::new()
$old=[Environment]::GetEnvironmentVariable('NVIDIA_TF32_OVERRIDE','Process')
try {
 [Environment]::SetEnvironmentVariable('NVIDIA_TF32_OVERRIDE',$null,'Process')
 & $NcuPath --version | Set-Content "$OutputDir\ncu-version.txt"
 foreach($n in $Sizes) { foreach($mode in @('vitamin_tf32_dbf','tf32_bn64_bk32')) {
  $label="$n-$mode"
  $arguments=@('--profile-from-start','off','--launch-count','1','--replay-mode','kernel',
    '--clock-control','none','--cache-control','none','--metrics',($metrics -join ','),
    '--csv','--page','raw','--log-file',"$OutputDir\$label.csv",'--export',"$OutputDir\$label",
    $exe,'--profile',$mode,"$n","$Seed")
  $record=[ordered]@{size=$n;seed=$Seed;implementation=$mode;exe=$NcuPath;arguments=$arguments;
    start_utc=[DateTimeOffset]::UtcNow.ToString('o');timeout_seconds=$TimeoutSeconds}
  $p=Start-Process -FilePath $NcuPath -ArgumentList ($arguments | ForEach-Object {'"'+$_+'"'}) -WindowStyle Hidden -PassThru `
       -RedirectStandardOutput "$OutputDir\$label.stdout.log" -RedirectStandardError "$OutputDir\$label.stderr.log"
  if (!$p.WaitForExit($TimeoutSeconds*1000)) {
   & taskkill.exe /PID $p.Id /T /F | Out-File "$OutputDir\$label.termination.log"
   $record['status']='TIMEOUT';$commands.Add($record)
   $commands | ConvertTo-Json -Depth 6 | Set-Content "$OutputDir\commands.json"
   throw "NCU timeout: $label; remaining profiling stopped"
  }
  $record['pid']=$p.Id;$record['exit_code']=$p.ExitCode;$record['end_utc']=[DateTimeOffset]::UtcNow.ToString('o')
  $commands.Add($record);$commands | ConvertTo-Json -Depth 6 | Set-Content "$OutputDir\commands.json"
  if ($p.ExitCode -ne 0) { throw "NCU failed for $label; no permission workaround" }
  Write-Host "PROFILE COMPLETE $label (excluded from benchmark TFLOPS)"
 }}
 'Six targeted launches; profiler durations are not benchmark timings.' | Set-Content "$OutputDir\PROFILE_COMPLETE.txt"
} catch { $_.Exception.Message | Set-Content "$OutputDir\PROFILE_INCOMPLETE.txt";throw }
finally { [Environment]::SetEnvironmentVariable('NVIDIA_TF32_OVERRIDE',$old,'Process') }
