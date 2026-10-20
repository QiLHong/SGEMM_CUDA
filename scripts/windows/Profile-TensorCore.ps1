#requires -Version 7.0
[CmdletBinding()]
param([string]$BuildDir='build-windows-tensorcore-sm89',[string]$OutputDir,
      [string]$NcuPath='C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.1\target\windows-desktop-win7-x64\ncu.exe',
      [int[]]$Sizes=@(2048,4096),
      [string[]]$Modes=@('k30_fp32','cublas_strict_fp32','vitamin_tf32_dbf','cublas_tf32','vitamin_fp16_dbf_rw','cublas_fp16'))
$ErrorActionPreference='Stop'
& "$PSScriptRoot\Initialize-Toolchain.ps1"
$repo=(Resolve-Path "$PSScriptRoot\..\..").Path
if (![IO.Path]::IsPathRooted($BuildDir)) { $BuildDir=Join-Path $repo $BuildDir }
if (!$OutputDir) { $OutputDir="results\windows\20261021-tensorcore\profile-$(Get-Date -Format yyyyMMdd-HHmmss)" }
if (![IO.Path]::IsPathRooted($OutputDir)) { $OutputDir=Join-Path $repo $OutputDir }
if (Test-Path -LiteralPath $OutputDir) { throw 'Refusing to overwrite profiler evidence' }
if (!(Test-Path -LiteralPath $NcuPath)) { throw 'Installed NCU not found; no installation attempted' }
$exe=Join-Path $BuildDir 'windows_tensor_benchmark.exe'
$manifest=Get-Content "$BuildDir\verified-build.json" -Raw | ConvertFrom-Json
if ($manifest.binary_sha256 -ne (Get-FileHash $exe -Algorithm SHA256).Hash) { throw 'Binary mismatch' }
New-Item -ItemType Directory -Path $OutputDir | Out-Null
Copy-Item -LiteralPath $PSCommandPath,"$BuildDir\verified-build.json" -Destination $OutputDir
$metrics=@('gpu__time_duration.sum','launch__registers_per_thread','launch__shared_mem_per_block',
    'launch__shared_mem_config_size','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active',
    'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active',
    'sm__warps_active.avg.pct_of_peak_sustained_active','smsp__warps_eligible.avg.per_cycle_active',
    'smsp__issue_active.avg.pct_of_peak_sustained_active',
    'smsp__warps_issue_stalled_barrier.avg.per_cycle_active',
    'smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active',
    'smsp__warps_issue_stalled_short_scoreboard.avg.per_cycle_active',
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
    'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
    'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
    'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum')
$old=[Environment]::GetEnvironmentVariable('NVIDIA_TF32_OVERRIDE','Process')
$commands=[System.Collections.Generic.List[object]]::new()
try {
    [Environment]::SetEnvironmentVariable('NVIDIA_TF32_OVERRIDE',$null,'Process')
    & $NcuPath --version | Set-Content "$OutputDir\ncu-version.txt"
    foreach ($n in $Sizes) { foreach ($mode in $Modes) {
        $label="$n-$mode"
        $args=@('--profile-from-start','off','--launch-count','1','--replay-mode','kernel',
            '--clock-control','none','--cache-control','none','--metrics',($metrics -join ','),
            '--csv','--page','raw','--log-file',"$OutputDir\$label.csv",
            '--export',"$OutputDir\$label",$exe,'--profile',$mode,"$n")
        $commands.Add([pscustomobject]@{size=$n;mode=$mode;exe=$NcuPath;arguments=$args})
        $commands | ConvertTo-Json -Depth 5 | Set-Content "$OutputDir\commands.json"
        Write-Host "Profiling $label (not performance-table timing)"
        & $NcuPath @args 2>&1 | Tee-Object "$OutputDir\$label.log"
        if ($LASTEXITCODE) { throw "NCU failed for $label; no permission workaround" }
    }}
    'COMPLETE; one sample per selected implementation/size; profiler durations excluded from benchmark' |
        Set-Content "$OutputDir\PROFILE_COMPLETE.txt"
} catch {
    $_.Exception.Message | Set-Content "$OutputDir\PROFILE_INCOMPLETE.txt"
    throw
} finally {
    [Environment]::SetEnvironmentVariable('NVIDIA_TF32_OVERRIDE',$old,'Process')
}
