#requires -Version 7.0
# 复用 Profile-Vitamin 的指标单位解析；这里只采两个独立 DBF binary。
[CmdletBinding()]
param(
    [string]$BuildDir = 'build-windows-dbf-register-sm89',
    [string]$OutputDir,
    [string]$NcuPath = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.1\target\windows-desktop-win7-x64\ncu.exe',
    [ValidateRange(1,10)][int]$ProfileRepeats = 2,
    [ValidateSet(2048,4096)][int[]]$Sizes = @(2048,4096)
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$metricMap = [ordered]@{
    'gpu__time_duration.sum' = 'ncu_duration_ms'
    'launch__registers_per_thread' = 'registers_per_thread'
    'launch__shared_mem_config_size' = 'shared_config_bytes_per_sm'
    'launch__shared_mem_per_block' = 'shared_total_bytes_per_block'
    'launch__shared_mem_per_block_static' = 'shared_static_bytes_per_block'
    'launch__shared_mem_per_block_dynamic' = 'shared_dynamic_bytes_per_block'
    'launch__occupancy_limit_registers' = 'register_limited_blocks_per_sm'
    'launch__occupancy_limit_shared_mem' = 'shared_limited_blocks_per_sm'
    'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum' = 'local_load_sectors'
    'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum' = 'local_store_sectors'
    'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum' = 'shared_load_wavefronts'
    'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum' = 'shared_store_wavefronts'
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum' = 'shared_load_bank_conflicts'
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum' = 'shared_store_bank_conflicts'
    'sm__warps_active.avg.pct_of_peak_sustained_active' = 'achieved_occupancy_pct'
    'smsp__warps_active.avg.per_cycle_active' = 'active_warps_per_smsp_active_cycle'
    'smsp__warps_eligible.avg.per_cycle_active' = 'eligible_warps_per_smsp_active_cycle'
    'smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active' = 'long_scoreboard_warps_per_smsp_active_cycle'
    'smsp__warps_issue_stalled_short_scoreboard.avg.per_cycle_active' = 'short_scoreboard_warps_per_smsp_active_cycle'
    'smsp__warps_issue_stalled_barrier.avg.per_cycle_active' = 'barrier_warps_per_smsp_active_cycle'
    'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active' = 'fp32_fma_pipe_active_pct'
    'smsp__issue_active.avg.pct_of_peak_sustained_active' = 'issue_active_pct'
    'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum' = 'thread_ffma_instructions'
    'smsp__inst_executed.sum' = 'warp_instructions'
    'dram__bytes_read.sum' = 'dram_read_bytes'
    'dram__bytes_write.sum' = 'dram_write_bytes'
    'lts__t_sector_hit_rate.pct' = 'l2_sector_hit_pct'
}

$kernelNames = @{default='sgemm_at_bcf_swizzling_dbf_rw_kernel';maxr128='sgemm_at_bcf_swizzling_dbf_rw_kernel'}

function Read-DbfRegisterNcuCsv {
    param([string]$Path, [string]$Mode, [int]$Size, [int]$Repetition,
          [System.Collections.IDictionary]$MetricAliases)
    $lines = @(Get-Content -LiteralPath $Path)
    $headerIndex = -1
    for ($i = 0; $i -lt $lines.Count; ++$i) {
        if ($lines[$i].StartsWith('"ID",')) { $headerIndex = $i; break }
    }
    if ($headerIndex -lt 0 -or $headerIndex + 2 -ge $lines.Count) {
        throw "Missing Nsight raw CSV header or data: $Path"
    }
    $data = @($lines[($headerIndex + 2)..($lines.Count - 1)] |
        Where-Object { $_.StartsWith('"0",') })
    if ($data.Count -ne 1) { throw "Expected one profiled ID=0 row: $Path" }
    $records = @(@($lines[$headerIndex],$lines[$headerIndex + 1],$data[0]) | ConvertFrom-Csv)
    $units = $records[0]; $row = $records[1]
    if ($Mode -eq 'cublas') {
        if ($row.'Kernel Name' -match 'sgemmK30|sgemmWarptiling|vitamin_cuda::') {
            throw "A custom kernel was captured for cuBLAS: $Path"
        }
    } else {
        $expected = [regex]::Escape($kernelNames[$Mode])
        if ($row.'Kernel Name' -notmatch "\b$expected(?=[<(])") {
            throw "Unexpected kernel for ${Mode}: $($row.'Kernel Name')"
        }
    }
    $wide = [ordered]@{
        size = $Size; mode = $Mode; profile_repeat = $Repetition
        kernel = $row.'Kernel Name'; source_csv = $Path
    }
    $long = [System.Collections.Generic.List[object]]::new()
    foreach ($metric in $MetricAliases.Keys) {
        $raw = $row.$metric
        if ([string]::IsNullOrWhiteSpace($raw)) { throw "Missing metric $metric in $Path" }
        $value = [double]::Parse($raw,
            [Globalization.NumberStyles]::Float -bor [Globalization.NumberStyles]::AllowThousands,
            [Globalization.CultureInfo]::InvariantCulture)
        if ([double]::IsNaN($value) -or [double]::IsInfinity($value)) {
            throw "Nonfinite metric $metric in $Path"
        }
        $unit = $units.$metric; $normalizedUnit = $unit
        if ($metric -eq 'gpu__time_duration.sum') {
            $scales = @{ nsecond = 1e-6; usecond = 1e-3; msecond = 1.0; second = 1e3 }
            if (!$scales.ContainsKey($unit)) { throw "Unknown duration unit: $unit" }
            $value *= $scales[$unit]; $normalizedUnit = 'msecond'
        } elseif ($unit -match '^(K|M|G)?byte(/block)?$') {
            $scales = @{ '' = 1.0; K = 1e3; M = 1e6; G = 1e9 }
            $value *= $scales[[string]$Matches[1]]
            $normalizedUnit = 'byte' + $Matches[2]
        }
        $wide[$MetricAliases[$metric]] = $value
        $long.Add([pscustomobject][ordered]@{
            size = $Size; mode = $Mode; profile_repeat = $Repetition
            metric = $metric; value = $value; unit = $normalizedUnit
            raw_value = $raw; raw_unit = $unit; source_csv = $Path
        })
    }
    # 同工作量下的服务量归一化；wavefront 是硬件工作包，不是 byte 或指令。
    $ffma = $wide['thread_ffma_instructions']
    if ($ffma -le 0) { throw "No FFMA work in selected GEMM kernel: $Path" }
    $wide['shared_load_wavefronts_per_1024_thread_ffma'] = $wide['shared_load_wavefronts'] * 1024 / $ffma
    $wide['warp_instructions_per_32_thread_ffma'] = $wide['warp_instructions'] * 32 / $ffma
    return @{ summary = [pscustomobject]$wide; metrics = $long.ToArray() }
}

& "$PSScriptRoot\Initialize-Toolchain.ps1"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
if (![IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
if (!(Test-Path -LiteralPath $NcuPath)) { throw "Missing existing NCU: $NcuPath" }
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
if (!$OutputDir) { $OutputDir = "results\windows\20261021-vitamin-followup\register-limit\profile-$(Get-Date -Format 'yyyyMMdd-HHmmss')" }
if (![IO.Path]::IsPathRooted($OutputDir)) { $OutputDir = Join-Path $repo $OutputDir }
if (Test-Path -LiteralPath $OutputDir) { throw "Refusing to overwrite: $OutputDir" }
New-Item -ItemType Directory -Path $OutputDir | Out-Null
Copy-Item -LiteralPath "$BuildDir\verified-build.json" -Destination $OutputDir
$commands = [System.Collections.Generic.List[object]]::new()
$exits = [System.Collections.Generic.List[object]]::new()
$summary = [System.Collections.Generic.List[object]]::new()
$metricsLong = [System.Collections.Generic.List[object]]::new()
$oldTf32 = [Environment]::GetEnvironmentVariable('NVIDIA_TF32_OVERRIDE','Process')
try {
    $env:NVIDIA_TF32_OVERRIDE = '0'
    & $NcuPath --version 2>&1 | Tee-Object -FilePath "$OutputDir\ncu-version.txt"
    if ($LASTEXITCODE) { throw 'NCU version query failed.' }
    [ordered]@{
        created_utc=[DateTimeOffset]::UtcNow.ToString('o')
        ncu_path=$NcuPath;variants=$verified.variants;sizes=$Sizes;profile_repeats=$ProfileRepeats
        configuration='strict_fp32';alpha=1;beta=0;seed=20261020
        warmup=10;launch_skip=11;launch_count=1
        launch_skip_scope='DBF name filter counts one preflight plus ten warmups; independent cuBLAS reference is excluded'
        replay_mode='kernel';clock_control='none';cache_control='none'
        NVIDIA_TF32_OVERRIDE='0';original_NVIDIA_TF32_OVERRIDE=$oldTf32
        metrics=@($metricMap.Keys)
        shared_config_metric='launch__shared_mem_config_size: confirmed in installed NCU LaunchStatistics.section and ProfilingGuide; hardware --query-metrics does not list launch metadata'
        caveat='Profiled events excluded from performance results; stalled warps are not time fractions; register cap also changes instruction scheduling'
    } | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath "$OutputDir\environment.json" -Encoding utf8
    foreach ($n in $Sizes) {
        for ($rep = 1; $rep -le $ProfileRepeats; ++$rep) {
            $order = if ($rep % 2) { @('default','maxr128') } else { @('maxr128','default') }
            foreach ($mode in $order) {
                $bench = "$BuildDir\windows_dbf_${mode}_benchmark.exe"
                $label = "$n-$mode-r$rep"
                $csvPath = Join-Path $OutputDir "ncu-$label.csv"
                $ncuArgs = @('--metrics',(@($metricMap.Keys) -join ','),
                    '--kernel-name-base','demangled',
                    '--kernel-name','regex:sgemm_at_bcf_swizzling_dbf_rw_kernel[<(]',
                    '--launch-skip','11','--launch-count','1','--replay-mode','kernel',
                    '--clock-control','none','--cache-control','none','--csv','--page','raw',
                    '--log-file',$csvPath,'--export',(Join-Path $OutputDir "ncu-$label"),
                    $bench,"$n",'1','1','10',(Join-Path $OutputDir "events-$label.csv"),'all')
                $commands.Add([pscustomobject]@{
                    size=$n;mode=$mode;profile_repeat=$rep;executable=$NcuPath;arguments=$ncuArgs
                })
                $commands | ConvertTo-Json -Depth 5 |
                    Set-Content -LiteralPath "$OutputDir\commands.json" -Encoding utf8
                Write-Host "Profiling DBF $label"
                & $NcuPath @ncuArgs 2>&1 | Tee-Object -FilePath "$OutputDir\console-$label.log"
                $code = $LASTEXITCODE
                $exits.Add([pscustomobject]@{command=$label;exit_code=$code})
                $exits | Export-Csv -LiteralPath "$OutputDir\exit-codes.csv" -NoTypeInformation -Encoding utf8
                if ($code) { throw "NCU failed $code at $label; raw evidence retained." }
                $parsed = Read-DbfRegisterNcuCsv -Path $csvPath -Mode $mode -Size $n `
                    -Repetition $rep -MetricAliases $metricMap
                $summary.Add($parsed.summary)
                foreach ($record in $parsed.metrics) { $metricsLong.Add($record) }
                $summary | Export-Csv -LiteralPath "$OutputDir\ncu-summary.csv" -NoTypeInformation -Encoding utf8
                $metricsLong | Export-Csv -LiteralPath "$OutputDir\ncu-metrics.csv" -NoTypeInformation -Encoding utf8
            }
        }
    }
    'PASS: all requested DBF profiles parsed; profiler timings are not normal performance results.' |
        Set-Content -LiteralPath "$OutputDir\PROFILE_PASS.txt" -Encoding utf8
} catch {
    $_.Exception.Message | Set-Content -LiteralPath "$OutputDir\PROFILE_INCOMPLETE.txt" -Encoding utf8
    throw
} finally {
    [Environment]::SetEnvironmentVariable('NVIDIA_TF32_OVERRIDE',$oldTf32,'Process')
}

