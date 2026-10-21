#requires -Version 7.0
# vitamin-cuda 五步 SIMT 路线与 K30 V2：只收集 profiler 证据，性能采用无 profiler 的 events 多轮结果。
# 复用 Profile-K30Sync.ps1 的指标和 CSV 单位归一化；不修改原 Python 环境或原始 kernel。
# 指标定义：https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-guide
# launch-skip 仅统计匹配 kernel：https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
[CmdletBinding()]
param(
    [string]$BuildDir = 'build-windows-vitamin-sm89-B',
    [string]$OutputDir,
    [string]$NcuPath,
    [ValidateRange(1,10)][int]$ProfileRepeats = 2,
    [ValidateSet(2048,4096)][int[]]$Sizes = @(2048,4096),
    [ValidateSet('sgemm_tiling','sgemm_at_tiling','sgemm_at_bcf_swizzling','sgemm_at_bcf_swizzling_rw','sgemm_at_bcf_swizzling_dbf_rw','k10','k30_v2','cublas')]
    [string[]]$Modes = @('sgemm_tiling','sgemm_at_tiling','sgemm_at_bcf_swizzling','sgemm_at_bcf_swizzling_rw','sgemm_at_bcf_swizzling_dbf_rw','k30_v2')
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

# 别名不省略单位：stalled/active/eligible 是每 SMSP active cycle 的平均 warp 数，不是百分比。
$metricMap = [ordered]@{
    'gpu__time_duration.sum' = 'ncu_duration_ms'
    'launch__registers_per_thread' = 'registers_per_thread'
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

$kernelNames = @{
    sgemm_tiling = 'sgemm_tiling_kernel'
    sgemm_at_tiling = 'sgemm_at_tiling_kernel'
    sgemm_at_bcf_swizzling = 'sgemm_at_bcf_swizzling_kernel'
    sgemm_at_bcf_swizzling_rw = 'sgemm_at_bcf_swizzling_rw_kernel'
    sgemm_at_bcf_swizzling_dbf_rw = 'sgemm_at_bcf_swizzling_dbf_rw_kernel'
    k10 = 'sgemmWarptiling'
    k30_v2 = 'sgemmK30'
}

function Read-VitaminNcuCsv {
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
$bench = Join-Path $BuildDir 'windows_vitamin_benchmark.exe'
if (!(Test-Path -LiteralPath $bench)) { throw "Missing benchmark: $bench" }
$cache = Join-Path $BuildDir 'CMakeCache.txt'
if (!(Test-Path -LiteralPath $cache) -or
    !(Select-String -LiteralPath $cache -Pattern '^SGEMM_VITAMIN_ORIGINAL_CONFIG:BOOL=OFF$' -Quiet)) {
    throw 'This profiler compares configuration B (strict FP32). Build with SGEMM_VITAMIN_ORIGINAL_CONFIG=OFF.'
}
if (!$NcuPath) {
    $known = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.1\target\windows-desktop-win7-x64\ncu.exe'
    if (Test-Path -LiteralPath $known) { $NcuPath = $known }
    elseif (Test-Path -LiteralPath 'C:\Program Files\NVIDIA Corporation') {
        $NcuPath = Get-ChildItem -LiteralPath 'C:\Program Files\NVIDIA Corporation' -Directory |
            Where-Object Name -Like 'Nsight Compute *' | Sort-Object Name -Descending |
            ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Filter ncu.exe -File -Recurse } |
            Select-Object -First 1 -ExpandProperty FullName
    }
}
if (!$NcuPath -or !(Test-Path -LiteralPath $NcuPath)) {
    throw 'No existing Nsight Compute CLI found. Preserve the events results; no installation is required to benchmark.'
}
$NcuPath = (Resolve-Path -LiteralPath $NcuPath).Path
if (!$OutputDir) {
    $stamp = [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow,
        'China Standard Time').ToString('yyyyMMdd-HHmmss-fff')
    $OutputDir = Join-Path $repo "results\windows\vitamin-profile-$stamp"
} elseif (![IO.Path]::IsPathRooted($OutputDir)) { $OutputDir = Join-Path $repo $OutputDir }
if (Test-Path -LiteralPath $OutputDir) { throw "Refusing to overwrite results: $OutputDir" }
New-Item -ItemType Directory -Path $OutputDir | Out-Null
$commands = [System.Collections.Generic.List[object]]::new()
$exits = [System.Collections.Generic.List[object]]::new()
$summary = [System.Collections.Generic.List[object]]::new()
$metricsLong = [System.Collections.Generic.List[object]]::new()
$oldTf32 = [Environment]::GetEnvironmentVariable('NVIDIA_TF32_OVERRIDE', 'Process')
try {
    $env:NVIDIA_TF32_OVERRIDE = '0'
    & $NcuPath --version 2>&1 | Tee-Object -FilePath (Join-Path $OutputDir 'ncu-version.txt')
    $versionExit = $LASTEXITCODE
    $exits.Add([pscustomobject]@{ command = 'version'; exit_code = $versionExit })
    $exits | Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $OutputDir 'exit-codes.csv')
    if ($versionExit -ne 0) { throw "NCU version query failed ($versionExit)." }
    [ordered]@{
        time_china = [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow,
            'China Standard Time').ToString('o')
        ncu_path = $NcuPath; binary = $bench
        binary_sha256 = (Get-FileHash -LiteralPath $bench -Algorithm SHA256).Hash
        configuration = 'strict_fp32'
        cmake_cache_sha256 = (Get-FileHash -LiteralPath $cache -Algorithm SHA256).Hash
        sizes = $Sizes; modes = $Modes; profile_repeats = $ProfileRepeats
        alpha = 1; beta = 0; seed = 20261020; warmup = 10; launch_count = 1
        custom_launch_skip = 11; cublas_launch_skip = 12
        launch_skip_assumption = 'Custom filter: one validation + 10 warmups; cuBLAS all kernels: one reference + one validation + 10 warmups. Inspect captured kernel names.'
        replay_mode = 'kernel'; clock_control = 'none'; cache_control = 'none'
        NVIDIA_TF32_OVERRIDE = $env:NVIDIA_TF32_OVERRIDE
        original_NVIDIA_TF32_OVERRIDE = $oldTf32
        CUDA_VISIBLE_DEVICES = $env:CUDA_VISIBLE_DEVICES
        metrics = @($metricMap.Keys)
        metric_reference = 'https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-guide'
        caveat = 'Profiled events include replay overhead. Local-memory sectors are traffic, not a direct compiler spill count. Stalled warp counts are not additive execution-time fractions.'
    } | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 (Join-Path $OutputDir 'environment.json')
    foreach ($n in $Sizes) {
        for ($rep = 1; $rep -le $ProfileRepeats; ++$rep) {
            $order = @($Modes)
            if ($rep % 2 -eq 0) { [array]::Reverse($order) }
            foreach ($mode in $order) {
                $label = "$n-$mode-r$rep"
                $csvPath = Join-Path $OutputDir "ncu-$label.csv"
                # 必须包含函数名末尾的模板/参数分隔符，rw 不会误匹配 dbf_rw。
                $filter = if ($mode -eq 'cublas') { 'regex:.*' }
                          else { 'regex:' + [regex]::Escape($kernelNames[$mode]) + '[<(]' }
                $skip = if ($mode -eq 'cublas') { '12' } else { '11' }
                $ncuArgs = @('--metrics', (@($metricMap.Keys) -join ','), '--kernel-name-base', 'demangled', '--kernel-name', $filter,
                    '--launch-skip', $skip, '--launch-count', '1', '--replay-mode', 'kernel',
                    '--clock-control', 'none', '--cache-control', 'none', '--csv', '--page', 'raw',
                    '--log-file', $csvPath, '--export', (Join-Path $OutputDir "ncu-$label"),
                    $bench, "$n", '1', '1', '10', (Join-Path $OutputDir "events-$label.csv"), $mode)
                $commands.Add([pscustomobject]@{
                    size = $n; mode = $mode; profile_repeat = $rep
                    executable = $NcuPath; arguments = $ncuArgs
                })
                $commands.ToArray() | ConvertTo-Json -Depth 5 |
                    Set-Content -Encoding utf8 (Join-Path $OutputDir 'commands.json')
                Write-Host "Profiling $label (one selected launch, clocks/caches unchanged)"
                & $NcuPath @ncuArgs 2>&1 | Tee-Object -FilePath (Join-Path $OutputDir "console-$label.log")
                $code = $LASTEXITCODE
                $exits.Add([pscustomobject]@{ command = $label; exit_code = $code })
                $exits | Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $OutputDir 'exit-codes.csv')
                if ($code -ne 0) {
                    throw "NCU failed ($code) for $label; inspect raw CSV and console log. Keep unprofiled events as fallback; no installation or permission change was attempted."
                }
                $parsed = Read-VitaminNcuCsv -Path $csvPath -Mode $mode -Size $n -Repetition $rep -MetricAliases $metricMap
                $summary.Add($parsed.summary)
                foreach ($record in $parsed.metrics) { $metricsLong.Add($record) }
                $summary | Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $OutputDir 'ncu-summary.csv')
                $metricsLong | Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $OutputDir 'ncu-metrics.csv')
            }
        }
    }
    'PASS: every requested profile and metric parsed; profiled events are not performance results.' |
        Set-Content -Encoding utf8 (Join-Path $OutputDir 'PROFILE_PASS.txt')
    Write-Host "Profile results: $OutputDir"
} catch {
    $_.Exception.Message | Set-Content -Encoding utf8 (Join-Path $OutputDir 'PROFILE_INCOMPLETE.txt')
    throw
} finally {
    [Environment]::SetEnvironmentVariable('NVIDIA_TF32_OVERRIDE', $oldTf32, 'Process')
}
