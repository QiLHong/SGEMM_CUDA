#requires -Version 7.0
# 仅采集 K10 / PAD8 的一条稳定阶段 kernel；性能结论仍以无 profiler 的多轮 events 为准。
[CmdletBinding()]
param(
    [string]$BuildDir = 'build-windows-sm89',
    [string]$NcuPath,
    [string]$OutputDir
)
$ErrorActionPreference = 'Stop'
# 原生命令的退出码由下文记录并检查；避免调用者的偏好设置提前中断日志归档。
$PSNativeCommandUseErrorActionPreference = $false

function Read-NcuMetrics {
    param([string]$Path, [string]$Mode)
    $lines = @(Get-Content -LiteralPath $Path)
    $headerIndex = -1
    for ($i = 0; $i -lt $lines.Count; ++$i) {
        if ($lines[$i].StartsWith('"ID",')) { $headerIndex = $i; break }
    }
    if ($headerIndex -lt 0 -or $headerIndex + 2 -ge $lines.Count) {
        throw "Missing Nsight CSV header, units or kernel row: $Path"
    }
    # --page raw 会先输出警告和 profiler 信息，再输出 header、单位行和 ID=0。
    $dataLines = @($lines[($headerIndex + 2)..($lines.Count - 1)] |
        Where-Object { $_.StartsWith('"0",') })
    if ($dataLines.Count -ne 1) { throw "Expected exactly one ID=0 kernel row: $Path" }
    $rows = @(@($lines[$headerIndex], $lines[$headerIndex + 1], $dataLines[0]) | ConvertFrom-Csv)
    $units = $rows[0]
    $row = $rows[1]
    if ($row.'Kernel Name' -notmatch 'sgemmWarptiling') { throw "Unexpected profiled kernel: $Path" }
    function Read-MetricNumber {
        param([string]$Metric)
        $value = $row.$Metric
        if ([string]::IsNullOrWhiteSpace($value)) { throw "Missing metric $Metric in $Path" }
        $number = [double]::Parse($value,
            [Globalization.NumberStyles]::Float -bor [Globalization.NumberStyles]::AllowThousands,
            [Globalization.CultureInfo]::InvariantCulture)
        if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) {
            throw "Nonfinite metric $Metric in $Path"
        }
        return $number
    }
    $duration = Read-MetricNumber 'gpu__time_duration.sum'
    $durationUnit = $units.'gpu__time_duration.sum'
    $toMilliseconds = @{ nsecond = 1e-6; usecond = 1e-3; msecond = 1.0; second = 1e3 }
    if (!$toMilliseconds.ContainsKey($durationUnit)) { throw "Unknown duration unit: $durationUnit" }
    [pscustomobject][ordered]@{
        mode = $Mode
        size = 4096
        kernel = $row.'Kernel Name'
        ncu_duration_ms = $duration * $toMilliseconds[$durationUnit]
        registers_per_thread = Read-MetricNumber 'launch__registers_per_thread'
        shared_store_bank_conflicts = Read-MetricNumber 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum'
        shared_load_bank_conflicts = Read-MetricNumber 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'
        issue_active_pct = Read-MetricNumber 'smsp__issue_active.avg.pct_of_peak_sustained_active'
        source_csv = $Path
    }
}

& "$PSScriptRoot\Initialize-Toolchain.ps1"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
if (![IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
$bench = Join-Path $BuildDir 'windows_benchmark.exe'
if (!(Test-Path -LiteralPath $bench)) { throw "Missing $bench. Run Build.ps1 -WithExperiment first." }
if (!$NcuPath) {
    $knownNcu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.1\target\windows-desktop-win7-x64\ncu.exe'
    if (Test-Path -LiteralPath $knownNcu) {
        $NcuPath = $knownNcu
    } else {
        $NcuPath = Get-ChildItem -LiteralPath 'C:\Program Files\NVIDIA Corporation' -Directory |
            Where-Object Name -Like 'Nsight Compute *' | Sort-Object Name -Descending |
            ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Filter ncu.exe -File -Recurse } |
            Select-Object -First 1 -ExpandProperty FullName
    }
}
if (!$NcuPath -or !(Test-Path -LiteralPath $NcuPath)) {
    throw 'Nsight Compute CLI was not found. Pass -NcuPath for an existing installation; the benchmark does not require profiling.'
}
$NcuPath = (Resolve-Path -LiteralPath $NcuPath).Path
if (!$OutputDir) {
    $stamp = [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow,
        'China Standard Time').ToString('yyyyMMdd-HHmmss-fff')
    $OutputDir = Join-Path $repo "results\windows\profile-$stamp"
} elseif (![IO.Path]::IsPathRooted($OutputDir)) { $OutputDir = Join-Path $repo $OutputDir }
if (Test-Path -LiteralPath $OutputDir) { throw "Refusing to overwrite existing results: $OutputDir" }
New-Item -ItemType Directory -Path $OutputDir | Out-Null

$metrics = @(
    'gpu__time_duration.sum',
    'launch__registers_per_thread',
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
    'smsp__issue_active.avg.pct_of_peak_sustained_active'
)
$commands = [System.Collections.Generic.List[object]]::new()
$exits = [System.Collections.Generic.List[object]]::new()
$summary = [System.Collections.Generic.List[object]]::new()
$oldTf32Override = [Environment]::GetEnvironmentVariable('NVIDIA_TF32_OVERRIDE', 'Process')
try {
    $env:NVIDIA_TF32_OVERRIDE = '0'
    & $NcuPath --version 2>&1 | Tee-Object -FilePath (Join-Path $OutputDir 'ncu-version.txt')
    $versionExit = $LASTEXITCODE
    $exits.Add([pscustomobject]@{ command = 'ncu --version'; exit_code = $versionExit })
    $exits | Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $OutputDir 'exit-codes.csv')
    if ($versionExit -ne 0) { throw "Nsight Compute version check failed; see $OutputDir. No installation was attempted." }
    [ordered]@{
        time_china = [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow,
            'China Standard Time').ToString('o')
        ncu_path = $NcuPath
        executable = $bench
        executable_sha256 = (Get-FileHash -LiteralPath $bench -Algorithm SHA256).Hash
        NVIDIA_TF32_OVERRIDE = $env:NVIDIA_TF32_OVERRIDE
        original_NVIDIA_TF32_OVERRIDE = $oldTf32Override
        CUDA_VISIBLE_DEVICES = $env:CUDA_VISIBLE_DEVICES
        size = 4096; beta = 3; rounds = 1; repeats = 1; warmup = 10
        launch_skip = 11; launch_count = 1
        clock_control = 'none'; cache_control = 'none'
        note = 'One profiled launch per mode. Profiler-instrumented CUDA-event times are not benchmark results.'
    } | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $OutputDir 'environment.json')
    foreach ($mode in 'k10','k10_pad8') {
        $label = $mode.Replace('_', '-')
        $csvPath = Join-Path $OutputDir "ncu-$label.csv"
        # 跳过一次正确性 launch + 10 次预热，只采集第一条计时 launch。
        $ncuArgs = @('--metrics', ($metrics -join ','), '--kernel-name', 'regex:sgemmWarptiling',
            '--launch-skip', '11', '--launch-count', '1', '--clock-control', 'none',
            '--cache-control', 'none', '--csv', '--page', 'raw', '--log-file', $csvPath,
            '--export', (Join-Path $OutputDir "ncu-$label"),
            $bench, '4096', '3', '1', '1', '10',
            (Join-Path $OutputDir "profile-$label-events.csv"), $mode)
        $commands.Add([pscustomobject]@{ mode = $mode; executable = $NcuPath; arguments = $ncuArgs })
        $commands.ToArray() | ConvertTo-Json -Depth 5 |
            Set-Content -Encoding utf8 (Join-Path $OutputDir 'commands.json')
        # 不设置 HOME，也不改变全局 profiler 配置；NCU 可自行回退到安装目录内的 sections。
        & $NcuPath @ncuArgs 2>&1 | Tee-Object -FilePath (Join-Path $OutputDir "ncu-$label-console.log")
        $profileExit = $LASTEXITCODE
        $exits.Add([pscustomobject]@{ command = $mode; exit_code = $profileExit })
        $exits | Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $OutputDir 'exit-codes.csv')
        if ($profileExit -ne 0) {
            throw "Nsight Compute failed ($profileExit) for $mode; inspect $csvPath and its console log. No installation was attempted."
        }
        $summary.Add((Read-NcuMetrics -Path $csvPath -Mode $mode))
        $summary | Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $OutputDir 'ncu-summary.csv')
    }
    Write-Host "Profile results: $OutputDir"
    Write-Host 'Use ncu-summary.csv for the selected metrics; use Run-Benchmarks.ps1 for unprofiled performance.'
} finally {
    [Environment]::SetEnvironmentVariable('NVIDIA_TF32_OVERRIDE', $oldTf32Override, 'Process')
}
