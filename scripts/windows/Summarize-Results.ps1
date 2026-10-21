#requires -Version 7.0
# 只汇总已有 CUDA event CSV，不启动 GPU。round_mean 是轮内样本的算术均值；
# 最终 median_ms 是这些轮均值的中位数，避免把所有重复样本当成独立实验轮。
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ResultDir,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$invariant = [Globalization.CultureInfo]::InvariantCulture
$resultPath = (Resolve-Path -LiteralPath $ResultDir).Path
if (!(Test-Path -LiteralPath $resultPath -PathType Container)) {
    throw "ResultDir is not a directory: $resultPath"
}
$outputNames = @('rounds.csv','summary.csv','paired.csv','paired-summary.csv')
foreach ($name in $outputNames) {
    $path = Join-Path $resultPath $name
    if ((Test-Path -LiteralPath $path) -and !$Force) {
        throw "Refusing to overwrite $path. Use -Force only to regenerate these summaries."
    }
}
$files = @(Get-ChildItem -LiteralPath $resultPath -File |
    Where-Object Name -Match '^timings?-[0-9]+\.csv$' | Sort-Object Name)
if (!$files.Count) { throw "No timings-N.csv or timing-N.csv files in $resultPath" }

function Read-Number {
    param($Row, [string]$Name)
    if (!$Row.PSObject.Properties[$Name]) { throw "Missing CSV column: $Name" }
    $number = [double]::Parse([string]$Row.$Name, [Globalization.NumberStyles]::Float, $invariant)
    if (![double]::IsFinite($number)) { throw "Non-finite CSV number in $Name" }
    return $number
}
function Read-Integer {
    param($Row, [string]$Name, [long]$Minimum = 0)
    $number = Read-Number $Row $Name
    if ($number -lt $Minimum -or $number -ne [Math]::Truncate($number) -or $number -gt [long]::MaxValue) {
        throw "Invalid nonnegative integer in $Name`: $number"
    }
    return [long]$number
}
function Get-Median {
    param([double[]]$Values)
    if (!$Values.Count) { throw 'Cannot compute median of no samples.' }
    $sorted = @($Values | Sort-Object)
    $middle = [int][Math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2) { return [double]$sorted[$middle] }
    return ([double]$sorted[$middle - 1] + [double]$sorted[$middle]) / 2.0
}
function Get-Maximum {
    param($Rows, [string]$Property)
    return [double](($Rows | Measure-Object -Property $Property -Maximum).Maximum)
}
function Get-PairedBootstrap {
    param([double[]]$Ratios)
    # 重采样单位是同轮的配对，不是轮内30个event。区间只描述这批已观测轮次；
    # 轮次可交换是工作假设，不覆盖WDDM系统误差、跨会话变化或潜在时间相关性。
    $logs = @($Ratios | ForEach-Object {
        if ($_ -le 0 -or ![double]::IsFinite($_)) { throw 'Invalid paired ratio.' }
        [Math]::Log($_)
    })
    $geomean = [Math]::Exp([double](($logs | Measure-Object -Average).Average))
    if ($logs.Count -lt 5) {
        return [pscustomobject]@{ ratio=$geomean; lower=$null; upper=$null; resamples=0 }
    }
    $random = [Random]::new(20261021)
    $resamples = 20000
    $estimates = [double[]]::new($resamples)
    for ($b = 0; $b -lt $resamples; ++$b) {
        $sum = 0.0
        for ($j = 0; $j -lt $logs.Count; ++$j) { $sum += $logs[$random.Next($logs.Count)] }
        $estimates[$b] = [Math]::Exp($sum / $logs.Count)
    }
    [Array]::Sort($estimates)
    # 线性插值的百分位区间；不把它称为排除干扰后的因果置信区间。
    $bounds = foreach ($p in 0.025,0.975) {
        $position = $p * ($resamples - 1)
        $lowerIndex = [int][Math]::Floor($position)
        $fraction = $position - $lowerIndex
        $estimates[$lowerIndex] * (1.0-$fraction) + $estimates[$lowerIndex+1] * $fraction
    }
    return [pscustomobject]@{ ratio=$geomean; lower=$bounds[0]; upper=$bounds[1]; resamples=$resamples }
}
function Export-InvariantCsv {
    param($Rows, [string]$Path)
    # PowerShell 的显示区域设置可以是中文或其他语言；磁盘中的小数始终用小数点。
    $Rows | ForEach-Object {
        $record = [ordered]@{}
        foreach ($property in $_.PSObject.Properties) {
            $value = $property.Value
            if ($value -is [double] -or $value -is [single]) {
                $record[$property.Name] = $value.ToString('G17', $invariant)
            } elseif ($value -is [int] -or $value -is [long]) {
                $record[$property.Name] = $value.ToString($invariant)
            } else {
                $record[$property.Name] = $value
            }
        }
        [pscustomobject]$record
    } | Export-Csv -LiteralPath $Path -NoTypeInformation -Encoding utf8
}

$samples = [Collections.Generic.List[object]]::new()
$sampleKeys = [Collections.Generic.HashSet[string]]::new()
foreach ($file in $files) {
    $fileSize = [int]([regex]::Match($file.Name, '^timings?-([0-9]+)\.csv$').Groups[1].Value)
    $rows = @(Import-Csv -LiteralPath $file.FullName)
    if (!$rows.Count) { throw "Empty timing CSV: $($file.FullName)" }
    foreach ($row in $rows) {
        $n = Read-Integer $row 'n' 128
        $impl = [string]$row.implementation
        if ($n -ne $fileSize -or $n % 128 -or $n -gt 4096) { throw "Invalid n in $($file.Name)" }
        if ($impl -notin @('k10','k10_pad8','k30_v1','k30_v2','strict')) { throw "Unknown implementation: $impl" }
        $round = Read-Integer $row 'round' 1
        $iteration = Read-Integer $row 'iteration' 1
        $key = "$n|$impl|$round|$iteration"
        if (!$sampleKeys.Add($key)) { throw "Duplicate timing sample: $key" }
        $eventMs = Read-Number $row 'event_ms'
        if ($eventMs -le 0) { throw "Nonpositive timing sample: $key" }
        if ($row.status -notin @('PASS','FAIL')) { throw "Invalid status for $key" }
        foreach ($hashColumn in 'a_hash','b_hash','c0_hash') {
            if (!$row.PSObject.Properties[$hashColumn] -or $row.$hashColumn -notmatch '^[0-9]+$') {
                throw "Missing/invalid $hashColumn for $key"
            }
        }
        $samples.Add([pscustomobject]@{
            n = $n; implementation = $impl; round = $round; iteration = $iteration
            order = Read-Integer $row 'order' 1
            event_ms = $eventMs; recorded_round_mean_ms = Read-Number $row 'round_mean_ms'
            alpha = Read-Number $row 'alpha'; beta = Read-Number $row 'beta'
            seed = Read-Integer $row 'seed'
            max_abs_error = Read-Number $row 'round_max_abs_error'
            max_rel_error = Read-Number $row 'round_max_rel_error'
            failure_count = Read-Integer $row 'round_failure_count'
            nonfinite_count = Read-Integer $row 'round_nonfinite_count'
            reference_nonfinite_count = Read-Integer $row 'reference_nonfinite_count'
            cpu_reference_count = Read-Integer $row 'cpu_reference_count' 1
            cpu_max_abs_error = Read-Number $row 'cpu_reference_max_abs_error'
            cpu_max_rel_error = Read-Number $row 'cpu_reference_max_rel_error'
            cpu_failure_count = Read-Integer $row 'cpu_reference_failure_count'
            cpu_nonfinite_count = Read-Integer $row 'cpu_reference_nonfinite_count'
            a_hash = [string]$row.a_hash; b_hash = [string]$row.b_hash; c0_hash = [string]$row.c0_hash
            status = [string]$row.status
            round_start_unix_ms = if ($row.PSObject.Properties['round_start_unix_ms']) {
                Read-Integer $row 'round_start_unix_ms' 1
            } else { $null }
            round_end_unix_ms = if ($row.PSObject.Properties['round_end_unix_ms']) {
                Read-Integer $row 'round_end_unix_ms' 1
            } else { $null }
        })
    }
}

# 每个尺寸的所有实现必须来自同一输入与数学问题；不能拼接不同 seed/beta 的 CSV。
foreach ($sizeGroup in ($samples | Group-Object n)) {
    $conditions = @($sizeGroup.Group | Select-Object alpha,beta,seed,a_hash,b_hash,c0_hash -Unique)
    if ($conditions.Count -ne 1) { throw "Mixed inputs/semantics for n=$($sizeGroup.Name)" }
}

$roundRows = [Collections.Generic.List[object]]::new()
foreach ($group in ($samples | Group-Object n,implementation,round)) {
    $members = @($group.Group | Sort-Object iteration)
    $first = $members[0]
    $orders = @($members.order | Select-Object -Unique)
    if ($orders.Count -ne 1) { throw "Inconsistent execution order for $($group.Name)" }
    if ($null -ne $first.round_start_unix_ms) {
        if (@($members.round_start_unix_ms | Select-Object -Unique).Count -ne 1 -or
            @($members.round_end_unix_ms | Select-Object -Unique).Count -ne 1 -or
            $first.round_end_unix_ms -lt $first.round_start_unix_ms) {
            throw "Inconsistent UTC window for $($group.Name)"
        }
    }
    for ($i = 0; $i -lt $members.Count; ++$i) {
        if ($members[$i].iteration -ne $i + 1) { throw "Missing iteration for $($group.Name)" }
    }
    $mean = [double](($members | Measure-Object -Property event_ms -Average).Average)
    foreach ($member in $members) {
        if ([Math]::Abs($member.recorded_round_mean_ms - $mean) -gt 1e-9 * [Math]::Max(1.0,$mean)) {
            throw "Recorded mean does not match event samples for $($group.Name)"
        }
    }
    $roundRows.Add([pscustomobject]@{
        n = $first.n; implementation = $first.implementation; round = $first.round
        order = $first.order; alpha = $first.alpha; beta = $first.beta; seed = $first.seed
        sample_count = $members.Count; mean_ms = $mean; median_sample_ms = Get-Median $members.event_ms
        min_sample_ms = [double](($members | Measure-Object event_ms -Minimum).Minimum)
        max_sample_ms = Get-Maximum $members 'event_ms'
        gflops = 2.0 * $first.n * $first.n * $first.n / ($mean * 1e6)
        max_abs_error = Get-Maximum $members 'max_abs_error'
        max_rel_error = Get-Maximum $members 'max_rel_error'
        failure_count = [long](Get-Maximum $members 'failure_count')
        nonfinite_count = [long](Get-Maximum $members 'nonfinite_count')
        reference_nonfinite_count = [long](Get-Maximum $members 'reference_nonfinite_count')
        cpu_reference_count = $first.cpu_reference_count
        cpu_max_abs_error = Get-Maximum $members 'cpu_max_abs_error'
        cpu_max_rel_error = Get-Maximum $members 'cpu_max_rel_error'
        cpu_failure_count = [long](Get-Maximum $members 'cpu_failure_count')
        cpu_nonfinite_count = [long](Get-Maximum $members 'cpu_nonfinite_count')
        status = if (@($members | Where-Object status -ne 'PASS').Count) { 'FAIL' } else { 'PASS' }
        round_start_unix_ms = $first.round_start_unix_ms
        round_end_unix_ms = $first.round_end_unix_ms
    })
}

$summaryRows = [Collections.Generic.List[object]]::new()
foreach ($group in ($roundRows | Group-Object n,implementation)) {
    $members = @($group.Group | Sort-Object round)
    $first = $members[0]
    for ($i = 0; $i -lt $members.Count; ++$i) {
        if ($members[$i].round -ne $i + 1) { throw "Missing round for $($group.Name)" }
    }
    $counts = @($members.sample_count | Select-Object -Unique)
    if ($counts.Count -ne 1) { throw "Different samples per round for $($group.Name)" }
    $median = Get-Median $members.mean_ms
    $minimum = [double](($members | Measure-Object mean_ms -Minimum).Minimum)
    $maximum = Get-Maximum $members 'mean_ms'
    $summaryRows.Add([pscustomobject]@{
        n = $first.n; implementation = $first.implementation
        alpha = $first.alpha; beta = $first.beta; seed = $first.seed
        round_count = $members.Count; samples_per_round = $first.sample_count
        total_event_samples = [long](($members | Measure-Object sample_count -Sum).Sum)
        median_ms = $median; min_ms = $minimum; max_ms = $maximum
        range_pct = 100.0 * ($maximum - $minimum) / $median
        median_gflops = 2.0 * $first.n * $first.n * $first.n / ($median * 1e6)
        median_tflops = 2.0 * $first.n * $first.n * $first.n / ($median * 1e9)
        k10_relative = 0.0; strict_relative = 0.0; strict_pct = 0.0
        max_abs_error = Get-Maximum $members 'max_abs_error'
        max_rel_error = Get-Maximum $members 'max_rel_error'
        # 验证统计在每轮内只数一次，不能把重复在每条 event 行的计数再乘 repeats。
        failed_elements_across_round_checks = [long](($members | Measure-Object failure_count -Sum).Sum)
        nonfinite_elements_across_round_checks = [long](($members | Measure-Object nonfinite_count -Sum).Sum)
        reference_nonfinite_count = [long](Get-Maximum $members 'reference_nonfinite_count')
        cpu_reference_count = $first.cpu_reference_count
        cpu_max_abs_error = Get-Maximum $members 'cpu_max_abs_error'
        cpu_max_rel_error = Get-Maximum $members 'cpu_max_rel_error'
        cpu_failure_count = [long](Get-Maximum $members 'cpu_failure_count')
        cpu_nonfinite_count = [long](Get-Maximum $members 'cpu_nonfinite_count')
        status = if (@($members | Where-Object status -ne 'PASS').Count) { 'FAIL' } else { 'PASS' }
    })
}
foreach ($sizeGroup in ($summaryRows | Group-Object n)) {
    $k10 = @($sizeGroup.Group | Where-Object implementation -eq 'k10')
    $strict = @($sizeGroup.Group | Where-Object implementation -eq 'strict')
    $names = @($sizeGroup.Group.implementation | Sort-Object)
    $actualSet = $names -join ','
    $stage1Set = @('k10','k10_pad8','k30_v2','strict') | Sort-Object
    $stage2Set = @('k10','k30_v1','k30_v2','strict') | Sort-Object
    if ($actualSet -ne ($stage1Set -join ',') -and $actualSet -ne ($stage2Set -join ',')) {
        throw "Need the complete all or sync implementation set, n=$($sizeGroup.Name)"
    }
    $roundCounts = @($sizeGroup.Group.round_count | Select-Object -Unique)
    $sampleCounts = @($sizeGroup.Group.samples_per_round | Select-Object -Unique)
    if ($roundCounts.Count -ne 1 -or $sampleCounts.Count -ne 1) {
        throw "Unbalanced implementation coverage for n=$($sizeGroup.Name)"
    }
    if ($k10[0].round_count -lt 5) {
        Write-Warning "n=$($sizeGroup.Name) has fewer than 5 rounds; use only as a pilot/check."
    }
    foreach ($row in $sizeGroup.Group) {
        $row.k10_relative = $k10[0].median_ms / $row.median_ms
        $row.strict_relative = $strict[0].median_ms / $row.median_ms
        $row.strict_pct = 100.0 * $row.strict_relative
    }
}

$pairedRows = [Collections.Generic.List[object]]::new()
foreach ($group in ($roundRows | Group-Object n,round)) {
    $pairDefinitions = if ($group.Group.implementation -contains 'k30_v1') {
        @(
            @{ comparison='v2_vs_v1'; baseline='k30_v1'; candidate='k30_v2' },
            @{ comparison='v2_vs_k10'; baseline='k10'; candidate='k30_v2' }
        )
    } else {
        @(@{ comparison='pad8_vs_k10'; baseline='k10'; candidate='k10_pad8' })
    }
    foreach ($definition in $pairDefinitions) {
        $baseline = @($group.Group | Where-Object implementation -eq $definition.baseline)
        $experiment = @($group.Group | Where-Object implementation -eq $definition.candidate)
        if ($baseline.Count -ne 1 -or $experiment.Count -ne 1) { throw "Missing pair: $($group.Name)" }
        $base = $baseline[0]; $trial = $experiment[0]
        $pairedRows.Add([pscustomobject]@{
            n = $base.n; round = $base.round; beta = $base.beta
            comparison = $definition.comparison
            baseline = $definition.baseline; candidate = $definition.candidate
            baseline_order = $base.order; candidate_order = $trial.order
            baseline_ms = $base.mean_ms; candidate_ms = $trial.mean_ms
            adjacent = [Math]::Abs($base.order - $trial.order) -eq 1
            # 保留第一阶段的列名；第二阶段通用字段在上面，不用K10标签冒充V1。
            k10_order = if ($definition.baseline -eq 'k10') { $base.order } else { $null }
            pad8_order = if ($definition.candidate -eq 'k10_pad8') { $trial.order } else { $null }
            k10_ms = if ($definition.baseline -eq 'k10') { $base.mean_ms } else { $null }
            pad8_ms = if ($definition.candidate -eq 'k10_pad8') { $trial.mean_ms } else { $null }
            # 吞吐提升与延迟变化分开写，避免把二者百分比误当成一样。
            speedup_ratio = $base.mean_ms / $trial.mean_ms
            speedup_pct = 100.0 * ($base.mean_ms / $trial.mean_ms - 1.0)
            latency_change_pct = 100.0 * ($trial.mean_ms / $base.mean_ms - 1.0)
            status = if ($base.status -eq 'PASS' -and $trial.status -eq 'PASS') { 'PASS' } else { 'FAIL' }
        })
    }
}
$pairedSummary = foreach ($group in ($pairedRows | Group-Object n,comparison)) {
    $members = @($group.Group)
    $bootstrap = Get-PairedBootstrap $members.speedup_ratio
    [pscustomobject]@{
        n = $members[0].n; pair_count = $members.Count
        comparison = $members[0].comparison
        baseline = $members[0].baseline; candidate = $members[0].candidate
        median_speedup_pct = Get-Median $members.speedup_pct
        min_speedup_pct = [double](($members | Measure-Object speedup_pct -Minimum).Minimum)
        max_speedup_pct = Get-Maximum $members 'speedup_pct'
        median_latency_change_pct = Get-Median $members.latency_change_pct
        faster_pair_count = @($members | Where-Object speedup_pct -gt 0).Count
        slower_pair_count = @($members | Where-Object speedup_pct -lt 0).Count
        tied_pair_count = @($members | Where-Object speedup_pct -eq 0).Count
        adjacent_pair_count = @($members | Where-Object adjacent -eq $true).Count
        geomean_speedup_pct = 100.0 * ($bootstrap.ratio - 1.0)
        conditional_ci95_low_pct = if ($null -ne $bootstrap.lower) { 100.0 * ($bootstrap.lower - 1.0) } else { $null }
        conditional_ci95_high_pct = if ($null -ne $bootstrap.upper) { 100.0 * ($bootstrap.upper - 1.0) } else { $null }
        bootstrap_resamples = $bootstrap.resamples
        bootstrap_seed = 20261021
        ci_scope = if ($bootstrap.resamples) {
            'conditional_on_observed_rounds_assuming_exchangeability_not_WDDM_or_between_session_uncertainty'
        } else { 'not_computed_fewer_than_5_pairs' }
        status = if (@($members | Where-Object status -ne 'PASS').Count) { 'FAIL' } else { 'PASS' }
    }
}

# 所有输入检查通过后才写输出；-Force 只重新生成四张可由原始 CSV 复算的表。
Export-InvariantCsv ($roundRows | Sort-Object n,round,order) (Join-Path $resultPath 'rounds.csv')
Export-InvariantCsv ($summaryRows | Sort-Object n,implementation) (Join-Path $resultPath 'summary.csv')
Export-InvariantCsv ($pairedRows | Sort-Object n,comparison,round) (Join-Path $resultPath 'paired.csv')
Export-InvariantCsv ($pairedSummary | Sort-Object n,comparison) (Join-Path $resultPath 'paired-summary.csv')
$summaryRows | Sort-Object n,implementation |
    Format-Table n,implementation,round_count,median_ms,median_tflops,strict_pct,range_pct,status -AutoSize
Write-Host "Summaries: $resultPath"
