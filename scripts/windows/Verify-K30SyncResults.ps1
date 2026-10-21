#requires -Version 7.0
# CPU-only 独立复算：直接读取原始 event/correctness/telemetry，不调用汇总脚本，
# 不运行 GPU，也不修改原始结果。JSON 目标必须尚不存在。
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ResultDir,
    [Parameter(Mandatory)][string]$Phase1Dir,
    [Parameter(Mandatory)][string]$PriorManifest,
    [Parameter(Mandatory)][string]$OutputPath
)
$ErrorActionPreference = 'Stop'
$inv = [Globalization.CultureInfo]::InvariantCulture
$ResultDir = (Resolve-Path -LiteralPath $ResultDir).Path
$Phase1Dir = (Resolve-Path -LiteralPath $Phase1Dir).Path
$PriorManifest = (Resolve-Path -LiteralPath $PriorManifest).Path
$OutputPath = [IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $OutputPath) { throw "Refusing to overwrite $OutputPath" }
$issues = [Collections.Generic.List[string]]::new()
function Number([string]$Text) {
    $value = [double]::Parse($Text,$inv)
    if (![double]::IsFinite($value)) { throw "Nonfinite value in audit input: $Text" }
    return $value
}
function Median([double[]]$Values) {
    $sorted = [double[]]$Values.Clone()
    [Array]::Sort($sorted)
    $mid = [int][Math]::Floor($sorted.Count/2)
    if ($sorted.Count%2) { return $sorted[$mid] }
    return ($sorted[$mid-1]+$sorted[$mid])/2
}
function Check-Close([double]$Actual,[double]$Expected,[string]$Label) {
    if ([Math]::Abs($Actual-$Expected) -gt 1e-10*[Math]::Max(1.0,[Math]::Abs($Expected))) {
        $issues.Add("Numerical mismatch: $Label")
    }
}
$meta = Get-Content -LiteralPath "$ResultDir/environment.json" -Raw | ConvertFrom-Json
$rawFiles = @(Get-ChildItem -LiteralPath $ResultDir -File -Filter 'timings-*.csv' | Sort-Object Name)
$correctFiles = @(Get-ChildItem -LiteralPath $ResultDir -File -Filter 'correctness-*.csv' | Sort-Object Name)
$raw = @($rawFiles | ForEach-Object { Import-Csv -LiteralPath $_.FullName })
$correct = @($correctFiles | ForEach-Object { Import-Csv -LiteralPath $_.FullName })
$savedSummary = @(Import-Csv "$ResultDir/summary.csv")
$savedPairs = @(Import-Csv "$ResultDir/paired-summary.csv")
$allowed = @('k10','k30_v1','k30_v2','strict')
$expectedRows = 3*4*[int]$meta.rounds*[int]$meta.repeats
if ($raw.Count -ne $expectedRows) { $issues.Add("Expected $expectedRows timing rows, found $($raw.Count).") }
if ($correct.Count -ne 72) { $issues.Add("Expected 72 correctness cases, found $($correct.Count).") }
if ($savedSummary.Count -ne 12 -or $savedPairs.Count -ne 6) { $issues.Add('Unexpected summary group count.') }
$keys = [Collections.Generic.HashSet[string]]::new()
foreach ($row in $raw) {
    if (!$keys.Add("$($row.n)|$($row.implementation)|$($row.round)|$($row.iteration)")) {
        $issues.Add('Duplicate raw event key.')
    }
}
$correctKeys = @($correct | ForEach-Object { "$($_.n)|$($_.beta)|$($_.implementation)" })
foreach ($n in 128,256,512,1024,2048,4096) {
    foreach ($beta in 0,1,3) {
        foreach ($impl in $allowed) {
            if (@($correctKeys | Where-Object { $_ -eq "$n|$beta|$impl" }).Count -ne 1) {
                $issues.Add("Missing or duplicate correctness case: $n/$beta/$impl")
            }
        }
    }
}
$allChecks = @($correct) + @($raw)
foreach ($row in $allChecks) {
    if ($row.status -ne 'PASS' -or (Number $row.round_failure_count) -ne 0 -or
        (Number $row.round_nonfinite_count) -ne 0 -or
        (Number $row.reference_nonfinite_count) -ne 0 -or
        (Number $row.cpu_reference_failure_count) -ne 0 -or
        (Number $row.cpu_reference_nonfinite_count) -ne 0 -or
        (Number $row.round_max_abs_error) -gt 0.01 -or
        (Number $row.cpu_reference_max_abs_error) -gt 0.01) {
        $issues.Add("Failed numerical verification in input: $($row.n)/$($row.implementation)")
    }
}

$rounds = foreach ($group in ($raw | Group-Object n,implementation,round)) {
    $rows = @($group.Group | Sort-Object { [int]$_.iteration })
    $sum = 0.0
    foreach ($row in $rows) { $sum += Number $row.event_ms }
    $mean = $sum/$rows.Count
    if ($rows.Count -ne [int]$meta.repeats) { $issues.Add("Incomplete round: $($group.Name)") }
    foreach ($row in $rows) { Check-Close (Number $row.round_mean_ms) $mean "round $($group.Name)" }
    [pscustomobject]@{
        n=[int]$rows[0].n; implementation=$rows[0].implementation
        round=[int]$rows[0].round; order=[int]$rows[0].order; mean_ms=$mean
        start=[long]$rows[0].round_start_unix_ms; end=[long]$rows[0].round_end_unix_ms
    }
}
if (@($rounds).Count -ne 12*[int]$meta.rounds) { $issues.Add('Unexpected number of rounds.') }
$summaryAudit = foreach ($group in ($rounds | Group-Object n,implementation)) {
    $first=$group.Group[0]; $values=[double[]]$group.Group.mean_ms
    $median=Median $values
    $saved=@($savedSummary | Where-Object { [int]$_.n-eq$first.n -and $_.implementation-eq$first.implementation })
    if ($saved.Count-ne1) { throw 'Missing summary row for audit.' }
    $tflops=2.0*$first.n*$first.n*$first.n/($median*1e9)
    Check-Close (Number $saved[0].median_ms) $median "$($group.Name) median"
    Check-Close (Number $saved[0].median_tflops) $tflops "$($group.Name) throughput"
    [pscustomobject]@{
        n=$first.n; implementation=$first.implementation; rounds=$values.Count
        median_round_ms=$median; tflops_at_median_ms=$tflops
        minimum_round_ms=($values | Measure-Object -Minimum).Minimum
        maximum_round_ms=($values | Measure-Object -Maximum).Maximum
    }
}
$inputs = foreach ($n in 1024,2048,4096) {
    $before=Import-Csv "$Phase1Dir/timings-$n.csv" | Select-Object -First 1
    $current=@($raw | Where-Object n -eq "$n")
    $matches=[ordered]@{n=$n}
    foreach ($field in 'a_hash','b_hash','c0_hash','alpha','beta','seed') {
        $agrees=@($current | Where-Object { $_.$field -ne $before.$field }).Count -eq 0
        $matches[$field]=$agrees
        if (!$agrees) { $issues.Add("Phase1 input mismatch: $n/$field") }
    }
    [pscustomobject]$matches
}
$slowEvents = foreach ($group in ($raw | Group-Object n,implementation)) {
    $values=[double[]]@($group.Group | ForEach-Object { Number $_.event_ms })
    $median=Median $values; $maximum=($values | Measure-Object -Maximum).Maximum
    [pscustomobject]@{
        n=[int]$group.Group[0].n; implementation=$group.Group[0].implementation
        samples=$values.Count; median_event_ms=$median; maximum_event_ms=$maximum
        max_over_event_median=$maximum/$median
        samples_over_1_5x_event_median=@($values | Where-Object { $_-gt1.5*$median }).Count
        dropped_samples=0
    }
}

$pairAudit = foreach ($saved in $savedPairs) {
    $n=[int]$saved.n; $ratios=[Collections.Generic.List[double]]::new()
    $adjacent=0; $latencies=[Collections.Generic.List[double]]::new()
    for ($r=1;$r-le[int]$meta.rounds;++$r) {
        $base=@($rounds | Where-Object { $_.n-eq$n -and $_.round-eq$r -and $_.implementation-eq$saved.baseline })
        $candidate=@($rounds | Where-Object { $_.n-eq$n -and $_.round-eq$r -and $_.implementation-eq$saved.candidate })
        if ($base.Count-ne1 -or $candidate.Count-ne1) { throw 'Missing pair.' }
        $ratios.Add($base[0].mean_ms/$candidate[0].mean_ms)
        $latencies.Add(100*($candidate[0].mean_ms/$base[0].mean_ms-1))
        $adjacent += [int]([Math]::Abs($base[0].order-$candidate[0].order)-eq1)
    }
    $logs=@($ratios | ForEach-Object { [Math]::Log($_) })
    $geomean=100*([Math]::Exp(($logs | Measure-Object -Average).Average)-1)
    $draws=[double[]]::new(20000); $rng=[Random]::new(20261021)
    for ($b=0;$b-lt20000;++$b) {
        $sum=0.0
        for ($j=0;$j-lt$logs.Count;++$j) { $sum+=$logs[$rng.Next($logs.Count)] }
        $draws[$b]=100*([Math]::Exp($sum/$logs.Count)-1)
    }
    [Array]::Sort($draws)
    # 20000个draw的2.5/97.5百分位，使用线性插值；与汇总脚本独立实现。
    $lo=$draws[499]*0.025+$draws[500]*0.975
    $hi=$draws[19499]*0.975+$draws[19500]*0.025
    $speedups=[double[]]@($ratios | ForEach-Object { 100*($_-1) })
    $computed=[ordered]@{
        n=$n; comparison=$saved.comparison; baseline=$saved.baseline; candidate=$saved.candidate
        pair_count=$ratios.Count; median_speedup_pct=Median $speedups
        min_speedup_pct=($speedups|Measure-Object -Minimum).Minimum
        max_speedup_pct=($speedups|Measure-Object -Maximum).Maximum
        median_latency_change_pct=Median $latencies.ToArray()
        geomean_speedup_pct=$geomean; conditional_ci95_low_pct=$lo; conditional_ci95_high_pct=$hi
        faster_pair_count=@($ratios|Where-Object{$_-gt1}).Count
        slower_pair_count=@($ratios|Where-Object{$_-lt1}).Count
        tied_pair_count=@($ratios|Where-Object{$_-eq1}).Count
        adjacent_pair_count=$adjacent
        bootstrap_seed=20261021; bootstrap_resamples=20000
    }
    foreach ($field in $computed.Keys) {
        if ($field -notin @('comparison','baseline','candidate')) {
            Check-Close (Number $saved.$field) ([double]$computed[$field]) "paired $n/$($saved.comparison)/$field"
        }
    }
    [pscustomobject]$computed
}

$zone=[TimeZoneInfo]::FindSystemTimeZoneById($meta.host_time_zone)
$telemetry=@(foreach ($row in Import-Csv "$ResultDir/telemetry.csv") {
    if (!$row.timestamp -or $row.timestamp-eq'timestamp') { continue }
    $local=[DateTime]::ParseExact($row.timestamp.Trim(),'yyyy/MM/dd HH:mm:ss.fff',$inv)
    $utc=[TimeZoneInfo]::ConvertTimeToUtc([DateTime]::SpecifyKind($local,[DateTimeKind]::Unspecified),$zone)
    $gpuProperty=$row.PSObject.Properties|Where-Object{$_.Name.Trim()-eq'index'}
    [pscustomobject]@{ms=([DateTimeOffset]$utc).ToUnixTimeMilliseconds();gpu=$gpuProperty.Value.Trim()}
})
$savedTelemetry=Import-Csv "$ResultDir/round-telemetry.csv"
$coverage=@(foreach ($round in $rounds) {
    $count=@($telemetry|Where-Object{$_.gpu-eq'0'-and$_.ms-ge$round.start-and$_.ms-le$round.end}).Count
    $saved=@($savedTelemetry|Where-Object{[int]$_.n-eq$round.n-and$_.implementation-eq$round.implementation-and[int]$_.round-eq$round.round})
    if ($saved.Count-ne1-or[int]$saved[0].telemetry_points-ne$count) { $issues.Add('Telemetry coverage mismatch.') }
    [pscustomobject]@{n=$round.n;implementation=$round.implementation;round=$round.round;points=$count}
})
$history=@(foreach ($row in Import-Csv -LiteralPath $PriorManifest) {
    $exists=Test-Path -LiteralPath $row.Path -PathType Leaf
    $actual=if($exists){(Get-FileHash -LiteralPath $row.Path -Algorithm SHA256).Hash}else{$null}
    $unchanged=$exists-and$actual-eq$row.Hash
    if(!$unchanged){$issues.Add("Historical file changed or missing: $($row.Path)")}
    [pscustomobject]@{path=$row.Path;expected_sha256=$row.Hash;actual_sha256=$actual;unchanged=$unchanged}
})
$auditFiles=@($rawFiles.FullName)+@($correctFiles.FullName)+@(
    "$ResultDir/environment.json","$ResultDir/summary.csv","$ResultDir/rounds.csv",
    "$ResultDir/paired-summary.csv","$ResultDir/telemetry.csv","$ResultDir/round-telemetry.csv",
    $PriorManifest,$PSCommandPath)
$audit=[ordered]@{
    schema_version=1; status=if($issues.Count){'FAIL'}else{'PASS'}
    audit_utc=[DateTimeOffset]::UtcNow.ToString('o')
    result_dir=$ResultDir; phase1_dir=$Phase1Dir
    method='Independent CPU-only recomputation from original events; no GPU, no input-file writes.'
    counts=@{correctness_cases=$correct.Count;event_samples=$raw.Count;rounds=@($rounds).Count;paired_comparisons=$pairAudit.Count}
    correctness=@{
        failed_cases=@($correct|Where-Object status -ne 'PASS').Count
        max_abs_error=($correct|Measure-Object round_max_abs_error -Maximum).Maximum
        max_cpu_abs_error=($correct|Measure-Object cpu_reference_max_abs_error -Maximum).Maximum
        nonfinite_count=($correct|Measure-Object round_nonfinite_count -Sum).Sum
    }
    phase1_inputs=$inputs; recomputed_summary=$summaryAudit; recomputed_paired_summary=$pairAudit
    slow_event_diagnostics=$slowEvents
    telemetry=@{
        raw_points=$telemetry.Count; windows=$coverage.Count
        windows_with_samples=@($coverage|Where-Object points -gt 0).Count
        windows_without_samples=@($coverage|Where-Object points -eq 0).Count
        coverage=$coverage
    }
    historical_results=@{checked=$history.Count;unchanged=@($history|Where-Object unchanged -eq $true).Count;files=$history}
    limitations=@(
        'Bootstrap intervals are conditional on observed paired rounds and exchangeability; not WDDM or between-session uncertainty.',
        'Telemetry is sparse and associated with warmup+timing+validation windows; empty windows remain unknown.',
        'CPU double reference is full at N128 and sampled at larger sizes; numerical PASS is not a sanitizer proof.',
        'No event outlier was removed; medians summarize round means, not pooled events.'
    )
    input_sha256=@($auditFiles|ForEach-Object{Get-FileHash -LiteralPath $_ -Algorithm SHA256}|Select-Object Path,Hash)
    issues=$issues.ToArray()
}
$parent=Split-Path -Parent $OutputPath
if (!(Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent | Out-Null }
$audit|ConvertTo-Json -Depth 10|Set-Content -LiteralPath $OutputPath -Encoding utf8
Write-Host "$($audit.status): audit saved to $OutputPath; historical files unchanged $(@($history|Where-Object unchanged -eq $true).Count)/$($history.Count)."
if($issues.Count){throw ($issues -join "`n")}
