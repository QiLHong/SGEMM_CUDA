#requires -Version 7.0
# 仅将实际落在每个实现的 UTC 轮次窗口内的 GPU 遥测点关联；不插值，也不假称逐 kernel 时钟。
[CmdletBinding()]
param([Parameter(Mandatory)][string]$ResultDir)
$ErrorActionPreference = 'Stop'
$ResultDir = (Resolve-Path -LiteralPath $ResultDir).Path
$output = Join-Path $ResultDir 'round-telemetry.csv'
if (Test-Path -LiteralPath $output) { throw "Refusing to overwrite $output" }
$metadata = Get-Content "$ResultDir\environment.json" -Raw | ConvertFrom-Json
$zone = [TimeZoneInfo]::FindSystemTimeZoneById($metadata.host_time_zone)
$samples = foreach ($row in Import-Csv "$ResultDir\telemetry.csv") {
    if (!$row.timestamp -or $row.timestamp -eq 'timestamp') { continue }
    $local = [DateTime]::ParseExact($row.timestamp.Trim(),'yyyy/MM/dd HH:mm:ss.fff',[Globalization.CultureInfo]::InvariantCulture)
    $utc = [TimeZoneInfo]::ConvertTimeToUtc([DateTime]::SpecifyKind($local,[DateTimeKind]::Unspecified),$zone)
    $data = @{}
    foreach ($property in $row.PSObject.Properties) { $data[$property.Name.Trim()] = $property.Value.Trim() }
    [pscustomobject]@{
        unix_ms = ([DateTimeOffset]$utc).ToUnixTimeMilliseconds()
        gpu = $data['index']
        temperature = $data['temperature.gpu']
        sm_clock = ($data['clocks.current.sm [MHz]'] -replace ' MHz$','')
        mem_clock = ($data['clocks.current.memory [MHz]'] -replace ' MHz$','')
        power = ($data['power.draw [W]'] -replace ' W$','')
        pstate = $data['pstate']
    }
}
function Get-Range($points,[string]$field) {
    $values = @($points | ForEach-Object {
        $v = 0.0
        if ([double]::TryParse([string]$_.$field,[Globalization.NumberStyles]::Float,
                [Globalization.CultureInfo]::InvariantCulture,[ref]$v)) { $v }
    })
    if (!$values.Count) { return @($null,$null) }
    $measure = $values | Measure-Object -Minimum -Maximum
    return @($measure.Minimum,$measure.Maximum)
}
$outputRows = foreach ($round in Import-Csv "$ResultDir\rounds.csv") {
    $start = [long]$round.round_start_unix_ms
    $end = [long]$round.round_end_unix_ms
    $points = @($samples | Where-Object { $_.gpu -eq '0' -and $_.unix_ms -ge $start -and $_.unix_ms -le $end })
    $temperature = Get-Range $points 'temperature'
    $sm = Get-Range $points 'sm_clock'
    $memory = Get-Range $points 'mem_clock'
    $power = Get-Range $points 'power'
    [pscustomobject]@{
        n=$round.n; implementation=$round.implementation; round=$round.round
        round_start_unix_ms=$start; round_end_unix_ms=$end; telemetry_points=$points.Count
        temperature_min_c=$temperature[0]; temperature_max_c=$temperature[1]
        sm_clock_min_mhz=$sm[0]; sm_clock_max_mhz=$sm[1]
        memory_clock_min_mhz=$memory[0]; memory_clock_max_mhz=$memory[1]
        power_min_w=$power[0]; power_max_w=$power[1]
        pstates=(@($points.pstate | Sort-Object -Unique) -join '|')
        scope='warmup+timing+validation window; no interpolation'
    }
}
$outputRows | Export-Csv -NoTypeInformation -Encoding utf8 $output
Write-Host "Telemetry: $($samples.Count) raw points; $($outputRows.Count) round windows."
