[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$SourceCommit,
    [Parameter(Mandatory=$true)][string]$Manifest,
    [string[]]$VerifyPaths = @('docs/zh/rtx5090/README.md'),
    [string]$Relay = 'D:\CUDA-Lab\cuda-learning\rtx5090-execution-20261022\publish-relay',
    [string]$Proxy = 'http://127.0.0.1:7877'
)
$ErrorActionPreference = 'Stop'
$branch = 'results/rtx5090-gemm-reproduction-20261022'
$ref = 'refs/heads/' + $branch
$origin = 'https://github.com/QiLHong/SGEMM_CUDA.git'
$remoteRepo = 'ssh://autodl-gpu/root/autodl-tmp/SGEMM_CUDA-rtx5090-20261022'
$sshCommand = 'C:/Windows/System32/OpenSSH/ssh.exe -F C:/Users/Administrator/.ssh/config -o BatchMode=yes'
$prefixes = @('docs/zh/rtx5090/', 'results/rtx5090/', 'scripts/rtx5090/', 'experiments/rtx5090/')

function Run-Git([string[]]$Arguments) {
    $output = & git -c "http.proxy=$Proxy" -c "core.sshCommand=$sshCommand" -C $Relay @Arguments 2>&1
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "Git $($Arguments[0]) failed with exit $code. Stop publication; inspect redacted diagnostics locally." }
    return (($output | ForEach-Object { $_.ToString() }) -join "`n").Trim()
}
function Allowed([string]$Path) {
    if ($Path -match '(^|/)\.\.(/|$)|[\r\n\\]' -or [IO.Path]::IsPathRooted($Path)) { return $false }
    foreach ($prefix in $prefixes) { if ($Path.StartsWith($prefix, [StringComparison]::Ordinal)) { return $true } }
    return $false
}
function Write-Json([string]$Path, $Value) {
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Path -Encoding utf8
}
function Remote-Sha {
    $line = Run-Git @('ls-remote', 'origin', $ref)
    if ($line -notmatch ('^([0-9a-f]{40})\s+' + [regex]::Escape($ref) + '$')) { throw 'Unexpected remote ref response' }
    return $Matches[1]
}

$gitDir = Run-Git @('rev-parse', '--absolute-git-dir')
$state = Join-Path $gitDir 'rtx5090-publish'
[IO.Directory]::CreateDirectory($state) | Out-Null
$lock = $null
$pendingPath = Join-Path $state "$SourceCommit.pending.json"
$receiptPath = Join-Path $state "$SourceCommit.verified.json"
try {
    $lock = [IO.File]::Open((Join-Path $state 'relay.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    if ((Run-Git @('remote', 'get-url', 'origin')) -ne $origin) { throw 'Unexpected origin (value suppressed)' }
    if (Run-Git @('status', '--porcelain')) { throw 'Relay worktree has changes; preserve and inspect them first' }
    if (!(Allowed $Manifest)) { throw 'Manifest outside task allowlist' }
    $before = Remote-Sha
    Write-Json $pendingPath @{status='FETCHING'; source_commit=$SourceCommit; remote_before=$before; branch=$branch; manifest=$Manifest; utc=[DateTime]::UtcNow.ToString('o')}
    Run-Git @('fetch', '--no-tags', $remoteRepo, $ref) | Out-Null
    $fetched = Run-Git @('rev-parse', 'FETCH_HEAD')
    if ($fetched -ne $SourceCommit) { throw 'Server HEAD changed; review it and rerun with its exact commit' }
    Run-Git @('fetch', '--no-tags', 'origin', $ref) | Out-Null
    $observed = Run-Git @('rev-parse', 'FETCH_HEAD')
    if ($observed -ne $before) { throw 'Remote branch changed during preflight; inspect before continuing' }
    Run-Git @('merge-base', '--is-ancestor', $before, $SourceCommit) | Out-Null
    $changes = Run-Git @('diff', '--name-only', $before, $SourceCommit)
    if ($changes) { foreach ($path in ($changes -split "`n")) { if (!(Allowed $path)) { throw "Commit changes non-task path: $path" } } }
    # Detached checkout preserves both histories; no merge, reset, rebase or force.
    Run-Git @('switch', '--detach', $SourceCommit) | Out-Null
    $bundlePath = Join-Path $state "$SourceCommit.bundle"
    if ($before -ne $SourceCommit -and !(Test-Path -LiteralPath $bundlePath)) {
        Run-Git @('bundle', 'create', $bundlePath, "${before}..HEAD") | Out-Null
    }
    Write-Json $pendingPath @{status='READY_TO_PUSH'; source_commit=$SourceCommit; remote_before=$before; branch=$branch; manifest=$Manifest; bundle=$bundlePath; utc=[DateTime]::UtcNow.ToString('o')}
    Run-Git @('push', 'origin', "HEAD:$ref") | Out-Null
    $after = Remote-Sha
    if ($after -ne $SourceCommit) { throw 'Remote SHA does not equal pushed source commit' }
    $readback = Join-Path $state ("readback-" + $SourceCommit)
    [IO.Directory]::CreateDirectory($readback) | Out-Null
    $verified = @()
    $toRead = @($Manifest) + @($VerifyPaths) | Select-Object -Unique
    foreach ($path in $toRead) {
        if (!(Allowed $path)) { throw "Readback path outside allowlist: $path" }
        $localFile = Join-Path $readback ($path -replace '/', [IO.Path]::DirectorySeparatorChar)
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($localFile)) | Out-Null
        $url = "https://raw.githubusercontent.com/QiLHong/SGEMM_CUDA/$SourceCommit/$path"
        # Fresh HTTPS bytes from GitHub, never a read of the server working tree.
        Invoke-WebRequest -Uri $url -Proxy $Proxy -OutFile $localFile -TimeoutSec 60
        $expectedBlob = Run-Git @('rev-parse', "${SourceCommit}:$path")
        $actualBlob = Run-Git @('hash-object', '--no-filters', '--', $localFile)
        if ($expectedBlob -ne $actualBlob) { throw "Readback Git blob mismatch: $path" }
        $verified += @{path=$path; git_blob=$actualBlob; sha256=(Get-FileHash -LiteralPath $localFile -Algorithm SHA256).Hash.ToLowerInvariant(); bytes=(Get-Item -LiteralPath $localFile).Length; url=$url}
    }
    $manifestFile = Join-Path $readback ($Manifest -replace '/', [IO.Path]::DirectorySeparatorChar)
    $expectedSha256 = @{}
    foreach ($line in Get-Content -LiteralPath $manifestFile -Encoding utf8) {
        if ($line -notmatch '^([0-9a-f]{64})  (.+)$') { throw 'Malformed manifest entry' }
        $expectedSha256[$Matches[2]] = $Matches[1]
    }
    foreach ($entry in $verified) {
        if ($entry.path -ne $Manifest) {
            if (!$expectedSha256.ContainsKey($entry.path)) { throw "Readback payload absent from selected checkpoint manifest: $($entry.path)" }
            if ($expectedSha256[$entry.path] -ne $entry.sha256) { throw "Readback manifest SHA256 mismatch: $($entry.path)" }
        }
    }
    $receipt = @{status='VERIFIED'; source_commit=$SourceCommit; remote_sha=$after; branch=$branch; push_exit_code=0; readback=$verified; readback_directory=$readback; utc=[DateTime]::UtcNow.ToString('o')}
    Write-Json $receiptPath $receipt
    Write-Json $pendingPath @{status='VERIFIED'; receipt=$receiptPath; source_commit=$SourceCommit}
    $receipt | ConvertTo-Json -Depth 12
} catch {
    Write-Json $pendingPath @{status='PENDING_OR_BLOCKED'; source_commit=$SourceCommit; branch=$branch; manifest=$Manifest; reason=$_.Exception.Message; utc=[DateTime]::UtcNow.ToString('o'); remote_backup_verified=$false}
    throw
} finally {
    if ($null -ne $lock) { $lock.Dispose() }
}

