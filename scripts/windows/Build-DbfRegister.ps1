#requires -Version 7.0
# 编译和静态导出，不启动GPU；两个目标仅实例化同一个原始DBF模板。
[CmdletBinding()]
param(
    [string]$BuildDir = 'build-windows-dbf-register-sm89',
    [string]$EvidenceDir,
    [string]$Python = 'D:\Anaconda\Anaconda\python.exe',
    [ValidateRange(1,16)][int]$Jobs = 2
)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\Initialize-Toolchain.ps1"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$source = Join-Path $repo 'experiments\vitamin-dbf-register'
if (![IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
if (!$EvidenceDir) { $EvidenceDir = "results\windows\20261021-vitamin-followup\register-limit\build-$(Get-Date -Format 'yyyyMMdd-HHmmss')" }
if (![IO.Path]::IsPathRooted($EvidenceDir)) { $EvidenceDir = Join-Path $repo $EvidenceDir }
if (Test-Path -LiteralPath $EvidenceDir) { throw "Refusing to overwrite evidence: $EvidenceDir" }
if (Test-Path -LiteralPath "$BuildDir\CMakeCache.txt") {
    $homeLine = @(Get-Content -LiteralPath "$BuildDir\CMakeCache.txt" | Where-Object { $_ -like 'CMAKE_HOME_DIRECTORY:INTERNAL=*' })
    if ($homeLine.Count -ne 1 -or [IO.Path]::GetFullPath($homeLine[0].Substring($homeLine[0].IndexOf('=')+1)) -ne [IO.Path]::GetFullPath($source)) {
        throw 'Build directory belongs to another project.'
    }
}
& $Python "$repo\scripts\extract_vitamin_simt.py" --vitamin "$repo\..\vitamin-cuda" `
    --output "$repo\src\kernels\vendor\vitamin_sgemm.cuh" `
    --manifest "$repo\src\kernels\vendor\provenance.json" --verify
if ($LASTEXITCODE) { throw 'Unchanged upstream device extraction verification failed.' }
New-Item -ItemType Directory -Path $EvidenceDir | Out-Null
$log = Join-Path $EvidenceDir 'build.log'
$configure = @('-S',$source,'-B',$BuildDir,'-G','Ninja',
    '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_TRY_COMPILE_CONFIGURATION=Release',
    '-DCMAKE_CUDA_ARCHITECTURES=89',"-DCMAKE_CUDA_COMPILER=$env:CUDACXX",
    "-DCMAKE_CUDA_HOST_COMPILER=$env:CUDAHOSTCXX","-DCMAKE_CXX_COMPILER=$env:CUDAHOSTCXX",
    "-DCUDAToolkit_ROOT=$env:CUDA_PATH")
& cmake @configure 2>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE) { throw 'DBF CMake configure failed.' }
& cmake --build $BuildDir --target windows_dbf_default_benchmark windows_dbf_maxr128_benchmark `
    --parallel $Jobs --verbose 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE) { throw 'DBF isolated build failed.' }
$commands = @(Get-Content -LiteralPath "$BuildDir\compile_commands.json" -Raw | ConvertFrom-Json)
if ($commands.Count -ne 2) { throw 'Expected exactly two isolated compile commands.' }
$variants = @()
foreach ($variant in 'default','maxr128') {
    $limit = if ($variant -eq 'default') { 0 } else { 128 }
    $command = @($commands | Where-Object command -Match "DBF_MAX_REGISTERS=$limit(\s|$)")
    if ($command.Count -ne 1) { throw "Missing unique compile command for $variant." }
    $text = $command[0].command
    if ($text -notmatch 'arch=compute_89,code=\[compute_89,sm_89\]' -or
        $text -match 'use_fast_math|allow-unsupported-compiler' -or
        (($text -match '--maxrregcount=128') -ne ($variant -eq 'maxr128'))) {
        throw "Unexpected CUDA flags for $variant."
    }
    $name = "windows_dbf_${variant}_benchmark.exe"
    $dest = Join-Path $EvidenceDir $variant
    New-Item -ItemType Directory -Path $dest | Out-Null
    Copy-Item -LiteralPath "$BuildDir\$name" -Destination $dest
    foreach ($dump in @(
        @{Option='--dump-resource-usage';Name='resources.txt'},
        @{Option='--dump-sass';Name='binary.sass'},
        @{Option='--dump-ptx';Name='binary.ptx'})) {
        & "$env:CUDA_PATH\bin\cuobjdump.exe" $dump.Option "$BuildDir\$name" |
            Set-Content -LiteralPath (Join-Path $dest $dump.Name) -Encoding utf8
        if ($LASTEXITCODE) { throw "cuobjdump failed: $variant $($dump.Option)" }
    }
    $resources = Get-Content -LiteralPath "$dest\resources.txt" -Raw
    $functions = @([regex]::Matches($resources,'(?m)^\s*Function\s+(\S+):'))
    if ($functions.Count -ne 1 -or $functions[0].Groups[1].Value -notmatch 'sgemm_at_bcf_swizzling_dbf_rw_kernel') {
        throw "Unexpected kernel instantiations in $variant binary."
    }
    $match = [regex]::Match($resources,'REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)')
    if (!$match.Success) { throw 'Resource parser failed.' }
    $variants += [ordered]@{
        variant=$variant; requested_register_limit=$limit
        compile_command=$text
        executable=(Join-Path $BuildDir $name)
        binary_sha256=(Get-FileHash -LiteralPath "$BuildDir\$name" -Algorithm SHA256).Hash
        registers=[int]$match.Groups[1].Value; stack_bytes=[int]$match.Groups[2].Value
        static_shared_bytes=[int]$match.Groups[3].Value; local_bytes=[int]$match.Groups[4].Value
        user_kernel_count=$functions.Count; kernel_symbol=$functions[0].Groups[1].Value
    }
}
Copy-Item -LiteralPath "$BuildDir\compile_commands.json","$BuildDir\CMakeCache.txt" -Destination $EvidenceDir
$sourcePaths = @("$repo\scripts\windows_dbf_register_benchmark.cu",
    "$source\CMakeLists.txt","$repo\src\kernels\vendor\vitamin_sgemm.cuh",
    "$repo\src\kernels\vendor\provenance.json","$repo\src\kernels\vendor\vitamin-LICENSE",
    "$PSScriptRoot\Build-DbfRegister.ps1")
$sourceHashes = @($sourcePaths | ForEach-Object {
    [ordered]@{path=$_;sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash}
})
$manifest = [ordered]@{
    created_utc=[DateTimeOffset]::UtcNow.ToString('o'); architecture='sm_89'
    precision='strict_fp32_no_fast_math'; only_device_factor='--maxrregcount=128'
    gpu_execution_performed=$false; evidence_dir=$EvidenceDir
    source_sha256=$sourceHashes; variants=$variants
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath "$EvidenceDir\verified-build.json" -Encoding utf8
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath "$BuildDir\verified-build.json" -Encoding utf8
$variants | ForEach-Object { [pscustomobject]$_ } |
    Select-Object variant,requested_register_limit,registers,stack_bytes,static_shared_bytes,local_bytes,user_kernel_count |
    Format-Table -AutoSize
Write-Host "CPU build/static evidence saved: $EvidenceDir"
