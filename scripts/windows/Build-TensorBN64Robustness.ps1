#requires -Version 7.0
[CmdletBinding()]
param([string]$BuildDir='build-windows-tensor-bn64-robustness-sm89',[string]$EvidenceDir,
      [string]$Python='D:\Anaconda\Anaconda\python.exe')
$ErrorActionPreference='Stop'
& "$PSScriptRoot\Initialize-Toolchain.ps1"
$repo=(Resolve-Path "$PSScriptRoot\..\..").Path
$source=Join-Path $repo 'experiments\tensorcore-bn64-robustness'
if (![IO.Path]::IsPathRooted($BuildDir)) { $BuildDir=Join-Path $repo $BuildDir }
if (!$EvidenceDir) { $EvidenceDir="results\windows\20261021-tensorcore-bn64-robustness\build-$(Get-Date -Format yyyyMMdd-HHmmss)" }
if (![IO.Path]::IsPathRooted($EvidenceDir)) { $EvidenceDir=Join-Path $repo $EvidenceDir }
if (Test-Path -LiteralPath $EvidenceDir) { throw "Refusing overwrite: $EvidenceDir" }
if (Test-Path -LiteralPath "$BuildDir\CMakeCache.txt") {
    $line=@(Get-Content "$BuildDir\CMakeCache.txt" | Where-Object { $_ -like 'CMAKE_HOME_DIRECTORY:INTERNAL=*' })
    if ($line.Count -ne 1 -or [IO.Path]::GetFullPath(($line[0] -split '=',2)[1]) -ne [IO.Path]::GetFullPath($source)) {
        throw 'Build directory belongs to another project'
    }
}
& $Python "$repo\scripts\extract_vitamin_tensor.py" --verify
if ($LASTEXITCODE) { throw 'Device extraction differs from upstream' }
& $Python "$repo\scripts\prepare_tensor_bn64.py" --verify
if ($LASTEXITCODE) { throw "BN64 derivation changed" }
New-Item -ItemType Directory -Path $EvidenceDir | Out-Null
& cmake -S $source -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_TRY_COMPILE_CONFIGURATION=Release -DCMAKE_CUDA_ARCHITECTURES=89 `
    "-DCMAKE_CUDA_COMPILER=$env:CUDACXX" "-DCMAKE_CUDA_HOST_COMPILER=$env:CUDAHOSTCXX" `
    "-DCMAKE_CXX_COMPILER=$env:CUDAHOSTCXX" "-DCUDAToolkit_ROOT=$env:CUDA_PATH" `
    2>&1 | Tee-Object "$EvidenceDir\build.log"
if ($LASTEXITCODE) { throw 'CMake configure failed' }
& cmake --build $BuildDir --parallel 2 --verbose 2>&1 | Tee-Object "$EvidenceDir\build.log" -Append
if ($LASTEXITCODE) { throw 'Build failed; evidence retained' }
$commands=@(Get-Content "$BuildDir\compile_commands.json" -Raw | ConvertFrom-Json)
if ((Get-Content "$BuildDir\CMakeCache.txt" -Raw) -notmatch 'CMAKE_BUILD_TYPE:STRING=Release') { throw 'Release configuration required' }
if ($commands.Count -ne 1 -or $commands[0].command -notmatch 'arch=compute_89,code=\[compute_89,sm_89\]' -or
    $commands[0].command -notmatch '(?:^| )-O3(?: |$)' -or $commands[0].command -match 'use_fast_math|allow-unsupported-compiler') { throw 'Unexpected compilation flags' }
$exe=Join-Path $BuildDir 'windows_tensor_bn64_robustness_benchmark.exe'
foreach ($dump in @(@{flag='--dump-sass';name='binary.sass'},@{flag='--dump-ptx';name='binary.ptx'},@{flag='--dump-resource-usage';name='resources.txt'})) {
    & "$env:CUDA_PATH\bin\cuobjdump.exe" $dump.flag $exe | Set-Content "$EvidenceDir\$($dump.name)" -Encoding utf8
    if ($LASTEXITCODE) { throw 'cuobjdump failed' }
}
$sass=Get-Content "$EvidenceDir\binary.sass" -Raw
if ($sass -notmatch 'HMMA' -or $sass -notmatch 'LDGSTS') { throw 'Missing Tensor Core / async copy machine code' }
Copy-Item -LiteralPath "$BuildDir\compile_commands.json","$BuildDir\CMakeCache.txt",$exe -Destination $EvidenceDir
Copy-Item -LiteralPath $source -Destination "$EvidenceDir\source-as-built" -Recurse
Copy-Item -LiteralPath "$repo\scripts\extract_vitamin_tensor.py","$repo\scripts\prepare_tensor_bn64.py",$PSCommandPath,"$PSScriptRoot\Initialize-Toolchain.ps1" -Destination "$EvidenceDir\source-as-built"
$dependencies=Get-ChildItem $source,"$repo\experiments\tensorcore","$repo\experiments\tensorcore-grid","$repo\experiments\tensorcore-ktile","$repo\experiments\tensorcore-bn64" -File -Recurse | ForEach-Object { [ordered]@{path=$_.FullName;sha256=(Get-FileHash $_.FullName -Algorithm SHA256).Hash} }
$dependencies | ConvertTo-Json | Set-Content "$BuildDir\dependencies.json"
Copy-Item "$BuildDir\dependencies.json" $EvidenceDir
Copy-Item "$repo\experiments\tensorcore" "$EvidenceDir\baseline-source" -Recurse
Copy-Item "$repo\experiments\tensorcore-grid" "$EvidenceDir\grid-harness-source" -Recurse
Copy-Item "$repo\experiments\tensorcore-ktile" "$EvidenceDir\ktile-source" -Recurse
Copy-Item "$repo\experiments\tensorcore-bn64" "$EvidenceDir\bn64-source" -Recurse
$manifest=[ordered]@{
    utc=[DateTimeOffset]::UtcNow.ToString('o');compile_command=$commands[0].command
    binary_sha256=(Get-FileHash $exe -Algorithm SHA256).Hash
    source_sha256=(Get-FileHash "$source\benchmark.cu" -Algorithm SHA256).Hash
    architecture='sm_89';fast_math=$false;upstream_commit=(& git -C "$repo\..\vitamin-cuda" rev-parse HEAD)
    nvcc=((& nvcc --version)-join "`n");host_compiler=$env:CUDAHOSTCXX
    evidence_dir=$EvidenceDir
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content "$EvidenceDir\verified-build.json" -Encoding utf8
$manifest | ConvertTo-Json -Depth 4 | Set-Content "$BuildDir\verified-build.json" -Encoding utf8
Write-Host "Built and verified Tensor Core instructions: $EvidenceDir"

