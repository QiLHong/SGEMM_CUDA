#requires -Version 7.0
[CmdletBinding()]
param([string]$BuildDir='build-windows-cutlass-4090d-sm89',
      [string]$CutlassDir='..\deps\cutlass-v3.5.1',
      [Parameter(Mandatory)][string]$EvidenceDir,
      [string]$Python='D:\Anaconda\Anaconda\python.exe')
$ErrorActionPreference='Stop'
& "$PSScriptRoot\Initialize-Toolchain.ps1"
$repo=(Resolve-Path "$PSScriptRoot\..\..").Path
$source=Join-Path $repo 'experiments\cutlass-rtx4090d'
foreach($name in @('BuildDir','EvidenceDir','CutlassDir')) {
    $value=Get-Variable $name -ValueOnly
    if(![IO.Path]::IsPathRooted($value)){Set-Variable $name (Join-Path $repo $value)}
}
$CutlassDir=(Resolve-Path -LiteralPath $CutlassDir).Path
$expected='f7b19de32c5d1f3cedfc735c2849f12b537522ee'
if((& git -C $CutlassDir rev-parse HEAD) -ne $expected){throw 'Expected pinned official CUTLASS v3.5.1'}
if((& git -C $CutlassDir status --porcelain)){throw 'CUTLASS dependency must be clean'}
if(Test-Path -LiteralPath $EvidenceDir){throw "Refusing overwrite: $EvidenceDir"}
if(Test-Path -LiteralPath "$BuildDir\CMakeCache.txt"){
    $homeLine=@(Get-Content "$BuildDir\CMakeCache.txt" | Where-Object {$_ -like 'CMAKE_HOME_DIRECTORY:INTERNAL=*'})
    if($homeLine.Count -ne 1 -or [IO.Path]::GetFullPath(($homeLine[0]-split '=',2)[1]) -ne [IO.Path]::GetFullPath($source)){throw 'Build directory belongs to another project'}
}
& $Python "$repo\scripts\extract_vitamin_tensor.py" --verify
if($LASTEXITCODE){throw 'Upstream extraction verification failed'}
& $Python "$repo\scripts\prepare_tensor_bn64.py" --verify
if($LASTEXITCODE){throw 'BN64 derivation verification failed'}
New-Item -ItemType Directory -Path $EvidenceDir | Out-Null
& cmake -S $source -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TRY_COMPILE_CONFIGURATION=Release -DCMAKE_CUDA_ARCHITECTURES=89 `
  "-DCUTLASS_ROOT=$CutlassDir" "-DCMAKE_CUDA_COMPILER=$env:CUDACXX" `
  "-DCMAKE_CUDA_HOST_COMPILER=$env:CUDAHOSTCXX" "-DCMAKE_CXX_COMPILER=$env:CUDAHOSTCXX" `
  "-DCUDAToolkit_ROOT=$env:CUDA_PATH" 2>&1 | Tee-Object "$EvidenceDir\build.log"
if($LASTEXITCODE){throw 'CMake configure failed; log retained'}
& cmake --build $BuildDir --parallel 2 --verbose 2>&1 | Tee-Object "$EvidenceDir\build.log" -Append
if($LASTEXITCODE){throw 'Build failed; log retained'}
$commands=@(Get-Content "$BuildDir\compile_commands.json" -Raw | ConvertFrom-Json)
if($commands.Count -ne 1 -or $commands[0].command -notmatch 'arch=compute_89,code=\[compute_89,sm_89\]' -or
  $commands[0].command -notmatch '(?:^| )-O3(?: |$)' -or $commands[0].command -match 'use_fast_math|allow-unsupported-compiler'){throw 'Unexpected compilation flags'}
$exe=Join-Path $BuildDir 'windows_cutlass_4090d_benchmark.exe'
foreach($dump in @(@{flag='--dump-sass';name='binary.sass'},@{flag='--dump-ptx';name='binary.ptx'},@{flag='--dump-resource-usage';name='resources.txt'})){
    & "$env:CUDA_PATH\bin\cuobjdump.exe" $dump.flag $exe | Set-Content "$EvidenceDir\$($dump.name)" -Encoding utf8
    if($LASTEXITCODE){throw 'cuobjdump failed'}
}
$sass=Get-Content "$EvidenceDir\binary.sass" -Raw
if($sass -notmatch 'HMMA' -or $sass -notmatch 'FFMA'){throw 'Expected TensorOp and SIMT instructions'}
Copy-Item -LiteralPath "$BuildDir\compile_commands.json","$BuildDir\CMakeCache.txt",$exe -Destination $EvidenceDir
Copy-Item -LiteralPath $source -Destination "$EvidenceDir\source-as-built" -Recurse
Copy-Item -LiteralPath $PSCommandPath,"$PSScriptRoot\Initialize-Toolchain.ps1" -Destination "$EvidenceDir\source-as-built"
$dependencyDirs=@($source,"$repo\experiments\tensorcore","$repo\experiments\tensorcore-grid","$repo\experiments\tensorcore-ktile","$repo\experiments\tensorcore-bn64","$repo\src\kernels","$CutlassDir\include")
$dependencies=Get-ChildItem $dependencyDirs -Recurse -File | ForEach-Object {[ordered]@{path=$_.FullName;sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}}
$dependencies | ConvertTo-Json | Set-Content "$BuildDir\dependencies.json"
Copy-Item -LiteralPath "$BuildDir\dependencies.json" -Destination $EvidenceDir
# Preserve exact pinned headers and license without changing the external checkout.
Compress-Archive -LiteralPath "$CutlassDir\include","$CutlassDir\LICENSE.txt" -DestinationPath "$EvidenceDir\cutlass-headers-source.zip"
$snapshot=Join-Path $EvidenceDir 'project-dependencies'
New-Item -ItemType Directory -Path $snapshot | Out-Null
foreach($dir in @('tensorcore','tensorcore-grid','tensorcore-ktile','tensorcore-bn64')){Copy-Item -LiteralPath "$repo\experiments\$dir" -Destination $snapshot -Recurse}
Copy-Item -LiteralPath "$repo\src\kernels" -Destination $snapshot -Recurse
$manifest=[ordered]@{
 utc=[DateTimeOffset]::UtcNow.ToString('o');compile_command=$commands[0].command
 binary_sha256=(Get-FileHash -LiteralPath $exe).Hash;source_sha256=(Get-FileHash "$source\benchmark.cu").Hash
 architecture='sm_89';fast_math=$false;cutlass_commit=$expected;cutlass_dir=$CutlassDir
 cutlass_headers_zip_sha256=(Get-FileHash "$EvidenceDir\cutlass-headers-source.zip").Hash
 nvcc=((& nvcc --version)-join "`n");host_compiler=$env:CUDAHOSTCXX;sgemm_commit=(& git -C $repo rev-parse HEAD)
 evidence_dir=$EvidenceDir
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content "$EvidenceDir\verified-build.json"
$manifest | ConvertTo-Json -Depth 4 | Set-Content "$BuildDir\verified-build.json"
& $exe --list | Set-Content "$EvidenceDir\configurations.txt"
if($LASTEXITCODE){throw 'Configuration listing failed'}
Write-Host "Built CUTLASS comparison and saved evidence: $EvidenceDir"
