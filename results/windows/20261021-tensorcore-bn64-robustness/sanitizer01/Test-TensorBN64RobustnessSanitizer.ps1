#requires -Version 7.0
[CmdletBinding()]
param([string]$BuildDir='build-windows-tensor-bn64-robustness-sm89',
 [Parameter(Mandatory)][string]$OutputDir,[int]$TimeoutSeconds=60,
 [string]$SanitizerPath='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\compute-sanitizer\compute-sanitizer.exe')
$ErrorActionPreference='Stop'
& "$PSScriptRoot\Initialize-Toolchain.ps1"
$repo=(Resolve-Path "$PSScriptRoot\..\..").Path
if (![IO.Path]::IsPathRooted($BuildDir)) {$BuildDir=Join-Path $repo $BuildDir}
if (![IO.Path]::IsPathRooted($OutputDir)) {$OutputDir=Join-Path $repo $OutputDir}
if(Test-Path -LiteralPath $OutputDir){throw 'Refusing overwrite'}
New-Item -ItemType Directory -Path $OutputDir | Out-Null
Copy-Item -LiteralPath $PSCommandPath -Destination $OutputDir
$exe=Join-Path $BuildDir 'windows_tensor_bn64_robustness_benchmark.exe'
$manifest=Get-Content "$BuildDir\verified-build.json" -Raw | ConvertFrom-Json
if($manifest.binary_sha256 -ne (Get-FileHash $exe).Hash){throw 'Binary mismatch'}
& $SanitizerPath --version | Set-Content "$OutputDir\version.txt"
# One legal case; stop the whole sanitizer item on attachment/error/timeout.
$arguments=@('--tool','memcheck','--launch-timeout','20','--target-processes','application-only',
 '--error-exitcode','99','--log-file',"$OutputDir\memcheck.log",$exe,'--case',
 '128','1024','32','20261020','1','1','1',"$OutputDir\small.csv")
$record=[ordered]@{mode='memcheck';shape='128x1024x32';seed=20261020;arguments=$arguments;
 start_utc=[DateTimeOffset]::UtcNow.ToString('o');timeout_seconds=$TimeoutSeconds}
$p=Start-Process -FilePath $SanitizerPath -ArgumentList ($arguments | ForEach-Object {'"'+$_+'"'}) -WindowStyle Hidden -PassThru `
 -RedirectStandardOutput "$OutputDir\stdout.log" -RedirectStandardError "$OutputDir\stderr.log"
if(!$p.WaitForExit($TimeoutSeconds*1000)){
 & taskkill.exe /PID $p.Id /T /F | Out-File "$OutputDir\termination.log"
 $record['status']='TIMEOUT_UNVERIFIED'
}else{
 $record['exit_code']=$p.ExitCode
 $text=if(Test-Path "$OutputDir\memcheck.log"){Get-Content "$OutputDir\memcheck.log" -Raw}else{''}
 $record['status']=if($p.ExitCode -eq 0 -and $text -match 'ERROR SUMMARY: 0 errors'){'PASS_MEMCHECK_ONLY'}else{'FAILED_OR_UNVERIFIED'}
}
$record['pid']=$p.Id;$record['end_utc']=[DateTimeOffset]::UtcNow.ToString('o')
$record | ConvertTo-Json -Depth 5 | Set-Content "$OutputDir\result.json"
Write-Host $record.status
