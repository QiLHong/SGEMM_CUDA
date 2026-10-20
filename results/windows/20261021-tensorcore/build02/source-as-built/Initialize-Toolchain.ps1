# 只修改当前 PowerShell 进程的环境，不写系统、用户或 Codex 配置。
[CmdletBinding()]
param([string]$CudaPath = $env:CUDA_PATH, [string]$MsvcVersion = '14.37')
$ErrorActionPreference = 'Stop'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (!(Test-Path -LiteralPath $vswhere)) { throw 'Missing vswhere. Install the VS C++ build tools first.' }
$vsPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vsPath) { throw 'No Visual Studio installation with MSVC x64 tools found.' }
$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvarsall.bat'
$toolset = Get-ChildItem -LiteralPath (Join-Path $vsPath 'VC\Tools\MSVC') -Directory |
    Where-Object Name -Like "$MsvcVersion*" | Sort-Object Name -Descending | Select-Object -First 1
if (!$toolset) { throw "Missing MSVC $MsvcVersion. CUDA 12.4 requires a supported 19.3x compiler." }
if (!$CudaPath) { throw 'CUDA_PATH is missing. Pass -CudaPath pointing to the Windows Toolkit.' }
$nvccPath = Join-Path $CudaPath 'bin\nvcc.exe'
if (!(Test-Path -LiteralPath $nvccPath)) { throw "nvcc not found: $nvccPath" }
$nvccVersion = (& $nvccPath --version) -join "`n"
if ($nvccVersion -notmatch 'release 12\.4,' -or $toolset.Name -notmatch '^14\.3[0-9]\.') {
    throw 'This script validates CUDA 12.4 + MSVC 19.3x only. Review the NVIDIA compatibility table before using another pair.'
}
# vcvarsall 在 cmd 子进程初始化，然后把环境导回本进程。路径始终有引号。
$environmentLines = & $env:ComSpec /d /s /c "`"$vcvars`" x64 -vcvars_ver=$MsvcVersion >nul && set"
if ($LASTEXITCODE -ne 0) { throw "vcvarsall failed ($LASTEXITCODE)" }
foreach ($line in $environmentLines) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}
$cmakeRoot = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake'
$cmakePath = Join-Path $cmakeRoot 'CMake\bin'
$ninjaPath = Join-Path $cmakeRoot 'Ninja'
if (!(Test-Path (Join-Path $cmakePath 'cmake.exe')) -or !(Test-Path (Join-Path $ninjaPath 'ninja.exe'))) {
    throw 'Missing VS CMake tools component (CMake and Ninja).'
}
$env:PATH = "$CudaPath\bin;$cmakePath;$ninjaPath;$env:PATH"
$env:CUDA_PATH = $CudaPath
$env:CUDACXX = $nvccPath
$env:CUDAHOSTCXX = Join-Path $toolset.FullName 'bin\Hostx64\x64\cl.exe'
if ((Get-Command cl.exe).Source -ne $env:CUDAHOSTCXX) { throw 'Selected cl.exe does not match the requested x64 toolset.' }
Write-Host "CUDA: $nvccPath"
Write-Host "MSVC: $env:CUDAHOSTCXX"
Write-Host "Windows SDK: $env:WindowsSdkDir $env:WindowsSDKVersion"
