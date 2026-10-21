#requires -Version 7.0
[CmdletBinding()]
param([Parameter(Mandatory)][string]$ResultDir,
      [string]$OutputDir, [string]$Python = 'D:\Anaconda\Anaconda\python.exe')
$ErrorActionPreference = 'Stop'
$arguments = @("$PSScriptRoot\..\summarize_vitamin_results.py",'--result-dir',$ResultDir)
if ($OutputDir) { $arguments += @('--output-dir',$OutputDir) }
& $Python @arguments
if ($LASTEXITCODE) { throw 'Raw-event/statistical audit failed; output retained for inspection.' }
