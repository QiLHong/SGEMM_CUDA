Set-Location 'D:\CUDA-Lab\cuda-learning\SGEMM_CUDA'
$ps7 = 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe'
$py = 'D:\Anaconda\Anaconda\python.exe'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "results\windows\tensorcore-bn64-robustness-$stamp"
$build = "build-windows-tensor-bn64-robustness-$stamp"
& $ps7 -NoProfile -File scripts\windows\Build-TensorBN64Robustness.ps1 -BuildDir $build -EvidenceDir "$out\build01"
if ($LASTEXITCODE) { throw 'build failed' }
& $ps7 -NoProfile -File scripts\windows\Run-TensorBN64Robustness.ps1 -BuildDir $build -OutputDir "$out\run" -PrepareOnly
if ($LASTEXITCODE) { throw 'plan failed' }
& scripts\windows\Initialize-Toolchain.ps1
& ".\$build\windows_tensor_bn64_robustness_benchmark.exe" --smoke "$out\smoke"
if ($LASTEXITCODE) { throw 'smoke failed' }
& $ps7 -NoProfile -File scripts\windows\Test-TensorBN64RobustnessSanitizer.ps1 -BuildDir $build -OutputDir "$out\sanitizer01"
# 阅读 result.json；附加超时只保留未验证，不无限等待或绕过权限。
& $ps7 -NoProfile -File scripts\windows\Run-TensorBN64Robustness.ps1 -BuildDir $build -OutputDir "$out\run" -ExecutePrepared -SmokeEvidenceDir "$out\smoke"
if ($LASTEXITCODE) { throw 'formal run failed; raw files retained' }
& $py scripts\summarize_tensor_bn64_robustness.py --run "$out\run" --output "$out\analysis"
if ($LASTEXITCODE) { throw 'summary failed' }
& $py scripts\verify_tensor_bn64_robustness.py --run "$out\run" --analysis "$out\analysis" --output "$out\verification"
if ($LASTEXITCODE) { throw 'independent verification failed' }
& $ps7 -NoProfile -File scripts\windows\Profile-TensorBN64Robustness.ps1 -BuildDir $build -OutputDir "$out\profile01"
if ($LASTEXITCODE) { throw 'profile failed; evidence retained' }
