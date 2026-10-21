$archive = 'results\windows\20261021-tensorcore-bn64-robustness'
$check = "results\windows\bn64-stats-check-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
& $py scripts\summarize_tensor_bn64_robustness.py --run "$archive\run" --output "$check\analysis"
& $py scripts\verify_tensor_bn64_robustness.py --run "$archive\run" --analysis "$check\analysis" --output "$check\verification"
& $py scripts\audit_tensor_bn64_robustness_evidence.py --archive $archive --build "$archive\build01" --profile "$archive\profile01" --output "$check\evidence"
