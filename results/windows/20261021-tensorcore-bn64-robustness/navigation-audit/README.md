# 导航与 CPU 索引核对

## 执行范围

本轮实际检查根 README、中文阅读路线、Windows 结果索引及第 26/27 章的本地 Markdown 链接。
外部 GitHub URL 没有联网请求，不能把本地核对写成远端网页全部验证。

## 修改前发现

- 221 个本地路径均存在；已核对的标题锚点均存在，未发现需要改写的本地路径。
- 其中 99 个链接指向未纳入 Git 的目标。本地已有第 24～27 章、SIMT stage3/followup
  记录和配套脚本，但这些文件未随起点 a54435d 发布。
- 因此修复方式是补充“本地已有、尚未发布”的准确可用性说明，保留真实路径；没有新增替代空壳，
  没有虚构缺失成绩，也没有自动提交发布。
- 根 README 的 57.998 TFLOPS 明确改为第 31 章历史单 seed 确认值；新增第 33 章及独立归档导航。

## 重新执行

在仓库根目录执行：

```powershell
python results/windows/20261021-tensorcore-bn64-robustness/navigation-audit/audit_grid.py
python results/windows/20261021-tensorcore-bn64-robustness/navigation-audit/audit_navigation.py
```

`audit_grid.py` 已运行通过：5760 个 CTA 映射、24576 个 CTA 内输出 fragment；四个正式尺寸与两个实现的
坐标覆盖、重复、遗漏、边界以及 16B 全局 copy 地址分量检查均通过。3072 单独为原版 576 CTA、BN64 1152 CTA。
这是一份 CPU 坐标和地址证明，不是 GPU race/synchronization 检查。

`audit_navigation.py` 已在第 33 章和归档 README 实际生成后执行并通过：229 个本地链接无缺失路径，
2 个标题锚点均存在。107 个链接目标尚未纳入 Git；29 个外部 URL 明确未联网检查。
结果见 `summary.json`、`links-after.json` 和实际读取的五个导航文件 SHA256。
路径存在检查不表示读取并核验了每个目标文件的内容。

`historical-evidence.json` 摘录旧构建和旧 NCU，只供与本轮新结果比较，不能填入本轮性能表。
