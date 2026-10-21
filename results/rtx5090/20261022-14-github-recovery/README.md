# 离机恢复演练

固定 GitHub 提交：`f14dc4aadb7d5d0d79f07fb2fc58a29a697314a0`。

在 Windows 本机新目录从 GitHub HTTPS 回读统计脚本、原始 event CSV、独立汇总、批次标记与两份所属检查点 SHA256 清单，共6文件。全部回读字节匹配该提交 Git blob；4个有效载荷均匹配所属清单 SHA256。使用现有 Python 执行回读脚本：1050个event、35个实现/轮组、每组30个event，重新计算的 JSON 对象与已上传独立汇总完全相同。

`recovery.json` 为完整证据；`commit-proof.json` 核对固定提交和 tree；`recomputed-summary.json` 是恢复后的统计。未使用GPU，未读取服务器工作文件，未修改GitHub或执行提交推送。

恢复验证成功只代表备份字节与统计可恢复。原批次 `exit_code=2`、`numeric_acceptance_pass=false` 保持原样，未转换为正确性通过。

初次恢复工具对不同历史清单中持续更新的发布日志错误要求哈希一致，已修正为按每个有效载荷所属检查点验证；第一次目录保留，第二次重新下载，不修改原数据。见 `attempt-history.json`。
