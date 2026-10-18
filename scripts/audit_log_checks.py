# 【文件导读】研究脚本共用的 Sanitizer 日志判定器
# 给初学者：正确性检查是在回答“算对了吗”；它与“跑得快吗”是两件事，PASS 也只覆盖实际测试过的情况。
# 用法：由 run_k30_audit.py、summarize_k31/k32_audit.py 导入 sanitizer_passed(content, tool)。
# 输入：完整日志文本与 memcheck/racecheck/synccheck 工具名；输出布尔值，不读写文件。
# 先要求 RESULT 行数值 PASS；再要求恰好一个工具对应的零错误汇总，racecheck 还要求零 warning/hazard。
# 用完整行匹配避免把 10 errors 当成 0 errors；未知工具名抛出异常。
# 回归入口：python3 scripts/test_audit_log_checks.py；本模块自身不运行 Compute Sanitizer。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

"""Strict Compute Sanitizer log checks shared by experiment drivers/reports."""
import re


# 不要只搜索 '0 errors' 子串，因为 '10 errors' 也包含它。
# 同时要求正确格式的 RESULT/PASS 与工具明确的零错误摘要；缺日志不能当作检查成功。
def sanitizer_passed(content, tool):
    """Require a numerical PASS and exactly one tool-specific zero summary.

    Substring checks for '0 errors' also match '10 errors', and race warnings
    can coexist with zero errors. Neither is a clean sanitizer result.
    """
    if not re.search(r"^RESULT,[^\n]*,PASS\s*$", content, re.MULTILINE):
        return False
    if tool == "racecheck":
        summaries = re.findall(
            r"^=+\s+RACECHECK SUMMARY:\s+(\d+) hazards? displayed "
            r"\((\d+) errors?, (\d+) warnings?\)\s*$", content, re.MULTILINE)
        return summaries == [("0", "0", "0")]
    if tool not in ("memcheck", "synccheck"):
        raise ValueError(f"Unsupported sanitizer: {tool}")
    summaries = re.findall(r"^=+\s+ERROR SUMMARY:\s+(\d+) errors?\s*$",
                           content, re.MULTILINE)
    return summaries == ["0"]
