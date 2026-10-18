#!/usr/bin/env python3
# 【文件导读】验证 Sanitizer 日志判定器拒绝伪 PASS
# 给初学者：这些小检查可以帮助理解索引或测试规则；不需要因此先跑一次大型 GPU 性能实验。
# 运行：python3 scripts/test_audit_log_checks.py；Python 标准库即可，不启动 GPU。
# 构造含 0/10 errors、race warnings、缺失或重复 summary 的日志，保护 audit_log_checks 的严格判定。
# 输入是文件内固定的教学数据/有限枚举范围；失败以断言或 unittest 非零退出体现。
# 结果打印终端，不生成 benchmark CSV；模型通过不等于设备 FP32、异步调度或吞吐验证。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

"""CPU-only regression tests: never convert sanitizer warnings into PASS."""
import unittest
from audit_log_checks import sanitizer_passed


class SanitizerLogTests(unittest.TestCase):
    result = "RESULT,control,128,3,0.1,1,0,PASS\n"

    def test_clean_memory_checks(self):
        for tool in ("memcheck", "synccheck"):
            self.assertTrue(sanitizer_passed(
                self.result + "========= ERROR SUMMARY: 0 errors\n", tool))

    def test_nonzero_memory_checks(self):
        for errors in (1, 10, 100):
            self.assertFalse(sanitizer_passed(
                self.result + f"========= ERROR SUMMARY: {errors} errors\n", "memcheck"))

    def test_race_summary(self):
        for hazards, errors, warnings in [(0, 0, 0), (1, 0, 1), (10, 10, 0)]:
            text = self.result + (f"========= RACECHECK SUMMARY: {hazards} hazards "
                                  f"displayed ({errors} errors, {warnings} warnings)\n")
            self.assertEqual(sanitizer_passed(text, "racecheck"), hazards == 0)

    def test_missing_or_duplicate_summary(self):
        summary = "========= ERROR SUMMARY: 0 errors\n"
        self.assertFalse(sanitizer_passed(self.result, "memcheck"))
        self.assertFalse(sanitizer_passed(summary, "memcheck"))
        self.assertFalse(sanitizer_passed(self.result + summary * 2, "memcheck"))
        self.assertFalse(sanitizer_passed(self.result + summary, "racecheck"))


if __name__ == "__main__":
    unittest.main()
