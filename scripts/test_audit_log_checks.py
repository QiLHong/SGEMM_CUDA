#!/usr/bin/env python3
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
