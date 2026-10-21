"""CPU-only audit of the isolated DBF register-cap harness and saved NCU data.

No binary, CUDA, profiler, or shell command is executed. Source comparisons use
explicit regions, rather than claiming that the two complete programs match.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import io
import json
from pathlib import Path
import re
import sys

from analyze_vitamin_evidence import profile_ranges, save_csv


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def region(text: str, first: str, end: str) -> str:
    if text.count(first) != 1:
        raise ValueError(f"Start anchor is not unique: {first!r}")
    start = text.index(first)
    stop = text.index(end, start + len(first))
    return text[start:stop]


# End markers are excluded. read_text normalizes CRLF to LF; no other text
# normalization is applied to the source regions.
REGIONS = {
    "seed_alpha_beta_tolerances": (
        "constexpr std::uint32_t Seed", "constexpr "),
    "input_hash_cpu_reference_and_error_functions": (
        "void cudaCheck(", "enum class Impl"),
    "buffers_events_blas_reference_helpers": (
        "struct DeviceBuffer", "void launch("),
    "blas_reporting_median_and_argument_helpers": (
        "void printBlas(", "int main("),
    "allocation_and_input_upload": (
        "    const size_t count = size_t(n)*n", "    BlasHandle "),
    "strict_reference_full_cpu_validation_reset": (
        '    printBlas("independent_reference"', "    auto run ="),
    "download_and_compare_lambda": (
        "    auto validate =", "    std::array<bool,"),
    "preflight": (
        "    upstreamPassed.fill(true);", "    Events events;"),
    "warmup_events_final_validation_and_output": (
        "      for (size_t slot = 0;", "    for (Impl impl : implementations) {\n      const auto& means"),
    "final_summary": (
        "    for (Impl impl : implementations) {\n      const auto& means", "    return 0;"),
}


def audit_sources(baseline: Path, candidate: Path, commands: Path, vendor: Path) -> dict:
    a = baseline.read_text(encoding="utf-8-sig")
    b = candidate.read_text(encoding="utf-8-sig")
    rows = []
    for label, (first, end) in REGIONS.items():
        # The three adjacent constant lines end before the different config.
        if label == "seed_alpha_beta_tolerances":
            left = "\n".join(a[a.index(first):].splitlines()[:3])
            right = "\n".join(b[b.index(first):].splitlines()[:3])
        else:
            left, right = region(a, first, end), region(b, first, end)
        rows.append({
            "region": label, "identical": left == right,
            "baseline_sha256_lf": digest(left.encode()),
            "candidate_sha256_lf": digest(right.encode()),
            "baseline_lines": len(left.splitlines()),
            "candidate_lines": len(right.splitlines()),
            "diff": list(difflib.unified_diff(left.splitlines(), right.splitlines()))[:30],
        })
    macro_lines = [
        {"line": number, "text": line}
        for number, line in enumerate(b.splitlines(), 1)
        if "DBF_MAX_REGISTERS" in line
    ]
    executable_macro_lines = [
        row for row in macro_lines
        if not row["text"].lstrip().startswith(("//", "#"))
    ]
    allowed_macro_uses = [
        'constexpr const char* Configuration = DBF_MAX_REGISTERS == 128 ? "dbf_maxr128" : "dbf_default";',
        "DBF_MAX_REGISTERS,attributes.numRegs,attributes.sharedSizeBytes,",
    ]
    macro_host_only = sorted(row["text"].strip() for row in executable_macro_lines) == sorted(allowed_macro_uses)
    vendor_has_macro = "DBF_MAX_REGISTERS" in vendor.read_text(encoding="utf-8-sig")
    command_rows = json.loads(commands.read_text(encoding="utf-8-sig"))
    if len(command_rows) != 2:
        raise ValueError("Expected exactly two isolated compile commands.")
    normalized_commands = []
    command_details = []
    for item in command_rows:
        command = item["command"]
        cap = "--maxrregcount=128" in command
        macro = re.findall(r"-DDBF_MAX_REGISTERS=(\d+)", command)
        if macro != (["128"] if cap else ["0"]):
            raise ValueError("Register limit and host label disagree.")
        if "--use_fast_math" in command or "compute_89" not in command or "sm_89" not in command:
            raise ValueError("Expected strict FP32 sm_89 compile configuration.")
        normalized = command.replace(" --maxrregcount=128", "")
        normalized = re.sub(r"-DDBF_MAX_REGISTERS=(?:0|128)", "-DDBF_MAX_REGISTERS=LABEL", normalized)
        normalized = re.sub(r"windows_dbf_(?:default|maxr128)_benchmark", "windows_dbf_VARIANT_benchmark", normalized)
        normalized_commands.append(normalized)
        command_details.append({"variant": "maxr128" if cap else "default", "command": command})
    commands_single_factor = normalized_commands[0] == normalized_commands[1]
    passed = all(row["identical"] for row in rows) and macro_host_only and not vendor_has_macro and commands_single_factor
    return {
        "status": "PASS" if passed else "FAIL",
        "source_regions": rows,
        "macro_uses": macro_lines,
        "macro_executable_uses_are_only_host_label_and_resource_printf": macro_host_only,
        "vendor_device_source_contains_cap_macro": vendor_has_macro,
        "compile_commands": command_details,
        "compile_commands_equal_after_removing_cap_host_label_and_target_output_names": commands_single_factor,
        "normalization": "Source regions: CRLF->LF only. Commands: remove --maxrregcount=128; normalize host DBF_MAX_REGISTERS label and target output names.",
        "scope": [
            "Source audit checks selected unchanged host regions; the complete harness programs differ.",
            "DBF device template remains <128,128,16,8,8> with 256 threads and n/128 by n/128 grid.",
            "Reference cuBLAS uses a distinct PEDANTIC handle in both programs.",
            "New harness runs only DBF; the external PowerShell script alternates binary processes.",
            "Resource/occupancy API queries occur before timing; no device body or template is changed by the host macro.",
            "The compiler register cap can change machine-code scheduling; this is not a pure occupancy-only experiment.",
        ],
    }


def analyze_profile(path: Path, output: Path) -> dict:
    if not (path.parent / "PROFILE_PASS.txt").is_file():
        raise ValueError("NCU profile is incomplete: missing PROFILE_PASS.txt.")
    text = path.read_text(encoding="utf-8-sig")
    raw = list(csv.DictReader(io.StringIO(text)))
    expected = {(str(n), mode, str(repeat)) for n in (2048, 4096)
                for mode in ("default", "maxr128") for repeat in (1, 2)}
    actual = {(r["size"], r["mode"], r["profile_repeat"]) for r in raw}
    if len(raw) != 8 or actual != expected:
        raise ValueError("Expected 2048/4096 x default/maxr128 x two profile repeats.")
    ranges = profile_ranges(text, "dbf_register_cap")
    save_csv(output / "profile-ranges.csv", ranges)
    by_key = {(r["size"], r["mode"], r["metric"]): r for r in ranges}
    pairs = []
    for n in (2048, 4096):
        for metric in sorted({r["metric"] for r in ranges}):
            base = by_key[(n, "default", metric)]
            cap = by_key[(n, "maxr128", metric)]
            pairs.append({
                "size": n, "metric": metric,
                "default_minimum": base["minimum"], "default_median": base["median"],
                "default_maximum": base["maximum"],
                "maxr128_minimum": cap["minimum"], "maxr128_median": cap["median"],
                "maxr128_maximum": cap["maximum"],
                "median_difference_cap_minus_default": cap["median"] - base["median"],
                "median_ratio_cap_over_default": cap["median"] / base["median"] if base["median"] else "",
            })
    save_csv(output / "profile-comparison.csv", pairs)
    return {
        "status": "PASS", "profile_launches": len(raw), "metric_range_rows": len(ranges),
        "profile_source": str(path.resolve()),
        "cautions": [
            "Two profiled launches per variant/size are mechanism evidence, not the unprofiled event performance table.",
            "Stalled-warp metrics count warps per SMSP active cycle; they are not elapsed-time percentages.",
            "clock-control=none and cache-control=none preserve state, which may vary across replay passes.",
            "Local sectors measure observed traffic; absence of spills also requires ptxas/resource/SASS evidence.",
            "Reported shared configuration is actual sampled launch metadata; do not infer it from architectural capacity.",
            "Register limits also change compiler scheduling, so occupancy alone cannot establish causation.",
        ],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=root / "scripts/windows_vitamin_benchmark.cu")
    parser.add_argument("--candidate", type=Path, default=root / "scripts/windows_dbf_register_benchmark.cu")
    parser.add_argument("--compile-commands", type=Path, default=root / "build-windows-dbf-register-sm89/compile_commands.json")
    parser.add_argument("--vendor", type=Path, default=root / "src/kernels/vendor/vitamin_sgemm.cuh")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = audit_sources(args.baseline, args.candidate, args.compile_commands, args.vendor)
    (args.output_dir / "source-consistency.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    inputs = [args.baseline, args.candidate, args.compile_commands, args.vendor, Path(__file__)]
    if args.profile:
        profile_report = analyze_profile(args.profile, args.output_dir)
        (args.output_dir / "profile-audit.json").write_text(json.dumps(profile_report, indent=2) + "\n", encoding="utf-8")
        inputs.extend([args.profile, args.profile.parent / "PROFILE_PASS.txt"])
    manifest = {
        "argv": sys.argv,
        "files": [{"path": str(path.resolve()), "sha256": digest(path.read_bytes())} for path in inputs],
        "cpu_only": True, "source_audit_status": report["status"],
    }
    (args.output_dir / "analysis-provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit("Source consistency audit failed; see saved evidence.")
    print("PASS: CPU source audit" + (" and eight saved NCU launches" if args.profile else ""))


if __name__ == "__main__":
    main()
