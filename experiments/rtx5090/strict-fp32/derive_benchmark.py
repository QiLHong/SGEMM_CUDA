#!/usr/bin/env python3
"""Derive host-only RTX5090 audit from the frozen portable FP32 benchmark."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

BASE_COMMIT = "a2b23dabadc79440bf9cbf63f0b566f7eec429ba"
SOURCE = "scripts/windows_benchmark.cu"
EXPECTED = "c65c79ccaef024e445fc4f0b6623eac330dd3766d81b581580296f97246196f9"
HEADERS = {
    "src/kernels/10_kernel_warptiling.cuh": "233313c753ba1efc0599ae4a570059f1a515b0ab1040b62df03e81a2af53f9b3",
    "src/kernels/30_kernel_mini_simt.cuh": "bed81ab2f52df0061af3c7e814ff43b66c851c634cfdb4fed5f22031ecabc4c4",
}
HEADERS_GIT_BLOB = {
    "src/kernels/10_kernel_warptiling.cuh": "053eb62df551563168ae6f644b87c4daea45535737bee1d0de9dcfa43be23d13",
    "src/kernels/30_kernel_mini_simt.cuh": "e437badfd59c602ec3c8d4985c91bcfb30f0bc873eabd31f4027acf0d89499ac",
}


def derive(source):
    s = source
    def replace(old, new, count=1):
        nonlocal s
        found = s.count(old)
        if found != count:
            raise RuntimeError(f"Source anchor count {found}, expected {count}: {old[:80]!r}")
        s = s.replace(old, new)

    replace('#include "../src/kernels/', '#include "../../../src/kernels/', 2)
    replace('constexpr float Alpha = 0.5f;', 'float Alpha = 0.5f; // Host-only --alpha override; historical default is unchanged.')
    replace('if (argc != 7 && argc != 8)', 'if (argc < 7)')
    replace('[all|sync|k10|k10_pad8|k30_v1|k30_v2|strict]', '[all|sync|paired|k10|k10_pad8|k30_v1|k30_v2|strict] [--alpha 0.5|1]', 2)
    replace('    const std::string selection = argc >= 8 ? argv[7] : "all";', '''    std::string selection = "all";
    int nextArg = 7;
    if (nextArg < argc && std::string(argv[nextArg]).rfind("--",0) != 0)
      selection = argv[nextArg++];
    bool alphaExplicit = false;
    while (nextArg < argc) {
      const std::string option = argv[nextArg++];
      if (option != "--alpha" || nextArg >= argc || alphaExplicit)
        throw std::runtime_error("Expected a single trailing --alpha 0.5|1 option.");
      const std::string value = argv[nextArg++];
      size_t end = 0;
      const float parsed = std::stof(value,&end);
      if (end != value.size() || !(parsed == 0.5f || parsed == 1.0f))
        throw std::runtime_error("--alpha must be exactly 0.5 or 1.");
      Alpha = parsed;
      alphaExplicit = true;
    }''')
    replace('    const bool sync = selection == "sync";', '    const bool sync = selection == "sync";\n    const bool paired = selection == "paired";')
    replace('const Impl selected = (all || sync)', 'const Impl selected = (all || sync || paired)')
    replace('        : sync ? std::vector<Impl>{Impl::K10,Impl::K30V1,Impl::K30,Impl::Strict}\n               : std::vector<Impl>{selected};', '''        : sync ? std::vector<Impl>{Impl::K10,Impl::K30V1,Impl::K30,Impl::Strict}
        : paired ? std::vector<Impl>{Impl::K10,Impl::Pad8,Impl::K30V1,Impl::K30,Impl::Strict}
                 : std::vector<Impl>{selected};''')
    replace('if (sync || selected == Impl::K30V1)', 'if (sync || paired || selected == Impl::K30V1)')

    # Move the original all/sync tables without changing their contents. New
    # paired table alternates both pairs and rotates cuBLAS first/middle/last.
    start = s.index('    // 每4轮覆盖 AB/CD')
    stop = s.index('    Events events;', start)
    old_tables = s[start:stop]
    s = s[:start] + s[stop:]
    new_tables = old_tables + '''    const std::array<std::array<Impl,5>,7> pairedOrders = {{
      {{Impl::K10,Impl::Pad8,Impl::K30V1,Impl::K30,Impl::Strict}},
      {{Impl::Strict,Impl::K30,Impl::K30V1,Impl::Pad8,Impl::K10}},
      {{Impl::K30V1,Impl::K30,Impl::Strict,Impl::K10,Impl::Pad8}},
      {{Impl::Pad8,Impl::K10,Impl::Strict,Impl::K30,Impl::K30V1}},
      {{Impl::Strict,Impl::K10,Impl::Pad8,Impl::K30V1,Impl::K30}},
      {{Impl::K30,Impl::K30V1,Impl::Pad8,Impl::K10,Impl::Strict}},
      {{Impl::K10,Impl::Pad8,Impl::Strict,Impl::K30,Impl::K30V1}}
    }};
    std::vector<std::vector<Impl>> plannedOrders;
    for (int round=0; round<rounds; ++round) {
      plannedOrders.push_back(all
          ? std::vector<Impl>(orders[round%4].begin(),orders[round%4].end())
          : sync ? std::vector<Impl>(syncOrders[round%4].begin(),syncOrders[round%4].end())
          : paired ? std::vector<Impl>(pairedOrders[round%7].begin(),pairedOrders[round%7].end())
                   : implementations);
    }
    const std::string orderPath = std::string(argv[6])+".order.csv";
    if (std::ifstream(orderPath).good())
      throw std::runtime_error("Order table already exists; choose a new output path.");
    // The full order is flushed and closed before any CUDA work.
    std::ofstream orderCsv(orderPath,std::ios::out);
    if (!orderCsv) throw std::runtime_error("Cannot create order CSV.");
    orderCsv << "n,alpha,beta,seed,round,order,implementation\\n";
    for (int round=0; round<rounds; ++round)
      for (size_t slot=0; slot<plannedOrders[round].size(); ++slot)
        orderCsv << n << ',' << Alpha << ',' << beta << ',' << Seed << ','
                 << round+1 << ',' << slot+1 << ',' << name(plannedOrders[round][slot]) << '\\n';
    orderCsv.close();
    if (!orderCsv) throw std::runtime_error("Failed closing order CSV.");
'''
    # Check output before creating either file, then insert the tables.
    replace('    // 不覆盖旧结果；脚本应为每次测量创建新目录。', '''    if (std::ifstream(argv[6]).good())
      throw std::runtime_error("Output already exists; choose a new CSV path.");
''' + new_tables + '    // 不覆盖旧结果；脚本应为每次测量创建新目录。')
    replace('''      std::vector<Impl> order = all
          ? std::vector<Impl>(orders[round%4].begin(),orders[round%4].end())
          : sync ? std::vector<Impl>(syncOrders[round%4].begin(),syncOrders[round%4].end())
                 : implementations;''', '      const auto& order = plannedOrders[round];')
    replace('"a_hash,b_hash,c0_hash,status,round_start_unix_ms,round_end_unix_ms\\n";', '''"a_hash,b_hash,c0_hash,status,round_start_unix_ms,round_end_unix_ms,"
           "direct_cpu_count,direct_cpu_max_abs_error,direct_cpu_max_rel_error,"
           "direct_cpu_failure_count,direct_cpu_nonfinite_count\\n";''')
    replace('    auto validate = [&]() {\n      CUDA_OK(cudaMemcpy(actual.data(),dc.ptr,bytes,cudaMemcpyDeviceToHost));\n      return compare(actual,expected);\n    };', '''    Errors directCpuErrors;
    auto validate = [&](Impl impl,int round) {
      CUDA_OK(cudaMemcpy(actual.data(),dc.ptr,bytes,cudaMemcpyDeviceToHost));
      directCpuErrors = checkCpuReference(n,beta,a,b,initial,actual,indices);
      std::printf("CPU_CANDIDATE,%s,round=%d,count=%zu,max_abs=%.9g,max_rel=%.9g,failures=%zu,nonfinite=%zu,%s\\n",
          name(impl),round,indices.size(),directCpuErrors.maxAbs,directCpuErrors.maxRel,
          directCpuErrors.failures,directCpuErrors.nonfinite,directCpuErrors.failures ? "FAIL" : "PASS");
      return compare(actual,expected);
    };''')
    replace('const Errors errors = validate();', 'const Errors errors = validate(impl,0);', 2)
    # Only the timed-loop occurrence gets its actual round number.
    replace('''        const Errors errors = validate(impl,0);''', '        const Errors errors = validate(impl,round+1);')
    replace('errors.failures ? "FAIL" : "PASS"', '(errors.failures || directCpuErrors.failures) ? "FAIL" : "PASS"', 3)
    replace('if (errors.failures) throw', 'if (errors.failures || directCpuErrors.failures) throw')
    replace('''        if (errors.failures)
          throw''', '''        if (errors.failures || directCpuErrors.failures)
          throw''')
    replace("<< ',' << roundStart << ',' << roundEnd << '\\n';", "<< ',' << roundStart << ',' << roundEnd\n              << ',' << indices.size() << ',' << directCpuErrors.maxAbs << ',' << directCpuErrors.maxRel\n              << ',' << directCpuErrors.failures << ',' << directCpuErrors.nonfinite << '\\n';")
    replace('    std::printf("TIMING,C0_D2D_reset_excluded,events_default_stream,"', '''    std::printf("HOST_ADAPTATION,alpha_explicit=%d,order_file=%s,input_generator=xorshift32_v1,input_hash=FNV1a64_float32_bytes\\n",
                int(alphaExplicit),orderPath.c_str());
    std::printf("TIMING,C0_D2D_reset_excluded,events_default_stream,"''')
    prefix = '''// RTX5090 unified host audit, derived from scripts/windows_benchmark.cu.
// Device headers and all/sync execution tables unchanged. Default alpha=0.5.
// New paired group and --alpha 0.5|1; CPU-double checks candidate outputs directly.
// Exact provenance and host-only diff accompany this file; no GPU result is implied.
'''
    return prefix + s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source_bytes = (args.repo / SOURCE).read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == EXPECTED, 'Unexpected source bytes'
    actual_header_hashes = {}
    for path, expected in HEADERS.items():
        actual = hashlib.sha256((args.repo / path).read_bytes()).hexdigest()
        assert actual in {expected, HEADERS_GIT_BLOB[path]}, path
        actual_header_hashes[path] = actual
    source = source_bytes.decode('utf-8').replace('\r\n', '\n')
    generated = derive(source)
    args.out.mkdir(parents=True, exist_ok=True)
    payloads = {
        'benchmark.cu': generated,
        'host-harness.diff': ''.join(difflib.unified_diff(source.splitlines(True), generated.splitlines(True), fromfile=SOURCE, tofile='experiments/rtx5090/strict-fp32/benchmark.cu')),
        'provenance.json': json.dumps({
            'base_source_commit': BASE_COMMIT,
            'source_path': SOURCE, 'source_sha256': EXPECTED,
            'derived_path': 'experiments/rtx5090/strict-fp32/benchmark.cu',
            'derived_sha256': hashlib.sha256(generated.encode('utf-8')).hexdigest(),
            'device_headers_unchanged': actual_header_hashes,
            'device_headers_windows_worktree_sha256': HEADERS,
            'device_headers_git_blob_sha256': HEADERS_GIT_BLOB,
            'materialized_repo_head': subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip(),
            'default_alpha': 0.5, 'allowed_explicit_alpha': [0.5,1],
            'beta': [0,1,3], 'seed': 20261020,
            'original_groups_preserved': ['all','sync'],
            'new_group': ['k10','k10_pad8','k30_v1','k30_v2','strict'],
            'modifications': ['relative header paths for isolated directory','host alpha option','five-way paired group and pre-generated order CSV','direct CPU-double candidate validation'],
            'gpu_execution': 'NOT_RUN_BY_GENERATOR',
        },ensure_ascii=False,indent=2)+'\n',
    }
    for name, content in payloads.items():
        path = args.out / name
        if path.exists():
            assert path.read_bytes() == content.encode('utf-8'), f'Refusing to overwrite different {path}'
        else:
            path.write_bytes(content.encode('utf-8'))
    print(json.dumps({'out':str(args.out),'files':list(payloads),'derived_sha256':hashlib.sha256(generated.encode('utf-8')).hexdigest()}))


if __name__ == '__main__':
    main()
