#!/usr/bin/env python3
"""Pure-CPU audit of the pinned RTX 4090 D comparison executable.

Read cuobjdump SASS/PTX/resources and the actual ptxas log. Never run a GPU.
Static counts/PC order are not dynamic counts, cycles, occupancy or overlap proof.
Outputs are created in a fresh directory, preserving the immutable build inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re


SASS_OPS = ('HMMA', 'FFMA', 'LDGSTS', 'LDGDEPBAR', 'DEPBAR', 'BAR',
            'LDG', 'LDS', 'STS', 'LDSM', 'LDL', 'STL', 'BRA', 'WARPSYNC')
TF32 = ('vitamin_tf32_dbf', 'tf32_bn64_bk32', 'cutlass_tf32_128x128',
        'cutlass_tf32_128x64_bk16')
SIMT = ('k10_fp32', 'k30_fp32', 'cutlass_simt_256x128')
CUTLASS = ('cutlass_tf32_128x128', 'cutlass_tf32_128x64_bk16', 'cutlass_simt_256x128')


def read(path):
    return Path(path).read_text(encoding='utf-8-sig')


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refusing to silently write an empty analysis: ' + str(path))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classify(symbol):
    if 'bn64_tf32' in symbol:
        return 'tf32_bn64_bk32'
    if 'vitamin_tf32' in symbol:
        return 'vitamin_tf32_dbf'
    if 'cutlass6Kernel' in symbol:
        tile = re.search(r'GemmShapeILi(\d+)ELi(\d+)ELi(\d+)E', symbol)
        assert tile, symbol
        return {('128', '128', '32'): 'cutlass_tf32_128x128',
                ('128', '64', '16'): 'cutlass_tf32_128x64_bk16',
                ('256', '128', '8'): 'cutlass_simt_256x128'}[tile.groups()]
    if 'sgemmWarptiling' in symbol:
        return 'k10_fp32'
    if 'sgemmK30' in symbol:
        assert 'ELi2EE' in symbol, 'Expected K30 V2'
        return 'k30_fp32'
    if 'prepareInputs' in symbol:
        return 'prepare_inputs_helper'
    raise ValueError('Unknown symbol: ' + symbol)


def sass_functions(text):
    blocks = list(re.finditer(r'^\s*Function\s*:\s*(\S+)', text, re.M))
    result = {}
    for index, match in enumerate(blocks):
        end = blocks[index + 1].start() if index + 1 < len(blocks) else len(text)
        body = text[match.end():end]
        name, rows = match[1], []
        for ins in re.finditer(r'^\s*/\*([0-9a-fA-F]+)\*/\s+(.*?)\s*;', body, re.M):
            op = re.match(r'(?:@(!?\w+)\s+)?([A-Z][A-Z0-9_.]*)\b', ins[2])
            assert op, ins[0]
            rows.append(dict(pc=int(ins[1], 16), opcode=op[2].split('.')[0],
                             full_opcode=op[2], predicate=op[1] or '',
                             constant_false=op[1] == '!PT', instruction=ins[2].strip(),
                             source_line=text.count('\n', 0, match.end() + ins.start()) + 1))
        assert rows, name
        mode = classify(name)
        assert mode not in result
        result[mode] = dict(symbol=name, rows=rows)
    assert set(result) == set(TF32 + SIMT + ('prepare_inputs_helper',)), set(result)
    return result


def ptx_functions(text):
    entries = list(re.finditer(r'^\s*(?:\.visible\s+)?\.entry\s+(\S+)\(', text, re.M))
    result = {}
    for index, match in enumerate(entries):
        end = entries[index + 1].start() if index + 1 < len(entries) else len(text)
        body = text[match.end():end]
        rows = []
        for ins in re.finditer(r'^\s*(?:@!?%\w+\s+)?([a-z][a-z0-9_.:]+)\s*([^;\n]*);', body, re.M):
            rows.append(dict(opcode=ins[1], instruction=(ins[1] + ' ' + ins[2]).strip(),
                             source_line=text.count('\n', 0, match.end() + ins.start()) + 1))
        result[classify(match[1])] = dict(symbol=match[1], rows=rows)
    return result


def resources(text, ptxas):
    result, current = {}, None
    for line in text.splitlines():
        match = re.match(r'\s*Function\s+(\S+):', line)
        if match:
            current = classify(match[1])
            result[current] = {'symbol': match[1]}
        elif current and 'REG:' in line:
            values = dict(re.findall(r'(REG|STACK|SHARED|LOCAL):(\d+)', line))
            result[current].update({k.lower() + '_bytes' if k != 'REG' else 'registers': int(v)
                                    for k, v in values.items()})
    current = None
    for line in ptxas.splitlines():
        match = re.search(r"Compiling entry function '(\S+)' for '(sm_\d+)'", line)
        if match:
            current = classify(match[1])
            result[current]['ptxas_arch'] = match[2]
        elif current:
            spills = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', line)
            if spills:
                for name, value in zip(('ptxas_stack_bytes', 'spill_store_bytes', 'spill_load_bytes'), spills.groups()):
                    result[current][name] = int(value)
            reg = re.search(r'Used (\d+) registers', line)
            if reg:
                assert result[current]['registers'] == int(reg[1]), current
    for mode, record in result.items():
        assert record['ptxas_arch'] == 'sm_89', (mode, record)
        assert record['stack_bytes'] == record['ptxas_stack_bytes']
        assert 'spill_load_bytes' in record and 'spill_store_bytes' in record
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, help='A completed harness timings CSV .meta.txt file')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.build / name for name in ('binary.sass', 'binary.ptx', 'resources.txt', 'build.log')]
    texts = [read(path) for path in paths]
    sass, ptx = sass_functions(texts[0]), ptx_functions(texts[1])
    resource = resources(texts[2], texts[3])
    assert set(sass) == set(ptx) == set(resource)
    assert set(re.findall(r'arch\s*=\s*(sm_\w+)', texts[0])) == {'sm_89'}
    assert set(re.findall(r'\.target\s+(sm_\w+)', texts[1])) == {'sm_89'}

    metadata = {}
    if args.metadata:
        paths.append(args.metadata)
        metadata = dict(line.split('=', 1) for line in read(args.metadata).splitlines() if '=' in line)
        assert metadata
    meta_suffix = {'cutlass_tf32_128x128': '128', 'cutlass_tf32_128x64_bk16': '64',
                   'cutlass_simt_256x128': 'simt'}
    summary, op_rows, ptx_rows, position_rows, loops, loop_positions = [], [], [], [], [], []
    evidence = {}
    for mode, function in sass.items():
        ins = function['rows']
        counts = Counter(i['opcode'] for i in ins)
        full = Counter(i['full_opcode'] for i in ins)
        false = Counter(i['opcode'] for i in ins if i['constant_false'])
        ptx_ops = Counter(i['opcode'] for i in ptx[mode]['rows'])
        record = {'implementation': mode, **resource[mode], 'static_instructions': len(ins),
                  **{op: counts[op] for op in SASS_OPS},
                  'constant_false_instructions': sum(false.values()),
                  'constant_false_LDS': false['LDS'],
                  'shared_note': 'cuobjdump SHARED is static only; dynamic bytes require launch metadata'}
        if mode in meta_suffix and metadata:
            suffix = meta_suffix[mode]
            record.update(launch_threads=int(metadata['cutlass_threads_' + suffix]),
                          dynamic_shared_bytes=int(metadata['cutlass_shared_' + suffix]))
        summary.append(record)
        for op, count in sorted(full.items()):
            op_rows.append(dict(implementation=mode, opcode=op, static_count=count,
                                constant_false_count=sum(i['full_opcode'] == op and i['constant_false'] for i in ins)))
        for op, count in sorted(ptx_ops.items()):
            ptx_rows.append(dict(implementation=mode, opcode=op, static_count=count))
        for index, row in enumerate(ins):
            if row['opcode'] in SASS_OPS:
                position_rows.append({'implementation': mode, 'instruction_index': index,
                                      **row, 'pc': hex(row['pc'])})
        for index, row in enumerate(ins):
            if row['opcode'] != 'BRA' or row['constant_false']:
                continue
            target = re.search(r'\b(0x[0-9a-f]+)\b', row['instruction'])
            if not target or int(target[1], 16) >= row['pc']:
                continue
            body = [i for i in ins if int(target[1], 16) <= i['pc'] <= row['pc']]
            body_counts = Counter(i['opcode'] for i in body if not i['constant_false'])
            if not body_counts['HMMA'] and not body_counts['FFMA']:
                continue
            loop_id = f"{mode}_{target[1]}_{hex(row['pc'])}"
            copies = [i['pc'] for i in body if i['opcode'] == 'LDGSTS' and not i['constant_false']]
            last_copy = max(copies) if copies else None
            waits = [i['pc'] for i in body if i['opcode'] == 'DEPBAR' and not i['constant_false']
                     and last_copy is not None and i['pc'] > last_copy]
            wait = min(waits) if waits else None
            between = sum(i['opcode'] == 'HMMA' and not i['constant_false'] and last_copy < i['pc'] < wait
                          for i in body) if wait is not None else None
            loops.append(dict(implementation=mode, loop_id=loop_id, start=target[1], end=hex(row['pc']),
                              instructions=len(body), **{op: body_counts[op] for op in SASS_OPS},
                              last_async_copy=hex(last_copy) if last_copy is not None else '',
                              first_wait_after_last_copy=hex(wait) if wait is not None else '',
                              hmma_between_last_copy_and_wait=between,
                              note='Static backward-BRA interval, not a dynamic schedule or overlap measurement'))
            for position, item in enumerate(body):
                loop_positions.append({'implementation':mode, 'loop_id':loop_id, 'loop_index':position,
                                       **item, 'pc':hex(item['pc'])})

        tensor_instructions = {op: count for op, count in full.items() if op.startswith(('HMMA', 'IMMA', 'BMMA', 'MMA'))}
        ptx_mma = {op: count for op, count in ptx_ops.items() if 'mma' in op}
        tma = {op: count for op, count in ptx_ops.items()
               if op.startswith(('cp.async.bulk', 'wgmma.', 'tcgen05.')) or 'tensor' in op}
        if mode in TF32:
            assert tensor_instructions and all(op.startswith('HMMA.') and '.TF32' in op for op in tensor_instructions), tensor_instructions
            assert ptx_mma and all('.tf32.' in op for op in ptx_mma), ptx_mma
        if mode in SIMT:
            assert not tensor_instructions and not ptx_mma, mode
            assert counts['FFMA'] > 0
        if mode in CUTLASS:
            assert counts['LDGSTS'] > 0 and any(op.startswith('cp.async.') for op in ptx_ops), mode
            assert 'OpMultiplyAddFastF32' not in function['symbol']
        assert not tma, tma
        evidence[mode] = dict(sass_tensor_instructions=tensor_instructions, ptx_mma_instructions=ptx_mma,
                              ptx_async_instructions={op: count for op, count in ptx_ops.items() if op.startswith('cp.async.')},
                              tma_wgmma_tcgen05_instructions=tma)

    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / 'resources-and-sass-counts.csv', summary)
    write_csv(args.output / 'sass-opcodes.csv', op_rows)
    write_csv(args.output / 'ptx-opcodes.csv', ptx_rows)
    write_csv(args.output / 'sass-instruction-positions.csv', position_rows)
    write_csv(args.output / 'sass-loop-candidates.csv', loops)
    write_csv(args.output / 'sass-loop-order.csv', loop_positions)
    paths.append(Path(__file__))
    hashes = [dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    report = dict(status='PASS', target='sm_89', function_count=len(sass), precision_evidence=evidence,
                  input_sha256=hashes, runtime_metadata_included=bool(metadata),
                  limits=['Static counts include all code paths, constant-false instructions reported separately.',
                          'Source tfloat32_t + OpMultiplyAdd proves the selected single-TF32 configuration; SASS verifies TF32 HMMA, but opcode alone cannot distinguish a hypothetical TF32x3 algorithm.',
                          'Static LDGSTS/HMMA/DEPBAR ordering demonstrates scheduling opportunity, not measured overlap or saved cycles.',
                          'The three CUTLASS configurations differ in multiple factors and are not a single-factor experiment or a library-wide optimum.',
                          'cuBLAS kernels live in the library and are not present in this executable dump.',
                          'SHARED=0 in cuobjdump does not mean zero shared memory when dynamic allocation is used.'])
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (args.output / 'STATIC_AUDIT_PASS.txt').write_text('PASS\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'target', 'function_count', 'runtime_metadata_included')}))


if __name__ == '__main__':
    main()
