#!/usr/bin/env python3
"""只读检查中文翻译是否改变原版执行文本；不是编译器或 GPU 测试。"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import subprocess
import sys

BASE = "5a7dcc513d951ba764d51bc9d587b3163f3a894d"
TRANSLATION = "aadcd8e"
# 字符串必须整体匹配，不能把 URL 或字符串中的 // 当成注释。
CPP_TOKEN = re.compile(
    r'(?P<raw>(?:u8|u|U|L)?R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\([\s\S]*?\)(?P=delimiter)")'
    r'|(?P<string>(?:u8|u|U|L)?(?:"(?:\\[\s\S]|[^"\\])*"|\'(?:\\[\s\S]|[^\'\\])*\'))'
    r'|(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)'
    r'|(?P<space>\s+)'
    r'|(?P<word>[A-Za-z_][A-Za-z_0-9]*|[0-9][A-Za-z_0-9.\']*)'
    r'|(?P<op>%:%:|>>=|<<=|->\*|\.\.\.|::|->|\.\*|\+\+|--|&&|\|\||<<|>>|<=|>=|==|!=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|##|\S)'
)


def cpp_tokens(text: str) -> list[str]:
    """忽略注释/排版，保留字面量、运算符和预处理指令的行终止。"""
    text = re.sub(r"\\\r?\n", "", text)
    tokens: list[str] = []
    line_start, directive = True, False
    for match in CPP_TOKEN.finditer(text):
        kind, value = match.lastgroup, match.group()
        if kind in ("space", "comment"):
            if "\n" in value:
                if directive:
                    tokens.append("<directive-end>")
                directive, line_start = False, True
            continue
        if line_start and value == "#":
            directive = True
        tokens.append(value)
        line_start = False
    if directive:
        tokens.append("<directive-end>")
    return tokens


class WithoutDocstrings(ast.NodeTransformer):
    """仅移除模块/函数/类的首条 docstring，不忽略普通字符串常量。"""
    def visit(self, node: ast.AST) -> ast.AST:
        node = super().visit(node)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr):
                value = body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body = body[1:]
        return node


def python_structure(text: str) -> str:
    return ast.dump(WithoutDocstrings().visit(ast.parse(text)), include_attributes=False)


def noncomment_lines(text: str) -> list[str]:
    # 对本仓库 shell/Make/CMake/YAML，仅忽略整行注释和空行。
    # 保留 shebang、前导空白、Makefile TAB、命令字符串和所有非注释文本。
    return [line.rstrip() for line in text.splitlines()
            if line.strip() and (not line.lstrip().startswith("#") or line.startswith("#!"))]


def fingerprint(path: str, data: bytes) -> object:
    if path.endswith((".cu", ".cuh")) or path.startswith(".vscode/"):
        return cpp_tokens(data.decode("utf-8"))
    if path.endswith(".py"):
        return python_structure(data.decode("utf-8"))
    if path.endswith((".sh", ".yml")) or path in ("CMakeLists.txt", "Makefile", ".gitignore"):
        return noncomment_lines(data.decode("utf-8"))
    return data


def git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def self_test() -> None:
    assert cpp_tokens('const char *s="https://a/b"; // old\n') == cpp_tokens('/*中文*/ const char *s="https://a/b";\n')
    assert cpp_tokens('x++;') != cpp_tokens('x+ +;')
    assert cpp_tokens('"old"') != cpp_tokens('"new"')
    assert cpp_tokens('R"tag(// /* literal */)tag"') == ['R"tag(// /* literal */)tag"']
    assert cpp_tokens('#define X 1\nx;') != cpp_tokens('#define X 1 x;')
    assert python_structure('def f():\n "old"\n return "keep"\n') == python_structure('def f():\n "中文"\n return "keep"\n')
    assert python_structure('print("old")') != python_structure('print("new")')
    assert noncomment_lines('# old\nx:\n\techo ok\n') == noncomment_lines('# 中文\nx:\n\techo ok\n')
    assert noncomment_lines('x:\n\techo ok\n') != noncomment_lines('x:\n echo ok\n')
    print('SELF-TEST PASS: token / docstring / Makefile checks')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=BASE, help='要比较的原版 commit，默认固定上游基线')
    parser.add_argument(
        '--translation', default=TRANSLATION,
        help='纯翻译 commit；默认 aadcd8e，后续算法/实验提交不参与翻译判定')
    parser.add_argument('--self-test', action='store_true', help='只测试检查器，不读取仓库')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    checked = 0
    try:
        base = git(root, 'rev-parse', '--verify', args.base + '^{commit}').decode().strip()
        translation = git(
            root, 'rev-parse', '--verify', args.translation + '^{commit}'
        ).decode().strip()
        entries = git(root, 'ls-tree', '-r', '-z', base).split(b'\0')
        for entry in entries:
            if not entry:
                continue
            metadata, name = entry.split(b'\t', 1)
            mode, kind, blob = metadata.decode().split()
            path = name.decode('utf-8')
            if kind != 'blob' or mode not in ('100644', '100755'):
                failures.append(path + ': unsupported entry type/mode')
                continue
            before = git(root, 'cat-file', 'blob', blob)
            target = 'README.en.md' if path == 'README.md' else path
            translated_entry = git(
                root, 'ls-tree', '-z', translation, '--', target
            ).rstrip(b'\0')
            if not translated_entry:
                failures.append(target + ': missing from translation commit')
                continue
            translated_metadata, translated_name = translated_entry.split(b'\t', 1)
            translated_mode, translated_kind, translated_blob = (
                translated_metadata.decode().split()
            )
            if translated_name.decode('utf-8') != target or translated_kind != 'blob':
                failures.append(target + ': unexpected translation entry')
                continue
            after = git(root, 'cat-file', 'blob', translated_blob)
            equal = before == after if path == 'README.md' else fingerprint(path, before) == fingerprint(path, after)
            if not equal:
                failures.append(target + ': non-comment content differs')
            # README 在翻译提交中另存为 README.en.md；其余文件模式也必须保持。
            if path != 'README.md' and translated_mode != mode:
                failures.append(path + ': tracked file mode differs')
            checked += 1
    except (OSError, RuntimeError, ValueError, SyntaxError, subprocess.TimeoutExpired) as error:
        print('CHECK BLOCKED:', error, file=sys.stderr)
        return 2
    for failure in failures:
        print('FAIL:', failure)
    print(
        f'Base: {base}\nTranslation: {translation}\n'
        f'Original files checked: {checked}\nFailures: {len(failures)}'
    )
    print('此检查不执行 CUDA、benchmark 或调优脚本；不是编译/GPU 正确性证明。')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
