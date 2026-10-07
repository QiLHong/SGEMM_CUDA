#!/usr/bin/env python3

import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from pathlib import Path

# 【学习补充】原绘图脚本保持算法、正则、标签和输出格式不变。
# 英文运行日志是解析协议，不在注释翻译时改动；自动生成的表格/图例也保留英文。
# 注意：执行本文件会覆盖仓库根目录的结果图，并更新 README 内标记包围的表格。

# 原作者保留的主题配置：改善外观并使用较高 DPI；当前这行未启用。
# sn.set_theme(style="whitegrid", font_scale=1.0, rc={"figure.dpi": 200})
# 可选的衬线字体设置（未启用）。
# plt.rcParams["font.family"] = "serif"
# 当前启用的默认样式、字体与导出背景。
matplotlib.style.use("fivethirtyeight")
matplotlib.style.use("seaborn-v0_8-talk")
matplotlib.rcParams["font.family"] = "monospace"
matplotlib.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.facecolor"] = "white"
# sn.set_context("talk")

# kernel 编号要与 runner.cu 对照；这里没有 12 号的绘图标签。
KERNEL_NAMES = {
    0: "cuBLAS",
    1: "Naive",
    2: "GMEM Coalescing",
    3: "SMEM Caching",
    4: "1D Blocktiling",
    5: "2D Blocktiling",
    6: "Vectorized Mem Access",
    7: "Avoid Bank Conflicts (Linearize)",
    8: "Avoid Bank Conflicts (Offset)",
    9: "Autotuning",
    10: "Warptiling",
    11: "Double Buffering",
}


def parse_file(file):
    """
    从日志提取矩阵尺寸和 GFLOPS；待解析的上游输出格式如下：

    Average elapsed time: (0.005661) s, performance: (24277.4) GFLOPS. size: (4096).
    """
    with open(file, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    data = {"size": [], "gflops": []}
    pattern = "Average elapsed time: \((.*?)\) s, performance: \((.*?)\) GFLOPS. size: \((.*?)\)."
    for line in lines:
        if r := re.match(pattern, line):
            data["size"].append(int(r.group(3)))
            data["gflops"].append(float(r.group(2)))
    return data


def plot(df: pd.DataFrame):
    """
    输入表包含 kernel、size、gflops 三列。
    为不同 kernel 绘制随矩阵尺寸变化的多条 GFLOPS 曲线。
    """
    save_dir = Path.cwd()

    plt.figure(figsize=(18, 10))
    colors = sn.color_palette("husl", len(df["kernel"].unique()))
    sn.lineplot(data=df, x="size", y="gflops", hue="kernel", palette=colors)
    # 叠加数据点，但不再生成一份图例。
    sn.scatterplot(data=df, x="size", y="gflops", hue="kernel", palette=colors, legend=False)

    # 刻度使用实际测量尺寸。
    plt.xticks(df["size"].unique())
    # 将 x 轴刻度旋转并对齐。
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # 原作者备注：在 x 轴刻度处添加短线（这里没有额外执行语句）。

    # 在各条曲线末端显示名称。
    # 【学习补充】下面按枚举序号 i 查找 kernel，原实现假定编号/顺序完整对应。
    # 只收集部分编号或顺序不同可能不满足这一假设，本分支不改绘图逻辑。
    for i, kernel in enumerate(df["kernel"].unique()):
        # 原注释说右对齐；实际参数 horizontalalignment='left'，按代码为准。
        plt.text(
            df[df["kernel"] == i]["size"].iloc[-1],
            df[df["kernel"] == i]["gflops"].iloc[-1] + 300,
            f"{i}:{KERNEL_NAMES[i]}",
            color=colors[i],
            horizontalalignment="left",
            weight="medium",
        )

    # 移除图例，改由曲线旁文字标识。
    plt.gca().get_legend().remove()

    plt.title("Performance of different kernels")
    plt.xlabel("Matrix size (square, one side)")
    plt.ylabel("GFLOPs/s")
    plt.tight_layout()

    plt.savefig(save_dir / "benchmark_results.png")


if __name__ == "__main__":
    results_dir = Path("benchmark_results")
    assert results_dir.is_dir()

    data = []
    for filename in results_dir.glob("*.txt"):
        # 预期文件名为 <kernel编号>_output.txt。
        if not filename.stem.split("_")[0].isdigit() and "_output" not in filename.stem:
            continue
        results_dict = parse_file(filename)
        kernel_nr = int(filename.stem.split("_")[0])
        for size, gflops in zip(results_dict["size"], results_dict["gflops"]):
            data.append({"kernel": kernel_nr, "size": size, "gflops": gflops})
    df = pd.DataFrame(data)

    plot(df)

    # README 只展示 4096 尺寸的排序结果；比值分母是同批日志里的 cuBLAS。
    # 所以必须存在 0 号 kernel 的 4096 行，不能把不同机器的日志随意混在一起。
    df = df[df["size"] == 4096].sort_values(by="gflops", ascending=True)[["kernel", "gflops"]]
    df["kernel"] = df["kernel"].map({k: f"{k}: {v}" for k, v in KERNEL_NAMES.items()})
    df["relperf"] = df["gflops"] / df[df["kernel"] == "0: cuBLAS"]["gflops"].iloc[0]
    df["relperf"] = df["relperf"].apply(lambda x: f"{x*100:.1f}%")
    df.columns = ["Kernel", "GFLOPs/s", "Performance relative to cuBLAS"]

    # 读取将要更新的 README。
    with open("README.md", "r") as f:
        readme = f.read()
    # 替换两条 benchmark_results 标记之间的旧结果；标记本身必须保持原名。
    readme = re.sub(
        r"<!-- benchmark_results -->.*<!-- benchmark_results -->",
        "<!-- benchmark_results -->\n{}\n<!-- benchmark_results -->".format(
            df.to_markdown(index=False)
        ),
        readme,
        flags=re.DOTALL,
    )
    # 写入新结果；不是只读绘图操作。
    with open("README.md", "w") as f:
        f.write(readme)
