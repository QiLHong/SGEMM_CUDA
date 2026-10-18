# 脚本怎么选

第一次来，不要挨个运行脚本。先读[代码与脚本使用索引](../docs/zh/23-代码与脚本使用索引.md)：
它解释全部 122 个代码/配置文件，包含术语、调用关系、最小 K30 路线、结果读法和危险入口。

## 常用顺序

```text
build     编译源码，得到可执行文件与资源日志
correctness  对同输入的严格 FP32 参考验算
memcheck / sync  检查访存、竞争与同步
fair      多轮交错计时，保存原始 CSV
profile / source  用 Nsight 解释硬件行为
analyze / summarize  读证据、统计并归档
```

当前最佳研究基线 K30 V2：从 [build_k30_audit.sh](build_k30_audit.sh) 和
[run_k30_audit.py](run_k30_audit.py) 开始。最小命令在上面的总索引中。
完整收尾复现使用包含 K10/K30/K32 的独立 harness，见[复现文档](../docs/zh/20-项目复现与简历讲述.md)。

每个文件顶部注明输入、输出、运行条件和覆盖范围，关键函数旁解释为什么这样做。
`.cu` 需要编译，`.sh` 用 Bash，`.py` 用 Python；`audit` 是编译出的程序，不是目录里的源文件。

## 先避开这些误用

- `kernel_9/10/11_autotuner.sh` 会原地改源码；先读，别直接跑。
- 旧绘图器会覆盖 README 和图片；不用它生成当前 strict FP32 研究结论。
- 输出目录已存在时换新名字，不删除旧结果。
- 只看到 GFLOPS 不等于实验成功；还要看 PASS、工具摘要、测试范围与日志是否齐全。
- 先串行验证，再计时；不要同时跑 profiler 与 benchmark。
