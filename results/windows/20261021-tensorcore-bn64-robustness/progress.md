# 进展

保存两仓库 Git 状态、差异、所有已跟踪及未忽略未跟踪文件实际字节 SHA256。未找到适用 AGENTS.md。尚未运行本轮 GPU 测量。

原始device提取与BN64派生文本验证均PASS；CPU完整CTA映射PASS（8192输出、12288shared float地址、无重复遗漏越界）。3072扩展网格枚举由导航审计独立保存。现有NCU直接路径版本确认通过；全指标查询已保存，部署sections警告使用安装目录fallback，没有更改HOME。

新host build01完成。Release/sm89/无fast-math确认；static01核对35个旧experiment文件原字节未变。新旧binary不同，但作者与BN64两函数normalized SASS+128bit编码字完全一致。寄存器127/102，spill均0。尚无本轮正式event。

Plan.csv已冻结756组；三小矩形smoke通过（9个辅助event，CPU每例全量131072点）。Sanitizer attach timeout20秒后仍等待target，总60秒TIMEOUT_UNVERIFIED；日志提示No attachable process。终止命令返回user name/password错误，事后Get-Process未发现工具/目标残留；不声称终止成功或内存检查通过。正式run已启动，未与profiler并行。

正式三batch结束，22680events独立复算PASS；576逐位比较0差异，4320参考检查0非有限。2048全部63轮领先双对照；3072作者63/63，库51/63；4096作者19/63。六次NCU成功，static/evidence编码一致。33章已写，全部36组合列出，正反counter和Sanitizer失败均保留。正在最终文档、导航及字节保全检查。

交付核验完成：实际读取并哈希2487个起点文件，除6个授权导航文件外全部未变；Git HEAD和index未变。源码zip逐项回读通过，正式event独立验证和6个NCU证据核验通过；Sanitizer保留TIMEOUT_UNVERIFIED。到此停止，不继续调参。
