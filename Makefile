.PHONY: all build debug clean profile bench cuobjdump

# 这些目标封装原有构建/性能分析命令；配方开头必须是制表符。
CMAKE := cmake

BUILD_DIR := build
BENCHMARK_DIR := benchmark_results

all: build

# 默认构建 Release。此 Makefile 调用子目录 make，不能混用 Ninja 生成的 build/。
build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Release ..
	@$(MAKE) -C $(BUILD_DIR)

# Debug 会启用设备调试选项，适合断点，不适合报告性能。
debug:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Debug ..
	@$(MAKE) -C $(BUILD_DIR)

# 注意：原命令会删除整个 build/，不要把自己的学习笔记放在里面。
clean:
	@rm -rf $(BUILD_DIR)

# 从二进制符号中找到 Warptiling 函数，供下面导出汇编。
FUNCTION := $$(cuobjdump -symbols build/sgemm | grep -i Warptiling | awk '{print $$NF}')

# SASS 是目标架构机器指令的反汇编，PTX 是较高层的虚拟指令表示。
# 原命令固定 sm_86；这里只增加说明，不替换架构或编译选项。
cuobjdump: build
	@cuobjdump -arch sm_86 -sass -fun $(FUNCTION) build/sgemm | c++filt > build/cuobjdump.sass
	@cuobjdump -arch sm_86 -ptx -fun $(FUNCTION) build/sgemm | c++filt > build/cuobjdump.ptx

# 用法：make profile KERNEL=<整数编号> PREFIX=<可选前缀>
# 需要 Nsight Compute/ncu 及设备性能计数器权限；profile 的耗时不能当正常延迟。
# --force-overwrite 会覆盖同名分析文件，运行前确认输出路径。
profile: build
	@ncu --set full --export $(BENCHMARK_DIR)/$(PREFIX)kernel_$(KERNEL) --force-overwrite $(BUILD_DIR)/sgemm $(KERNEL)

# 会运行批量测试并调用绘图脚本；后者会改写 README 的结果表和结果图片。
bench: build
	@bash gen_benchmark_results.sh
