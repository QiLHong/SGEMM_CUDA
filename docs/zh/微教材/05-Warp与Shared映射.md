# 第 5 章 Warp 与 Shared 映射

[上一章](04-分块与寄存器小矩阵.md) · [目录](README.md) · [下一章](06-同步与双缓冲.md)

前面已经解释数据复用。现在进一步问：同样的 64×64 输出，交给同样的 32 个
lane，是否只有一种分法？答案是否定的，而这种分配会影响 shared 的消费方式。

## 1 先把三个映射分开

```text
输出所有权 ownership：lane 7 负责哪些 C 坐标
共享物理布局 layout：逻辑 A[m,k] 存在 shared 的哪个地址
片段装载 iterator：本轮把哪些地址装进本线程的哪些寄存器
```

它们相互配合，但不是同一个对象。只改 layout 时，读写必须一致；只改 ownership
时，必须重新检查输出覆盖及加载模式；源码的 fragment 编号也不是最终物理 R 编号。
这解释了为什么简单 XOR 或交换 lane 并不天然优化整个微内核。

## 2 用真实的 K30 公式逐步推导

K30 使用 128×128 CTA，四个 warp 各负责 64×64。设 w 为 block 内 warp 编号，
l 为 lane 编号，均从 0 开始。代码中的行列基点是：

```text
row = 64*(w/2) + 4*(2*(l/16)+l%2)
col = 64*(w%2) + 4*((l%16)/2)
```

所有除法都是整数除法。第 i 个 A 行、第 j 个 B 列对应：

```text
m(i) = row + 16*(i/4) + i%4       i=0..15
n(j) = col + 32*(j/4) + j%4       j=0..7
acc[i*8+j] 负责 C[m(i),n(j)]，再加 block 的矩阵偏移
```

先代入 w=0、l=0：row=0、col=0，所以 M 集合是
`0..3,16..19,32..35,48..51`，N 集合是 `0..3,32..35`。
输出是两个集合的笛卡尔积，16×8=128 个元素，并非一整块连续 16×8 矩形。

再代入相邻 lane，观察变化：

|lane|row 基点|col 基点|相对 lane 0 的变化|
|---:|---:|---:|---|
|0|0|0|基准|
|1|4|0|M 坐标加 4|
|2|0|4|N 坐标加 4|
|15|4|28|M 加 4，N 加 28|
|16|8|0|M 坐标加 8|
|31|12|28|M 加 12，N 加 28|

于是 M 方向的四个基点 0/4/8/12，与步长 16 的四组拼满 0..63；N 方向的八个
基点 0/4/…/28，与步长 32 的两组拼满 0..63。四个 warp 再覆盖 CTA 四个象限。
这比只检查 `32*128=4096` 更强：我们说明了为什么既不重叠，也不遗漏。

## 3 与 K10 比较的是什么

当前 K10 同样让每 lane 保存 128 个输出，但输入片段组织不同：
TM=8、TN=4、WNITER=4，对应每 lane A8、B16。
以 warp 0 的 lane 0 为例，K10 的 M 为 0..7，N 为
`0..3,16..19,32..35,48..51`。K30 则是 A16/B8，并改变 lane 分布。

两者的输出总量相同，甚至每个 k 的标量 fragment 总数都为 24；但同一条 warp
指令中，不同 lane 在请求什么数据并不相同。因此“寄存器数组长度一样”不能证明
shared 服务成本或编译调度一样。

本项目中，新的 K30 组织降低了主循环 shared-load wavefront 约 40%，但单阶段
V1 仍然更慢。应把它解释为后续组合设计的一个条件，不是 layout 一项单独带来
全部吞吐收益。完整证据在[第 8 章](08-用K30串起整个项目.md)。

## 4 物理布局如何与逻辑坐标对应

K30 的 A shared 布局为 `As[k*132+m]`，B 为 `Bs[k*128+n]`。
例如逻辑 A[m=3,k=2] 在 A shared 的元素偏移是 `2*132+3=267`。
读取时必须再用这个公式找回它，不是把整个 global A 做了一份永久转置。

Iterator 将线程已知的 row/col 与当前 k 转成这些地址，advance 移到下一个 k。
它的价值可能是组织寻址与生命周期，但“写成 struct”本身不会保证少一条指令。
最终还是要看 ptxas 生成了什么。

## 5 零 conflict 为什么不是终点

shared-load conflict 为零，只排除了一类额外服务成本。一次宽读取仍可能需要
多个基本服务 wavefront，不同 lane 地址组合会影响这个需求。
所以要分别问：发出了多少 LDS、每次需要多少服务量、额外冲突又有多少。

不要反过来推断“wavefront 少就一定快”。V1 的失败已经说明，其他调度与同步
成本可能抵消它。硬件计数是解释的一环，不是最终接受标准。

## 6 源码精读：沿着一个 accumulator 找到它的所有输入

现在把公式接回 [K30 实现](../../../src/kernels/30_kernel_mini_simt.cuh)。
建议按下面顺序阅读，而不是从文件第一行一路硬读到最后：

```text
Shape 定义常量
  → FragmentIterator 定义“谁算哪个 C”
  → SharedLayoutA/B 定义“数据存在哪”
  → WarpIteratorA/B 定义“这轮取哪些输入”
  → compute 更新 accumulator
  → epilogue 把 accumulator 放回 C
```

### 6.1 先把模板参数替换成数字

当前审计入口的 `Small=k30::Shape<128,128,16,64,64,4,4,128>` 展开如下：

|源码常量|数值|用途|
|---|---:|---|
|M/N/K|128/128/16|CTA 的输出宽高与一次 K tile 长度，不是完整矩阵尺寸|
|WarpM/WarpN|64/64|一个 warp 的逻辑输出区域|
|ThreadM/ThreadN|4/4|fragment 的基本四元素分组约束|
|Rows=WM/4|16|每 lane 的 A fragment 元素数|
|Cols=WN/8|8|每 lane 的 B fragment 元素数|
|Rows*Cols|128|每 lane 的输出 accumulator 数|
|ALoads/BLoads|4/4|每线程每 K tile 各搬运多少个 float4|

**TM=TN=4 不代表这个线程总共只算 4×4 个输出。** 它是分组形状；
实际所有权由 Rows、Cols 与 m(i)/n(j) 共同定义。这也是只看模板名字容易误读
微内核的地方。`static_assert` 会检查 warp 覆盖、整除和分组条件，但不会替
主机检查传入矩阵是否满足所有运行时边界与对齐契约。

### 6.2 一个具体的 Shared → Register → C 追踪

取 blockIdx=(0,0)、tid=2，即 warp 0、lane 2。由 FragmentIterator 得到
row=0、col=4。选择 i=5、j=6：

```text
m(5) = 0 + 16 + 1 = 17
n(6) = 4 + 32 + 2 = 38
reg(5,6) = 5*8+6 = 46
```

所以 acc[46] 始终负责 C[17,38]。在第一段 K 的 k=2，这一轮的数据流是：

```text
A[17,2] → As[2*132+17] = As[281] → fa[5] ┐
                                         ├→ acc[46] += fa[5]*fb[6]
B[2,38] → Bs[2*128+38] = Bs[294] → fb[6] ┘
```

沿源码逐个检查对应实现：

```cpp
// SharedLayoutA / SharedLayoutB
__device__ static int offset(int m,int k) { return k*Stride+m; }
__device__ static int offset(int k,int n) { return k*Stride+n; }

// FragmentIterator
__device__ static int reg(int i,int j) { return i*S::Cols+j; }

// compute
acc[FragmentIterator<S>::reg(i,j)]+=fa[i]*fb[j];
```

这里把不同结构体中的相关行放在一起展示；它们不是一个可以独立编译的结构体。
上述代码中的两个 Stride 分别是 132 和 128。

### 6.3 WarpIterator 到底“迭代”什么

A 的 iterator 从 `base+offset(map.row,0)` 开始，load 取
`ptr[(i/4)*16+i%4]`，advance 执行 `ptr+=132`。
B 的 iterator 从 `base+offset(0,map.col)` 开始，load 取
`ptr[(j/4)*32+j%4]`，advance 执行 `ptr+=128`。

因此每 advance 一次，输出所属的行/列不变，只切换到下一个 k。
`compute` 为当前 shared tile 新建 iterator，循环 16 次。下一 tile 又从它自己的
shared 基址重新开始，但 acc 不清零。这让“沿 shared 的 k 前进”和“沿全局 K
tile 前进”成为两个清楚的层次。

这并非魔法：常量表达式、内联和展开可能使它与直接索引生成一样的 SASS。
它首先让映射关系能被检查；性能价值需要另看地址指令、load 调度和寄存器分配。

### 6.4 这两个输入又是谁从 global 搬来的

读 [GlobalIterator](../../../src/kernels/30_kernel_mini_simt.cuh#L41)：
A 每线程一次搬四个连续 k，`ar=tid/4`、`ak=(tid%4)*4`，再以 32 行为步长搬四组。
B 每线程搬四个连续 n，`bk=tid/32`、`bn=(tid%32)*4`，再以 4 个 k 行为步长搬四组。

因此第一段中，A[17,2] 来自 tid=68 的第一个 float4 的 z 分量；
B[2,38] 来自 tid=73 的第一个 float4 的 z 分量。消费者 tid=2 与两位搬运者
都不同。这把第三章的合作加载思想延续到了 K30，而不是每个线程独立做完全部工作。

A 装载沿 global 的 k 连续，写入 shared 时 x/y/z/w 分别放到不同 k 行；
B 装载沿 n 连续，写入 shared 时仍保持四个连续 n，所以可以整组 float4 store。
由此能理解：A/B 数学地位对称，不代表最合适的物理搬运方式对称。

### 6.5 最后怎么从 acc[46] 回到 C

epilogue 的 j 按 4 前进。i=5、j=4 时，目标起点是 C[17,36]，r=44，
一次处理 acc[44..47]，其中第三项就是 acc[46]，写到 C[17,38]。
这里仍用完整 N 作为 global 行跨度，不是 128 或 8。

四列一组的输出地址满足当前契约的 16 字节对齐；这不等于支持任意 N 或子矩阵
偏移。源码只对 M 行做了特定保护，没有通用 N/K 尾部处理。

本章动手项：将 tid=2 改成 tid=3，再追踪 i=5、j=6。输出行变成 21，列仍是 38。
不要先想着“怎样改快”，先确保能从任意 acc 编号追到 A/B/C 三个坐标。

## 7 练习

1. K30 的 warp 1、lane 0 相对 warp 0、lane 0 改变哪些坐标？
2. K30 lane 31 在 warp 0 中负责的最大 M/N 坐标分别是多少？
3. 如果所有 lane 的输出个数之和正确，为什么还不能保证结果正确？

### 参考答案

1. w=1 使 col 加 64，row 不变，即负责右上象限。
2. M 最大 `12+48+3=63`，N 最大 `28+32+3=63`。
3. 可能重复覆盖某些位置，同时漏掉同样数量的位置。还需验证坐标集合、唯一性、
   输入读取、计算过程和写回。练习脚本枚举 K10/K30 的整个 CTA 输出集合。

源码入口：[K30 FragmentIterator](../../../src/kernels/30_kernel_mini_simt.cuh)、
[K10 processFromSmem](../../../src/kernels/10_kernel_warptiling.cuh)。

[下一章 同步与双缓冲](06-同步与双缓冲.md)
