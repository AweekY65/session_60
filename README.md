# Buddy Memory Allocator 模拟器

一个用 Rust 实现的**固定大小内存池伙伴分配器（Buddy Memory Allocator）模拟器**。
内存池、各 order 的空闲链表和全部统计状态都只存在于本地进程内存中，由普通的
Rust 容器（`Vec` / `BTreeSet` / `BTreeMap`）支撑；**不依赖任何操作系统专用内存
管理服务（不调用 `mmap`/`brk`/`malloc` 来分配被模拟的块）、数据库或外部服务**，
核心的分配、拆分、合并算法全部在 [`src/lib.rs`](src/lib.rs) 内实现。项目无任何
第三方依赖，属性测试中的随机数发生器也在工程内实现（`src/sim.rs`）。

## 快速开始

```bash
cargo test            # 运行全部自动化测试与 property-based tests
cargo run --example demo   # 1 MiB 内存池上的随机负载演示，打印统计信息
```

## 核心概念

初始化时指定 2 的幂大小的内存池 `pool_size` 与同样为 2 的幂的最小块
`min_block_size`。定义 order：

```
order k 的块大小 = min_block_size × 2^k
max_order        = log2(pool_size / min_block_size)
```

初始时整个池子是一个 `max_order` 的空闲块。`free_lists[k]` 保存所有 order `k`
的空闲块起始偏移。

### 伙伴（buddy）地址计算

一个偏移为 `x`、大小为 `s`（2 的幂）的块，其伙伴的偏移为：

```
buddy(x) = x XOR s
```

因为块天然按 `s` 对齐：

- 若 `x` 是这一对中的“左块”（`x / s` 为偶数），则伙伴在右侧 `x + s`；
- 若 `x` 是“右块”，则伙伴在左侧 `x - s`。

两个伙伴大小相同、地址连续，合并后恰好是一个 order+1 的块，其起始为
`min(x, buddy(x))`。模拟地址带一个任意的 `base`，内部全部换算成相对偏移计算，
所以 `base` 不必是 2 的幂。

### `allocate(size)`：最小适配 + 递归拆分

1. 求能容纳请求的最小 order：`k = ceil_log2(ceil(size / min_block_size))`。
2. 从 order `k` 开始向上找第一个非空空闲链表：
   - 若一直找到 `max_order` 都没有空闲块 → `OutOfMemory`。
3. 从该链表取一个块，若它的 order 高于 `k`，则递归对半拆分：
   左半块继续向目标 order 拆，右半块（伙伴）挂到低一级空闲链表。
4. 返回左半块（最终 order `k`）的模拟地址，并记录请求大小。

返回的实际块大小满足 `size ≤ block_size < 2 × size`（在最小块粒度上）。

### `free(address)`：校验 + 递归合并

释放前依次检测非法情形（见下文错误类型）。合法释放后：

1. 从已分配表移除块，放入对应 order 的空闲链表；
2. 当 `order < max_order` 时计算伙伴：
   - 若伙伴**正好**在同一 order 的空闲链表中 → 摘下伙伴，二者合并为
     order+1 块（取较小起始），继续向上；
   - 否则停止（伙伴仍被占用，或与更大的空闲块合并了）。

只合并真正的伙伴、不合并任意相邻块，保证每个 order 下绝不会同时存在一对
相邻的空闲伙伴（合并极大化），从而尽可能减少外部碎片。

### 非法释放检测

`free` 对以下情况分别返回明确的错误（均为 `FreeError`，且不修改任何状态）：

| 情形 | 错误 |
|---|---|
| 地址在 `[base, base+pool_size)` 之外 | `OutOfBounds` |
| 地址在池内但未按最小块对齐 | `Misaligned` |
| 地址落在某个**已分配**块内部（非起始地址） | `NotBlockStart`（错误中给出真正的块起始地址） |
| 起始地址此前已释放过 | `DoubleFree` |
| 其它不是任何块起始的对齐地址（空闲块内部等） | `InvalidAddress` |

double free 的识别依据一个“当前空闲块起始地址集合”：只要该偏移对应一个
现存空闲块的起点，就说明它已被释放过。

## 统计信息（`stats()`）

`Stats` 快照包含：

- `pool_size`：池总容量（恒定）；
- `allocated_bytes` / `free_bytes`：已分配块（按块大小计）与空闲块字节数，
  二者恒满足 `allocated_bytes + free_bytes == pool_size`（**空间守恒**）；
- `requested_bytes`：所有请求的原始大小之和；
- `internal_fragmentation = allocated_bytes - requested_bytes`，
  以及比例 `internal_fragmentation_ratio()`；
- `allocated_blocks` / `free_blocks`；
- `free_blocks_per_order: Vec<usize>`：**各 order 的空闲块数量**；
- `largest_free_block`：当前最大空闲块；
- `external_fragmentation_ratio() = 1 - largest_free_block / free_bytes`：
  常用的外部碎片度量，越接近 1 表示空闲空间越分散；
- 累计计数：`total_allocations`、`total_frees`、
  `failed_allocations`、`failed_frees`。

> 内部碎片来自块大小相对请求的向上取整；外部碎片来自空闲块被伙伴边界切碎，
> 可能“总空闲够但没有连续大块”，此时分配返回 `OutOfMemory`。

## 复杂度

设 `M = max_order + 1`（块大小层级数，`M = log2(pool_size/min_block) + 1`），
`n` 为相关 order 空闲链表中的块数。空闲链表用 `BTreeSet` 实现，取/放块为
O(log n)，伙伴查找是 O(1) 地址运算。

| 操作 | 时间复杂度 | 说明 |
|---|---|---|
| `allocate` | O(M·log n) | 最多跨 M 级找块 + M 次拆分插入；实践中 M 很小（如 1 MiB/64 B ⇒ 15） |
| `free` | O(M·log n) | 最多 M 级伙伴合并 |
| `stats` | O(F + A) | F 个空闲块、A 个已分配块，汇总一次 |
| `check_invariants` | O((F+A) log(F+A)) | 仅测试/调试用的全量不变量校验 |

空间复杂度 O(F + A)，即当前块数量级（不超过 `pool_size / min_block_size`）。

## 随机操作序列

`BuddyAllocator::run_random_workload(&mut Rng, ops, max_size, free_prob_percent)`
在分配器上执行确定性的随机 allocate/free 序列（只释放当前存活块）。
[`Rng`] 是工程内实现的 xorshift64\*，**同一种子必然产生同一序列**，便于复现。
演示见 `examples/demo.rs`。

## 不变量保证

每次操作后（测试中每一步都调用 `check_invariants()` 显式校验）：

1. **空间守恒**：所有已分配块与空闲块大小之和恰为 `pool_size`；
2. **不重叠**：任意两个已分配块区间互不相交；
3. **完全覆盖**：已分配块 + 空闲块首尾相接铺满整个池子，无缝隙；
4. **合并极大化**：不存在两个同 order 的相邻空闲伙伴；
5. 空闲记账集合与各空闲链表严格一致。

## 测试

```bash
cargo test
```

- `tests/buddy.rs`：确定性测试——初始化结构、最小/最大分配、连续拆分
  （1024 个最小块顺序拿满）、递归合并（全满后逆序释放回到单块）、
  “邻居但非伙伴不合并”、内存耗尽与外部碎片、double free、越界/非对齐/
  非起始地址释放、内部碎片与空间守恒、任意 `base` 地址等。
- `tests/property.rs`：**property-based tests**（不依赖 proptest/quickcheck，
  用内置 RNG + 固定种子矩阵保证可复现），对照一个独立的**逐字节占用参考模型**，
  在数千~数万次随机操作的每一步之后断言：已分配块绝不重叠、空间守恒、
  与参考模型逐字节一致、合并极大化、per-order 空闲字节之和等于 `free_bytes`，
  全部释放后恢复单块初始形态；覆盖小/中/大池、单 order 池、最小请求反复
  填满/排空、近池容量大请求等场景。

所有测试均在终端通过 `cargo test` 运行，无需网络或第三方依赖。
