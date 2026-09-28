//! # Buddy Memory Allocator 模拟器
//!
//! 一个完全在本地进程内存中运行的伙伴系统（buddy system）内存分配器模拟器。
//!
//! * 内存池、各 order 的空闲链表（这里用有序集合实现以获得确定性的遍历顺序）
//!   以及所有统计状态都保存在 [`BuddyAllocator`] 结构体内部，仅由普通的
//!   Rust 容器（`Vec` / `BTreeSet` / `BTreeMap`）支撑，不使用任何操作系统
//!   专用内存管理服务（不调用 `mmap`/`brk`/`malloc` 来“分配被模拟的块”），
//!   也不依赖数据库或外部服务。
//! * 核心分配（最小适配 + 递归拆分）与释放（递归伙伴合并）算法全部在本文件实现。
//!
//! ## 设计概述
//!
//! 初始化时给定总容量 `pool_size`（2 的幂）与最小块 `min_block_size`（2 的幂，
//! 且不超过 `pool_size`）。整个池子被视为一个 `max_order` 级别的块，
//! 其中 order `k` 的块大小为 `min_block_size * 2^k`。
//!
//! * [`BuddyAllocator::allocate`]：从能满足请求的最小 order 开始找块；该 order
//!   没有空闲块时，从更高 order“借”一个块递归对半拆分，拆出的另一半挂到
//!   低一级的空闲链表中。
//! * [`BuddyAllocator::free`]：校验通过后把块放回对应 order 的空闲链表，
//!   然后不断计算它的伙伴（buddy）：若伙伴正好也在同一 order 空闲，则二者
//!   合并成高一级块继续向上合并，否则停止。
//!
//! 所有地址都是“模拟地址”：相对偏移 `0..pool_size` 加上调用方指定的 `base`。

use std::collections::{BTreeMap, BTreeSet};

pub mod sim;
pub use sim::{Op, OpResult, Rng, RunReport};

/// 分配失败的原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocError {
    /// 请求大小为 0。
    ZeroSize,
    /// 单个请求超过内存池总容量，任何块都不可能容纳。
    RequestTooLarge {
        /// 请求字节数。
        requested: usize,
        /// 内存池容量。
        pool_size: usize,
    },
    /// 没有足够大的连续空闲块（外部碎片 / 内存耗尽）。
    OutOfMemory {
        /// 请求字节数。
        requested: usize,
    },
}

impl std::fmt::Display for AllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AllocError::ZeroSize => write!(f, "allocate size must be greater than zero"),
            AllocError::RequestTooLarge {
                requested,
                pool_size,
            } => write!(
                f,
                "requested {requested} bytes but the whole pool is only {pool_size} bytes"
            ),
            AllocError::OutOfMemory { requested } => write!(
                f,
                "out of memory: no contiguous free block can satisfy {requested} bytes"
            ),
        }
    }
}

impl std::error::Error for AllocError {}

/// 释放失败的原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreeError {
    /// 地址完全落在内存池范围之外。
    OutOfBounds {
        /// 被释放的地址。
        address: usize,
    },
    /// 地址在池内，但没有按最小块大小对齐（不可能是任何块的起始地址）。
    Misaligned {
        /// 被释放的地址。
        address: usize,
    },
    /// 地址是某个**已分配**块的内部地址，而非该块的起始地址。
    NotBlockStart {
        /// 被释放的地址。
        address: usize,
        /// 真正拥有该区间的已分配块的起始地址。
        block_start: usize,
    },
    /// 该起始地址此前已经被释放过（再次释放即 double free）。
    DoubleFree {
        /// 被重复释放的地址。
        address: usize,
    },
    /// 地址不是任何当前块（已分配或空闲）的起始地址。
    InvalidAddress {
        /// 被释放的地址。
        address: usize,
    },
}

impl std::fmt::Display for FreeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FreeError::OutOfBounds { address } => {
                write!(f, "address {address:#x} is outside the memory pool")
            }
            FreeError::Misaligned { address } => write!(
                f,
                "address {address:#x} is not aligned to the minimum block size"
            ),
            FreeError::NotBlockStart {
                address,
                block_start,
            } => write!(
                f,
                "address {address:#x} lies inside an allocated block starting at {block_start:#x}; \
                 free must use the block start address"
            ),
            FreeError::DoubleFree { address } => {
                write!(f, "double free detected at {address:#x}")
            }
            FreeError::InvalidAddress { address } => write!(
                f,
                "address {address:#x} is not the start of any live or free block"
            ),
        }
    }
}

impl std::error::Error for FreeError {}

/// 一个已分配块的记录：块所在 order 与调用方请求的字节数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Allocation {
    /// 块 order，实际占用 `min_block_size << order` 字节。
    order: usize,
    /// 用户请求的原始大小，用于内部碎片统计。
    requested: usize,
}

/// 分配器的统计快照，参见 [`BuddyAllocator::stats`]。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stats {
    /// 内存池总容量（字节，恒定）。
    pub pool_size: usize,
    /// 当前已分配块数。
    pub allocated_blocks: usize,
    /// 当前空闲块数（所有 order 合计）。
    pub free_blocks: usize,
    /// 已分配块**实际占用**的字节数（块大小之和）。
    pub allocated_bytes: usize,
    /// 空闲块字节数之和。
    pub free_bytes: usize,
    /// 内部碎片字节数 = `allocated_bytes - 所有请求大小之和`。
    pub internal_fragmentation: usize,
    /// 当前所有已分配请求的原始字节数之和。
    pub requested_bytes: usize,
    /// 各 order 的空闲块数量，`free_blocks_per_order[k]` 即 order `k` 的空闲块数。
    pub free_blocks_per_order: Vec<usize>,
    /// 当前最大的空闲块大小（字节）；池子全满时为 0。
    pub largest_free_block: usize,
    /// 自初始化以来成功分配的总次数。
    pub total_allocations: u64,
    /// 自初始化以来成功释放的总次数。
    pub total_frees: u64,
    /// 自初始化以来分配失败（[`AllocError`]）的总次数。
    pub failed_allocations: u64,
    /// 自初始化以来非法释放（[`FreeError`]）的总次数。
    pub failed_frees: u64,
}

impl Stats {
    /// 内部碎片占“已分配块实际空间”的比例 `[0, 1)`；当前没有任何分配时为 0。
    pub fn internal_fragmentation_ratio(&self) -> f64 {
        if self.allocated_bytes == 0 {
            0.0
        } else {
            self.internal_fragmentation as f64 / self.allocated_bytes as f64
        }
    }

    /// 外部碎片的常用度量：`1 - largest_free_block / free_bytes`，取值 `[0, 1)`。
    ///
    /// 越接近 1 表示空闲空间越分散（即便总空闲量足够，也可能无法满足大请求）。
    /// 没有任何空闲空间时为 0。
    pub fn external_fragmentation_ratio(&self) -> f64 {
        if self.free_bytes == 0 {
            0.0
        } else {
            1.0 - self.largest_free_block as f64 / self.free_bytes as f64
        }
    }
}

/// 固定大小内存池上的 Buddy Memory Allocator 模拟器。
///
/// 所有状态都是进程本地的普通数据结构，见[模块文档](self)。
#[derive(Debug, Clone)]
pub struct BuddyAllocator {
    /// 模拟内存池的基地址（模拟地址空间中的起始值）。
    base: usize,
    /// 内存池总容量，2 的幂。
    pool_size: usize,
    /// 最小块大小，2 的幂。
    min_block_size: usize,
    /// 最大 order（整块池对应的 order）。
    max_order: usize,
    /// `free_lists[k]`：order `k` 的所有空闲块**起始偏移**集合。
    ///
    /// 用 [`BTreeSet`] 而非裸链表：取块/放块/伙伴判定都是 O(log n)，
    /// 且遍历顺序确定，便于测试与复现。
    free_lists: Vec<BTreeSet<usize>>,
    /// 已分配块：起始偏移 -> 记录。
    allocated: BTreeMap<usize, Allocation>,
    /// 所有“当前对应一个空闲块起始地址”的偏移集合，用于精确识别 double free。
    freed_starts: BTreeSet<usize>,
    /// 累计统计计数器。
    total_allocations: u64,
    total_frees: u64,
    failed_allocations: u64,
    failed_frees: u64,
}

impl BuddyAllocator {
    /// 创建一个分配器。
    ///
    /// # 参数
    /// * `base`：模拟地址空间中内存池的起始地址（可取任意值，内部按相对偏移计算）。
    /// * `pool_size`：内存池总字节数，必须是 2 的幂。
    /// * `min_block_size`：最小块字节数，必须是 2 的幂且 `<= pool_size`。
    ///
    /// # Panics
    /// 参数不满足上述约束时 panic（属于编程错误，构造只发生一次，直接 fail-fast）。
    pub fn new(base: usize, pool_size: usize, min_block_size: usize) -> Self {
        assert!(pool_size >= 1, "pool_size must be >= 1");
        assert!(
            pool_size.is_power_of_two(),
            "pool_size must be a power of two, got {pool_size}"
        );
        assert!(min_block_size >= 1, "min_block_size must be >= 1");
        assert!(
            min_block_size.is_power_of_two(),
            "min_block_size must be a power of two, got {min_block_size}"
        );
        assert!(
            min_block_size <= pool_size,
            "min_block_size ({min_block_size}) must not exceed pool_size ({pool_size})"
        );

        // max_order 满足 pool_size == min_block_size << max_order
        let max_order = (pool_size / min_block_size).trailing_zeros() as usize;
        let mut free_lists = vec![BTreeSet::new(); max_order + 1];
        free_lists[max_order].insert(0);

        let mut freed_starts = BTreeSet::new();
        freed_starts.insert(0);

        Self {
            base,
            pool_size,
            min_block_size,
            max_order,
            free_lists,
            allocated: BTreeMap::new(),
            freed_starts,
            total_allocations: 0,
            total_frees: 0,
            failed_frees: 0,
            failed_allocations: 0,
        }
    }

    /// 内存池基地址（模拟地址）。
    pub fn base(&self) -> usize {
        self.base
    }

    /// 内存池总容量（字节）。
    pub fn pool_size(&self) -> usize {
        self.pool_size
    }

    /// 最小块大小（字节）。
    pub fn min_block_size(&self) -> usize {
        self.min_block_size
    }

    /// 最大 order；order `k` 的块大小为 `min_block_size * 2^k`。
    pub fn max_order(&self) -> usize {
        self.max_order
    }

    /// order `k` 对应的块字节数。
    pub fn block_size(&self, order: usize) -> usize {
        assert!(order <= self.max_order, "order {order} out of range");
        self.min_block_size << order
    }

    /// 容纳 `size` 字节所需的最小 order：满足 `min_block_size * 2^k >= size`。
    fn required_order(&self, size: usize) -> usize {
        // ceil(size / min_block_size) 向上取整到 2 的幂后取 log2。
        let units = size.div_ceil(self.min_block_size);
        ceil_log2(units) as usize
    }

    /// 分配 `size` 字节，成功返回块起始的**模拟地址**（`base + offset`）。
    ///
    /// 采用最小适配：从能容纳 `size` 的最小 order 开始；该 order 无空闲块时，
    /// 沿更高 order 寻找一个空闲块并递归对半拆分，直到目标 order。
    pub fn allocate(&mut self, size: usize) -> Result<usize, AllocError> {
        if size == 0 {
            self.failed_allocations += 1;
            return Err(AllocError::ZeroSize);
        }
        if size > self.pool_size {
            self.failed_allocations += 1;
            return Err(AllocError::RequestTooLarge {
                requested: size,
                pool_size: self.pool_size,
            });
        }

        let target = self.required_order(size);

        // 1) 在 [target, max_order] 中找一个有空闲块的最低 order。
        let mut source_order = target;
        while source_order <= self.max_order && self.free_lists[source_order].is_empty() {
            source_order += 1;
        }
        if source_order > self.max_order {
            self.failed_allocations += 1;
            return Err(AllocError::OutOfMemory { requested: size });
        }

        // 2) 取出该 order 中偏移最小的空闲块（确定性的选择）。
        let block = self.free_lists[source_order]
            .pop_first()
            .expect("non-empty");
        self.freed_starts.remove(&block);

        // 3) 递归对半拆分：每次把当前块分成两个 order-1 的伙伴，
        //    右半部分挂回空闲链表，左半部分继续向下拆。
        let current = block;
        let mut order = source_order;
        while order > target {
            order -= 1;
            let buddy = current + self.block_size(order);
            self.free_lists[order].insert(buddy);
            self.freed_starts.insert(buddy);
        }

        // 4) current 现在是 order == target 的块，标记为已分配。
        self.allocated.insert(
            current,
            Allocation {
                order: target,
                requested: size,
            },
        );
        self.total_allocations += 1;
        Ok(self.base + current)
    }

    /// 释放此前由 [`allocate`](Self::allocate) 返回的块起始地址。
    ///
    /// 校验通过后归还块，并沿伙伴关系递归合并：伙伴在同一 order 空闲时
    /// 合并成 order+1 的块继续，直到无法合并或到达 `max_order`。
    pub fn free(&mut self, address: usize) -> Result<(), FreeError> {
        // --- 校验阶段（不修改任何状态） ---

        // 1) 超出内存池范围。
        if address < self.base || address >= self.base + self.pool_size {
            self.failed_frees += 1;
            return Err(FreeError::OutOfBounds { address });
        }
        let offset = address - self.base;

        // 2) 未按最小块对齐：不可能是任何块的起始地址。
        if !offset.is_multiple_of(self.min_block_size) {
            self.failed_frees += 1;
            return Err(FreeError::Misaligned { address });
        }

        // 3) 落在某个已分配块内部（非起始地址）。
        if let Some((&start, alloc)) = self.find_allocated_owner(offset) {
            if start != offset {
                self.failed_frees += 1;
                return Err(FreeError::NotBlockStart {
                    address,
                    block_start: self.base + start,
                });
            }
            // 正常释放路径：start == offset，下面执行。
            let alloc = *alloc;
            return self.release_block(offset, alloc);
        }

        // 4) 不是已分配块。freed_starts 记录了所有当前空闲块起始地址：
        //    命中即说明该地址之前已释放过 -> double free。
        if self.freed_starts.contains(&offset) {
            self.failed_frees += 1;
            return Err(FreeError::DoubleFree { address });
        }

        // 5) 其余对齐地址：处于某个空闲块内部，或根本不对应任何块起始。
        self.failed_frees += 1;
        Err(FreeError::InvalidAddress { address })
    }

    /// 找到包含偏移 `offset` 的已分配块（若有）。
    fn find_allocated_owner(&self, offset: usize) -> Option<(&usize, &Allocation)> {
        // 起始地址 <= offset 的最后一个已分配块；再判断其上界是否覆盖 offset。
        self.allocated
            .range(..=offset)
            .next_back()
            .filter(|(&start, alloc)| offset < start + self.block_size(alloc.order))
    }

    /// 真正执行归还与递归伙伴合并（offset 已确认是 alloc 描述的已分配块起点）。
    fn release_block(&mut self, offset: usize, alloc: Allocation) -> Result<(), FreeError> {
        self.allocated.remove(&offset);
        self.total_frees += 1;

        let mut current = offset;
        let mut order = alloc.order;

        // 递归合并：伙伴在同一 order 也空闲时，二者合并为 order+1 块继续向上。
        while order < self.max_order {
            let block_size = self.block_size(order);
            // 相对偏移的伙伴地址：异或一个块大小。
            // 因为偏移从 0 开始且所有块按 block_size 对齐，异或即等价于
            // “偶数号伙伴 +size / 奇数号伙伴 -size”。
            let buddy = current ^ block_size;

            if self.free_lists[order].remove(&buddy) {
                self.freed_starts.remove(&buddy);
                current = current.min(buddy);
                order += 1;
            } else {
                break;
            }
        }

        let inserted = self.free_lists[order].insert(current);
        debug_assert!(inserted, "free list must not already contain {current}");
        self.freed_starts.insert(current);

        Ok(())
    }

    /// 当前统计快照，包含分配/空闲字节、内部碎片、各 order 空闲块数量等。
    pub fn stats(&self) -> Stats {
        let mut allocated_blocks = 0usize;
        let mut allocated_bytes = 0usize;
        let mut requested_bytes = 0usize;
        for alloc in self.allocated.values() {
            allocated_blocks += 1;
            allocated_bytes += self.block_size(alloc.order);
            requested_bytes += alloc.requested;
        }

        let mut free_blocks = 0usize;
        let mut largest_free_block = 0usize;
        let mut free_blocks_per_order = Vec::with_capacity(self.max_order + 1);
        for (order, list) in self.free_lists.iter().enumerate() {
            free_blocks += list.len();
            free_blocks_per_order.push(list.len());
            if !list.is_empty() {
                largest_free_block = largest_free_block.max(self.block_size(order));
            }
        }

        let free_bytes = self.pool_size - allocated_bytes;

        Stats {
            pool_size: self.pool_size,
            allocated_blocks,
            free_blocks,
            allocated_bytes,
            free_bytes,
            internal_fragmentation: allocated_bytes - requested_bytes,
            requested_bytes,
            free_blocks_per_order,
            largest_free_block,
            total_allocations: self.total_allocations,
            total_frees: self.total_frees,
            failed_allocations: self.failed_allocations,
            failed_frees: self.failed_frees,
        }
    }

    /// 检查模拟器内部不变量。所有公开操作返回后都应成立；属性测试每一步都会调用。
    ///
    /// 检查内容：
    /// 1. 空间守恒：所有已分配块与空闲块大小之和恰为 `pool_size`；
    /// 2. 所有块起始/大小对齐合法；
    /// 3. 块与块之间互不重叠，且完整覆盖 `[0, pool_size)`；
    /// 4. 空闲块与已分配块不相交；
    /// 5. 任意两个同 order 的相邻空闲伙伴不能同时存在（说明有未完成的合并）；
    /// 6. `freed_starts` 恰好等于所有空闲块起始地址的集合。
    ///
    /// 返回 `Err(描述)` 表示不变量被破坏（应当永远不会发生）。
    pub fn check_invariants(&self) -> Result<(), String> {
        // 收集所有块 (start, size)：已分配块 + 各 order 空闲块。
        let mut segments: Vec<(usize, usize)> = Vec::new();
        for (&start, alloc) in &self.allocated {
            let size = self.block_size(alloc.order);
            if start % self.min_block_size != 0 || start.checked_add(size) != Some(start + size) {
                return Err(format!("allocated block at {start} misaligned/overflow"));
            }
            segments.push((start, size));
        }
        for (order, list) in self.free_lists.iter().enumerate() {
            let size = self.block_size(order);
            for &start in list {
                if start % size != 0 {
                    return Err(format!(
                        "free block at {start} not aligned to its size {size}"
                    ));
                }
                segments.push((start, size));
            }
        }

        segments.sort_unstable();

        // 覆盖与不重叠：段必须首尾相接铺满整个池子。
        let mut cursor = 0usize;
        for (start, size) in segments {
            if start != cursor {
                return Err(format!(
                    "coverage gap/overlap: expected block at {cursor}, found one at {start}"
                ));
            }
            cursor += size;
        }
        if cursor != self.pool_size {
            return Err(format!(
                "blocks cover {cursor} bytes but pool is {} bytes",
                self.pool_size
            ));
        }

        // 伙伴合并极大性：同 order 空闲块中，不能有一对相邻伙伴同时空闲。
        for (order, list) in self.free_lists.iter().enumerate() {
            if order == self.max_order {
                break; // 整块池没有伙伴可合并
            }
            let size = self.block_size(order);
            for &start in list {
                let buddy = start ^ size;
                if buddy != start && list.contains(&buddy) {
                    return Err(format!(
                        "free buddies at {start} and {buddy} (order {order}) were not coalesced"
                    ));
                }
            }
        }

        // freed_starts 与空闲块集合严格一致。
        let free_union: BTreeSet<usize> = self
            .free_lists
            .iter()
            .flat_map(|l| l.iter().copied())
            .collect();
        if free_union != self.freed_starts {
            return Err("freed_starts does not match the union of free lists".to_string());
        }

        Ok(())
    }
}

/// 返回最小的 `k` 使 `2^k >= n`（要求 `n >= 1`）。
///
/// 等价于 `n.next_power_of_two().ilog2()`。
fn ceil_log2(n: usize) -> u32 {
    debug_assert!(n >= 1);
    n.next_power_of_two().trailing_zeros()
}
