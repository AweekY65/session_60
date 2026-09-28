//! 随机操作序列支持：确定性伪随机数发生器 + 随机 allocate/free 负载驱动器。
//!
//! 随机数发生器（xorshift64*）在工程内实现，不依赖 `rand` 等外部 crate，
//! 因此“支持随机操作序列”这一能力同样完全在本地进程内完成且结果可复现：
//! 同一个种子必然产生同一个操作序列。

use crate::{AllocError, BuddyAllocator};

/// 基于 xorshift64* 的确定性伪随机数发生器（Vigna, 2014）。
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// 用任意种子构造；种子 0 会被映射为固定的非零值（xorshift 不允许全 0 状态）。
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    /// 生成下一个 64 位随机数。
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// 均匀返回 `[0, bound)` 内的值；`bound` 必须非零。
    pub fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// 以概率 `numerator/denominator` 返回 true。
    pub fn chance(&mut self, numerator: u32, denominator: u32) -> bool {
        debug_assert!(numerator <= denominator && denominator > 0);
        (self.next_u64() % denominator as u64) < numerator as u64
    }
}

/// 一次随机操作的记录，便于回放与调试。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// 请求分配 `size` 字节。
    Alloc { size: usize },
    /// 释放当前仍然存活的第 `index` 个分配（指向 [`RunReport::live`] 的逻辑槽位语义见驱动）。
    Free { live_index: usize },
}

/// 单个操作的执行结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpResult {
    /// 分配成功，返回模拟地址。
    Allocated(usize),
    /// 分配失败（例如内存耗尽）。
    AllocFailed(AllocError),
    /// 释放成功。
    Freed,
    /// 没有存活块可释放（该次 free 被跳过，不计入非法释放）。
    NothingToFree,
}

/// 一次随机负载运行的汇总。
#[derive(Debug, Clone)]
pub struct RunReport {
    /// 执行的操作总数。
    pub ops: usize,
    /// 成功分配次数。
    pub allocations: usize,
    /// 成功释放次数。
    pub frees: usize,
    /// 分配失败次数（内存耗尽是正常现象）。
    pub alloc_failures: usize,
    /// 运行结束时仍然存活的块地址。
    pub live: Vec<usize>,
}

impl BuddyAllocator {
    /// 在当前分配器上执行一段确定性的随机 allocate/free 操作序列。
    ///
    /// # 参数
    /// * `rng`：随机源（同一种子 => 同一序列，可复现）；
    /// * `ops`：操作次数；
    /// * `max_size`：每次分配请求在 `[1, max_size]` 内均匀取值；
    /// * `free_prob_percent`：当存在存活块时选择 free 的概率（0..=100）。
    ///
    /// 驱动只会释放“当前存活”的块，因此不会产生非法释放
    /// （非法路径由专门的单元测试覆盖）。返回每个操作的结果与汇总信息。
    pub fn run_random_workload(
        &mut self,
        rng: &mut Rng,
        ops: usize,
        max_size: usize,
        free_prob_percent: u32,
    ) -> (Vec<OpResult>, RunReport) {
        assert!(max_size >= 1, "max_size must be >= 1");
        assert!(free_prob_percent <= 100);

        let mut results = Vec::with_capacity(ops);
        let mut live: Vec<usize> = Vec::new();
        let mut allocations = 0usize;
        let mut frees = 0usize;
        let mut alloc_failures = 0usize;

        for _ in 0..ops {
            let choose_free = !live.is_empty() && rng.chance(free_prob_percent, 100);
            if choose_free {
                let idx = rng.below(live.len());
                let address = live.swap_remove(idx);
                match self.free(address) {
                    Ok(()) => {
                        frees += 1;
                        results.push(OpResult::Freed);
                    }
                    // 驱动只释放存活块，正常情况下不可能失败；把错误透传给调用方，
                    // 用 panic 暴露任何意外状态破坏。
                    Err(e) => panic!("workload free of live block {address:#x} failed: {e}"),
                }
            } else {
                let size = 1 + rng.below(max_size);
                match self.allocate(size) {
                    Ok(address) => {
                        allocations += 1;
                        live.push(address);
                        results.push(OpResult::Allocated(address));
                    }
                    Err(e) => {
                        alloc_failures += 1;
                        results.push(OpResult::AllocFailed(e));
                    }
                }
            }
        }

        let report = RunReport {
            ops,
            allocations,
            frees,
            alloc_failures,
            live,
        };
        (results, report)
    }
}
