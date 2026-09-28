//! Property-based tests（不依赖 proptest/quickcheck 等外部 crate）：
//!
//! 用工程内的确定性 RNG 跑大量随机 allocate/free 序列，并在**每一步之后**
//! 对照一个独立的“字节占用参考模型”校验不变量：
//!
//! 1. 已分配块两两**绝不重叠**；
//! 2. 空间守恒：已分配字节 + 空闲字节 == pool_size，且参考位图被占用的字节
//!    恰好都属于某个已分配块（不多不少）；
//! 3. 任意两个同 order 的相邻空闲伙伴不能共存（合并始终极大化）；
//! 4. 成功分配返回的块大小 `>= size` 且严格位于池内；
//! 5. 成功 free 的区间与参考位图在操作前完全一致；
//! 6. 全部释放后池子回到“单个 max-order 空闲块”的初始形态。
//!
//! 用固定种子矩阵复现，而不是随机种子，保证 `cargo test` 结果确定可复现。

use buddy_allocator::{BuddyAllocator, Rng};

/// 独立参考模型：逐字节的占用标记（0=空闲，1=属于一个已分配块）。
struct ReferenceModel {
    bitmap: Vec<u8>,
}

impl ReferenceModel {
    fn new(pool_size: usize) -> Self {
        Self {
            bitmap: vec![0u8; pool_size],
        }
    }

    /// 占用 [start, start+len)，要求之前全部为 0（即绝不重叠）。
    fn occupy(&mut self, start: usize, len: usize) {
        for b in &mut self.bitmap[start..start + len] {
            assert_eq!(*b, 0, "overlap: byte inside another allocation");
            *b = 1;
        }
    }

    /// 释放 [start, start+len)，要求之前全部为 1。
    fn release(&mut self, start: usize, len: usize) {
        for b in &mut self.bitmap[start..start + len] {
            assert_eq!(*b, 1, "freeing byte that was not allocated");
            *b = 0;
        }
    }

    fn used_bytes(&self) -> usize {
        self.bitmap.iter().map(|&b| b as usize).sum()
    }
}

/// 测试模型独立计算“请求 size 应得到的块大小”，
/// 不读取分配器私有状态：ceil 到最小块，再向上取整到 2 的幂。
fn expected_block_size(size: usize, min_block: usize) -> usize {
    let units = size.div_ceil(min_block);
    min_block * units.next_power_of_two()
}

/// 单次属性测试参数。
struct Cfg {
    pool_size: usize,
    min_block: usize,
    ops: usize,
    max_request: usize,
    free_prob: u32,
}

fn run_property(cfg: &Cfg, seed: u64) {
    let base = 0x1000_0000usize;
    let mut alloc = BuddyAllocator::new(base, cfg.pool_size, cfg.min_block);
    let mut reference = ReferenceModel::new(cfg.pool_size);
    let mut rng = Rng::new(seed);

    // 存活块：(偏移, 实际块大小)。块大小由测试侧独立计算，
    // 因此这里同时验证了分配器的 order 选择与最小适配语义。
    let mut live: Vec<(usize, usize)> = Vec::new();

    for step in 0..cfg.ops {
        let do_free = !live.is_empty() && rng.chance(cfg.free_prob, 100);
        if do_free {
            let idx = rng.below(live.len());
            let (offset, block_len) = live.swap_remove(idx);

            // 操作前：该块区间在参考模型中必须全部占用。
            assert_eq!(
                reference.bitmap[offset..offset + block_len],
                vec![1u8; block_len],
                "seed {seed} step {step}: free region disagrees with reference"
            );

            alloc
                .free(base + offset)
                .expect("free of a live block must succeed");
            reference.release(offset, block_len);
        } else {
            let size = 1 + rng.below(cfg.max_request);
            match alloc.allocate(size) {
                Ok(address) => {
                    let offset = address - base;
                    let block_len = expected_block_size(size, cfg.min_block);

                    assert!(block_len >= size, "block smaller than request");
                    assert!(offset + block_len <= cfg.pool_size, "block out of pool");
                    // 逐字节占用；与任何已有块重叠时会在这里 panic。
                    reference.occupy(offset, block_len);
                    live.push((offset, block_len));
                }
                Err(_) => {
                    // 分配失败（内存耗尽）是合法结果；下方对账会确认状态一致。
                }
            }
        }

        // 每一步之后：分配器内部不变量 + 与参考模型逐项对账。
        alloc
            .check_invariants()
            .unwrap_or_else(|e| panic!("seed {seed} step {step}: invariants broken: {e}"));

        let stats = alloc.stats();
        // 空间守恒。
        assert_eq!(
            stats.allocated_bytes + stats.free_bytes,
            cfg.pool_size,
            "seed {seed} step {step}: space not conserved"
        );
        // 参考位图占用量必须等于分配器记账的已分配字节（双重独立账）。
        assert_eq!(
            reference.used_bytes(),
            stats.allocated_bytes,
            "seed {seed} step {step}: reference bitmap disagrees with allocated_bytes"
        );
        // 区间级不重叠检查。
        let mut intervals = live.clone();
        intervals.sort_unstable();
        for w in intervals.windows(2) {
            assert!(
                w[0].0 + w[0].1 <= w[1].0,
                "seed {seed} step {step}: live blocks overlap"
            );
        }
        // 统计自洽：各 order 空闲块字节之和 == free_bytes。
        let free_sum: usize = stats
            .free_blocks_per_order
            .iter()
            .enumerate()
            .map(|(o, n)| n * (cfg.min_block << o))
            .sum();
        assert_eq!(free_sum, stats.free_bytes, "per-order free bytes mismatch");
    }

    // 收尾：全部释放后回到初始形态。
    for (offset, block_len) in live.drain(..) {
        alloc.free(base + offset).expect("final free succeeds");
        reference.release(offset, block_len);
    }
    alloc.check_invariants().expect("invariants after draining");
    let stats = alloc.stats();
    assert_eq!(stats.allocated_bytes, 0);
    assert_eq!(stats.free_bytes, cfg.pool_size);
    assert_eq!(stats.requested_bytes, 0);
    assert_eq!(stats.free_blocks, 1);
    assert_eq!(stats.largest_free_block, cfg.pool_size);
    assert_eq!(reference.used_bytes(), 0);
}

/// 小规模：多种子、覆盖全部 order 边界，频繁分配/释放。
#[test]
fn prop_random_small_pool_many_seeds() {
    let cfg = Cfg {
        pool_size: 1024,
        min_block: 1,
        ops: 3_000,
        max_request: 256,
        free_prob: 45,
    };
    for seed in 1..=40u64 {
        run_property(&cfg, seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    }
}

/// 中规模：最小块较大，内部碎片场景更多。
#[test]
fn prop_random_medium_pool_coarse_blocks() {
    let cfg = Cfg {
        pool_size: 1 << 14, // 16 KiB
        min_block: 64,
        ops: 5_000,
        max_request: 2048,
        free_prob: 50,
    };
    for seed in 1..=20u64 {
        run_property(&cfg, seed * 7919 + 11);
    }
}

/// 大规模：上万次操作、数千个最小块，验证大规模随机序列下
/// 不重叠、空间守恒与递归合并。
#[test]
fn prop_random_large_scale() {
    let cfg = Cfg {
        pool_size: 1 << 16, // 64 KiB, 4096 个最小块
        min_block: 16,
        ops: 20_000,
        max_request: 4096,
        free_prob: 55,
    };
    for &seed in &[1u64, 42, 20260928, 0xDEAD_BEEF, u64::MAX] {
        run_property(&cfg, seed);
    }
}

/// 极限压力：请求大小恒为 1，把池子打满再随机释放，反复多个波次。
#[test]
fn prop_tiny_requests_fill_and_drain() {
    let cfg = Cfg {
        pool_size: 2048,
        min_block: 1,
        ops: 8_000,
        max_request: 1,
        free_prob: 48,
    };
    for seed in 1..=15u64 {
        run_property(&cfg, seed ^ 0xA5A5_5A5A);
    }
}

/// 极端参数：池与最小块相同（只有一个 order，没有拆分/合并可言）。
#[test]
fn prop_single_order_pool() {
    let cfg = Cfg {
        pool_size: 128,
        min_block: 128,
        ops: 2_000,
        max_request: 128,
        free_prob: 50,
    };
    for seed in 1..=10u64 {
        run_property(&cfg, seed * 131 + 7);
    }
}

/// 请求大小恒接近池容量：大块分配/失败/释放的交替。
#[test]
fn prop_near_full_requests() {
    let cfg = Cfg {
        pool_size: 4096,
        min_block: 1,
        ops: 3_000,
        max_request: 4096,
        free_prob: 50,
    };
    for &seed in &[3u64, 17, 99991] {
        run_property(&cfg, seed);
    }
}
