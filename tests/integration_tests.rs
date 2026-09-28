//! Integration tests: targeted scenarios plus a large-scale deterministic
//! randomized stress test. The PRNG is a tiny self-contained xorshift so the
//! "random" sequences are fully reproducible.

use buddy_allocator::{
    AllocError, Allocation, BuddyAllocator, FreeError, InitError,
};

/// Deterministic 64-bit xorshift RNG (xorshift64*).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform value in `0..n` (n > 0) without modulo bias concerns mattering
    /// for test purposes.
    fn below(&mut self, n: u64) -> usize {
        (self.next_u64() % n) as usize
    }
}

#[test]
fn new_validates_arguments() {
    assert_eq!(
        BuddyAllocator::new(0, 16).unwrap_err(),
        InitError::BadPoolSize
    );
    assert_eq!(
        BuddyAllocator::new(1023, 16).unwrap_err(),
        InitError::BadPoolSize
    );
    assert_eq!(
        BuddyAllocator::new(1024, 0).unwrap_err(),
        InitError::BadMinBlockSize
    );
    assert_eq!(
        BuddyAllocator::new(1024, 17).unwrap_err(),
        InitError::BadMinBlockSize
    );
    let a = BuddyAllocator::new(1024, 1024).unwrap();
    assert_eq!(a.max_order(), 0);
    assert_eq!(a.stats().free_blocks_per_order.len(), 1);
}

#[test]
fn minimum_sized_request_uses_order_zero() {
    let mut a = BuddyAllocator::new(128, 32).unwrap();
    let x = a.allocate(1).unwrap();
    assert_eq!(x.order, 0);
    assert_eq!(x.size, 1);
    assert_eq!(x.addr, 0);
    // 128 -> 64+64 -> 32+32 leaves free blocks at orders 0..2.
    let free = a.free_blocks();
    assert_eq!(free, vec![(32, 0), (64, 1)]);
    a.verify().unwrap();
    a.free(0).unwrap();
    assert_eq!(a.free_blocks(), vec![(0, 2)]);
}

#[test]
fn maximum_sized_request_consumes_entire_pool() {
    let mut a = BuddyAllocator::new(1024, 16).unwrap();
    let x = a.allocate(1024).unwrap();
    assert_eq!(x.order, a.max_order());
    assert_eq!(x.addr, 0);
    assert_eq!(a.stats().free_bytes, 0);
    assert_eq!(a.allocate(1).unwrap_err(), AllocError::OutOfMemory);
    a.free(0).unwrap();
    assert_eq!(a.stats().free_bytes, 1024);
    a.verify().unwrap();
}

#[test]
fn exact_order_boundary_requests_round_up_to_matching_order() {
    let mut a = BuddyAllocator::new(1024, 16).unwrap();
    // 16 -> order 0, 17 -> order 1, 32 -> order 1, 33 -> order 2 ...
    assert_eq!(a.allocate(16).unwrap().order, 0);
    assert_eq!(a.allocate(17).unwrap().order, 1);
    assert_eq!(a.allocate(32).unwrap().order, 1);
    assert_eq!(a.allocate(33).unwrap().order, 2);
    assert_eq!(a.allocate(256).unwrap().order, 4);
    assert_eq!(a.allocate(257).unwrap().order, 5);
    a.verify().unwrap();
}

#[test]
fn continuous_splits_produce_expected_free_blocks() {
    // Pool 1024 (16-byte min). One order-0 alloc forces splits through six
    // orders; each split leaves exactly one free block of the child order.
    let mut a = BuddyAllocator::new(1024, 16).unwrap();
    let x = a.allocate(1).unwrap();
    assert_eq!(x.order, 0);
    for order in 0..=5usize {
        let expected = a.block_size(order);
        assert_eq!(
            a.stats().free_blocks_per_order[order].count,
            1,
            "expected one leftover free block of size {expected}"
        );
    }
    assert_eq!(a.stats().free_blocks_per_order[6].count, 0);
    a.verify().unwrap();
}

#[test]
fn recursive_merge_rebuilds_all_orders() {
    // Fill the pool with 8 order-0 blocks, then free every left/right pair in
    // an order that drives coalescing up each level one by one.
    let mut a = BuddyAllocator::new(128, 16).unwrap();
    let blocks: Vec<Allocation> = (0..8).map(|_| a.allocate(16).unwrap()).collect();

    a.free(blocks[0].addr).unwrap();
    a.free(blocks[1].addr).unwrap(); // merge order 0 -> 1 at [0,32)
    assert_eq!(a.free_blocks(), vec![(0, 1)]);

    a.free(blocks[2].addr).unwrap();
    a.free(blocks[3].addr).unwrap(); // [32,64), merges with [0,32)
    assert_eq!(a.free_blocks(), vec![(0, 2)]);

    a.free(blocks[4].addr).unwrap();
    a.free(blocks[5].addr).unwrap();
    a.free(blocks[6].addr).unwrap();
    a.free(blocks[7].addr).unwrap();
    assert_eq!(a.free_blocks(), vec![(0, 3)]);
    a.verify().unwrap();
}

#[test]
fn fragmented_free_space_still_satisfies_coalesced_requests() {
    // External fragmentation is repaired automatically: free two buddies and a
    // request spanning both must succeed.
    let mut a = BuddyAllocator::new(64, 16).unwrap();
    let p = a.allocate(16).unwrap();
    let q = a.allocate(16).unwrap();
    let r = a.allocate(16).unwrap();
    a.free(p.addr).unwrap();
    a.free(q.addr).unwrap(); // [0,32) coalesced
    let big = a.allocate(32).unwrap();
    assert_eq!(big.addr, 0);
    assert_eq!(big.order, 1);
    a.free(r.addr).unwrap();
    a.free(big.addr).unwrap();
    assert_eq!(a.free_blocks(), vec![(0, 2)]);
}

#[test]
fn memory_exhaustion_and_recovery() {
    let mut a = BuddyAllocator::new(64, 16).unwrap();
    let mut blocks = Vec::new();
    for _ in 0..4 {
        blocks.push(a.allocate(16).unwrap());
    }
    assert_eq!(a.allocate(1).unwrap_err(), AllocError::OutOfMemory);
    assert_eq!(a.stats().free_bytes, 0);

    // Free half; only half-sized requests work again.
    a.free(blocks[2].addr).unwrap();
    a.free(blocks[3].addr).unwrap();
    assert_eq!(a.allocate(33).unwrap_err(), AllocError::OutOfMemory);
    let recovered = a.allocate(32).unwrap();
    assert_eq!(recovered.addr, 32);
    blocks.truncate(2);
    blocks.push(recovered);
    for b in blocks {
        a.free(b.addr).unwrap();
    }
    assert_eq!(a.stats().free_bytes, 64);
    a.verify().unwrap();
}

#[test]
fn every_invalid_free_variant_is_detected() {
    let mut a = BuddyAllocator::new(64, 16).unwrap();

    assert_eq!(a.free(64), Err(FreeError::AddressOutOfRange));
    assert_eq!(a.free(1000), Err(FreeError::AddressOutOfRange));
    assert_eq!(a.free(3), Err(FreeError::NotBlockAligned));

    let big = a.allocate(32).unwrap();
    assert_eq!(a.free(big.addr + 1), Err(FreeError::NotBlockStart));
    assert_eq!(a.free(big.addr + 16), Err(FreeError::NotBlockStart));

    // Split-created free block the caller never owned.
    assert_eq!(a.free(32), Err(FreeError::NotAllocated));

    a.free(big.addr).unwrap();
    assert_eq!(a.free(big.addr), Err(FreeError::DoubleFree));
    a.verify().unwrap();
}

#[test]
fn statistics_are_consistent() {
    let mut a = BuddyAllocator::new(1024, 16).unwrap();
    let mut total_requested = 0usize;
    let mut total_granted = 0usize;
    for &size in &[1usize, 100, 500, 17] {
        if let Ok(x) = a.allocate(size) {
            total_requested += size;
            total_granted += a.block_size(x.order);
        }
    }
    let s = a.stats();
    assert_eq!(s.total_bytes, 1024);
    assert_eq!(s.allocated_bytes, total_requested);
    assert_eq!(s.used_bytes, total_granted);
    assert_eq!(s.free_bytes, 1024 - total_granted);
    assert_eq!(
        s.internal_fragmentation_bytes,
        total_granted - total_requested
    );
    let per_order_free: usize = s
        .free_blocks_per_order
        .iter()
        .map(|fc| fc.count * fc.block_size)
        .sum();
    assert_eq!(per_order_free, s.free_bytes);
    a.verify().unwrap();
}

// ---------------------------------------------------------------------------
// Large-scale randomized stress tests
// ---------------------------------------------------------------------------

/// Run one deterministic random workload and assert invariants continuously.
fn run_stress(pool_size: usize, min_block: usize, steps: usize, seed: u64) {
    let mut alloc = BuddyAllocator::new(pool_size, min_block).unwrap();
    let mut rng = Rng::new(seed);

    // Live allocations tracked independently.
    let mut live: Vec<Allocation> = Vec::new();

    for step in 0..steps {
        match rng.below(100) {
            // 55% allocate, 40% free a random live block, 5% bogus free.
            0..=54 => {
                // Weight size distribution: mostly small, occasional large.
                let size = match rng.below(10) {
                    0..=5 => 1 + rng.below(min_block as u64 * 2),
                    6..=8 => 1 + rng.below(pool_size as u64 / 4),
                    _ => 1 + rng.below(pool_size as u64),
                };
                match alloc.allocate(size) {
                    Ok(a) => {
                        // Independence from every existing block.
                        let new_end = a.addr + alloc.block_size(a.order);
                        for b in &live {
                            let b_end = b.addr + alloc.block_size(b.order);
                            assert!(
                                a.addr >= b_end || new_end <= b.addr,
                                "step {step}: overlap between {}..{new_end} and {}..{b_end}",
                                a.addr,
                                b.addr
                            );
                        }
                        live.push(a);
                    }
                    Err(AllocError::ZeroSize | AllocError::RequestTooLarge) => {
                        panic!("stress workload generated out-of-contract size {size}")
                    }
                    Err(AllocError::OutOfMemory) => {
                        // Claimed OOM must agree with the free lists: no block
                        // of a sufficient order may exist.
                        let target = {
                            let units = size.div_ceil(min_block);
                            (units.next_power_of_two()).trailing_zeros() as usize
                        };
                        let s = alloc.stats();
                        assert!(
                            s.free_blocks_per_order[target..]
                                .iter()
                                .all(|fc| fc.count == 0),
                            "step {step}: spurious OOM for size {size}"
                        );
                    }
                }
            }
            55..=94 => {
                if !live.is_empty() {
                    let i = rng.below(live.len() as u64);
                    let a = live.swap_remove(i);
                    alloc
                        .free(a.addr)
                        .unwrap_or_else(|e| panic!("step {step}: free of {a:?} failed: {e}"));
                }
            }
            _ => {
                // Invalid free attempts must always be rejected.
                let addr = rng.below((pool_size as u64) + 16);
                if let Ok(()) = alloc.free(addr) {
                    // Only legal if it actually named a live block; sync model.
                    let pos = live
                        .iter()
                        .position(|a| a.addr == addr)
                        .expect("free succeeded for an address not in the live model");
                    live.swap_remove(pos);
                }
            }
        }

        // Invariant check every step is affordable at these pool sizes.
        alloc.verify().unwrap_or_else(|e| panic!("step {step}: {e}"));

        // Statistics must agree with the independent model.
        let s = alloc.stats();
        let model_requested: usize = live.iter().map(|a| a.size).sum();
        let model_granted: usize = live
            .iter()
            .map(|a| alloc.block_size(a.order))
            .sum();
        assert_eq!(s.allocated_bytes, model_requested, "step {step}");
        assert_eq!(s.used_bytes, model_granted, "step {step}");
        assert_eq!(s.free_bytes + s.used_bytes, pool_size, "step {step}");
    }

    // Drain everything: coalescing must restore the intact pool.
    while let Some(a) = live.pop() {
        alloc
            .free(a.addr)
            .unwrap_or_else(|e| panic!("drain free of {a:?} failed: {e}"));
        alloc.verify().unwrap();
    }
    let s = alloc.stats();
    assert_eq!(s.free_bytes, pool_size);
    assert_eq!(s.used_bytes, 0);
    assert_eq!(s.allocated_bytes, 0);
    assert_eq!(s.free_blocks_per_order[alloc.max_order()].count, 1);
}

#[test]
fn stress_small_pool_fine_granularity() {
    run_stress(256, 1, 20_000, 0xDEAD_BEEF);
}

#[test]
fn stress_medium_pool() {
    run_stress(4096, 16, 20_000, 0x1234_5678_9ABC_DEF0);
}

#[test]
fn stress_large_pool_coarse_granularity() {
    run_stress(1 << 16, 256, 20_000, 0xCAFE_F00D_BAAD_F00D);
}

#[test]
fn stress_multiple_seeds_cover_varying_shapes() {
    for seed in 0..8u64 {
        run_stress(1024, 16, 5_000, seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    }
}

#[test]
fn stress_alternating_bulk_alloc_and_free() {
    // Deterministic pattern that repeatedly exhausts and drains the pool.
    let pool_size = 2048usize;
    let mut a = BuddyAllocator::new(pool_size, 16).unwrap();
    let mut rng = Rng::new(777);
    for round in 0..200 {
        let mut live = Vec::new();
        // Fill until even a minimum-sized request fails: intermediate OOMs at
        // larger sizes simply mean "try smaller", not total exhaustion.
        let mut consecutive_skips = 0;
        loop {
            let size = 1 + rng.below(128);
            match a.allocate(size) {
                Ok(x) => {
                    live.push(x);
                    consecutive_skips = 0;
                }
                Err(AllocError::OutOfMemory) => {
                    // Only genuinely exhausted when no min-block is available;
                    // inspect stats rather than allocating a probe block.
                    if a.stats().free_blocks_per_order[0].count == 0
                        && a.stats().free_bytes == 0
                    {
                        break;
                    }
                    consecutive_skips += 1;
                    assert!(consecutive_skips < 10_000, "fill phase stalled");
                }
                Err(e) => panic!("unexpected error {e}"),
            }
        }
        a.verify().unwrap();
        assert_eq!(a.stats().free_bytes, 0);
        assert_eq!(a.allocate(1).unwrap_err(), AllocError::OutOfMemory);

        while let Some(x) = live.pop() {
            a.free(x.addr).unwrap();
        }
        let s = a.stats();
        assert_eq!(s.free_bytes, pool_size, "round {round}");
        assert_eq!(s.free_blocks(), vec![(0, a.max_order())], "round {round}");
    }
}
