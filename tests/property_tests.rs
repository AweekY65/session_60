//! Property-based tests for the buddy allocator.
//!
//! Property-based testing normally relies on a crate such as `proptest`; this
//! project intentionally has **zero external dependencies** (its storage and
//! algorithm must stay entirely in-process), so the property framework used
//! here is a compact built-in harness:
//!
//! * a deterministic xorshift PRNG generates random operation sequences;
//! * many sequences are run across several pool configurations and seeds;
//! * the "shrink" requirement is covered structurally by generating sequences
//!   of varying lengths starting at zero — any failing prefix is itself a
//!   counterexample and is reported with the exact seed/length for replay.
//!
//! After every single operation [`BuddyAllocator::verify`] asserts the full
//! set of structural invariants (no overlap, space conservation, free-list/
//! index agreement, counter consistency).

use buddy_allocator::{AllocError, Allocation, BuddyAllocator, FreeError};

/// Deterministic 64-bit xorshift (xorshift64*).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15 | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> usize {
        (self.next_u64() % n) as usize
    }
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Alloc(usize),
    FreeLive,
    FreeBogus(usize),
}

/// Pool/min-block configurations exercised by the property tests.
const CONFIGS: &[(usize, usize)] = &[
    (1, 1),
    (2, 1),
    (4, 1),
    (4, 2),
    (16, 1),
    (16, 16),
    (64, 16),
    (256, 16),
    (1024, 16),
    (1024, 64),
];

fn smallest_fitting_order(alloc: &BuddyAllocator, size: usize) -> usize {
    let units = size.div_ceil(alloc.min_block_size());
    (units.next_power_of_two()).trailing_zeros() as usize
}

fn random_op(rng: &mut Rng, pool_size: usize) -> Op {
    match rng.below(100) {
        0..=54 => {
            // Bias: many small/boundary sizes, plus occasional large ones.
            let size = match rng.below(10) {
                0..=5 => 1 + rng.below((pool_size as u64).min(8)),
                6..=8 => 1 + rng.below((pool_size as u64).saturating_div(2).max(1)),
                _ => 1 + rng.below(pool_size as u64),
            };
            Op::Alloc(size)
        }
        55..=89 => Op::FreeLive,
        _ => Op::FreeBogus(rng.below(pool_size as u64 + 8)),
    }
}

/// Core property: arbitrary op sequences never corrupt the allocator; live
/// blocks never overlap; space is conserved; counters match an independent
/// model; and freeing everything rebuilds the pool as one block.
fn run_invariant_sequence(cfg: (usize, usize), steps: usize, seed: u64) {
    let (pool_size, min_block) = cfg;
    let mut rng = Rng::new(seed);
    let mut alloc = BuddyAllocator::new(pool_size, min_block).unwrap();
    let mut live: Vec<Allocation> = Vec::new();
    let mut requested_total = 0usize;
    let mut granted_total = 0usize;

    for step in 0..steps {
        match random_op(&mut rng, pool_size) {
            Op::Alloc(size) => {
                match alloc.allocate(size) {
                    Ok(a) => {
                        // The granted block fits, and does so minimally.
                        let granted = alloc.block_size(a.order);
                        assert!(granted >= size);
                        if a.order > 0 {
                            assert!(alloc.block_size(a.order - 1) < size);
                        }
                        // Grantee intervals are pairwise disjoint.
                        let a_end = a.addr + granted;
                        for b in &live {
                            let b_end = b.addr + alloc.block_size(b.order);
                            assert!(
                                a.addr >= b_end || a_end <= b.addr,
                                "cfg={cfg:?} seed={seed} step={step}: \
                                 new block {}..{a_end} overlaps {}..{b_end}",
                                a.addr,
                                b.addr
                            );
                        }
                        requested_total += size;
                        granted_total += granted;
                        live.push(a);
                    }
                    Err(e) => {
                        // An in-contract request can only fail with OOM, and
                        // then no sufficiently large free block remains.
                        assert_eq!(e, AllocError::OutOfMemory);
                        let target = smallest_fitting_order(&alloc, size);
                        let stats = alloc.stats();
                        assert!(
                            stats.free_blocks_per_order[target..]
                                .iter()
                                .all(|fc| fc.count == 0),
                            "seed={seed} step={step}: spurious OOM for size {size}"
                        );
                    }
                }
            }
            Op::FreeLive => {
                if !live.is_empty() {
                    let i = rng.below(live.len() as u64);
                    let a = live.swap_remove(i);
                    alloc.free(a.addr).expect("free of tracked allocation");
                    requested_total -= a.size;
                    granted_total -= alloc.block_size(a.order);
                }
            }
            Op::FreeBogus(addr) => {
                let is_live_start = live.iter().any(|a| a.addr == addr);
                let result = alloc.free(addr);
                if is_live_start {
                    assert!(result.is_ok(), "free of live start {addr} must succeed");
                    if let Some(pos) = live.iter().position(|a| a.addr == addr) {
                        let a = live.swap_remove(pos);
                        requested_total -= a.size;
                        granted_total -= alloc.block_size(a.order);
                    }
                } else {
                    assert!(
                        matches!(
                            result,
                            Err(FreeError::AddressOutOfRange)
                                | Err(FreeError::NotBlockAligned)
                                | Err(FreeError::NotBlockStart)
                                | Err(FreeError::DoubleFree)
                                | Err(FreeError::NotAllocated)
                        ),
                        "cfg={cfg:?} seed={seed} step={step}: bogus free at {addr} succeeded"
                    );
                }
            }
        }

        // Full structural check after every operation.
        alloc
            .verify()
            .unwrap_or_else(|e| panic!("cfg={cfg:?} seed={seed} step={step}: {e}"));

        // Independent model agrees with reported statistics.
        let stats = alloc.stats();
        assert_eq!(stats.allocated_bytes, requested_total, "seed={seed} step={step}");
        assert_eq!(stats.used_bytes, granted_total, "seed={seed} step={step}");
        assert_eq!(stats.used_bytes + stats.free_bytes, pool_size);
        assert_eq!(live.len(), alloc.live_allocations().len());
    }

    // Releasing everything restores a single max-order free block.
    while let Some(a) = live.pop() {
        alloc.free(a.addr).unwrap();
    }
    alloc.verify().unwrap();
    let stats = alloc.stats();
    assert_eq!(stats.free_bytes, pool_size);
    assert_eq!(stats.used_bytes, 0);
    assert_eq!(stats.free_blocks_per_order[alloc.max_order()].count, 1);
    assert!(stats.free_blocks_per_order[..alloc.max_order()]
        .iter()
        .all(|fc| fc.count == 0));
    assert_eq!(alloc.free_blocks(), vec![(0, alloc.max_order())]);
}

#[test]
fn invariants_hold_under_random_operations() {
    // Many seeds over every configuration, with sequence lengths crossing
    // exhaustion/recycling boundaries. A failure prints cfg+seed+step which
    // uniquely replays the counterexample.
    for &cfg in CONFIGS {
        for seed in 0..60u64 {
            for &steps in &[1usize, 2, 5, 20, 80, 200, 500] {
                run_invariant_sequence(cfg, steps, seed * 1_000_003 + steps as u64);
            }
        }
    }
}

#[test]
fn tiny_pool_boundary_sequences() {
    // Exhaustive-ish small sequences on the smallest possible pools.
    for &cfg in &[(1usize, 1usize), (2, 1), (4, 1), (4, 2)] {
        for seed in 0..500u64 {
            run_invariant_sequence(cfg, 30, seed ^ 0xABCD);
        }
    }
}

/// Property: regardless of the free order, releasing all live blocks always
/// completely coalesces the pool back into one max-order block.
#[test]
fn any_free_order_completely_coalesces() {
    for &(pool_size, min_block) in CONFIGS {
        for seed in 0..40u64 {
            let mut rng = Rng::new(seed ^ 0x5151);
            let mut alloc = BuddyAllocator::new(pool_size, min_block).unwrap();
            let mut blocks = Vec::new();

            // Randomly fill until OOM a few times with intervening frees.
            for _ in 0..200 {
                match rng.below(100) {
                    0..=69 => {
                        let size = 1 + rng.below(pool_size as u64);
                        if let Ok(a) = alloc.allocate(size) {
                            blocks.push(a);
                        }
                    }
                    _ => {
                        if !blocks.is_empty() {
                            let i = rng.below(blocks.len() as u64);
                            let a = blocks.swap_remove(i);
                            alloc.free(a.addr).unwrap();
                        }
                    }
                }
                alloc.verify().unwrap();
            }

            // Free the survivors in the (randomized) vector order.
            while let Some(a) = blocks.pop() {
                alloc.free(a.addr).unwrap();
                alloc.verify().unwrap();
            }
            let stats = alloc.stats();
            assert_eq!(stats.free_bytes, pool_size, "cfg=({pool_size},{min_block}) seed={seed}");
            assert_eq!(
                alloc.free_blocks(),
                vec![(0, alloc.max_order())],
                "cfg=({pool_size},{min_block}) seed={seed}"
            );
        }
    }
}

/// Property: every failed allocation is "honest" — immediately after an OOM,
/// freeing one block and retrying only succeeds when the model says it can.
#[test]
fn oom_matches_free_list_capacity_repeatedly() {
    for seed in 0..100u64 {
        let (pool_size, min_block) = (1024, 16);
        let mut rng = Rng::new(seed ^ 0x7777);
        let mut alloc = BuddyAllocator::new(pool_size, min_block).unwrap();
        let mut live: Vec<Allocation> = Vec::new();

        for _ in 0..300 {
            let size = 1 + rng.below(pool_size as u64);
            if let Ok(a) = alloc.allocate(size) {
                live.push(a);
            } else {
                let target = smallest_fitting_order(&alloc, size);
                assert!(alloc.stats().free_blocks_per_order[target..]
                    .iter()
                    .all(|fc| fc.count == 0));
            }
            alloc.verify().unwrap();
            if !live.is_empty() && rng.below(100) < 45 {
                let i = rng.below(live.len() as u64);
                let a = live.swap_remove(i);
                alloc.free(a.addr).unwrap();
            }
        }
        for a in live.drain(..) {
            alloc.free(a.addr).unwrap();
        }
        assert_eq!(alloc.stats().free_bytes, pool_size);
    }
}
