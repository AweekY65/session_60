//! 确定性的自动化测试：初始化、最小/最大分配、连续拆分、递归合并、
//! 内存耗尽以及各类非法释放。

use buddy_allocator::{AllocError, BuddyAllocator, FreeError};

const BASE: usize = 0x4000_0000;

fn mk(pool: usize, min: usize) -> BuddyAllocator {
    let a = BuddyAllocator::new(BASE, pool, min);
    a.check_invariants().unwrap();
    a
}

#[test]
fn init_creates_one_max_order_free_block() {
    let a = mk(1024, 64);
    let s = a.stats();
    assert_eq!(s.pool_size, 1024);
    assert_eq!(s.free_bytes, 1024);
    assert_eq!(s.allocated_bytes, 0);
    assert_eq!(s.free_blocks, 1);
    // orders 0..=4（64,128,...,1024），初始只有 order 4 上一个块。
    assert_eq!(s.free_blocks_per_order, vec![0, 0, 0, 0, 1]);
    assert_eq!(s.largest_free_block, 1024);
    assert_eq!(a.max_order(), 4);
    assert_eq!(a.block_size(0), 64);
}

#[test]
#[should_panic(expected = "power of two")]
fn rejects_non_power_of_two_pool() {
    let _ = BuddyAllocator::new(0, 1000, 8);
}

#[test]
#[should_panic(expected = "must not exceed")]
fn rejects_min_block_larger_than_pool() {
    let _ = BuddyAllocator::new(0, 64, 128);
}

#[test]
fn min_allocation_uses_one_min_block_and_internal_fragmentation() {
    let mut a = mk(1024, 64);
    // 请求 1 字节也要给一个最小块（64 B），内部碎片 63 B。
    let p = a.allocate(1).unwrap();
    assert_eq!(p, BASE);
    let s = a.stats();
    assert_eq!(s.allocated_blocks, 1);
    assert_eq!(s.allocated_bytes, 64);
    assert_eq!(s.requested_bytes, 1);
    assert_eq!(s.internal_fragmentation, 63);
    assert_eq!(s.free_bytes, 960);
    // 连续拆分：为了得到 order 0，1024 的块一路拆到 64，
    // 在 order 0/1/2/3 各留下一个伙伴空闲块（64+128+256+512=960），
    // order 4 的源块被摘走。
    assert_eq!(s.free_blocks_per_order, vec![1, 1, 1, 1, 0]);
    assert_eq!(s.free_blocks, 4);
    a.check_invariants().unwrap();
}

#[test]
fn exact_size_requests_have_no_internal_fragmentation() {
    let mut a = mk(1024, 64);
    let p = a.allocate(64).unwrap();
    assert_eq!(p, BASE);
    let s = a.stats();
    assert_eq!(s.internal_fragmentation, 0);
    // 256 恰好是 order 2，不需要内部碎片。
    a.allocate(256).unwrap();
    assert_eq!(a.stats().internal_fragmentation, 0);
    a.check_invariants().unwrap();
}

#[test]
fn max_allocation_owns_whole_pool_and_then_exhausts() {
    let mut a = mk(1024, 16);
    let p = a.allocate(1024).unwrap();
    assert_eq!(p, BASE);
    let s = a.stats();
    assert_eq!(s.allocated_bytes, 1024);
    assert_eq!(s.free_bytes, 0);
    assert_eq!(s.free_blocks, 0);
    assert_eq!(s.largest_free_block, 0);
    // 池满：哪怕 1 字节也分不出来。
    assert_eq!(
        a.allocate(1).unwrap_err(),
        AllocError::OutOfMemory { requested: 1 }
    );
    // 请求超过池容量是另一类错误。
    assert_eq!(
        a.allocate(1025).unwrap_err(),
        AllocError::RequestTooLarge {
            requested: 1025,
            pool_size: 1024
        }
    );
    assert_eq!(a.allocate(0).unwrap_err(), AllocError::ZeroSize);
    a.check_invariants().unwrap();
}

#[test]
fn out_of_memory_due_to_external_fragmentation() {
    // 4 个最小块全部占满 -> 无法满足 2 个最小块大小的请求，
    // 尽管“只要”释放其中两个相邻块就能满足，当前确实 OOM。
    let mut a = mk(256, 64);
    let mut live = Vec::new();
    for _ in 0..4 {
        live.push(a.allocate(64).unwrap());
    }
    assert_eq!(
        a.allocate(65).unwrap_err(),
        AllocError::OutOfMemory { requested: 65 }
    );
    // 释放两个不相邻的块（0 和 2）：空闲 128 B 但不连续，仍然无法满足 128 B。
    a.free(live[0]).unwrap();
    a.free(live[2]).unwrap();
    assert_eq!(
        a.allocate(128).unwrap_err(),
        AllocError::OutOfMemory { requested: 128 }
    );
    // 外部碎片度量此时应为 1（空闲空间完全分散成最小块）。
    let s = a.stats();
    assert_eq!(s.free_bytes, 128);
    assert_eq!(s.largest_free_block, 64);
    assert!((s.external_fragmentation_ratio() - 0.5).abs() < 1e-9);
    a.check_invariants().unwrap();
}

#[test]
fn sequential_splits_hand_out_deterministic_addresses() {
    // 1024 / 1 = 1024 个最小块；依次全部分配，地址应严格递增。
    let mut a = mk(1024, 1);
    for i in 0..1024 {
        let p = a.allocate(1).unwrap();
        assert_eq!(p, BASE + i, "allocation {i}");
    }
    assert!(a.allocate(1).is_err());
    a.check_invariants().unwrap();
}

#[test]
fn recursive_coalescing_reverses_full_exhaustion() {
    let mut a = mk(1024, 1);
    let mut live = (0..1024)
        .map(|_| a.allocate(1).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(a.stats().free_bytes, 0);

    // 按逆序释放（每次释放的块与其已空闲的左伙伴合并），最终应只剩一整块。
    while let Some(p) = live.pop() {
        a.free(p).unwrap();
    }
    let s = a.stats();
    assert_eq!(s.allocated_bytes, 0);
    assert_eq!(s.free_bytes, 1024);
    assert_eq!(s.free_blocks, 1);
    let mut expected = vec![0usize; 10];
    expected.push(1);
    assert_eq!(s.free_blocks_per_order, expected);
    assert_eq!(s.largest_free_block, 1024);
    a.check_invariants().unwrap();
}

#[test]
fn coalescing_only_merges_buddies_not_arbitrary_neighbors() {
    let mut a = mk(256, 64);
    let p0 = a.allocate(64).unwrap(); // offset 0
    let p1 = a.allocate(64).unwrap(); // offset 64
    let p2 = a.allocate(64).unwrap(); // offset 128
    let _p3 = a.allocate(64).unwrap(); // offset 192

    // 释放 0 和 2：二者相邻于 128 边界，但**不是伙伴**（0 的伙伴是 64，
    // 128 的伙伴是 192），且 64/192 仍占用，不能合并。
    a.free(p0).unwrap();
    a.free(p2).unwrap();
    let s = a.stats();
    assert_eq!(s.free_blocks, 2);
    assert_eq!(s.free_blocks_per_order[0], 2);
    assert_eq!(s.largest_free_block, 64);

    // 释放 64：先与 0 合并成 [0,128) order1；128 是 order0 空闲，
    // 但 order1 块的伙伴是 [128,256)，而 192 仍占用，故停在 order1。
    a.free(p1).unwrap();
    let s = a.stats();
    assert_eq!(s.largest_free_block, 128);
    assert_eq!(s.free_bytes, 192);
    assert_eq!(s.free_blocks_per_order[1], 1);
    assert_eq!(s.free_blocks_per_order[0], 1); // offset 128
    a.check_invariants().unwrap();
}

#[test]
fn allocate_after_partial_coalesce_reuses_space() {
    let mut a = mk(256, 64);
    let p0 = a.allocate(64).unwrap();
    let p1 = a.allocate(64).unwrap();
    a.free(p0).unwrap();
    a.free(p1).unwrap(); // 合并回 order 1（128 B 块）
    a.check_invariants().unwrap();

    // 100 B 需要 128 B 块，应直接复用合并出来的块（offset 0）。
    let q = a.allocate(100).unwrap();
    assert_eq!(q, BASE);
    let s = a.stats();
    assert_eq!(s.allocated_bytes, 128);
    assert_eq!(s.internal_fragmentation, 28);
    assert_eq!(s.free_bytes, 128);
    a.check_invariants().unwrap();
}

#[test]
fn double_free_is_detected() {
    let mut a = mk(256, 16);
    let p = a.allocate(32).unwrap();
    a.free(p).unwrap();
    assert_eq!(a.free(p).unwrap_err(), FreeError::DoubleFree { address: p });
    // 第三次仍然报 double free。
    assert_eq!(a.free(p).unwrap_err(), FreeError::DoubleFree { address: p });
    // 失败计数累计，但状态未被破坏。
    assert_eq!(a.stats().failed_frees, 2);
    a.check_invariants().unwrap();

    // 合并后原块起始地址变成“空闲块内部地址”，再次释放不再是 double free，
    // 而是非法起始地址（InvalidAddress）。
    let q = a.allocate(16).unwrap();
    a.free(q).unwrap(); // 与 p 所在空闲块的关系：此处 p 已是 order1 空闲块的一部分
    let addr_inside = BASE + 16;
    // BASE+16 落在空闲块内部（对齐但不是块起点）。
    assert!(matches!(
        a.free(addr_inside).unwrap_err(),
        FreeError::InvalidAddress { .. }
    ));
    a.check_invariants().unwrap();
}

#[test]
fn out_of_bounds_free_is_detected() {
    let mut a = mk(256, 16);
    let _ = a.allocate(16).unwrap();
    // 基地址之前。
    assert!(matches!(
        a.free(BASE - 1).unwrap_err(),
        FreeError::OutOfBounds { .. }
    ));
    // 池子末地址（开区间，最后一个合法偏移是 pool_size - 1）。
    assert!(matches!(
        a.free(BASE + 256).unwrap_err(),
        FreeError::OutOfBounds { .. }
    ));
    // 远离池子。
    assert!(matches!(
        a.free(0xDEAD_BEEF).unwrap_err(),
        FreeError::OutOfBounds { .. }
    ));
    a.check_invariants().unwrap();
}

#[test]
fn misaligned_free_is_detected() {
    let mut a = mk(256, 16);
    let p = a.allocate(16).unwrap();
    assert!(matches!(
        a.free(p + 1).unwrap_err(),
        FreeError::Misaligned { .. }
    ));
    assert!(matches!(
        a.free(p + 7).unwrap_err(),
        FreeError::Misaligned { .. }
    ));
    a.check_invariants().unwrap();
}

#[test]
fn freeing_interior_of_allocated_block_is_detected() {
    let mut a = mk(1024, 16);
    let p = a.allocate(512).unwrap(); // 占用 [p, p+512)
    for d in [16usize, 32, 256, 512 - 16] {
        match a.free(p + d) {
            Err(FreeError::NotBlockStart {
                address,
                block_start,
            }) => {
                assert_eq!(address, p + d);
                assert_eq!(block_start, p);
            }
            other => panic!("expected NotBlockStart, got {other:?}"),
        }
    }
    // 对齐但越过块尾的地址属于其它（空闲）块，错误类型不同。
    assert!(matches!(
        a.free(p + 512).unwrap_err(),
        FreeError::InvalidAddress { .. } | FreeError::DoubleFree { .. }
    ));
    // 块本身仍然完好，可以正常释放。
    a.free(p).unwrap();
    a.check_invariants().unwrap();
}

#[test]
fn stats_space_conservation_under_mixed_lifetimes() {
    let mut a = mk(4096, 32);
    let p1 = a.allocate(100).unwrap();
    let p2 = a.allocate(1000).unwrap();
    let p3 = a.allocate(4000).unwrap_err(); // 装不下
    assert!(matches!(p3, AllocError::OutOfMemory { .. }));
    a.free(p2).unwrap();
    let p4 = a.allocate(1000).unwrap(); // 复用 p2 腾出的空间
    assert_eq!(p4, p2);

    let s = a.stats();
    assert_eq!(s.allocated_bytes + s.free_bytes, s.pool_size);
    // 内部碎片 = 已分配块字节 - 请求字节，恒非负。
    assert!(s.internal_fragmentation <= s.allocated_bytes);
    assert_eq!(s.total_allocations, 3);
    assert_eq!(s.total_frees, 1);
    assert_eq!(s.failed_allocations, 1);
    a.check_invariants().unwrap();

    a.free(p1).unwrap();
    a.free(p4).unwrap();
    let s = a.stats();
    assert_eq!(s.allocated_bytes, 0);
    assert_eq!(s.free_bytes, 4096);
    assert_eq!(s.free_blocks, 1);
    a.check_invariants().unwrap();
}

#[test]
fn base_address_is_arbitrary_and_offsets_hold() {
    // 非 0、非 2 的幂的基地址也必须正确（内部按相对偏移做 buddy 运算）。
    let mut a = BuddyAllocator::new(0x1234_5600, 128, 32);
    let p0 = a.allocate(32).unwrap();
    let p1 = a.allocate(32).unwrap();
    assert_eq!(p0, 0x1234_5600);
    assert_eq!(p1, 0x1234_5620);
    a.free(p0).unwrap();
    a.free(p1).unwrap();
    assert_eq!(a.stats().free_blocks, 1);
    a.check_invariants().unwrap();
}
