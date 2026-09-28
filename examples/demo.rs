//! 端到端演示：
//!
//! ```bash
//! cargo run --example demo
//! ```
//!
//! 在一个 1 MiB 的模拟内存池上跑一段确定性的随机 allocate/free 序列，
//! 打印各阶段统计信息。所有状态都在本进程内存中，无任何外部依赖。

use buddy_allocator::{BuddyAllocator, Rng};

fn print_stats(tag: &str, a: &BuddyAllocator) {
    let s = a.stats();
    println!("--- {tag} ---");
    println!(
        "pool={} B  allocated={} B ({} blocks)  free={} B ({} blocks)",
        s.pool_size, s.allocated_bytes, s.allocated_blocks, s.free_bytes, s.free_blocks
    );
    println!("  requested:                {} B", s.requested_bytes);
    println!(
        "  internal fragmentation:   {} B ({:.2}% of allocated)",
        s.internal_fragmentation,
        s.internal_fragmentation_ratio() * 100.0
    );
    println!(
        "  external fragmentation:   {:.2}% (largest free block = {} B)",
        s.external_fragmentation_ratio() * 100.0,
        s.largest_free_block
    );
    println!("  free blocks per order:    {:?}", s.free_blocks_per_order);
    println!(
        "  counters: alloc ok={}, alloc fail={}, free ok={}, free fail={}",
        s.total_allocations, s.failed_allocations, s.total_frees, s.failed_frees
    );
    a.check_invariants().expect("invariants hold");
}

fn main() {
    const POOL: usize = 1 << 20; // 1 MiB
    const MIN_BLOCK: usize = 64;
    const BASE: usize = 0x1000_0000;

    let mut allocator = BuddyAllocator::new(BASE, POOL, MIN_BLOCK);
    print_stats("initial", &allocator);

    // 1) 几个确定性的分配，展示拆分与内部碎片。
    let a = allocator.allocate(1).unwrap();
    let b = allocator.allocate(100).unwrap();
    let c = allocator.allocate(5000).unwrap();
    println!("allocated 1 B @ {a:#x}, 100 B @ {b:#x}, 5000 B @ {c:#x}");
    print_stats("after 3 allocations", &allocator);

    allocator.free(b).unwrap();
    println!("freed {b:#x}");
    print_stats("after freeing 100 B block", &allocator);

    // 2) 固定种子的大规模随机序列（可复现）。
    let mut rng = Rng::new(20260928);
    let (_, report) = allocator.run_random_workload(&mut rng, 20_000, 4096, 45);
    println!(
        "random workload: {} ops, {} ok allocs, {} failed allocs, {} frees, {} still live",
        report.ops,
        report.allocations,
        report.alloc_failures,
        report.frees,
        report.live.len()
    );
    print_stats("after random workload", &allocator);

    // 3) 释放所有存活块后应恢复到初始状态：整池一个 max-order 空闲块。
    for address in report.live {
        allocator.free(address).unwrap();
    }
    allocator.free(a).unwrap();
    allocator.free(c).unwrap();
    print_stats("after freeing everything", &allocator);

    let s = allocator.stats();
    assert_eq!(s.allocated_bytes, 0);
    assert_eq!(s.free_bytes, POOL);
    assert_eq!(s.free_blocks, 1);
    assert_eq!(s.free_blocks_per_order.last(), Some(&1));
    println!("all memory reclaimed; buddy coalescing restored a single max-order block.");
}
