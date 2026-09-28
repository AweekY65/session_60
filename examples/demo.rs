//! Small walk-through of the buddy allocator API.
//!
//! Run with `cargo run --example demo`.

use buddy_allocator::BuddyAllocator;

fn main() {
    // A 1 KiB pool with 16-byte minimum blocks (orders 0..=6).
    let mut alloc = BuddyAllocator::new(1024, 16).expect("valid configuration");

    println!("== initial ==");
    println!("{}", alloc.stats());

    let a = alloc.allocate(1).unwrap();
    let b = alloc.allocate(100).unwrap();
    let c = alloc.allocate(500).unwrap();
    println!("== three allocations: 1 B, 100 B, 500 B ==");
    println!("  a = {a:?}");
    println!("  b = {b:?}");
    println!("  c = {c:?}");
    println!("{}", alloc.stats());

    alloc.free(b.addr).unwrap();
    alloc.free(a.addr).unwrap();
    println!("== after freeing a and b (coalesced) ==");
    println!("{}", alloc.stats());

    alloc.free(c.addr).unwrap();
    println!("== after freeing c: pool fully restored ==");
    println!("{}", alloc.stats());
}
