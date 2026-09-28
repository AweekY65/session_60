//! # Buddy Memory Allocator simulator
//!
//! A self-contained simulation of a binary buddy allocator. Everything lives in
//! ordinary process-local heap memory (`Vec<u8>` plus a few vectors/hash maps):
//! there is no dependence on OS memory-management services, databases or any
//! other external service. The allocation algorithm itself is implemented in
//! this file.
//!
//! ## Model
//!
//! The pool has total size `2^max_order * min_block_size` bytes and is managed
//! at `max_order + 1` granularities ("orders"). Order `o` blocks are
//! `min_block_size << o` bytes large. Every block is naturally aligned to its
//! own size.
//!
//! * [`BuddyAllocator::allocate`] locates the smallest order that fits the
//!   request, pops a free block from that order's free list (recursively
//!   splitting larger blocks when necessary) and records the resulting block.
//! * [`BuddyAllocator::free`] validates the request, returns the block and
//!   recursively coalesces it with its buddy while the buddy is a single free
//!   block of the matching order.
//! * [`BuddyAllocator::stats`] reports allocated / free bytes, internal
//!   fragmentation and the number of free blocks per order.
//! * [`BuddyAllocator::verify`] checks all structural invariants (no overlap,
//!   space conservation, free-list consistency) and is used heavily by tests.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

/// One successful allocation: the user-visible handle returned by
/// [`BuddyAllocator::allocate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Allocation {
    /// Offset of the block's first byte inside the pool. Always aligned to
    /// the block's order size.
    pub addr: usize,
    /// Bytes requested by the caller (`1..=pool_size`).
    pub size: usize,
    /// Order actually granted; the block spans `block_size(order)` bytes.
    pub order: usize,
}

/// Errors that can occur while initializing an allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitError {
    /// `min_block_size` must be a non-zero power of two.
    BadMinBlockSize,
    /// The pool size must be a non-zero power of two at least as large as
    /// `min_block_size`.
    BadPoolSize,
}

/// Errors that can occur during [`BuddyAllocator::allocate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocError {
    /// A zero-sized request is meaningless for a block allocator.
    ZeroSize,
    /// The request exceeds the whole pool and can never succeed.
    RequestTooLarge,
    /// The pool is fragmented/exhausted: no block of a sufficient order exists.
    OutOfMemory,
}

/// Errors that can occur during [`BuddyAllocator::free`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreeError {
    /// The address lies outside the pool's address range.
    AddressOutOfRange,
    /// The address is not aligned to `min_block_size` and therefore cannot be a
    /// block start.
    NotBlockAligned,
    /// The address points inside a live allocated block but is not its start.
    NotBlockStart,
    /// The address already names a free block: it was freed before and has not
    /// been reallocated since.
    DoubleFree,
    /// The address is validly aligned and in range, but no allocation or free
    /// block starts there (e.g. a stale/unknown handle).
    NotAllocated,
}

/// Number of free blocks for a single order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FreeCount {
    /// The order this entry describes.
    pub order: usize,
    /// Block size in bytes of this order.
    pub block_size: usize,
    /// Free blocks available at this order.
    pub count: usize,
}

/// Point-in-time statistics produced by [`BuddyAllocator::stats`].
#[derive(Debug, Clone, PartialEq)]
pub struct Stats {
    /// Total pool size in bytes.
    pub total_bytes: usize,
    /// Sum of the *requested* sizes of live allocations.
    pub allocated_bytes: usize,
    /// Bytes covered by blocks granted to live allocations
    /// (`allocated_bytes + internal_fragmentation_bytes`).
    pub used_bytes: usize,
    /// Bytes currently owned by free blocks.
    pub free_bytes: usize,
    /// Wasted bytes inside allocated blocks:
    /// `used_bytes - allocated_bytes`.
    pub internal_fragmentation_bytes: usize,
    /// Per-order free block counts, indexed by order.
    pub free_blocks_per_order: Vec<FreeCount>,
    /// Internal fragmentation as a fraction of *used* bytes (`0.0` when
    /// nothing is allocated).
    pub internal_fragmentation_ratio: f64,
    /// Allocated-requested bytes as a fraction of the total pool.
    pub utilization: f64,
}

/// A binary buddy allocator over a fixed, power-of-two sized in-process pool.
#[derive(Debug)]
pub struct BuddyAllocator {
    /// Raw pool storage. Only process-local heap memory; kept `Vec<u8>` so the
    /// simulator is self-contained and byte-addressable.
    pool: Vec<u8>,
    /// Smallest possible block (order 0); a power of two.
    min_block_size: usize,
    /// Highest order; `pool.len() == block_size(max_order)`.
    max_order: usize,
    /// `free_lists[o]` holds the start addresses of free order-`o` blocks.
    /// Order among the entries is irrelevant, so a `Vec` used as a stack is
    /// enough; membership lookups go through `free_order`.
    free_lists: Vec<Vec<usize>>,
    /// Start address -> order for every free block.
    free_order: HashMap<usize, usize>,
    /// Start address -> live allocation record.
    allocations: BTreeMap<usize, Allocation>,
    /// Addresses freed by the user and not yet reallocated. Used to recognise
    /// double frees distinctly from other invalid releases.
    recently_freed: HashSet<usize>,
    /// Running sum of requested bytes.
    allocated_bytes: usize,
    /// Running sum of granted block bytes.
    used_bytes: usize,
}

impl BuddyAllocator {
    /// Create a pool of `pool_size` bytes (a power of two) whose smallest
    /// block is `min_block_size` bytes (a power of two, `<= pool_size`).
    pub fn new(pool_size: usize, min_block_size: usize) -> Result<Self, InitError> {
        if min_block_size == 0 || !min_block_size.is_power_of_two() {
            return Err(InitError::BadMinBlockSize);
        }
        if pool_size == 0 || !pool_size.is_power_of_two() || pool_size < min_block_size {
            return Err(InitError::BadPoolSize);
        }
        let max_order = (pool_size / min_block_size).trailing_zeros() as usize;

        let mut free_lists = vec![Vec::new(); max_order + 1];
        free_lists[max_order].push(0);
        let mut free_order = HashMap::new();
        free_order.insert(0, max_order);

        Ok(Self {
            pool: vec![0u8; pool_size],
            min_block_size,
            max_order,
            free_lists,
            free_order,
            allocations: BTreeMap::new(),
            recently_freed: HashSet::new(),
            allocated_bytes: 0,
            used_bytes: 0,
        })
    }

    /// Total pool size in bytes.
    pub fn pool_size(&self) -> usize {
        self.pool.len()
    }

    /// Smallest block size (order 0) in bytes.
    pub fn min_block_size(&self) -> usize {
        self.min_block_size
    }

    /// Highest order managed by the pool.
    pub fn max_order(&self) -> usize {
        self.max_order
    }

    /// Byte size of blocks at `order`.
    pub fn block_size(&self, order: usize) -> usize {
        self.min_block_size << order
    }

    /// Smallest order whose blocks can hold `size` bytes.
    ///
    /// `size` must be `1..=pool_size`; larger/smaller callers are rejected by
    /// [`allocate`](Self::allocate) first.
    fn order_for_size(&self, size: usize) -> usize {
        // Round `size` up to the next multiple of the minimum block, then find
        // the power-of-two block that contains it.
        let units = size.div_ceil(self.min_block_size);
        // `units.next_power_of_two()` is at most the number of min-blocks in
        // the pool because `size <= pool_size`.
        (units.next_power_of_two()).trailing_zeros() as usize
    }

    /// Allocate a block of at least `size` bytes.
    ///
    /// The smallest fitting order is chosen; larger free blocks are split
    /// recursively until a block of exactly that order is available.
    pub fn allocate(&mut self, size: usize) -> Result<Allocation, AllocError> {
        if size == 0 {
            return Err(AllocError::ZeroSize);
        }
        if size > self.pool.len() {
            return Err(AllocError::RequestTooLarge);
        }

        let target = self.order_for_size(size);

        // Locate the smallest order >= target with a free block.
        let mut source_order = target;
        while source_order <= self.max_order && self.free_lists[source_order].is_empty() {
            source_order += 1;
        }
        if source_order > self.max_order {
            return Err(AllocError::OutOfMemory);
        }

        // Pop the block we will grant.
        let addr = self.free_lists[source_order].pop().expect("non-empty list");
        self.free_order.remove(&addr);

        // Recursively split it down to the target order. Each split divides a
        // block into two aligned buddies of the next-smaller order; one half is
        // kept for further splitting, the other half becomes a free block.
        let mut cur_order = source_order;
        while cur_order > target {
            cur_order -= 1;
            let buddy_addr = addr + self.block_size(cur_order);
            self.free_lists[cur_order].push(buddy_addr);
            self.free_order.insert(buddy_addr, cur_order);
        }

        let alloc = Allocation {
            addr,
            size,
            order: target,
        };
        self.allocations.insert(addr, alloc);
        self.recently_freed.remove(&addr);
        self.allocated_bytes += size;
        self.used_bytes += self.block_size(target);
        Ok(alloc)
    }

    /// Release a block previously returned by [`allocate`](Self::allocate).
    ///
    /// The address must be the **start** of a live allocation. After removal
    /// the block is recursively merged with its free buddy at each order,
    /// minimizing external fragmentation.
    pub fn free(&mut self, addr: usize) -> Result<(), FreeError> {
        // 1. Range: every valid block start addresses an in-pool byte.
        if addr >= self.pool.len() {
            return Err(FreeError::AddressOutOfRange);
        }

        // 2. Interior address: any address strictly inside a live allocated
        //    block that is not its start is rejected as a non-start release
        //    (checked before alignment so an unaligned offset into a live block
        //    is reported as the more meaningful error).
        if self
            .allocations
            .values()
            .any(|a| a.addr < addr && addr < a.addr + self.block_size(a.order))
        {
            return Err(FreeError::NotBlockStart);
        }

        // 3. Alignment: outside live blocks, a block start must be a multiple
        //    of the minimum block size.
        if addr % self.min_block_size != 0 {
            return Err(FreeError::NotBlockAligned);
        }

        // 4. A handle that was already released (even if it has since been
        //    absorbed into a larger coalesced free block) is a double free.
        if self.recently_freed.contains(&addr) {
            return Err(FreeError::DoubleFree);
        }

        // 5. A free block the caller never held (created by an internal split)
        //    is not a valid release target.
        if self.free_order.contains_key(&addr) {
            return Err(FreeError::NotAllocated);
        }

        // 6. Otherwise the address must name a live allocation start.
        let alloc = match self.allocations.remove(&addr) {
            Some(a) => a,
            None => return Err(FreeError::NotAllocated),
        };

        self.allocated_bytes -= alloc.size;
        self.used_bytes -= self.block_size(alloc.order);

        // Recursive coalescing: while the current block's buddy (a) exists at
        // this order, (b) is free and (c) is free at exactly this order, merge
        // into the aligned parent and repeat one order up.
        let mut addr = addr;
        let mut order = alloc.order;
        while order < self.max_order {
            let buddy = addr ^ self.block_size(order);
            match self.free_order.get(&buddy).copied() {
                Some(buddy_order) if buddy_order == order => {
                    // Remove the buddy from its free list.
                    let list = &mut self.free_lists[order];
                    let pos = list
                        .iter()
                        .position(|&a| a == buddy)
                        .expect("buddy recorded in free list");
                    list.swap_remove(pos);
                    self.free_order.remove(&buddy);
                    // Parent starts at the lower of the two aligned buddies.
                    addr = addr.min(buddy);
                    order += 1;
                }
                _ => break,
            }
        }

        self.free_lists[order].push(addr);
        self.free_order.insert(addr, order);
        self.recently_freed.insert(alloc.addr);
        Ok(())
    }

    /// Immutable view of the raw pool backing storage (mainly for demo /
    /// inspection purposes).
    pub fn pool(&self) -> &[u8] {
        &self.pool
    }

    /// Snapshot of all live allocations, ordered by start address.
    pub fn live_allocations(&self) -> Vec<Allocation> {
        self.allocations.values().copied().collect()
    }

    /// Snapshot of free blocks as `(address, order)` pairs.
    pub fn free_blocks(&self) -> Vec<(usize, usize)> {
        let mut blocks: Vec<(usize, usize)> = self
            .free_order
            .iter()
            .map(|(&addr, &order)| (addr, order))
            .collect();
        blocks.sort_unstable();
        blocks
    }

    /// Current statistics: allocated/free bytes, internal fragmentation and
    /// per-order free block counts.
    pub fn stats(&self) -> Stats {
        let total = self.pool.len();
        let free_bytes: usize = self
            .free_order
            .values()
            .map(|&o| self.block_size(o))
            .sum();
        let internal = self.used_bytes - self.allocated_bytes;
        let free_blocks_per_order = (0..=self.max_order)
            .map(|order| FreeCount {
                order,
                block_size: self.block_size(order),
                count: self.free_lists[order].len(),
            })
            .collect();

        Stats {
            total_bytes: total,
            allocated_bytes: self.allocated_bytes,
            used_bytes: self.used_bytes,
            free_bytes,
            internal_fragmentation_bytes: internal,
            free_blocks_per_order,
            internal_fragmentation_ratio: if self.used_bytes == 0 {
                0.0
            } else {
                internal as f64 / self.used_bytes as f64
            },
            utilization: self.allocated_bytes as f64 / total as f64,
        }
    }

    /// Verify every structural invariant, returning `Err(reason)` if violated.
    ///
    /// Checks:
    /// * the backing pool has the declared size;
    /// * free-list entries agree with the free-block index;
    /// * all blocks lie in range and are correctly aligned;
    /// * free blocks are pairwise disjoint and tile the pool together with
    ///   the allocated blocks (space conservation, no overlap);
    /// * the running byte counters agree with the block records.
    pub fn verify(&self) -> Result<(), String> {
        let total = self.pool.len();
        if total != self.block_size(self.max_order) {
            return Err(format!(
                "pool size {total} != order-{} block size {}",
                self.max_order,
                self.block_size(self.max_order)
            ));
        }

        // Free list <-> index consistency.
        for (o, list) in self.free_lists.iter().enumerate() {
            if list.len() != list.iter().collect::<HashSet<_>>().len() {
                return Err(format!("duplicate addresses in free list order {o}"));
            }
            for &addr in list {
                match self.free_order.get(&addr) {
                    Some(&fo) if fo == o => {}
                    Some(&fo) => {
                        return Err(format!(
                            "free block {addr} in order {o} but indexed as order {fo}"
                        ))
                    }
                    None => {
                        return Err(format!(
                            "free block {addr} present in list order {o} but not indexed"
                        ))
                    }
                }
                let size = self.block_size(o);
                if addr % size != 0 {
                    return Err(format!("free block {addr} not aligned to its size {size}"));
                }
                if addr.checked_add(size).map_or(true, |end| end > total) {
                    return Err(format!("free block {addr}..{} out of range", addr + size));
                }
            }
        }
        if self.free_order.len()
            != self.free_lists.iter().map(|l| l.len()).sum::<usize>()
        {
            return Err("free index/free list count mismatch".to_string());
        }

        // Allocated blocks: alignment, range, mutual non-overlap.
        let mut prev_end: Option<usize> = None;
        let mut sum_req = 0usize;
        let mut sum_granted = 0usize;
        for (&addr, alloc) in &self.allocations {
            if alloc.addr != addr {
                return Err(format!("allocation key {addr} != record addr {}", alloc.addr));
            }
            if alloc.size == 0 || alloc.size > self.block_size(alloc.order) {
                return Err(format!("allocation at {addr} has bad size {}", alloc.size));
            }
            if alloc.order > self.max_order {
                return Err(format!("allocation at {addr} has order > max_order"));
            }
            let size = self.block_size(alloc.order);
            if alloc.size <= size / 2 && alloc.order > 0 {
                return Err(format!(
                    "allocation at {addr} uses order {} but request {} fits order below",
                    alloc.order, alloc.size
                ));
            }
            if addr % size != 0 {
                return Err(format!("allocation at {addr} not aligned to {size}"));
            }
            let end = addr + size;
            if end > total {
                return Err(format!("allocation {addr}..{end} out of range"));
            }
            if let Some(pe) = prev_end {
                if addr < pe {
                    return Err(format!("allocation at {addr} overlaps previous block ending {pe}"));
                }
            }
            prev_end = Some(end);
            sum_req += alloc.size;
            sum_granted += size;
        }

        // Tiling: merge all blocks (free + allocated) into one sorted interval
        // walk and require them to exactly cover [0, total) without gaps or
        // overlaps. This simultaneously proves non-overlap and conservation.
        #[derive(Clone, Copy)]
        struct Interval {
            start: usize,
            end: usize,
        }
        let mut intervals: Vec<Interval> = self
            .allocations
            .values()
            .map(|a| Interval {
                start: a.addr,
                end: a.addr + self.block_size(a.order),
            })
            .chain(self.free_order.iter().map(|(&addr, &o)| Interval {
                start: addr,
                end: addr + self.block_size(o),
            }))
            .collect();
        intervals.sort_unstable_by_key(|i| i.start);

        let mut cursor = 0usize;
        for iv in &intervals {
            if iv.start != cursor {
                return Err(format!(
                    "gap/overlap at {}: expected block to start at {cursor}",
                    iv.start
                ));
            }
            if iv.end <= iv.start {
                return Err("empty or backwards block interval".to_string());
            }
            cursor = iv.end;
        }
        if cursor != total {
            return Err(format!("blocks cover {cursor} bytes, pool is {total}"));
        }

        if sum_req != self.allocated_bytes {
            return Err(format!(
                "allocated_bytes counter {} != sum of requests {sum_req}",
                self.allocated_bytes
            ));
        }
        if sum_granted != self.used_bytes {
            return Err(format!(
                "used_bytes counter {} != sum of granted blocks {sum_granted}",
                self.used_bytes
            ));
        }
        if self.used_bytes + self.stats().free_bytes != total {
            return Err("used + free != total (space not conserved)".to_string());
        }

        Ok(())
    }
}

impl fmt::Display for InitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InitError::BadMinBlockSize => write!(
                f,
                "min_block_size must be a non-zero power of two"
            ),
            InitError::BadPoolSize => write!(
                f,
                "pool_size must be a non-zero power of two and >= min_block_size"
            ),
        }
    }
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocError::ZeroSize => write!(f, "cannot allocate zero bytes"),
            AllocError::RequestTooLarge => {
                write!(f, "requested size exceeds the pool size")
            }
            AllocError::OutOfMemory => {
                write!(f, "out of memory: no free block of a sufficient order")
            }
        }
    }
}

impl std::error::Error for InitError {}
impl std::error::Error for AllocError {}
impl std::error::Error for FreeError {}

impl fmt::Display for FreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FreeError::AddressOutOfRange => write!(f, "free address lies outside the pool"),
            FreeError::NotBlockAligned => {
                write!(f, "free address is not aligned to the minimum block size")
            }
            FreeError::NotBlockStart => {
                write!(f, "free address points inside an allocated block, not its start")
            }
            FreeError::DoubleFree => {
                write!(f, "double free: block at that address is already free")
            }
            FreeError::NotAllocated => {
                write!(f, "no allocated block starts at that address")
            }
        }
    }
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Buddy allocator stats")?;
        writeln!(f, "  total pool  : {:>10} bytes", self.total_bytes)?;
        writeln!(f, "  allocated   : {:>10} bytes (requested)", self.allocated_bytes)?;
        writeln!(f, "  used        : {:>10} bytes (granted blocks)", self.used_bytes)?;
        writeln!(f, "  free        : {:>10} bytes", self.free_bytes)?;
        writeln!(
            f,
            "  internal frag: {:>9} bytes ({:.2}% of used)",
            self.internal_fragmentation_bytes,
            self.internal_fragmentation_ratio * 100.0
        )?;
        writeln!(f, "  utilization : {:>9.2}%", self.utilization * 100.0)?;
        writeln!(f, "  free blocks per order:")?;
        for fc in &self.free_blocks_per_order {
            writeln!(
                f,
                "    order {:>2} (block {:>8} B): {}",
                fc.order, fc.block_size, fc.count
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_creates_single_max_order_block() {
        let a = BuddyAllocator::new(1024, 16).unwrap();
        a.verify().unwrap();
        assert_eq!(a.max_order(), 6);
        assert_eq!(a.block_size(6), 1024);
        assert_eq!(a.free_blocks(), vec![(0, 6)]);

        let s = a.stats();
        assert_eq!(s.total_bytes, 1024);
        assert_eq!(s.free_bytes, 1024);
        assert_eq!(s.allocated_bytes, 0);
        assert_eq!(s.free_blocks_per_order[6].count, 1);
    }

    #[test]
    fn init_rejects_bad_sizes() {
        assert_eq!(
            BuddyAllocator::new(1024, 0),
            Err(InitError::BadMinBlockSize)
        );
        assert_eq!(
            BuddyAllocator::new(1024, 300),
            Err(InitError::BadMinBlockSize)
        );
        assert_eq!(
            BuddyAllocator::new(1000, 16),
            Err(InitError::BadPoolSize)
        );
        assert_eq!(
            BuddyAllocator::new(8, 16),
            Err(InitError::BadPoolSize)
        );
        assert!(BuddyAllocator::new(1, 1).is_ok());
    }

    #[test]
    fn allocate_min_and_max_sizes() {
        let mut a = BuddyAllocator::new(1024, 16).unwrap();

        let small = a.allocate(1).unwrap();
        assert_eq!(small.order, 0);
        assert_eq!(small.addr, 0);
        // Splitting 1024 all the way down leaves one free block at every
        // intermediate order and a 16-byte sibling at order 0.
        assert_eq!(a.block_size(0), 16);
        a.verify().unwrap();

        // Largest possible request: must consume the sibling tree root after
        // the 1-byte block lives in the other half.
        let big = a.allocate(512).unwrap();
        assert_eq!(big.order, 5);
        assert_eq!(big.addr, 512);
        a.verify().unwrap();
        assert_eq!(a.stats().free_bytes, 0);

        // Nothing left for even a minimum block.
        assert_eq!(a.allocate(1), Err(AllocError::OutOfMemory));

        assert!(a.free(small.addr).is_ok());
        assert!(a.free(big.addr).is_ok());
        a.verify().unwrap();
        assert_eq!(a.free_blocks(), vec![(0, 6)]);
        assert_eq!(a.stats().free_bytes, 1024);
    }

    #[test]
    fn split_cascade_populates_each_order() {
        // Pool 64, min block 16 => orders 0..=2 (16/32/64).
        let mut a = BuddyAllocator::new(64, 16).unwrap();
        let x = a.allocate(16).unwrap();
        assert_eq!(x.addr, 0);
        // 64 -> 32+32, keep [0,32), free [32,64); then 32 -> 16+16, keep
        // [0,16), free [16,32).
        assert_eq!(a.free_blocks(), vec![(16, 0), (32, 1)]);
        a.verify().unwrap();
    }

    #[test]
    fn coalesce_rebuilds_the_whole_pool() {
        let mut a = BuddyAllocator::new(64, 16).unwrap();
        let blocks: Vec<_> = (0..4).map(|_| a.allocate(16).unwrap()).collect();
        assert_eq!(a.stats().free_bytes, 0);

        // Freeing in an order that forces left/right merging at every step.
        a.free(blocks[0].addr).unwrap();
        assert_eq!(a.free_blocks(), vec![(0, 0)]);
        a.free(blocks[1].addr).unwrap();
        assert_eq!(a.free_blocks(), vec![(0, 1)]);
        a.free(blocks[3].addr).unwrap();
        assert_eq!(a.free_blocks(), vec![(0, 1), (48, 0)]);
        a.free(blocks[2].addr).unwrap();
        assert_eq!(a.free_blocks(), vec![(0, 2)]);
        a.verify().unwrap();
    }

    #[test]
    fn coalescing_promotes_allocation_to_higher_order() {
        // Two min-blocks freed together form a 32-byte hole that a later
        // 17-byte request must use (no falsely fragmented failure).
        let mut a = BuddyAllocator::new(64, 16).unwrap();
        let x = a.allocate(16).unwrap(); // addr 0
        let y = a.allocate(16).unwrap(); // addr 16
        let z = a.allocate(16).unwrap(); // addr 32
        a.free(x.addr).unwrap();
        a.free(y.addr).unwrap(); // coalesces into [0,32)
        let big = a.allocate(20).unwrap();
        assert_eq!(big.order, 1);
        assert_eq!(big.addr, 0);
        a.free(z.addr).unwrap();
        a.free(big.addr).unwrap();
        assert_eq!(a.free_blocks(), vec![(0, 2)]);
    }

    #[test]
    fn zero_and_oversized_requests_are_rejected() {
        let mut a = BuddyAllocator::new(64, 16).unwrap();
        assert_eq!(a.allocate(0), Err(AllocError::ZeroSize));
        assert_eq!(a.allocate(65), Err(AllocError::RequestTooLarge));
    }

    #[test]
    fn invalid_frees_are_detected() {
        let mut a = BuddyAllocator::new(64, 16).unwrap();

        // Out of range (also covers "past the end" and huge addresses).
        assert_eq!(a.free(64), Err(FreeError::AddressOutOfRange));
        assert_eq!(a.free(usize::MAX), Err(FreeError::AddressOutOfRange));

        // On a fresh allocator unaligned addresses lie in free space and can
        // never name a block start.
        assert_eq!(a.free(1), Err(FreeError::NotBlockAligned));
        assert_eq!(a.free(15), Err(FreeError::NotBlockAligned));
        assert_eq!(a.free(63), Err(FreeError::NotBlockAligned));

        let x = a.allocate(16).unwrap(); // [0, 16)

        // Any address strictly inside the live block is a non-start release,
        // aligned or not.
        assert_eq!(a.free(1), Err(FreeError::NotBlockStart));
        assert_eq!(a.free(15), Err(FreeError::NotBlockStart));

        // Unknown but aligned in-range address inside a free region.
        // After allocating [0,16), [16,64) is free (as 16 + 32). Address 16
        // is the start of a free block but the user never received it.
        assert_eq!(a.free(16), Err(FreeError::NotAllocated));

        // Double free.
        a.free(x.addr).unwrap();
        assert_eq!(a.free(x.addr), Err(FreeError::DoubleFree));

        // Reallocation clears the "recently freed" state: freeing the new live
        // block must work, and freeing again is once more a double free.
        let y = a.allocate(16).unwrap();
        assert_eq!(y.addr, x.addr);
        a.free(y.addr).unwrap();
        assert_eq!(a.free(y.addr), Err(FreeError::DoubleFree));
        a.verify().unwrap();
    }

    #[test]
    fn stale_handle_after_coalesce_is_still_a_double_free() {
        let mut a = BuddyAllocator::new(64, 16).unwrap();
        let x = a.allocate(16).unwrap(); // 0
        let y = a.allocate(16).unwrap(); // 16
        a.free(x.addr).unwrap();
        a.free(y.addr).unwrap(); // both absorbed into the coalesced [0,32) hole
        // The block at 16 no longer exists as a free block, yet its stale
        // handle must still be recognized as a double free.
        assert_eq!(a.free(16), Err(FreeError::DoubleFree));
        assert_eq!(a.free(0), Err(FreeError::DoubleFree));
    }

    #[test]
    fn interior_free_after_split_is_rejected() {
        let mut a = BuddyAllocator::new(64, 16).unwrap();
        let big = a.allocate(32).unwrap(); // [0,32)
        assert_eq!(big.addr, 0);
        assert_eq!(a.free(8), Err(FreeError::NotBlockStart));
        assert_eq!(a.free(16), Err(FreeError::NotBlockStart));
        assert_eq!(a.free(31), Err(FreeError::NotBlockStart));
    }

    #[test]
    fn internal_fragmentation_is_tracked() {
        let mut a = BuddyAllocator::new(64, 16).unwrap();
        let x = a.allocate(1).unwrap();
        let y = a.allocate(17).unwrap();
        let s = a.stats();
        assert_eq!(s.allocated_bytes, 18);
        assert_eq!(s.used_bytes, 16 + 32);
        assert_eq!(s.internal_fragmentation_bytes, 30);
        assert!((s.internal_fragmentation_ratio - 30.0 / 48.0).abs() < 1e-9);
        a.free(x.addr).unwrap();
        a.free(y.addr).unwrap();
        let s = a.stats();
        assert_eq!(s.internal_fragmentation_bytes, 0);
        assert_eq!(s.internal_fragmentation_ratio, 0.0);
    }

    #[test]
    fn exhaustive_min_block_allocations_are_disjoint_and_tiling() {
        // Pool 16, min block 1: 16 single-byte allocations tile the pool.
        let mut a = BuddyAllocator::new(16, 1).unwrap();
        let mut allocs = Vec::new();
        for _ in 0..16 {
            allocs.push(a.allocate(1).unwrap());
        }
        assert_eq!(a.allocate(1), Err(AllocError::OutOfMemory));
        // All addresses 0..16 exactly once.
        let mut addrs: Vec<usize> = allocs.iter().map(|x| x.addr).collect();
        addrs.sort_unstable();
        assert_eq!(addrs, (0..16).collect::<Vec<_>>());
        a.verify().unwrap();
        for x in allocs {
            a.free(x.addr).unwrap();
            a.verify().unwrap();
        }
        assert_eq!(a.free_blocks(), vec![(0, 4)]);
    }

    #[test]
    fn pool_with_min_block_one_supports_all_power_of_two_sizes() {
        let mut a = BuddyAllocator::new(8, 1).unwrap();
        let x = a.allocate(3).unwrap();
        assert_eq!(x.order, 2); // 4-byte block
        assert_eq!(a.allocate(4).unwrap_err(), AllocError::OutOfMemory);
        a.free(x.addr).unwrap();
        let y = a.allocate(4).unwrap();
        assert_eq!(y.order, 2);
        a.free(y.addr).unwrap();
        a.verify().unwrap();
    }
}
