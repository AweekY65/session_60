# Buddy Memory Allocator

A self-contained **binary buddy memory allocator simulator** written in Rust.
The whole allocator — the memory pool, the per-order free lists, the allocation
index and all statistics — lives in ordinary, process-local heap memory
(`Vec<u8>` / `Vec` / `HashMap`). It does **not** use any OS-specific memory
management service, `mmap`/`brk`, a database, or any other external service.
The buddy split/coalesce algorithm is implemented from scratch in
[`src/lib.rs`](src/lib.rs).

## Concepts and buddy arithmetic

The pool size and the minimum block size are both powers of two, with
`pool_size >= min_block_size`. Blocks are managed at `max_order + 1`
granularities ("orders"):

| order | block size                     |
|-------|--------------------------------|
| 0     | `min_block_size`               |
| 1     | `min_block_size * 2`           |
| `k`   | `min_block_size << k`          |
| max   | `pool_size`                    |

Every block of order `k` is naturally aligned to its own size, so its starting
address is a multiple of `block_size(k)`.

**Buddy of a block.** Two blocks of the same order are buddies when they are the
two halves obtained by splitting one block of the next order. Because every
block is aligned to its size, the buddy address is a single XOR away:

```text
buddy_addr(addr, order) = addr XOR block_size(order)
```

The aligned parent formed by merging a pair starts at
`min(addr, buddy_addr)` (equivalently `addr & !block_size(order)`).

### `allocate(size)`

1. Reject zero-sized and larger-than-pool requests.
2. Choose the **smallest** fitting order:
   `order = log2(next_power_of_two(ceil(size / min_block_size)))`.
3. Find the lowest order `>= target` whose free list is non-empty.
4. Pop that block and **split recursively** down to the target order. Each
   split divides a block into two aligned buddies: one half keeps being split,
   the other is pushed onto the corresponding lower-order free list.
5. Record the block as allocated and update the byte counters.

### `free(addr)`

1. Validate the request:
   * `AddressOutOfRange` — address outside `[0, pool_size)`;
   * `NotBlockStart` — address is strictly inside a live allocated block;
   * `NotBlockAligned` — in range but not a multiple of `min_block_size`;
   * `DoubleFree` — the handle was already released (this is also detected when
     the block has since been absorbed into a larger coalesced free block);
   * `NotAllocated` — aligned and in range, but names a free block produced by
     an internal split that the caller never owned.
2. Remove the allocation and insert its block into the matching free list.
3. **Recursively coalesce**: while the block's buddy is free *at exactly the
   same order*, remove the buddy from its free list, merge into the aligned
   parent, and repeat one order up. Coalescing minimizes external
   fragmentation automatically.

### Statistics

`BuddyAllocator::stats()` returns:

* `allocated_bytes` — sum of the **requested** sizes of live allocations;
* `used_bytes` — bytes covered by the granted blocks;
* `free_bytes` — bytes currently in free blocks (`total - used`);
* `internal_fragmentation_bytes` — `used_bytes - allocated_bytes`, i.e. bytes
  wasted because a request is rounded up to a power-of-two block;
* `internal_fragmentation_ratio` — internal fragmentation divided by used bytes;
* `utilization` — requested bytes divided by the pool size;
* `free_blocks_per_order` — number of free blocks at every order.

In addition, `BuddyAllocator::verify()` checks every structural invariant:
free-list/index agreement, block alignment and range, pairwise non-overlap of
all blocks, and that allocated and free blocks **tile the pool exactly**
(space conservation). It is invoked after every single operation in the test
suites.

## Complexity

Let `M = max_order = log2(pool_size / min_block_size)`.

| operation                              | time                              | extra space |
|----------------------------------------|-----------------------------------|-------------|
| `allocate(size)` (splits + list push/pop) | `O(M)` worst case, `O(1)` when a block of the target order is free | none (list updates) |
| `free(addr)` (validation + coalescing) | `O(M)` worst case, `O(1)` when the buddy is not free | none |
| `stats()`                              | `O(number of free blocks)`        | — |
| `verify()`                             | `O(n log n)` (`n` = total blocks, dominated by sorting the interval walk) | — |

Free lists use `Vec` as stacks (`pop`/`push` = `O(1)`) plus a `swap_remove`
when unlinking a merged buddy, so coalescing stays linear in the number of
merges — at most `M`.

## Layout

```text
Cargo.toml
src/lib.rs                 # allocator implementation + unit tests
tests/integration_tests.rs # targeted tests + large-scale deterministic RNG stress tests
tests/property_tests.rs    # property-based tests (built-in deterministic harness)
examples/demo.rs           # short API walk-through
```

## Requirements

* A Rust toolchain with Cargo (edition 2021).
* **No external dependencies** (neither runtime nor dev-dependencies): the
  property-based tests use a small deterministic xorshift-driven harness
  included in the repository, so `cargo test` works fully offline.

## Running the tests

```bash
cargo test                # unit + integration + property-based tests
cargo test --release      # same, optimized (large stress tests run faster)
```

Useful filters:

```bash
cargo test stress         # only the large-scale randomized stress tests
cargo test property       # only the property-based suites
cargo run --example demo  # print stats across a small allocate/free sequence
```

The test suites cover:

* minimum- and maximum-sized allocations;
* continuous split cascades and recursive merging at every order;
* internal-fragmentation accounting and per-order free counts;
* memory exhaustion and recovery;
* every invalid-free variant (out of range, unaligned, non-start interior
  address, double free including stale handles after coalescing, unknown
  addresses);
* large-scale randomized allocate/free workloads (20k steps per run across
  several pool shapes and seeds, plus repeated exhaust/drain cycles) with a
  full invariant check after **every** step;
* property-based tests that generate thousands of random operation sequences
  over many pool configurations/seeds and assert that live blocks never
  overlap, space is always conserved (`used + free == pool_size`), counters
  always agree with an independent model, OOM results always agree with the
  free lists, and freeing everything always rebuilds the pool as a single
  max-order free block. Every failure reports the exact configuration, seed
  and step, which replays deterministically.
