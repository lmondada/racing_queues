# Racing Queues

Racing Queues to keep track of overlaps between two queues.

A racing queue is formed of two queues (the left and the right queue).
Each queue can be viewed as forming an interval from the minimum in the queue
to the maximum.
At any time, `drain_overlap` will pop and return all elements from either
queue that belong to the intersection of the queue intervals.

When restricted to either the left or right queue, all insertions must be
made in non-decreasing order.

To be memory efficient, racing queues automatically discard elements that
become too small as larger elements are inserted left or right.

### Example
```rust
use racing_queues::{RacingQueues, Entry};
let mut r = RacingQueues::new();
r.push_left(1);
r.push_left(2);
r.push_right(3);
let entries = vec![Entry::Left(1), Entry::Left(2), Entry::Right(3)];
assert_eq!(r, entries.into());
r.push_left(4);
r.push_right(4);
let overlap: Vec<_> = r.drain_overlap().collect();
assert_eq!(overlap, vec![Entry::Right(3), Entry::Left(4), Entry::Right(4)]);
```

This will panic, as the third element is not in non-decreasing order:
```rust
use racing_queues::{RacingQueues, Entry, Side};
let mut r = RacingQueues::new();
r.push_left(2);
// This is fine
r.push_right(1);
// This isn't
r.push_left(1);
```