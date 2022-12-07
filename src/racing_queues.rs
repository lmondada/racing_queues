use std::{collections::VecDeque, mem};

use crate::Entry;

/// Racing Queues to keep track of overlaps between two queues
/// 
/// A racing queue is formed of two queues (the left and the right queue).
/// Each queue can be viewed as forming an interval from the minimum in the queue
/// to the maximum.
/// At any time, `drain_overlap` will pop and return all elements from either
/// queue that belong to the intersection of the queue intervals.

/// When restricted to either the left or right queue, all insertions must be
/// made in non-decreasing order.
///
/// To be memory efficient, racing queues automatically discard elements that
/// become too small as larger elements are inserted left or right.
/// 
/// # Example
/// ```
/// use racing_queues::{RacingQueues, Entry};
/// let mut r = RacingQueues::new();
/// r.push_left(1);
/// r.push_left(2);
/// r.push_right(3);
/// let entries = vec![Entry::Left(1), Entry::Left(2), Entry::Right(3)];
/// assert_eq!(r, entries.into());
/// r.push_left(4);
/// r.push_right(4);
/// let overlap: Vec<_> = r.drain_overlap().collect();
/// assert_eq!(overlap, vec![Entry::Right(3), Entry::Left(4), Entry::Right(4)]);
/// ```
/// 
/// This will panic, as the third element is not in non-decreasing order:
/// ```should_panic
/// use racing_queues::{RacingQueues, Entry, Side};
/// let mut r = RacingQueues::new();
/// r.push_left(2);
/// // This is fine
/// r.push_right(1);
/// // This isn't
/// r.push_left(1);
/// ```
#[derive(Clone, PartialEq, Debug)]
pub struct RacingQueues<T> {
    queues: [VecDeque<T>; 2],
}

impl<T: Ord> RacingQueues<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, v: Entry<T>) {
        let side = v.side();
        let insert_into = &mut self.queues[side.index()];
        if Some(v.value()) < insert_into.back() {
            panic!("Cannot insert smaller value")
        }
        insert_into.push_back(v.into_value());

        // To save memory, delete tail (i.e. elements too small to ever be overlapping)
        let mut queues = self.queues_mut();
        queues.sort_by(|a, b| a.front().cmp(&b.front()));
        let [small, large] = queues;
        if let Some(large_min) = large.front() {
            let small_max = (0..small.len())
                .find(|ind| small[*ind] > *large_min)
                .unwrap_or(small.len());
            // we want to leave small_max - 1, the last element that is smaller than large_min
            if small_max > 1 {
                small.drain(..(small_max - 1));
            }
        }
    }

    pub fn push_left(&mut self, v: T) {
        self.push(Entry::Left(v))
    }

    pub fn push_right(&mut self, v: T) {
        self.push(Entry::Right(v))
    }

    pub fn drain_overlap(&mut self) -> impl Iterator<Item = Entry<T>> + '_ {
        let mut queues = self.queues_mut();
        queues.sort_by(|a, b| a.back().cmp(&b.back()));
        let [small, large] = queues;
        // For small, delete tail
        while small.front() < large.front() {
            small.pop_front();
        }
        // For large, find out the interval [0..overlap) to drain
        let overlap = if let Some(max_in_small) = small.back() {
            // Find overlap s.t. large[overlap - 1] <= small.back() < large[overlap]
            match large.binary_search(max_in_small) {
                Ok(ind) => {
                    // edge past this entry
                    let v = &large[ind];
                    ((ind + 1)..large.len())
                        .find(|ind| large[*ind] != *v)
                        .unwrap_or(large.len())
                }
                Err(ind) => ind,
            }
        } else {
            0
        };
        // Drain the whole of small
        let it1 = small.drain(..);
        // Drain [0..overlap) of large
        let it2 = large.drain(..overlap);
        LeftRightIter::new(it1, it2)
    }

    pub fn iter(&self) -> impl Iterator<Item = Entry<&T>> + '_ {
        LeftRightIter::new(self.queues[0].iter(), self.queues[1].iter())
    }

    pub fn into_iter(self) -> impl Iterator<Item = Entry<T>> {
        let [q1, q2] = self.queues;
        LeftRightIter::new(q1.into_iter(), q2.into_iter())
    }

    fn queues_mut(&mut self) -> [&mut VecDeque<T>; 2] {
        let (q1, q2) = self.queues.split_at_mut(1);
        let q1 = q1.first_mut().expect("We know self.queues is len 2");
        let q2 = q2.first_mut().expect("We know self.queues is len 2");
        [q1, q2]
    }
}

impl<T> Default for RacingQueues<T> {
    fn default() -> Self {
        Self {
            queues: Default::default(),
        }
    }
}

impl<T: Ord> FromIterator<Entry<T>> for RacingQueues<T> {
    fn from_iter<I: IntoIterator<Item = Entry<T>>>(iter: I) -> Self {
        let mut ret = Self::default();
        for e in iter.into_iter() {
            ret.push(e);
        }
        ret
    }
}

impl<T: Ord> From<RacingQueues<T>> for Vec<Entry<T>> {
    fn from(queues: RacingQueues<T>) -> Self {
        queues.into_iter().collect()
    }
}

impl<T: Ord> From<Vec<Entry<T>>> for RacingQueues<T> {
    fn from(vec: Vec<Entry<T>>) -> Self {
        vec.into_iter().collect()
    }
}

struct LeftRightIter<I1: Iterator, I2: Iterator> {
    it1: I1,
    it2: I2,
    next1: Option<I1::Item>,
    next2: Option<I2::Item>,
}

impl<V: Ord, I1: Iterator<Item = V>, I2: Iterator<Item = V>> Iterator for LeftRightIter<I1, I2> {
    type Item = Entry<V>;

    fn next(&mut self) -> Option<Self::Item> {
        match (&self.next1, &self.next2) {
            (None, None) => None,
            (Some(_), None) => {
                mem::replace(&mut self.next1, self.it1.next()).map(|v| Entry::Left(v))
            }
            (None, Some(_)) => {
                mem::replace(&mut self.next2, self.it2.next()).map(|v| Entry::Right(v))
            }
            (Some(v1), Some(v2)) => {
                if *v1 <= *v2 {
                    mem::replace(&mut self.next1, self.it1.next()).map(|v| Entry::Left(v))
                } else {
                    mem::replace(&mut self.next2, self.it2.next()).map(|v| Entry::Right(v))
                }
            }
        }
    }
}

impl<I1: Iterator, I2: Iterator> LeftRightIter<I1, I2> {
    fn new(mut it1: I1, mut it2: I2) -> Self {
        let next1 = it1.next();
        let next2 = it2.next();
        Self {
            it1,
            it2,
            next1,
            next2,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Side;

    use super::*;

    const LEFT: Side = Side::Left(());
    const RIGHT: Side = Side::Right(());

    #[test]
    fn racing_queue_simple() {
        let mut r = RacingQueues::new();
        r.push_left(1);
        r.push_left(2);
        assert_eq!(r, vec![Entry::Left(1), Entry::Left(2)].into());
        r.push_right(3);
        assert_eq!(r, vec![Entry::Left(2), Entry::Right(3)].into());
    }

    #[test]
    fn racing_queue_several_identical() {
        let mut r = RacingQueues::new();
        r.push_left(1);
        r.push_left(2);
        r.push_left(2);
        r.push_left(2);
        r.push_left(2);
        assert_eq!(
            r,
            vec![
                Entry::Left(1),
                Entry::Left(2),
                Entry::Left(2),
                Entry::Left(2),
                Entry::Left(2)
            ]
            .into()
        );
        r.push_right(1);
        assert_eq!(r.queues, [vec![1, 2, 2, 2, 2], vec![1]]);
        assert_eq!(
            r.drain_overlap().collect::<Vec<_>>(),
            [LEFT.entry(1), RIGHT.entry(1)]
        );
        assert_eq!(r.queues, [vec![2, 2, 2, 2], vec![]]);
    }

    #[test]
    fn racing_queue_internals() {
        let mut r = RacingQueues::new();
        r.push_left(1);
        r.push_left(2);
        r.push_right(0);
        r.push_right(1);
        assert_eq!(r.queues, [vec![1, 2], vec![1]]);
        r.push_right(1);
        assert_eq!(r.queues, [vec![1, 2], vec![1, 1]]);
        r.push_right(4);
        println!("{:?}", r);
        assert_eq!(r.queues, [vec![1, 2], vec![1, 1, 4]]);
        assert_eq!(
            r.drain_overlap().collect::<Vec<_>>(),
            [LEFT.entry(1), RIGHT.entry(1), RIGHT.entry(1), LEFT.entry(2)]
        );
    }
}
