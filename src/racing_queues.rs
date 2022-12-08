use std::{collections::VecDeque, mem, ops::Range};

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

    pub fn overlap(&self) -> impl Iterator<Item = Entry<&T>> + '_ {
        let [r0, r1] = self.overlap_range();
        LeftRightIter::new(
            r0.map(|i| &self.queues[0][i]),
            r1.map(|i| &self.queues[1][i]),
        )
    }

    fn overlap_range(&self) -> [Range<usize>; 2] {
        let mut start_range = [0; 2];
        let mut end_range = [0; 2];

        // Compare tops of queues to find end of overlap
        let mut inds = [0, 1];
        inds.sort_by(|a, b| self.queues[*a].back().cmp(&self.queues[*b].back()));
        let [small, large] = inds;

        // For small queue, end of overlap is the end of the queue
        end_range[small] = self.queues[small].len();
        // For large queue, end of overlap is overlap s.t.
        // large[overlap - 1] <= small.back() < large[overlap]
        end_range[large] = if let Some(max_of_small) = self.queues[small].back() {
            match self.queues[large].binary_search(max_of_small) {
                Ok(ind) => {
                    // edge past this entry
                    let v = &self.queues[large][ind];
                    ((ind + 1)..self.queues[large].len())
                        .find(|ind| self.queues[large][*ind] != *v)
                        .unwrap_or(self.queues[large].len())
                }
                Err(ind) => ind,
            }
        } else {
            0
        };

        // Compare bottoms of queues to find beginning of overlap
        let mut inds = [0, 1];
        inds.sort_by(|a, b| self.queues[*a].front().cmp(&self.queues[*b].front()));
        let [small, large] = inds;

        // For large queue, beginning of overlap is the beginning of the queue
        start_range[large] = 0;
        // For small queue, beginning of overlap is overlap s.t.
        // small[overlap - 1] < large.front() <= small[overlap]
        start_range[small] = if let Some(min_of_large) = self.queues[large].front() {
            match self.queues[small].binary_search(min_of_large) {
                Ok(ind) => {
                    // edge past this entry
                    let v = &self.queues[small][ind];
                    (0..ind).rev().find(|ind| self.queues[small][*ind] != *v).unwrap_or(0)
                }
                Err(ind) => ind,
            }
        } else {
            self.queues[small].len()
        };

        [0, 1].map(|i| start_range[i]..end_range[i])
    }

    pub fn drain_overlap(&mut self) -> impl Iterator<Item = Entry<T>> + '_ {
        let ranges = self.overlap_range();
        // For efficiency, we start by draining the tail with elements too small
        let mut its = self.queues_mut().into_iter().zip(ranges).map(|(q, r)| {
            q.drain(0..r.start);
            let r = 0..(r.end - r.start);
            q.drain(r)
        }).collect::<Vec<_>>().into_iter();
        let it0 = its.next().unwrap();
        let it1 = its.next().unwrap();

        LeftRightIter::new(it0, it1)
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
        assert_eq!(r.queues, [vec![1, 2], vec![1, 1, 4]]);
        assert_eq!(
            r.drain_overlap().collect::<Vec<_>>(),
            [LEFT.entry(1), RIGHT.entry(1), RIGHT.entry(1), LEFT.entry(2)]
        );
    }

    #[test]
    fn racing_queue_fully_included() {
        let mut r = RacingQueues::new();
        r.push_left(1);
        r.push_left(3);
        r.push_left(5);
        r.push_right(2);
        r.push_right(3);
        assert_eq!(
            r.drain_overlap().collect::<Vec<_>>(),
            [RIGHT.entry(2), LEFT.entry(3), RIGHT.entry(3)]
        );
    }
}
