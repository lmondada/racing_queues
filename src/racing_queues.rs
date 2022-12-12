use std::{
    cmp::{self, Reverse},
    collections::VecDeque,
    mem,
    ops::{Index, Range},
};

use crate::Entry;

/// Racing Queues to keep track of overlaps between two queues
///
/// A racing queue is formed of two queues (the left and the right queue).
/// Each queue can be viewed as forming an interval from the minimum in the queue
/// to the maximum.
/// At any time, `drain_overlap` will pop and return all elements from either
/// queue that belong to the intersection of the queue intervals.

/// When restricted to either the left or right queue, all insertions should be
/// made in non-decreasing order. Non-ordered pushes are allowed but inefficient
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
/// let overlap: Vec<_> = r.drain_overlap(None).collect();
/// assert_eq!(overlap, vec![Entry::Right(3), Entry::Left(4), Entry::Right(4)]);
/// ```
#[derive(Clone, PartialEq, Debug)]
pub struct RacingQueues<T> {
    queues: [VecDeque<T>; 2],
}

/// First index in sorted queue that is either `Greater` or `Less` than T
///
/// Note: little quirk: if ordering == Less, then ind is offset by one
/// (because actual index could be -1)
/// This works out nicely because ranges are [), ie including start but excluding end
fn first_past<'a, T: Ord>(queue: &'a VecDeque<T>, limit: &T, ordering: cmp::Ordering) -> usize {
    match ordering {
        cmp::Ordering::Less => queue.partition_point(|v| v < limit),
        cmp::Ordering::Greater => queue.partition_point(|v| v <= limit),
        cmp::Ordering::Equal => panic!("Invalid Ordering"),
    }
}

// A VecDeque with reversed ordering and indexing
#[derive(PartialEq, Eq)]
struct RevVecDeque<T>(VecDeque<T>);

impl<T> Index<usize> for RevVecDeque<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[self.0.len() - index - 1]
    }
}

impl<T: Ord> Ord for RevVecDeque<T> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        Reverse(self).cmp(&Reverse(other))
    }
}
impl<T: Ord> PartialOrd for RevVecDeque<T> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(&other))
    }
}

impl<T: Ord> RacingQueues<T> {
    pub fn new() -> Self {
        Self::default()
    }

    // Push values in (more or less) increasing order
    // Any values can be pushed, but small values are inefficient
    pub fn push(&mut self, v: Entry<T>) {
        let side = v.side();
        let insert_into = &mut self.queues[side.index()];
        let idx = insert_into.partition_point(|x| x < v.value());
        insert_into.insert(idx, v.into_value());
        // insert_into.push_back(v.into_value());
    }

    pub fn push_left(&mut self, v: T) {
        self.push(Entry::Left(v))
    }

    pub fn push_right(&mut self, v: T) {
        self.push(Entry::Right(v))
    }

    /// Jump the queue!
    ///
    /// Only allowed if the entry is smaller than entries in both queues
    pub fn push_front(&mut self, v: Entry<T>) {
        let side = v.side();
        let min = cmp::min(self.queues[0].front(), self.queues[1].front());
        if min.is_some() && v.value() > min.unwrap() {
            panic!("Cannot insert larger value at the front")
        }
        let insert_into = &mut self.queues[side.index()];
        insert_into.push_front(v.into_value());
    }

    pub fn overlap(&self, limit: Option<&T>) -> impl Iterator<Item = Entry<&T>> + '_ {
        let [r0, r1] = self.overlap_range(limit);
        LeftRightIter::new(
            r0.map(|i| &self.queues[0][i]),
            r1.map(|i| &self.queues[1][i]),
        )
    }

    fn overlap_range(&self, limit: Option<&T>) -> [Range<usize>; 2] {
        let mut start_range = [0; 2];
        let mut end_range = [0; 2];

        // Compare tops of queues to find end of overlap
        let mut inds = [0, 1];
        inds.sort_by(|a, b| self.queues[*a].back().cmp(&self.queues[*b].back()));
        let [small, large] = inds;

        // For small queue, end of overlap is the end of the queue, or limit
        end_range[small] = if let Some(limit) = limit {
            first_past(&self.queues[small], limit, cmp::Ordering::Greater)
        } else {
            self.queues[small].len()
        };
        // For large queue, end of overlap is overlap s.t.
        // large[overlap - 1] <= small.back() < large[overlap]
        end_range[large] = if let Some(max_of_small) = self.queues[small].back() {
            let max_of_small = limit
                .map(|limit| cmp::min(limit, max_of_small))
                .unwrap_or(max_of_small);
            first_past(&self.queues[large], max_of_small, cmp::Ordering::Greater)
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
            first_past(&self.queues[small], min_of_large, cmp::Ordering::Less)
        } else {
            self.queues[small].len()
        };

        // Make sure empty ranges still satisfy start <= end
        for i in [0, 1] {
            start_range[i] = cmp::min(start_range[i], end_range[i]);
        }

        [0, 1].map(|i| start_range[i]..end_range[i])
    }

    pub fn drain_overlap(&mut self, limit: Option<&T>) -> impl Iterator<Item = Entry<T>> + '_ {
        let ranges = self.overlap_range(limit);
        let mut its = self
            .queues_mut()
            .into_iter()
            .zip(ranges)
            .map(|(q, r)| {
                // For efficiency, we start by draining the tail with elements too small
                q.drain(0..r.start);
                let r = 0..(r.end - r.start);
                q.drain(r)
            })
            .collect::<Vec<_>>()
            .into_iter();
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
        assert_eq!(
            r,
            vec![Entry::Left(1), Entry::Left(2), Entry::Right(3)].into()
        );
        r.push_left(3);
        assert_eq!(
            r.drain_overlap(None).collect::<Vec<_>>(),
            vec![LEFT.entry(3), RIGHT.entry(3)]
        )
    }

    #[test]
    fn racing_queue_identical() {
        let mut r = RacingQueues::new();
        r.push_left(2);
        r.push_left(2);
        assert_eq!(
            r,
            vec![
                Entry::Left(2),
                Entry::Left(2),
            ]
            .into()
        );
        r.push_right(2);
        assert_eq!(r.queues, [vec![2, 2], vec![2]]);
        assert_eq!(
            r.drain_overlap(Some(&2)).collect::<Vec<_>>(),
            [
                LEFT.entry(2),
                LEFT.entry(2),
                RIGHT.entry(2),
            ]
        );
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
            r.drain_overlap(None).collect::<Vec<_>>(),
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
        assert_eq!(r.queues, [vec![1, 2], vec![0, 1]]);
        r.push_right(1);
        assert_eq!(r.queues, [vec![1, 2], vec![0, 1, 1]]);
        r.push_right(4);
        assert_eq!(r.queues, [vec![1, 2], vec![0, 1, 1, 4]]);
        assert_eq!(
            r.drain_overlap(None).collect::<Vec<_>>(),
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
            r.drain_overlap(None).collect::<Vec<_>>(),
            [RIGHT.entry(2), LEFT.entry(3), RIGHT.entry(3)]
        );
    }

    #[test]
    fn more_queuing_tests() {
        let mut r = RacingQueues::new();
        r.push_left(1);
        r.push_left(3);
        r.push_left(5);
        assert_eq!(r.overlap(None).collect::<Vec<_>>(), []);
        let r2 = r.clone();
        r.drain_overlap(None);
        assert_eq!(r, r2);
        r.push_right(3);
        assert_eq!(
            r.overlap(None)
                .map(|e| e.map(|u| u.clone()))
                .collect::<Vec<_>>(),
            [LEFT.entry(3), RIGHT.entry(3)]
        );
        r.push_right(6);
        assert_eq!(
            r.overlap(None)
                .map(|e| e.map(|u| u.clone()))
                .collect::<Vec<_>>(),
            [LEFT.entry(3), RIGHT.entry(3), LEFT.entry(5)]
        );
        assert_eq!(
            r.overlap(Some(&5))
                .map(|e| e.map(|u| u.clone()))
                .collect::<Vec<_>>(),
            [LEFT.entry(3), RIGHT.entry(3), LEFT.entry(5)]
        );
        assert_eq!(
            r.overlap(Some(&4))
                .map(|e| e.map(|u| u.clone()))
                .collect::<Vec<_>>(),
            [LEFT.entry(3), RIGHT.entry(3)]
        );
        assert_eq!(
            r.overlap(Some(&2))
                .map(|e| e.map(|u| u.clone()))
                .collect::<Vec<_>>(),
            []
        );
        r.drain_overlap(None);
        assert_eq!(r.queues, [vec![], vec![6]]);
    }

    #[test]
    fn disjoint_overlap() {
        let mut r = RacingQueues::new();
        r.push_left(1);
        r.push_left(3);
        r.push_left(5);
        r.push_right(6);
        assert_eq!(r.overlap(None).count(), 0);
    }
}
