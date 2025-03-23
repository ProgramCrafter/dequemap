use ftree::FenwickTree;
use std::borrow::Borrow;
use std::iter::{Flatten, FlatMap, Map};
use std::ops::{Bound, RangeBounds};

// Constants
const NODE_CAPACITY: usize = 64;

// Main structure
pub struct DequeMap<K, V> {
    sublists: Vec<Vec<(K, V)>>,
    fenwick: FenwickTree<usize>,
    node_capacity: usize,
}

fn only_key<K, V>((k, _): (K, V)) -> K {k}
fn only_val<K, V>((_, v): (K, V)) -> V {v}

// Iterator types
pub type IntoItems<K, V>  = Flatten<std::vec::IntoIter<Vec<(K, V)>>>;
pub type IntoKeys<K, V>   = Map<IntoItems<K, V>, fn((K, V)) -> K>;
pub type IntoValues<K, V> = Map<IntoItems<K, V>, fn((K, V)) -> V>;

pub struct IterMap<'a, K: 'a, V: 'a>(FlatMap<std::slice::Iter<'a, Vec<(K, V)>>, std::slice::Iter<'a, (K, V)>, fn(&'a Vec<(K, V)>) -> std::slice::Iter<'a, (K, V)>>);
pub struct IterMut<'a, K: 'a, V: 'a>(FlatMap<std::slice::IterMut<'a, Vec<(K, V)>>, std::slice::IterMut<'a, (K, V)>, fn(&'a mut Vec<(K, V)>) -> std::slice::IterMut<'a, (K, V)>>);
pub struct Keys<'a, K: 'a, V: 'a>(IterMap<'a, K, V>);
pub struct Values<'a, K: 'a, V: 'a>(IterMap<'a, K, V>);
pub struct ValuesMut<'a, K: 'a, V: 'a>(IterMut<'a, K, V>);
pub struct RangeMap<'a, K: 'a, V: 'a> {
    current_front_iter: Option<std::slice::Iter<'a, (K, V)>>,
    remaining_sublists: std::slice::Iter<'a, Vec<(K, V)>>,
    len: usize,
}
pub struct RangeMut<'a, K: 'a, V: 'a> {
    current_front_iter: Option<std::slice::IterMut<'a, (K, V)>>,
    remaining_sublists: std::slice::IterMut<'a, Vec<(K, V)>>,
    len: usize,
}
pub struct CursorMap<'a, K: 'a, V: 'a> {
    sublists: &'a [Vec<(K, V)>],
    sublist_idx: usize,
    pos: usize,
}

// Entry types
pub enum Entry<'a, K: 'a, V: 'a> {
    Vacant(VacantEntry<'a, K, V>),
    Occupied(OccupiedEntry<'a, K, V>),
}

pub struct VacantEntry<'a, K: 'a, V: 'a> {
    map: &'a mut DequeMap<K, V>,
    key: K,
    sublist_idx: usize,
}

pub struct OccupiedEntry<'a, K: 'a, V: 'a> {
    map: &'a mut DequeMap<K, V>,
    sublist_idx: usize,
    pos: usize,
}

impl<K, V> DequeMap<K, V>
where
    K: Ord,
{
    // --- Auxiliary Functions ---

    fn find_sublist_for_key<Q>(&self, key: &Q) -> (usize, Option<usize>)
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let i = self.sublists.partition_point(|sublist| {
            if sublist.is_empty() { true } else { sublist[0].0.borrow() < key }
        });
        let consider = if i == 0 { 0 } else { i - 1 };
        let sublist = &self.sublists[consider];
        if sublist.is_empty() || key < sublist[0].0.borrow() {
            (consider, None)
        } else if key > sublist.last().unwrap().0.borrow() {
            (consider, None)
        } else {
            match sublist.binary_search_by_key(&key, |(k, _)| k.borrow()) {
                Ok(j) => (consider, Some(j)),
                Err(_j) => (consider, None),
            }
        }
    }

    fn find_sublist_for_index(&self, index: usize) -> Option<(usize, usize)> {
        if index >= self.len() {
            None
        } else {
            let mut low = 0;
            let mut high = self.sublists.len();
            while low < high {
                let mid = low + (high - low) / 2;
                if self.fenwick.prefix_sum(mid, 0) <= index {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            let sublist_idx = low - 1;
            let offset = self.fenwick.prefix_sum(sublist_idx, 0);
            let pos = index - offset;
            Some((sublist_idx, pos))
        }
    }

    fn split_sublist(&mut self, idx: usize) {
        let sublist = &mut self.sublists[idx];
        let mid = sublist.len() / 2;
        let new_sublist = sublist.split_off(mid);
        self.sublists.insert(idx + 1, new_sublist);
        let sizes = self.sublists.iter().map(|s| s.len());
        self.fenwick = FenwickTree::from_iter(sizes);
    }

    fn rebuild_fenwick(&mut self) {
        let sizes = self.sublists.iter().map(|s| s.len());
        self.fenwick = FenwickTree::from_iter(sizes);
    }
    
    fn find_lower_bound<Q>(&self, key: &Q) -> (usize, usize)
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let idx = self.sublists.partition_point(|sublist| {
            sublist.is_empty() || sublist[0].0.borrow() < key
        });
        let consider = idx.saturating_sub(1);
        let sublist = &self.sublists[consider];
        let pos = sublist.partition_point(|(k, _)| k.borrow() < key);
        if pos < sublist.len() {
            (consider, pos)
        } else if consider + 1 < self.sublists.len() {
            (consider + 1, 0)
        } else {
            (self.sublists.len(), 0)
        }
    }

    fn find_upper_bound<Q>(&self, key: &Q) -> (usize, usize)
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let idx = self.sublists.partition_point(|sublist| {
            sublist.is_empty() || sublist[0].0.borrow() <= key
        });
        let consider = if idx == 0 { 0 } else { idx - 1 };
        let sublist = &self.sublists[consider];
        let pos = sublist.partition_point(|(k, _)| k.borrow() <= key);
        if pos < sublist.len() {
            (consider, pos)
        } else if consider + 1 < self.sublists.len() {
            (consider + 1, 0)
        } else {
            (self.sublists.len(), 0)
        }
    }

    // --- Public API ---

    pub fn append(&mut self, other: &mut Self) {
        let fast_merge_ok = match (self.last_key_value(), other.first_key_value()) {
            (Some((x,_)), Some((y,_))) => x < y,
            _                  => true
        };
        
        if !fast_merge_ok {
            for _ in 0..other.len() {
                let (k, v) = other.pop_first().unwrap();
                self.insert(k, v);
            }
            
            return;
        }
        
        let other_sublists = std::mem::replace(&mut other.sublists, vec![Vec::with_capacity(NODE_CAPACITY)]);
        self.sublists.extend(other_sublists);
        self.rebuild_fenwick();
        other.fenwick = FenwickTree::new();
        other.fenwick.push(0);
    }

    pub fn clear(&mut self) {
        self.sublists = vec![Vec::with_capacity(NODE_CAPACITY)];
        self.fenwick = FenwickTree::new();
        self.fenwick.push(0);
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (_, pos) = self.find_sublist_for_key(key);
        pos.is_some()
    }

    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        for sublist in &self.sublists {
            if let Some((k, v)) = sublist.first() {
                return Some((k, v));
            }
        }
        None
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (sublist_idx, pos) = self.find_sublist_for_key(key);
        pos.map(|j| &self.sublists[sublist_idx][j].1)
    }

    pub fn get_index(&self, idx: usize) -> Option<(&K, &V)> {
        self.find_sublist_for_index(idx)
            .map(|(sublist_idx, pos)| &self.sublists[sublist_idx][pos])
            .map(|(k, v)| (k, v))
    }

    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (sublist_idx, pos) = self.find_sublist_for_key(key);
        pos.map(|j| &self.sublists[sublist_idx][j])
            .map(|(k, v)| (k, v))
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (sublist_idx, pos) = self.find_sublist_for_key(key);
        pos.map(move |j| &mut self.sublists[sublist_idx][j].1)
    }

    pub fn get_mut_index(&mut self, index: usize) -> Option<&mut V> {
        self.find_sublist_for_index(index)
            .map(move |(sublist_idx, pos)| &mut self.sublists[sublist_idx][pos].1)
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let (sublist_idx, pos) = self.find_sublist_for_key(&key);
        let sublist = &mut self.sublists[sublist_idx];
        if let Some(j) = pos {
            let old_value = std::mem::replace(&mut sublist[j].1, value);
            Some(old_value)
        } else {
            let insert_pos = sublist.partition_point(|(k, _)| k < &key);
            sublist.insert(insert_pos, (key, value));
            self.fenwick.add_at(sublist_idx, 1);
            if sublist.len() > self.node_capacity {
                self.split_sublist(sublist_idx);
            }
            None
        }
    }
    
    pub fn consume(self) -> IntoItems<K, V> {
        self.sublists.into_iter().flatten()
    }

    pub fn into_keys(self) -> IntoKeys<K, V> {
        self.consume().map(only_key)
    }

    pub fn into_values(self) -> IntoValues<K, V> {
        self.consume().map(only_val)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> IterMap<'_, K, V> {
        IterMap(self.sublists.iter().flat_map(|sublist| sublist.iter()))
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut(self.sublists.iter_mut().flat_map(|sublist| sublist.iter_mut()))
    }

    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys(self.iter())
    }

    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        self.sublists.last().and_then(|sublist| sublist.last().map(|(k, v)| (k, v)))
    }

    pub fn len(&self) -> usize {
        self.fenwick.prefix_sum(self.sublists.len(), 0)
    }

    pub fn new() -> Self {
        let mut this = DequeMap {
            sublists: Vec::new(),
            fenwick: FenwickTree::new(),
            node_capacity: NODE_CAPACITY,
        };
        this.clear();
        this
    }

    pub fn pop_first(&mut self) -> Option<(K, V)> {
        for i in 0..self.sublists.len() {
            if !self.sublists[i].is_empty() {
                let (k, v) = self.sublists[i].remove(0);
                self.fenwick.sub_at(i, 1);
                if self.sublists[i].is_empty() && i > 0 {
                    self.sublists.remove(i);
                    self.rebuild_fenwick();
                }
                return Some((k, v));
            }
        }
        None
    }

    pub fn pop_index(&mut self, index: usize) -> (K, V) {
        let (sublist_idx, pos) = self.find_sublist_for_index(index)
            .expect("Index out of bounds");
        let sublist = &mut self.sublists[sublist_idx];
        let (k, v) = sublist.remove(pos);
        self.fenwick.sub_at(sublist_idx, 1);
        if sublist.is_empty() && sublist_idx > 0 {
            self.sublists.remove(sublist_idx);
            self.rebuild_fenwick();
        }
        (k, v)
    }

    pub fn pop_last(&mut self) -> Option<(K, V)> {
        if let Some(i) = (0..self.sublists.len()).rev().find(|&i| !self.sublists[i].is_empty()) {
            let sublist = &mut self.sublists[i];
            let (k, v) = sublist.pop().unwrap();
            self.fenwick.sub_at(i, 1);
            if sublist.is_empty() && i > 0 {
                self.sublists.remove(i);
                self.rebuild_fenwick();
            }
            Some((k, v))
        } else {
            None
        }
    }

    pub fn range<Q, R>(&self, range: R) -> RangeMap<K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
        R: RangeBounds<Q>,
    {
        let start = match range.start_bound() {
            Bound::Included(key) => self.find_lower_bound(key),
            Bound::Excluded(key) => self.find_upper_bound(key),
            Bound::Unbounded => (0, 0),
        };
        let end = match range.end_bound() {
            Bound::Included(key) => self.find_upper_bound(key),
            Bound::Excluded(key) => self.find_lower_bound(key),
            Bound::Unbounded => (self.sublists.len(), 0),
        };
        
        let start_i = self.fenwick.prefix_sum(start.0, start.1);
        let end_i   = self.fenwick.prefix_sum(end.0, end.1);
        let mut sublists = self.sublists[start.0..end.0].iter();
        let first_iter = sublists.next().map(|n| n[start.1..].iter());
        
        RangeMap {
            current_front_iter: first_iter,
            remaining_sublists: sublists,
            len: end_i - start_i,
        }
    }

    pub fn range_mut<Q, R>(&mut self, range: R) -> RangeMut<'_, K, V>
    where
        Q: Ord + ?Sized,
        K: Borrow<Q>,
        R: RangeBounds<Q>,
    {
        let start = match range.start_bound() {
            Bound::Included(key) => self.find_lower_bound(key),
            Bound::Excluded(key) => self.find_upper_bound(key),
            Bound::Unbounded => (0, 0),
        };
        let end = match range.end_bound() {
            Bound::Included(key) => self.find_upper_bound(key),
            Bound::Excluded(key) => self.find_lower_bound(key),
            Bound::Unbounded => (self.sublists.len(), 0),
        };
        
        let start_i = self.fenwick.prefix_sum(start.0, start.1);
        let end_i   = self.fenwick.prefix_sum(end.0, end.1);
        let mut sublists = self.sublists[start.0..end.0].iter_mut();
        let first_iter = sublists.next().map(|n| n[start.1..].iter_mut());
        
        RangeMut {
            current_front_iter: first_iter,
            remaining_sublists: sublists,
            len: end_i - start_i,
        }
    }

    pub fn range_mut_idx<R>(&mut self, range: R) -> RangeMut<'_, K, V>
    where
        R: RangeBounds<usize>,
    {
        let start = match range.start_bound() {
            Bound::Included(&i) => i,
            Bound::Excluded(&i) => i + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&i) => i + 1,
            Bound::Excluded(&i) => i,
            Bound::Unbounded => self.len(),
        };
        let (start_idx, start_pos) = self.find_sublist_for_index(start).unwrap_or((0, 0));
        let (end_idx, _end_pos) = self.find_sublist_for_index(end - 1)
            .map_or((self.sublists.len() - 1, self.sublists.last().map_or(0, |s| s.len())),
                    |(idx, pos)| (idx, pos + 1));
        
        let mut sublists = self.sublists[start_idx..end_idx].iter_mut();
        let first_iter = sublists.next().map(|n| n[start_pos..].iter_mut());
        
        RangeMut {
            current_front_iter: first_iter,
            remaining_sublists: sublists,
            len: end.saturating_sub(start),
        }
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (sublist_idx, pos) = self.find_sublist_for_key(key);
        if let Some(j) = pos {
            let sublist = &mut self.sublists[sublist_idx];
            let (_, v) = sublist.remove(j);
            self.fenwick.sub_at(sublist_idx, 1);
            if sublist.is_empty() && sublist_idx > 0 {
                self.sublists.remove(sublist_idx);
                self.rebuild_fenwick();
            }
            Some(v)
        } else {
            None
        }
    }

    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (sublist_idx, pos) = self.find_sublist_for_key(key);
        if let Some(j) = pos {
            let sublist = &mut self.sublists[sublist_idx];
            let kv = sublist.remove(j);
            self.fenwick.sub_at(sublist_idx, 1);
            if sublist.is_empty() && sublist_idx > 0 {
                self.sublists.remove(sublist_idx);
                self.rebuild_fenwick();
            }
            Some(kv)
        } else {
            None
        }
    }

    pub fn retain<F, Q>(&mut self, mut f: F)
    where
        F: FnMut(&Q, &mut V) -> bool,
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        for i in 0..self.sublists.len() {
            let sublist = &mut self.sublists[i];
            let old_len = sublist.len();
            
            let mut new_sublist = Vec::new();
            for (k, mut v) in sublist.drain(..) {
                if f(k.borrow(), &mut v) {
                    new_sublist.push((k, v));
                }
            }
            *sublist = new_sublist;
            
            let new_len = sublist.len();
            if old_len != new_len {
                self.fenwick.sub_at(i, old_len - new_len);
            }
            if sublist.is_empty() && i > 0 {
                self.sublists.remove(i);
                self.rebuild_fenwick();
                break; // Rebuild invalidates indices, so restart or handle carefully
            }
        }
    }

    pub fn split_off<Q>(&mut self, key: &Q) -> Self
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (sublist_idx, pos) = self.find_sublist_for_key(key);
        let split_pos = pos.unwrap_or_else(|| self.sublists[sublist_idx].partition_point(|(k, _)| k.borrow() < key));
        let mut new_sublists = self.sublists.split_off(sublist_idx);
        if split_pos > 0 {
            let remaining = new_sublists[0].split_off(split_pos);
            self.sublists.push(remaining);
        }
        let mut new_fenwick = FenwickTree::new();
        for s in &new_sublists {
            new_fenwick.push(s.len());
        }
        self.rebuild_fenwick();
        DequeMap {
            sublists: new_sublists,
            fenwick: new_fenwick,
            node_capacity: NODE_CAPACITY,
        }
    }

    pub fn values(&self) -> Values<'_, K, V> {
        Values(self.iter())
    }

    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut(self.iter_mut())
    }

    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        let (sublist_idx, pos) = self.find_sublist_for_key(&key);
        if let Some(j) = pos {
            Entry::Occupied(OccupiedEntry {
                map: self,
                sublist_idx,
                pos: j,
            })
        } else {
            Entry::Vacant(VacantEntry {
                map: self,
                key,
                sublist_idx,
            })
        }
    }

    pub fn first_entry(&mut self) -> Option<OccupiedEntry<'_, K, V>> {
        for i in 0..self.sublists.len() {
            if !self.sublists[i].is_empty() {
                return Some(OccupiedEntry {
                    map: self,
                    sublist_idx: i,
                    pos: 0,
                });
            }
        }
        None
    }

    pub fn last_entry(&mut self) -> Option<OccupiedEntry<'_, K, V>> {
        let maybe_i = (0..self.sublists.len()).rev().find(|&i| !self.sublists[i].is_empty());
        if let Some(i) = maybe_i {
            let pos = self.sublists[i].len() - 1;
            Some(OccupiedEntry {
                map: self,
                sublist_idx: i,
                pos,
            })
        } else {
            None
        }
    }

    pub fn lower_bound<Q>(&self, bound: Bound<&Q>) -> CursorMap<'_, K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (sublist_idx, pos) = match bound {
            Bound::Included(key) => {
                let (idx, pos) = self.find_sublist_for_key(key);
                if let Some(p) = pos { (idx, p) } else { (idx, self.sublists[idx].partition_point(|(k, _)| k.borrow() < key)) }
            }
            Bound::Excluded(key) => {
                let (idx, pos) = self.find_sublist_for_key(key);
                let p = pos.unwrap_or_else(|| self.sublists[idx].partition_point(|(k, _)| k.borrow() < key));
                if p < self.sublists[idx].len() { (idx, p + 1) } else { (idx + 1, 0) }
            }
            Bound::Unbounded => (0, 0),
        };
        CursorMap {
            sublists: &self.sublists,
            sublist_idx,
            pos,
        }
    }

    pub fn rank<Q>(&self, value: &Q) -> usize
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (sublist_idx, pos) = self.find_sublist_for_key(value);
        let offset = self.fenwick.prefix_sum(sublist_idx, 0);
        let sublist = &self.sublists[sublist_idx];
        let rank_in_sublist = pos.unwrap_or_else(|| sublist.partition_point(|(k, _)| k.borrow() < value));
        offset + rank_in_sublist
    }
}

impl<K: Ord, V, const N: usize> From<[(K, V); N]> for DequeMap<K, V> {
    fn from(arr: [(K, V); N]) -> Self {
        let mut map = DequeMap::new();
        for (k, v) in arr {
            map.insert(k, v);
        }
        map
    }
}

impl<K: Ord + Clone, V: Clone> Clone for DequeMap<K, V> {
    fn clone(&self) -> Self {
        DequeMap {
            sublists: self.sublists.clone(),
            fenwick: self.fenwick.clone(),
            node_capacity: self.node_capacity,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.sublists.clone_from(&source.sublists);
        self.fenwick.clone_from(&source.fenwick);
        self.node_capacity = source.node_capacity;
    }
}

// --- Iterator Implementations ---

impl<'a, K, V> Iterator for IterMap<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(k, v)| (k, v))
    }
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(k, v)| (&*k, v))
    }
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(k, _)| k)
    }
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(_, v)| v)
    }
}

impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(_, v)| v)
    }
}

impl<'a, K, V> Iterator for RangeMap<'a, K, V> {
    type Item = (&'a K, &'a V);
    
    fn next(&mut self) -> Option<Self::Item> {
        self.len = self.len.checked_sub(1)?;    // return if == 0
        
        loop {
            if let Some((k, v)) = self.current_front_iter.as_mut().unwrap().next() {
                return Some((k, v));
            }
            self.current_front_iter = self.remaining_sublists.next().map(|sl| sl.iter());
        }
    }
}

impl<'a, K, V> Iterator for RangeMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.len = self.len.checked_sub(1)?;    // return if == 0
        
        loop {
            if let Some((k, v)) = self.current_front_iter.as_mut().unwrap().next() {
                return Some((&*k, v));
            }
            self.current_front_iter = self.remaining_sublists.next().map(|sl| sl.iter_mut());
        }
    }
}

impl<'a, K: 'a, V: 'a> Iterator for CursorMap<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        if self.sublist_idx >= self.sublists.len() {
            return None;
        }
        let sublist = &self.sublists[self.sublist_idx];
        if self.pos >= sublist.len() {
            self.sublist_idx += 1;
            self.pos = 0;
            self.next()
        } else {
            let (k, v) = &sublist[self.pos];
            self.pos += 1;
            Some((k, v))
        }
    }
}
impl<'a, K: 'a, V: 'a> CursorMap<'a, K, V> {
    pub fn key(&self) -> Option<&K> {
        let mut sublists_it = self.sublists[self.sublist_idx..].iter();
        let mut sublist = sublists_it.next()?;
        let mut pos = self.pos;
        
        while pos >= sublist.len() {
            pos = 0;
            sublist = sublists_it.next()?;
        }
        
        sublist.get(pos).map(|(k, _)| k)
    }
}

// --- Entry Implementations ---

impl<'a, K: Ord, V> VacantEntry<'a, K, V> {
    pub fn insert(self, value: V) -> &'a mut V {
        let sublist = &mut self.map.sublists[self.sublist_idx];
        let insert_pos = sublist.partition_point(|(k, _)| k < &self.key);
        sublist.insert(insert_pos, (self.key, value));
        self.map.fenwick.add_at(self.sublist_idx, 1);
        if sublist.len() > self.map.node_capacity {
            self.map.split_sublist(self.sublist_idx);
        }
        &mut self.map.sublists[self.sublist_idx][insert_pos].1
    }

    pub fn key(&self) -> &K {
        &self.key
    }

    pub fn into_key(self) -> K {
        self.key
    }
}

impl<'a, K: Ord, V> OccupiedEntry<'a, K, V> {
    pub fn get(&self) -> &V {
        &self.map.sublists[self.sublist_idx][self.pos].1
    }

    pub fn get_mut(&mut self) -> &mut V {
        &mut self.map.sublists[self.sublist_idx][self.pos].1
    }

    pub fn into_mut(self) -> &'a mut V {
        &mut self.map.sublists[self.sublist_idx][self.pos].1
    }

    pub fn key(&self) -> &K {
        &self.map.sublists[self.sublist_idx][self.pos].0
    }

    pub fn remove(self) -> V {
        let sublist = &mut self.map.sublists[self.sublist_idx];
        let (_, v) = sublist.remove(self.pos);
        self.map.fenwick.sub_at(self.sublist_idx, 1);
        if sublist.is_empty() && self.sublist_idx > 0 {
            self.map.sublists.remove(self.sublist_idx);
            self.map.rebuild_fenwick();
        }
        v
    }

    pub fn remove_entry(self) -> (K, V) {
        let sublist = &mut self.map.sublists[self.sublist_idx];
        let kv = sublist.remove(self.pos);
        self.map.fenwick.sub_at(self.sublist_idx, 1);
        if sublist.is_empty() && self.sublist_idx > 0 {
            self.map.sublists.remove(self.sublist_idx);
            self.map.rebuild_fenwick();
        }
        kv
    }

    pub fn insert(&mut self, value: V) -> V {
        std::mem::replace(&mut self.map.sublists[self.sublist_idx][self.pos].1, value)
    }
}

impl<'a, K: Ord, V> Entry<'a, K, V> {
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default),
        }
    }
}

