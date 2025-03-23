use std::iter::{Flatten, FlatMap, Map};
use std::ops::{Bound, RangeBounds};
use std::borrow::Borrow;
use std::cmp::Ordering;
use ftree::FenwickTree;

const NODE_CAPACITY: usize = 64;

// Main structure
#[derive(Debug)]
pub struct DequeMap<K, V> {
    sublists: Vec<Vec<(K, V)>>,
    fenwick: FenwickTree<usize>,
    node_capacity: usize,
}


/// Until #[feature(substr_range)] is stabilized
#[inline]
fn slice_element_offset<T>(origin: &[T], element: &T) -> Option<usize> {
    let t_size = std::mem::size_of::<T>();
    if t_size == 0 {
        panic!("elements are zero-sized");
    }
    
    let self_start = origin.as_ptr().addr();
    let elem_start = std::ptr::from_ref(element).addr();
    let byte_offset = elem_start.wrapping_sub(self_start);
    if byte_offset % t_size != 0 {
        return None;
    }
    
    let offset = byte_offset / t_size;
    if offset < origin.len() { Some(offset) } else { None }
}

fn reborrow<'a, K: 'a, V: 'a>((r,s): &'a (K, V)) -> (&'a K, &'a V) {
    (r, s)
}


// Iterator types
pub type IntoItems<K, V>  = Flatten<std::vec::IntoIter<Vec<(K, V)>>>;
pub type IntoKeys<K, V>   = Map<IntoItems<K, V>, fn((K, V)) -> K>;
pub type IntoValues<K, V> = Map<IntoItems<K, V>, fn((K, V)) -> V>;
type Iter<'a, K, V>       = FlatMap<std::slice::Iter<'a, Vec<(K, V)>>, std::slice::Iter<'a, (K, V)>, fn(&'a Vec<(K, V)>) -> std::slice::Iter<'a, (K, V)>>;
pub type IterMap<'a, K, V>   = Map<Iter<'a, K, V>, fn(&'a (K, V)) -> (&'a K, &'a V)>;
pub type Keys<'a, K, V>      = Map<Iter<'a, K, V>, fn(&'a (K, V)) -> &'a K>;
pub type Values<'a, K, V>    = Map<Iter<'a, K, V>, fn(&'a (K, V)) -> &'a V>;
pub type ValuesMut<'a, K, V> = Map<IterMut<'a, K, V>, fn(&'a mut (K, V)) -> &'a mut V>;

type IterMut<'a, K, V> = FlatMap<std::slice::IterMut<'a, Vec<(K, V)>>, std::slice::IterMut<'a, (K, V)>, fn(&'a mut Vec<(K, V)>) -> std::slice::IterMut<'a, (K, V)>>;
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
        let consider = i.saturating_sub(1);
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

    fn split_sublist(&mut self, idx: usize) {
        let sublist = &mut self.sublists[idx];
        let mid = sublist.len() / 2;
        let new_sublist = sublist.split_off(mid);
        self.sublists.insert(idx + 1, new_sublist);
        self.rebuild_fenwick();
    }

    fn rebuild_fenwick(&mut self) {
        let sizes = self.sublists.iter().map(|s| s.len());
        self.fenwick = FenwickTree::from_iter(sizes);
    }
    
    pub fn validate_buckets(&self) where (K, V): std::fmt::Debug {
        assert!(!self.sublists.is_empty());
        assert_eq!(self.sublists.iter().map(|sl| sl.len()).sum::<usize>(), self.len());
        
        assert!(self.keys().is_sorted());
    }
    
    /// The comparator function should return an order code whether `Node`
    /// with specified index is Less, Equal or Greater than required one.
    /// 
    /// If a suitable node is found then [`Result::Ok`] is returned,
    /// referring to it. Alternatively, [`Result::Err`] is returned showing
    /// where a matching node could be inserted while maintaining sorted
    /// order.
    fn locate_node_by<P>(&self, mut cmp: P) -> Result<(usize, &Vec<(K, V)>), usize>
    where
        P: FnMut(usize, &Vec<(K, V)>) -> Ordering
    {
        self.sublists
            .binary_search_by(|node| {
                cmp(slice_element_offset(&self.sublists, node).unwrap(),
                    node)
            })
            .map(|node_id| (node_id, &self.sublists[node_id]))
    }
    
    /// For 0 <= idx < self.len(), locates node containing that element.
    fn locate_node_with_idx_inbounds(&self, idx: usize) -> (usize, &Vec<(K, V)>) {
        self.locate_node_by(|node_id, contained| {
            let prior_elements = self.fenwick.prefix_sum(node_id, 0);
            if prior_elements > idx {
                Ordering::Greater
            } else if prior_elements + contained.len() > idx {
                Ordering::Equal
            } else {
                Ordering::Less
            }
        }).expect("locate_node_by failed to produce node by index")
    }
    
    /// For a key, locates a node which contains it, or could do so potentially.
    fn locate_node_with_key<Q>(&self, key: &Q) -> (usize, &Vec<(K, V)>) where K: Borrow<Q>, Q: Ord + ?Sized {
        match self.locate_node_by(|_, contained| {
            if contained.is_empty() {  // invariant 2. it's the first sublist
                Ordering::Less
            } else if contained.last().unwrap().0.borrow() < key {
                Ordering::Less
            } else if contained[0].0.borrow() > key {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }) {
            // contained[0] <= key <= contained[-1]
            Ok((node_id, contained)) => (node_id, contained),
            Err(mut n) => {
                // The key should be inserted prior to self.sublists[n].
                // In self.sublists[n-1], that is, if n is nonzero.
                n = n.saturating_sub(1);
                (n, &self.sublists[n])
            }
        }
    }
    
    /// Returns (sublist_idx, place) such that all elements with strictly smaller indices are lower than key,
    /// and >= indices have elements greater than or equal to key.
    fn find_lower_bound<Q>(&self, key: &Q) -> (usize, usize) where K: Borrow<Q>, Q: Ord + ?Sized {
        let (sublist_idx, node) = self.locate_node_with_key(key);
        match node.binary_search_by_key(&key, |(k,_)| k.borrow()) {
            Ok(j)  => (sublist_idx, j),
            Err(j) => (sublist_idx, j)
        }
    }
    /// Returns (sublist_idx, place) such that all elements with strictly smaller indices are lower than or equal to key,
    /// and >= indices have elements greater than key.
    fn find_upper_bound<Q>(&self, key: &Q) -> (usize, usize) where K: Borrow<Q>, Q: Ord + ?Sized {
        let upper_node: Result<_, _> = self.locate_node_by(|_, contained| {
            if contained.is_empty() {
                Ordering::Less
            } else if contained.last().unwrap().0.borrow() <= key {
                Ordering::Less
            } else if contained[0].0.borrow() > key {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
        match upper_node {
            Ok((sublist_idx, node)) => match node.binary_search_by_key(&key, |(k,_)| k.borrow()) {
                Ok(j)  => (sublist_idx, j + 1),
                Err(j) => (sublist_idx, j),
            },
            Err(sublist_place) => {
                (sublist_place, 0)
            }
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
        
        self.sublists.extend(other.sublists.extract_if(.., |sl| !sl.is_empty()));
        self.rebuild_fenwick();
        other.clear();
    }

    pub fn clear(&mut self) {
        self.sublists = vec![Vec::with_capacity(NODE_CAPACITY)];
        self.fenwick = FenwickTree::from_iter([0]);
    }

    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        self.sublists.iter().find_map(|sl| sl.first()).map(reborrow)
    }
    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        self.sublists.last().and_then(|sublist| sublist.last().map(|(k, v)| (k, v)))
    }
    pub fn first_entry(&mut self) -> Option<OccupiedEntry<'_, K, V>> {
        match self.sublists.iter().position(|sl| !sl.is_empty()) {
            None => None,
            Some(sublist_idx) => Some(OccupiedEntry { map: self, sublist_idx, pos: 0 })
        }
    }
    pub fn last_entry(&mut self) -> Option<OccupiedEntry<'_, K, V>> {
        match self.sublists.iter().rposition(|sl| !sl.is_empty()) {
            None => None,
            Some(sublist_idx) => {
                let pos = self.sublists[sublist_idx].len() - 1;
                Some(OccupiedEntry { map: self, sublist_idx, pos })
            }
        }
    }

    pub fn get_index(&self, idx: usize) -> Option<(&K, &V)> {
        if idx >= self.len() {return None;}
        let (sublist_idx, node) = self.locate_node_with_idx_inbounds(idx);
        node.get(idx - self.fenwick.prefix_sum(sublist_idx, 0)).map(reborrow)
    }
    pub fn get_mut_index(&mut self, idx: usize) -> Option<&mut V> {
        if idx >= self.len() {return None;}
        let (sublist_idx, _) = self.locate_node_with_idx_inbounds(idx);
        let node = &mut self.sublists[sublist_idx];
        node.get_mut(idx - self.fenwick.prefix_sum(sublist_idx, 0)).map(|(_,v)| v)
    }

    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)> where K: Borrow<Q>, Q: Ord + ?Sized {
        let (_, node) = self.locate_node_with_key(key);
        let j = node.binary_search_by_key(&key, |(k,_)| k.borrow());
        j.ok().and_then(|j| node.get(j)).map(reborrow)
    }
    pub fn get<Q>(&self, key: &Q) -> Option<&V> where K: Borrow<Q>, Q: Ord + ?Sized {
        self.get_key_value(key).map(|(_, v)| v)
    }
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V> where K: Borrow<Q>, Q: Ord + ?Sized {
        let (sublist_idx, node) = self.locate_node_with_key(key);
        let j = node.binary_search_by_key(&key, |(k,_)| k.borrow());
        let node = &mut self.sublists[sublist_idx];
        j.ok().and_then(|j| node.get_mut(j)).map(|(_, v)| v)
    }
    pub fn contains_key<Q>(&self, key: &Q) -> bool where K: Borrow<Q>, Q: Ord + ?Sized {
        self.get_key_value(key).is_some()
    }
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        let (sublist_idx, node) = self.locate_node_with_key(&key);
        if let Ok(j) = node.binary_search_by_key(&&key, |(k,_)| k) {
            Entry::Occupied(OccupiedEntry { map: self, sublist_idx, pos: j })
        } else {
            Entry::Vacant(VacantEntry { map: self, key, sublist_idx })
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let (sublist_idx, _) = self.locate_node_with_key(&key);
        let sublist = &mut self.sublists[sublist_idx];
        match sublist.binary_search_by_key(&&key, |(k,_)| k) {
            Ok(j) => {
                let old_value = std::mem::replace(&mut sublist[j].1, value);
                Some(old_value)
            },
            Err(j) => {
                sublist.insert(j, (key, value));
                self.fenwick.add_at(sublist_idx, 1);
                if sublist.len() > self.node_capacity {
                    self.split_sublist(sublist_idx);
                }
                None
            }
        }
    }
    
    pub fn consume(self) -> IntoItems<K, V> {
        self.sublists.into_iter().flatten()
    }
    pub fn into_keys(self) -> IntoKeys<K, V> {
        self.consume().map(|(k, _)| k)
    }
    pub fn into_values(self) -> IntoValues<K, V> {
        self.consume().map(|(_, v)| v)
    }
    fn inner_iter(&self) -> Iter<'_, K, V> {
        self.sublists.iter().flat_map(|sublist| sublist.iter())
    }
    pub fn iter(&self) -> IterMap<'_, K, V> {
        self.inner_iter().map(reborrow)
    }
    pub fn keys(&self) -> Keys<'_, K, V> {
        self.inner_iter().map(|(k, _)| k)
    }
    fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        self.sublists.iter_mut().flat_map(|sublist| sublist.iter_mut())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
    
    fn pop_pair(&mut self, sublist_idx: usize, item_idx: usize) -> (K, V) {
        let node = &mut self.sublists[sublist_idx];
        let pair = node.remove(item_idx);
        self.fenwick.sub_at(sublist_idx, 1);
        if node.is_empty() && sublist_idx != 0 {
            self.sublists.remove(sublist_idx);
            self.rebuild_fenwick();
            // TODO: pull some elements
        }
        pair
    }

    pub fn pop_first(&mut self) -> Option<(K, V)> {
        let i = self.sublists.iter().position(|sl| !sl.is_empty())?;
        Some(self.pop_pair(i, 0))
    }
    pub fn pop_index(&mut self, index: usize) -> (K, V) {
        assert!(index < self.len(), "index out of bounds");
        let (sublist_idx, _) = self.locate_node_with_idx_inbounds(index);
        self.pop_pair(sublist_idx, index - self.fenwick.prefix_sum(sublist_idx, 0))
    }
    pub fn pop_last(&mut self) -> Option<(K, V)> {
        let i = self.sublists.iter().rposition(|sl| !sl.is_empty())?;
        Some(self.pop_pair(i, self.sublists[i].len() - 1))
    }
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)> where K: Borrow<Q>, Q: Ord + ?Sized {
        let (sublist_idx, node) = self.locate_node_with_key(key);
        let j = node.binary_search_by_key(&key, |(k,_)| k.borrow()).ok()?;
        Some(self.pop_pair(sublist_idx, j))
    }
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V> where K: Borrow<Q>, Q: Ord + ?Sized {
        self.remove_entry(key).map(|(_, v)| v)
    }

    pub fn range<Q, R>(&self, range: R) -> RangeMap<K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
        R: RangeBounds<Q>,
    {
        let (start_sublist, start_place) = match range.start_bound() {
            Bound::Included(key) => self.find_lower_bound(key),
            Bound::Excluded(key) => self.find_upper_bound(key),
            Bound::Unbounded => (0, 0),
        };
        let (end_sublist, end_place) = match range.end_bound() {
            Bound::Included(key) => self.find_upper_bound(key),
            Bound::Excluded(key) => self.find_lower_bound(key),
            Bound::Unbounded => (self.sublists.len(), 0),
        };
        if end_sublist == 0 && end_place == 0 {return RangeMap::empty();}
        let end_sublist_incl = end_sublist - (end_place == 0) as usize;
        
        let start_i = self.fenwick.prefix_sum(start_sublist, start_place);
        let end_i   = self.fenwick.prefix_sum(end_sublist, end_place);
        let mut sublists = self.sublists[start_sublist..=end_sublist_incl].iter();
        let first_iter = sublists.next().map(|n| n[start_place..].iter());
        
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
        let (start_sublist, start_place) = match range.start_bound() {
            Bound::Included(key) => self.find_lower_bound(key),
            Bound::Excluded(key) => self.find_upper_bound(key),
            Bound::Unbounded => (0, 0),
        };
        let (end_sublist, end_place) = match range.end_bound() {
            Bound::Included(key) => self.find_upper_bound(key),
            Bound::Excluded(key) => self.find_lower_bound(key),
            Bound::Unbounded => (self.sublists.len(), 0),
        };
        if end_sublist == 0 && end_place == 0 {return RangeMut::empty();}
        let end_sublist_incl = end_sublist - (end_place == 0) as usize;
        
        let start_i = self.fenwick.prefix_sum(start_sublist, start_place);
        let end_i   = self.fenwick.prefix_sum(end_sublist, end_place);
        let mut sublists = self.sublists[start_sublist..=end_sublist_incl].iter_mut();
        let first_iter = sublists.next().map(|n| n[start_place..].iter_mut());
        
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
            Bound::Unbounded => usize::MAX,
        }.min(self.len());
        if start >= end {
            return RangeMut::empty();
        }
        
        let (start_node, _) = self.locate_node_with_idx_inbounds(start);
        let (end_node_incl, _) = self.locate_node_with_idx_inbounds(end - 1);
        let start_pos = start - self.fenwick.prefix_sum(start_node, 0);
        let mut sublists = self.sublists[start_node..=end_node_incl].iter_mut();
        let first_iter = sublists.next().map(|n| n[start_pos..].iter_mut());
        
        RangeMut {
            current_front_iter: first_iter,
            remaining_sublists: sublists,
            len: end.saturating_sub(start),
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
            sublist.retain_mut(|(k, v)| f((*k).borrow(), v));
        }
        self.sublists.retain(|sl| !sl.is_empty());
        if self.sublists.is_empty() {
            self.sublists.push(Vec::with_capacity(NODE_CAPACITY));
        }
        self.rebuild_fenwick();
    }

    /// Splits the collection into two at the given key. Returns everything after the given key, including the key.
    pub fn split_off<Q>(&mut self, key: &Q) -> Self where K: Borrow<Q>, Q: Ord + ?Sized {
        let (sublist_idx, pos) = self.find_lower_bound(key);
        // self.sublists[0..sublist_idx] are ours
        // self.sublists[sublist_idx][..pos] is ours
        
        let mut right = Self::new();
        if let Some(sl) = self.sublists.get_mut(sublist_idx) {
            right.sublists[0].append(&mut sl.split_off(pos));
            right.sublists.append(&mut self.sublists.split_off(sublist_idx + 1));
        }
        if self.sublists.last().unwrap().is_empty() && self.sublists.len() > 1 {
            let _ = self.sublists.pop();
        }
        right.rebuild_fenwick();
        self.rebuild_fenwick();
        right
    }

    pub fn values(&self) -> Values<'_, K, V> {
        self.inner_iter().map(|(_, v)| v)
    }

    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        self.iter_mut().map(|(_, v)| v)
    }

    pub fn lower_bound<Q>(&self, bound: Bound<&Q>) -> CursorMap<'_, K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (sublist_idx, pos) = match bound {
            Bound::Included(key) => self.find_lower_bound(key),
            Bound::Excluded(key) => self.find_upper_bound(key),
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
impl<'a, K, V> RangeMap<'a, K, V> {
    fn empty() -> Self {
        Self {
            current_front_iter: Some([].iter()),
            remaining_sublists: [].iter(),
            len: 0
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
impl<'a, K, V> RangeMut<'a, K, V> {
    fn empty() -> Self {
        Self {
            current_front_iter: Some([].iter_mut()),
            remaining_sublists: [].iter_mut(),
            len: 0
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
        let VacantEntry{mut sublist_idx, map, key} = self;
        
        let sublist = &mut map.sublists[sublist_idx];
        let j = match sublist.binary_search_by_key(&&key, |(k,_)| k) {
            Ok(_) => unreachable!(),
            Err(mut j) => {
                sublist.insert(j, (key, value));
                map.fenwick.add_at(sublist_idx, 1);
                if sublist.len() > map.node_capacity {
                    map.split_sublist(sublist_idx);
                    if j >= map.sublists[sublist_idx].len() {
                        j -= map.sublists[sublist_idx].len();
                        sublist_idx += 1;
                    }
                    j
                } else {
                    j
                }
            }
        };
        &mut map.sublists[sublist_idx][j].1
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
        self.map.pop_pair(self.sublist_idx, self.pos).1
    }

    pub fn remove_entry(self) -> (K, V) {
        self.map.pop_pair(self.sublist_idx, self.pos)
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

