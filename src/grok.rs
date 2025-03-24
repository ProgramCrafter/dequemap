#![forbid(unsafe_code)]

use std::ops::{Bound, RangeBounds, Index, IndexMut};
use std::iter::{Flatten, FlatMap, Map};
use std::borrow::Borrow;
use std::cmp::Ordering;
use ftree::FenwickTree;

const NODE_CAPACITY: usize = 64;

/// An ordered map based on a two-dimensional sorted list, similar to some deque implementations.
#[derive(Debug)]
pub struct LiqueMap<K, V> {
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
pub struct RangeMap<'a, K, V> {
    current_front_iter: Option<std::slice::Iter<'a, (K, V)>>,
    current_back_iter: Option<std::slice::Iter<'a, (K, V)>>,
    remaining_sublists: std::slice::Iter<'a, Vec<(K, V)>>,
    len: usize,
}
pub struct RangeMut<'a, K, V> {
    current_front_iter: Option<std::slice::IterMut<'a, (K, V)>>,
    current_back_iter: Option<std::slice::IterMut<'a, (K, V)>>,
    remaining_sublists: std::slice::IterMut<'a, Vec<(K, V)>>,
    len: usize,
}
pub struct CursorMap<'a, K, V> {
    sublists: &'a [Vec<(K, V)>],
    sublist_idx: usize,
    pos: usize,
}

// Entry types
pub enum Entry<'a, K, V> {
    Vacant(VacantEntry<'a, K, V>),
    Occupied(OccupiedEntry<'a, K, V>),
}

pub struct VacantEntry<'a, K, V> {
    map: &'a mut LiqueMap<K, V>,
    key: K,
    sublist_idx: usize,
}

pub struct OccupiedEntry<'a, K, V> {
    map: &'a mut LiqueMap<K, V>,
    sublist_idx: usize,
    pos: usize,
}

impl<K, V> LiqueMap<K, V>
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
    
    /// Checks whether the internal invariants are maintained correctly.
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

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// If a key from `other` is already present in `self`, the respective
    /// value from `self` will be overwritten with the respective value from `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut a = LiqueMap::new();
    /// a.insert(2, "b");
    /// a.insert(3, "c"); // Note: Key (3) also present in b.
    ///
    /// let mut b = LiqueMap::new();
    /// b.insert(3, "d"); // Note: Key (3) also present in a.
    /// b.insert(4, "e");
    /// b.insert(5, "f");
    ///
    /// a.append(&mut b);
    ///
    /// assert_eq!(a.len(), 4);
    /// assert_eq!(b.len(), 0);
    ///
    /// assert_eq!(a[&2], "b");
    /// assert_eq!(a[&3], "d"); // Note: "c" has been overwritten.
    /// assert_eq!(a[&4], "e");
    /// assert_eq!(a[&5], "f");
    /// ```
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

    /// Clears the map, removing all elements.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut a = LiqueMap::new();
    /// a.insert(1, "a");
    /// a.clear();
    /// assert!(a.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.sublists = vec![Vec::with_capacity(NODE_CAPACITY)];
        self.fenwick = FenwickTree::from_iter([0]);
    }

    /// Returns the first key-value pair in the map.
    /// The key in this pair is the minimum key in the map.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut map = LiqueMap::new();
    /// assert_eq!(map.first_key_value(), None);
    /// map.insert(1, "b");
    /// map.insert(2, "a");
    /// assert_eq!(map.first_key_value(), Some((&1, &"b")));
    /// ```
    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        self.sublists.iter().find_map(|sl| sl.first()).map(reborrow)
    }
    /// Returns the last key-value pair in the map.
    /// The key in this pair is the maximum key in the map.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut map = LiqueMap::new();
    /// map.insert(1, "b");
    /// map.insert(2, "a");
    /// assert_eq!(map.last_key_value(), Some((&2, &"a")));
    /// ```
    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        self.sublists.last().and_then(|sublist| sublist.last().map(|(k, v)| (k, v)))
    }
    /// Returns the first entry in the map for in-place value manipulation.
    /// The key of this entry is the minimum key in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut map = LiqueMap::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// if let Some(mut entry) = map.first_entry() {
    ///     if *entry.key() > 0 {
    ///         entry.insert("first");
    ///     }
    /// }
    /// assert_eq!(*map.get(&1).unwrap(), "first");
    /// assert_eq!(*map.get(&2).unwrap(), "b");
    /// ```
    pub fn first_entry(&mut self) -> Option<OccupiedEntry<'_, K, V>> {
        match self.sublists.iter().position(|sl| !sl.is_empty()) {
            None => None,
            Some(sublist_idx) => Some(OccupiedEntry { map: self, sublist_idx, pos: 0 })
        }
    }
    /// Returns the last entry in the map for in-place value manipulation.
    /// The key of this entry is the maximum key in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut map = LiqueMap::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// if let Some(mut entry) = map.last_entry() {
    ///     if *entry.key() > 0 {
    ///         entry.insert("last");
    ///     }
    /// }
    /// assert_eq!(*map.get(&1).unwrap(), "a");
    /// assert_eq!(*map.get(&2).unwrap(), "last");
    /// ```
    pub fn last_entry(&mut self) -> Option<OccupiedEntry<'_, K, V>> {
        match self.sublists.iter().rposition(|sl| !sl.is_empty()) {
            None => None,
            Some(sublist_idx) => {
                let pos = self.sublists[sublist_idx].len() - 1;
                Some(OccupiedEntry { map: self, sublist_idx, pos })
            }
        }
    }

    /// Returns a reference to the pair ranked idx-th in sort order in the map, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut map = LiqueMap::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// assert_eq!(map.get_index(2), None);
    /// assert_eq!(map.get_index(0), Some((&1, &"a")));
    /// assert_eq!(map[&1], "a");
    /// ```
    pub fn get_index(&self, idx: usize) -> Option<(&K, &V)> {
        if idx >= self.len() {return None;}
        let (sublist_idx, node) = self.locate_node_with_idx_inbounds(idx);
        node.get(idx - self.fenwick.prefix_sum(sublist_idx, 0)).map(reborrow)
    }
    /// Returns a mutable reference to the value attached to idx-th ranked key in the map, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut map = LiqueMap::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// assert_eq!(map.get_mut_index(4), None);
    /// *map.get_mut_index(0).unwrap() = "mu";
    /// assert_eq!(map.get(&1), Some(&"mu"));
    /// ```
    pub fn get_mut_index(&mut self, idx: usize) -> Option<&mut V> {
        if idx >= self.len() {return None;}
        let (sublist_idx, _) = self.locate_node_with_idx_inbounds(idx);
        let node = &mut self.sublists[sublist_idx];
        node.get_mut(idx - self.fenwick.prefix_sum(sublist_idx, 0)).map(|(_,v)| v)
    }

    /// Returns the key-value pair corresponding to the supplied key.
    ///
    /// The supplied key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut map = LiqueMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get_key_value(&1), Some((&1, &"a")));
    /// assert_eq!(map.get_key_value(&2), None);
    /// ```
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)> where K: Borrow<Q>, Q: Ord + ?Sized {
        let (_, node) = self.locate_node_with_key(key);
        let j = node.binary_search_by_key(&key, |(k,_)| k.borrow());
        j.ok().and_then(|j| node.get(j)).map(reborrow)
    }
    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut map = LiqueMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&2), None);
    /// ```
    pub fn get<Q>(&self, key: &Q) -> Option<&V> where K: Borrow<Q>, Q: Ord + ?Sized {
        self.get_key_value(key).map(|(_, v)| v)
    }
    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut map = LiqueMap::new();
    /// map.insert(1, "a");
    /// if let Some(x) = map.get_mut(&1) {
    ///     *x = "b";
    /// }
    /// assert_eq!(map[&1], "b");
    /// ```
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V> where K: Borrow<Q>, Q: Ord + ?Sized {
        let (sublist_idx, node) = self.locate_node_with_key(key);
        let j = node.binary_search_by_key(&key, |(k,_)| k.borrow());
        let node = &mut self.sublists[sublist_idx];
        j.ok().and_then(|j| node.get_mut(j)).map(|(_, v)| v)
    }
    /// Returns `true` if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut map = LiqueMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    pub fn contains_key<Q>(&self, key: &Q) -> bool where K: Borrow<Q>, Q: Ord + ?Sized {
        self.get_key_value(key).is_some()
    }
    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut count: LiqueMap<&str, usize> = LiqueMap::new();
    ///
    /// // count the number of occurrences of letters in the vec
    /// for x in ["a", "b", "a", "c", "a", "b"] {
    ///     count.entry(x).and_modify(|curr| *curr += 1).or_insert(1);
    /// }
    ///
    /// assert_eq!(count["a"], 3);
    /// assert_eq!(count["b"], 2);
    /// assert_eq!(count["c"], 1);
    /// ```
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        let (sublist_idx, node) = self.locate_node_with_key(&key);
        if let Ok(j) = node.binary_search_by_key(&&key, |(k,_)| k) {
            Entry::Occupied(OccupiedEntry { map: self, sublist_idx, pos: j })
        } else {
            Entry::Vacant(VacantEntry { map: self, key, sublist_idx })
        }
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, `None` is returned.
    ///
    /// If the map did have this key present, the value is replaced, and the old
    /// value is returned. The earlier key is preserved, though; this matters for
    /// types that can be `==` without being identical.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut map = LiqueMap::new();
    /// assert_eq!(map.insert(37, "a"), None);
    /// assert!(!map.is_empty());
    ///
    /// map.insert(37, "b");
    /// assert_eq!(map.insert(37, "c"), Some("b"));
    /// assert_eq!(map[&37], "c");
    /// assert_eq!(map.len(), 1);
    /// ```
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
    
    /// Converts the map into iterator over its (key, value) pairs, in sorted order.
    pub fn consume(self) -> IntoItems<K, V> {
        self.sublists.into_iter().flatten()
    }
    /// Converts the map into iterator over its keys in sorted order.
    /// The values are dropped at an unspecified moment.
    pub fn into_keys(self) -> IntoKeys<K, V> {
        self.consume().map(|(k, _)| k)
    }
    /// Converts the map into iterator over its values, returned in ascending order of keys.
    /// The keys are dropped at an unspecified moment.
    pub fn into_values(self) -> IntoValues<K, V> {
        self.consume().map(|(_, v)| v)
    }
    /// Gets an iterator over the key-value pairs of the map - i.e. &(K, V) - in sorted order.
    fn inner_iter(&self) -> Iter<'_, K, V> {
        self.sublists.iter().flat_map(|sublist| sublist.iter())
    }
    /// Gets an iterator over the entries of the map - i.e. (&K, &V) - sorted by key.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut map = LiqueMap::new();
    /// map.insert(3, "c");
    /// map.insert(2, "b");
    /// map.insert(1, "a");
    ///
    /// for (key, value) in map.iter() {
    ///     println!("{key}: {value}");
    /// }
    ///
    /// let (first_key, first_value) = map.iter().next().unwrap();
    /// assert_eq!((*first_key, *first_value), (1, "a"));
    /// ```
    pub fn iter(&self) -> IterMap<'_, K, V> {
        self.inner_iter().map(reborrow)
    }
    /// Gets an iterator over the keys of the map, in sorted order.
    pub fn keys(&self) -> Keys<'_, K, V> {
        self.inner_iter().map(|(k, _)| k)
    }
    fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        self.sublists.iter_mut().flat_map(|sublist| sublist.iter_mut())
    }
    /// Gets an iterator over the values of the map, returned in ascending order of associated keys.
    pub fn values(&self) -> Values<'_, K, V> {
        self.inner_iter().map(|(_, v)| v)
    }
    /// Gets a mutable iterator over the values of the map, returned in ascending order of associated keys.
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        self.iter_mut().map(|(_, v)| v)
    }

    /// Returns `true` if the map contains no elements.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut a = LiqueMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1, "a");
    /// assert!(!a.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Returns the number of elements in the map.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut a = LiqueMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.fenwick.prefix_sum(self.sublists.len(), 0)
    }

    /// Makes a new, empty LiqueMap.
    ///
    /// Allocates NODE_CAPACITY locations for key-value pairs, plus auxillary information.
    /// 
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut map = LiqueMap::new();
    /// map.insert(1, "a");  // entries can now be inserted
    /// ```
    pub fn new() -> Self {
        let mut this = LiqueMap {
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
            // TODO: try pulling some elements from adjacent sublists instead
        }
        pair
    }

    /// Removes and returns the first element in the map, if any.
    /// The key of this element is the minimum key that was in the map.
    ///
    /// # Examples
    ///
    /// Draining elements in ascending order, while keeping a usable map each iteration.
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut map = LiqueMap::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// while let Some((key, _val)) = map.pop_first() {
    ///     // note: map is not borrowed from, we can use it safely
    ///     assert!(map.iter().all(|(k, _v)| *k > key));
    /// }
    /// assert!(map.is_empty());
    /// ```
    pub fn pop_first(&mut self) -> Option<(K, V)> {
        let i = self.sublists.iter().position(|sl| !sl.is_empty())?;
        Some(self.pop_pair(i, 0))
    }
    /// Removes the i-th key-value pair from the map and returns it.
    ///
    /// # Panics
    ///
    /// Panics when `index >= map.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut map = LiqueMap::new();
    ///
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// assert_eq!(map.pop_index(1), (2, "b"));
    /// assert_eq!(map.pop_index(0), (1, "a"));
    /// assert!(map.is_empty());
    /// ```
    pub fn pop_index(&mut self, index: usize) -> (K, V) {
        assert!(index < self.len(), "index out of bounds");
        let (sublist_idx, _) = self.locate_node_with_idx_inbounds(index);
        self.pop_pair(sublist_idx, index - self.fenwick.prefix_sum(sublist_idx, 0))
    }
    /// Removes and returns the last element in the map, if any.
    /// The key of this element is the maximum key that was in the map.
    ///
    /// # Examples
    ///
    /// Draining elements in descending order, while keeping a usable map each iteration.
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut map = LiqueMap::new();
    /// map.insert(1, "a");
    /// map.insert(4, "b");
    /// while let Some((key, _val)) = map.pop_last() {
    ///     // note: map is not borrowed from, we can use it safely
    ///     assert!(map.iter().all(|(k, _v)| *k < key));
    /// }
    /// assert!(map.is_empty());
    /// ```
    pub fn pop_last(&mut self) -> Option<(K, V)> {
        let i = self.sublists.iter().rposition(|sl| !sl.is_empty())?;
        Some(self.pop_pair(i, self.sublists[i].len() - 1))
    }
    /// Removes a key from the map, returning the stored key and value if the key
    /// was previously in the map.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut map = LiqueMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove_entry(&1), Some((1, "a")));
    /// assert_eq!(map.remove_entry(&1), None);
    /// ```
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)> where K: Borrow<Q>, Q: Ord + ?Sized {
        let (sublist_idx, node) = self.locate_node_with_key(key);
        let j = node.binary_search_by_key(&key, |(k,_)| k.borrow()).ok()?;
        Some(self.pop_pair(sublist_idx, j))
    }
    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut map = LiqueMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove(&1), Some("a"));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V> where K: Borrow<Q>, Q: Ord + ?Sized {
        self.remove_entry(key).map(|(_, v)| v)
    }

    /// Constructs an iterator over a sub-range of elements in the map.
    /// The simplest way is to use the range syntax `min..max`, thus `range(min..max)` will
    /// yield elements from min (inclusive) to max (exclusive).
    /// The range may also be entered as `(Bound<T>, Bound<T>)`, so for example
    /// `range((Excluded(4), Included(10)))` will yield a left-exclusive, right-inclusive
    /// range from 4 to 10.
    ///
    /// # Panics
    ///
    /// Panics if range `start > end`.
    /// Panics if range `start == end` and both bounds are `Excluded`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use liquemap::LiqueMap;
    /// use std::ops::Bound::Included;
    ///
    /// let mut map = LiqueMap::new();
    /// map.insert(3, "a");
    /// map.insert(5, "b");
    /// map.insert(8, "c");
    /// for (&key, &value) in map.range((Included(&4), Included(&8))) {
    ///     println!("{key}: {value}");
    /// }
    /// assert_eq!(Some((&5, &"b")), map.range(4..).next());
    /// ```
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
        let start_i = self.fenwick.prefix_sum(start_sublist, start_place);
        let end_i   = self.fenwick.prefix_sum(end_sublist, end_place);
        if start_i >= end_i {return RangeMap::empty();}
        let end_sublist_incl = end_sublist - (end_place == 0) as usize;
        let end_place_o = end_place.checked_sub(1).unwrap_or(self.sublists[end_sublist_incl].len() - 1);
        let mut sublists = self.sublists[start_sublist..=end_sublist_incl].iter();
        
        let first_iter = sublists.next().map(|n| {
            if end_sublist_incl == start_sublist {
                n[start_place..=end_place_o].iter()
            } else {
                n[start_place..].iter()
            }
        });
        let back_iter = sublists.next_back().map(|n| n[..=end_place_o].iter());
        
        RangeMap {
            current_front_iter: first_iter,
            current_back_iter: back_iter,
            remaining_sublists: sublists,
            len: end_i - start_i,
        }
    }

    /// Constructs an iterator over a sub-range of elements in the map.
    /// It yields immutable references to keys and mutable ones to values, allowing to
    /// mutate the map's contents but not mess with keys ordering.
    /// The range may also be entered as `(Bound<T>, Bound<T>)`. For example,
    /// `range((Excluded(4), Included(10)))` will yield a left-exclusive, right-inclusive
    /// range from 4 to 10.
    ///
    /// # Panics
    ///
    /// May panic if range `start > end`.
    /// May panic if range `start == end` and both bounds are `Excluded`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use liquemap::LiqueMap;
    /// use std::ops::Bound::{Excluded, Unbounded};
    ///
    /// let mut map: LiqueMap<&str, i32> =
    ///     [("Alice", 0), ("Bob", 0), ("Carol", 0), ("Cheryl", 0)].into();
    /// for (_, balance) in map.range_mut("B".."Cheryl") {
    ///     *balance += 100;
    /// }
    /// for (_, balance) in map.range_mut::<&str, _>((Excluded("Bob"), Unbounded)).rev() {
    ///     *balance += 60;
    /// }
    /// for (name, balance) in map.iter() {
    ///     println!("{name} => {balance}");
    /// }
    /// assert_eq!(map["Alice"], 0);
    /// assert_eq!(map["Bob"], 100);
    /// assert_eq!(map["Carol"], 160);
    /// assert_eq!(map["Cheryl"], 60);
    /// ```
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
        let start_i = self.fenwick.prefix_sum(start_sublist, start_place);
        let end_i   = self.fenwick.prefix_sum(end_sublist, end_place);
        if start_i >= end_i {return RangeMut::empty();}
        
        let end_sublist_incl = end_sublist - (end_place == 0) as usize;
        let end_place_o = end_place.checked_sub(1).unwrap_or(self.sublists[end_sublist_incl].len() - 1);
        let mut sublists = self.sublists[start_sublist..=end_sublist_incl].iter_mut();
        let first_iter = sublists.next().map(|n| {
            if end_sublist_incl == start_sublist {
                n[start_place..=end_place_o].iter_mut()
            } else {
                n[start_place..].iter_mut()
            }
        });
        let back_iter = sublists.next_back().map(|n| n[..=end_place_o].iter_mut());
        
        RangeMut {
            current_front_iter: first_iter,
            current_back_iter: back_iter,
            remaining_sublists: sublists,
            len: end_i - start_i,
        }
    }

    /// Constructs an immutable iterator over a sub-range of elements in the map, .
    /// The range may also be entered as `(Bound<T>, Bound<T>)`. For example,
    /// `range((Excluded(4), Included(10)))` will yield a left-exclusive, right-inclusive
    /// range from 4 to 10.
    ///
    /// # Panics
    ///
    /// May panic if range `start > end`.
    /// May panic if range `start == end` and both bounds are `Excluded`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use liquemap::LiqueMap;
    /// use std::ops::Bound::{Excluded, Unbounded};
    ///
    /// let mut map: LiqueMap<&str, i32> =
    ///     [("Alice", 10), ("Bob", 20), ("Carol", 30), ("Cheryl", 40)].into();
    /// let mut total_balance = 0;
    /// for (_, balance) in map.range_idx(1..3) {
    ///     total_balance += balance;
    /// }
    /// assert_eq!(total_balance, 50);
    /// ```
    pub fn range_idx<R>(&self, range: R) -> RangeMap<'_, K, V>
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
            return RangeMap::empty();
        }
        
        let (start_node, _) = self.locate_node_with_idx_inbounds(start);
        let (end_node_incl, _) = self.locate_node_with_idx_inbounds(end - 1);
        let start_place = start - self.fenwick.prefix_sum(start_node, 0);
        let end_place = end - self.fenwick.prefix_sum(end_node_incl, 0);
        let mut sublists = self.sublists[start_node..=end_node_incl].iter();
        let first_iter = sublists.next().map(|n| {
            if end_node_incl == start_node {
                n[start_place..end_place].iter()
            } else {
                n[start_place..].iter()
            }
        });
        let back_iter = sublists.next_back().map(|n| n[..end_place].iter());
        
        RangeMap {
            current_front_iter: first_iter,
            current_back_iter: back_iter,
            remaining_sublists: sublists,
            len: end.saturating_sub(start),
        }
    }

    /// Constructs an iterator over a sub-range of elements in the map at given indices.
    /// It yields immutable references to keys and mutable ones to values, allowing to
    /// mutate the map's contents but not mess with keys ordering.
    /// The range may also be entered as `(Bound<T>, Bound<T>)`. For example,
    /// `range((Excluded(4), Included(10)))` will yield a left-exclusive, right-inclusive
    /// range from 4 to 10.
    ///
    /// # Panics
    ///
    /// May panic if range `start > end`.
    /// May panic if range `start == end` and both bounds are `Excluded`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use liquemap::LiqueMap;
    /// use std::ops::Bound::{Excluded, Unbounded};
    ///
    /// let mut map: LiqueMap<&str, i32> =
    ///     [("Alice", 0), ("Bob", 0), ("Carol", 0), ("Cheryl", 0)].into();
    /// for (_, balance) in map.range_mut_idx(1..3).rev() {
    ///     *balance += 100;
    /// }
    /// for (_, balance) in map.range_mut_idx((Excluded(1), Unbounded)) {
    ///     *balance += 60;
    /// }
    /// for (name, balance) in map.iter() {
    ///     println!("{name} => {balance}");
    /// }
    /// assert_eq!(map["Alice"], 0);
    /// assert_eq!(map["Bob"], 100);
    /// assert_eq!(map["Carol"], 160);
    /// assert_eq!(map["Cheryl"], 60);
    /// ```
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
        let start_place = start - self.fenwick.prefix_sum(start_node, 0);
        let end_place = end - self.fenwick.prefix_sum(end_node_incl, 0);
        let mut sublists = self.sublists[start_node..=end_node_incl].iter_mut();
        let first_iter = sublists.next().map(|n| {
            if end_node_incl == start_node {
                n[start_place..end_place].iter_mut()
            } else {
                n[start_place..].iter_mut()
            }
        });
        let back_iter = sublists.next_back().map(|n| n[..end_place].iter_mut());
        
        RangeMut {
            current_front_iter: first_iter,
            current_back_iter: back_iter,
            remaining_sublists: sublists,
            len: end.saturating_sub(start),
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all pairs `(k, v)` for which `f(&k, &mut v)` returns `false`.
    /// The elements are visited in ascending key order.
    ///
    /// # Examples
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let mut map: LiqueMap<_, _> = [(0, 0), (2, 20), (3, 30), (5, 50), (6, 60)].into();
    /// // Keep only the elements with even-numbered keys.
    /// map.retain(|&k, _| k % 2 == 0);
    /// assert!(map.consume().eq(vec![(0, 0), (2, 20), (6, 60)]));
    /// ```
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

    /// Splits the collection into two at the given key.
    /// 
    /// Returns everything after the given key, including the key.
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


    /// Returns a [`CursorMap`] pointing at the first element that is above the
    /// given bound.
    ///
    /// If no such element exists then a cursor pointing at the "ghost"
    /// non-element is returned.
    ///
    /// Passing [`Bound::Unbounded`] will return a cursor pointing at the first
    /// element of the map.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use liquemap::LiqueMap;
    /// use std::ops::Bound;
    ///
    /// let mut a = LiqueMap::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    /// a.insert(3, "c");
    /// a.insert(4, "c");
    /// let cursor = a.lower_bound(Bound::Excluded(&2));
    /// assert_eq!(cursor.key(), Some(&3));
    /// ```
    pub fn lower_bound<Q>(&self, bound: Bound<&Q>) -> CursorMap<'_, K, V> where K: Borrow<Q>, Q: Ord + ?Sized {
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

    /// Returns the position in which the given element would fall in the already-existing sorted
    /// order (equivalently, number of map keys strictly less than the given one).
    ///
    /// The value may be any borrowed form of the set's element type,
    /// but the ordering on the borrowed form *must* match the
    /// ordering on the element type.
    ///
    /// # Examples
    ///
    /// ```
    /// use liquemap::LiqueMap;
    ///
    /// let map: LiqueMap<_, _> = [(1, "a"), (2, "b"), (3, "c")].into();
    /// assert_eq!(map.rank(&1), 0);
    /// assert_eq!(map.rank(&3), 2);
    /// assert_eq!(map.rank(&4), 3);
    /// assert_eq!(map.rank(&100), 3);
    /// ```
    pub fn rank<Q>(&self, value: &Q) -> usize where K: Borrow<Q>, Q: Ord + ?Sized {
        let (sublist_idx, pos) = self.find_sublist_for_key(value);
        let offset = self.fenwick.prefix_sum(sublist_idx, 0);
        let sublist = &self.sublists[sublist_idx];
        let rank_in_sublist = pos.unwrap_or_else(|| sublist.partition_point(|(k, _)| k.borrow() < value));
        offset + rank_in_sublist
    }
}

impl<K: Ord, V, const N: usize> From<[(K, V); N]> for LiqueMap<K, V> {
    fn from(arr: [(K, V); N]) -> Self {
        let mut map = LiqueMap::new();
        for (k, v) in arr {
            map.insert(k, v);
        }
        map
    }
}

impl<K: Ord + Clone, V: Clone> Clone for LiqueMap<K, V> {
    fn clone(&self) -> Self {
        LiqueMap {
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


impl<K, Q, V> Index<&Q> for LiqueMap<K, V> where K: Borrow<Q> + Ord, Q: Ord + ?Sized {
    type Output = V;
    fn index(&self, index: &Q) -> &V {
        self.get(index).unwrap()
    }
}
impl<K, Q, V> IndexMut<&Q> for LiqueMap<K, V> where K: Borrow<Q> + Ord, Q: Ord + ?Sized {
    fn index_mut(&mut self, index: &Q) -> &mut V {
        self.get_mut(index).unwrap()
    }
}

// --- Iterator Implementations ---

impl<'a, K, V> RangeMap<'a, K, V> {
    fn empty() -> Self {
        Self {
            current_front_iter: Some([].iter()),
            current_back_iter: Some([].iter()),
            remaining_sublists: [].iter(),
            len: 0
        }
    }
    #[inline]
    fn make_front(&mut self) {
        match self.remaining_sublists.next() {
            Some(it) => {self.current_front_iter = Some(it.iter());}
            None     => {self.current_front_iter = self.current_back_iter.take();}
        }
    }
    #[inline]
    fn make_back(&mut self) {
        match self.remaining_sublists.next_back() {
            Some(it) => {println!("take_back:ok"); self.current_back_iter = Some(it.iter());}
            None     => {println!("take_back:fr"); self.current_back_iter = self.current_front_iter.take();}
        }
    }
}

impl<'a, K, V> Iterator for RangeMap<'a, K, V> {
    type Item = (&'a K, &'a V);
    
    fn count(self) -> usize {
        self.len
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
    
    fn next(&mut self) -> Option<Self::Item> {
        self.len = self.len.checked_sub(1)?;    // return if == 0
        if self.current_front_iter.is_none() {self.make_front();}
        loop {
            let front_it = self.current_front_iter.as_mut().unwrap();
            if let Some((k, v)) = front_it.next() {
                return Some((k, v));
            }
            self.make_front();
        }
    }
}
impl<'a, K, V> DoubleEndedIterator for RangeMap<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.len = self.len.checked_sub(1)?;    // return if == 0
        if self.current_back_iter.is_none() {self.make_back();}
        loop {
            let back_it = self.current_back_iter.as_mut().unwrap();
            if let Some((k, v)) = back_it.next_back() {
                return Some((k, v));
            }
            self.make_back();
        }
    }
}

// --- Mutable Iterator Implementations ---

impl<'a, K, V> RangeMut<'a, K, V> {
    fn empty() -> Self {
        Self {
            current_front_iter: Some([].iter_mut()),
            current_back_iter: Some([].iter_mut()),
            remaining_sublists: [].iter_mut(),
            len: 0
        }
    }
    fn make_front(&mut self) {
        match self.remaining_sublists.next() {
            Some(it) => {self.current_front_iter = Some(it.iter_mut());}
            None     => {self.current_front_iter = self.current_back_iter.take();}
        }
    }
    fn make_back(&mut self) {
        match self.remaining_sublists.next_back() {
            Some(it) => {self.current_back_iter = Some(it.iter_mut());}
            None     => {self.current_back_iter = self.current_front_iter.take();}
        }
    }
}

impl<'a, K, V> Iterator for RangeMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    
    fn count(self) -> usize {
        self.len
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
    
    fn next(&mut self) -> Option<Self::Item> {
        self.len = self.len.checked_sub(1)?;    // return if == 0
        if self.current_front_iter.is_none() {self.make_front();}
        loop {
            let front_it = self.current_front_iter.as_mut().unwrap();
            if let Some((k, v)) = front_it.next() {
                return Some((k, v));
            }
            self.make_front();
        }
    }
}
impl<'a, K, V> DoubleEndedIterator for RangeMut<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.len = self.len.checked_sub(1)?;    // return if == 0
        if self.current_back_iter.is_none() {self.make_back();}
        loop {
            let back_it = self.current_back_iter.as_mut().unwrap();
            if let Some((k, v)) = back_it.next_back() {
                return Some((k, v));
            }
            self.make_back();
        }
    }
}

// --- Ref Iterator Implementations ---

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
    pub fn and_modify(mut self, effect: impl FnOnce(&mut V)) -> Self {
        match &mut self {
            Entry::Occupied(entry) => {effect(entry.get_mut())},
            Entry::Vacant(_) => {},
        }
        self
    }
}

