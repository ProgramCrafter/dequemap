#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use std::ops::Bound;

use std::collections::BTreeMap as BeeMap;
use liquemap::LiqueMap as TestMap;

#[derive(Debug, Arbitrary)]
enum Command {
    Append { other: Vec<(u8, u8)> },
    Clear,
    ContainsKey { key: u8 },
    FirstKeyValue,
    Get { key: u8 },
    GetIndex { index: usize },
    GetKeyValue { key: u8 },
    GetMut { key: u8, value: u8 },
    GetMutIndex { index: usize, value: u8 },
    Insert { key: u8, value: u8 },
    IsEmpty,
    Iter,
    // IterMut,
    Keys,
    LastKeyValue,
    Len,
    PopFirst,
    PopIndex { index: usize },
    PopLast,
    Range { start: MyBound, end: MyBound },
    RangeMut { start: MyBound, end: MyBound },
    RangeMutIdx { start: usize, end: usize },
    Remove { key: u8 },
    RemoveEntry { key: u8 },
    Retain { threshold: u8 },
    RetainVal { threshold: u8 },
    SplitOff { key: u8 },
    Values,
    ValuesMut,
    Entry { key: u8, value: u8 },
    FirstEntry,
    LastEntry,
    LowerBound { bound: MyBound },
    Rank { key: u8 },
    Clone,
    CloneFrom { source: Vec<(u8, u8)> },
}

#[derive(Debug, Arbitrary)]
enum MyBound {
    Included(u8),
    Excluded(u8),
    Unbounded,
}

impl From<MyBound> for Bound<u8> {
    fn from(b: MyBound) -> Self {
        match b {
            MyBound::Included(x) => Bound::Included(x),
            MyBound::Excluded(x) => Bound::Excluded(x),
            MyBound::Unbounded => Bound::Unbounded,
        }
    }
}
impl<'a> From<&'a MyBound> for Bound<&'a u8> {
    fn from(b: &'a MyBound) -> Self {
        match b {
            MyBound::Included(x) => Bound::Included(x),
            MyBound::Excluded(x) => Bound::Excluded(x),
            MyBound::Unbounded => Bound::Unbounded,
        }
    }
}

fn validate_bounds(start: Bound<u8>, end: Bound<u8>) -> bool {
    match (start, end) {
        (Bound::Excluded(s), Bound::Excluded(e)) if s == e => {false},
        (Bound::Included(s) | Bound::Excluded(s), Bound::Included(e) | Bound::Excluded(e)) if s > e => {false},
        _ => {true},
    }
}

fuzz_target!(|data: &[u8]| {
    let mut unstructured = Unstructured::new(data);
    let commands = match Vec::<Command>::arbitrary(&mut unstructured) {
        Ok(c) => c,
        Err(_) => return,
    };

    let mut map: TestMap<u8, u8> = TestMap::new();
    let mut btree_map: BeeMap<u8, u8> = BeeMap::new();
    let mut cloned_map;

    for command in commands {
        map.validate_buckets();
        
        if std::env::var("RUST_BACKTRACE").is_ok() {
            println!("{command:?}");
        }
        
        match command {
            Command::Append { other } => {
                let mut other_map = TestMap::new();
                let mut other_btree = BeeMap::new();
                for (k, v) in other {
                    other_map.insert(k, v);
                    other_btree.insert(k, v);
                }
                map.append(&mut other_map);
                for (k, v) in other_btree.into_iter() {
                    btree_map.insert(k, v);
                }
            }
            Command::Clear => {
                map.clear();
                btree_map.clear();
                assert_eq!(map.len(), 0);
            }
            Command::ContainsKey { key } => {
                assert_eq!(map.contains_key(&key), btree_map.contains_key(&key));
            }
            Command::FirstKeyValue => {
                assert_eq!(
                    map.first_key_value(),
                    btree_map.first_key_value()
                );
            }
            Command::Get { key } => {
                assert_eq!(map.get(&key), btree_map.get(&key));
            }
            Command::GetIndex { index } => {
                let map_val = map.get_index(index);
                let btree_val = btree_map.iter().nth(index);
                assert_eq!(map_val, btree_val);
            }
            Command::GetKeyValue { key } => {
                assert_eq!(map.get_key_value(&key), btree_map.get_key_value(&key));
            }
            Command::GetMut { key, value } => {
                let mut had = false;
                if let Some(v) = map.get_mut(&key) {
                    *v = value;
                    had = true;
                }
                if let Some(v) = btree_map.get_mut(&key) {
                    *v = value;
                    assert!(had);
                } else {
                    assert!(!had);
                }
            }
            Command::GetMutIndex { index, value } => {
                let mut had = false;
                if let Some(v) = map.get_mut_index(index) {
                    *v = value;
                    had = true;
                }
                if let Some(v) = btree_map.values_mut().nth(index) {
                    *v = value;
                    assert!(had);
                } else {
                    assert!(!had);
                }
            }
            Command::Insert { key, value } => {
                assert_eq!(map.insert(key, value), btree_map.insert(key, value));
            }
            Command::IsEmpty => {
                assert_eq!(map.is_empty(), btree_map.is_empty());
            }
            Command::Iter => {
                assert!(map.iter().eq(btree_map.iter()));
            }
            /*
            Command::IterMut => {
                let map_items: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
                let btree_items: Vec<_> = btree_map.iter().map(|(k, v)| (*k, *v)).collect();
                assert_eq!(map_items, btree_items);

                map.iter_mut().for_each(|(_, v)| *v = v.wrapping_add(1));
                btree_map.iter_mut().for_each(|(_, v)| *v = v.wrapping_add(1));
            }
            */
            Command::Keys => {
                assert!(map.keys().eq(btree_map.keys()));
            }
            Command::LastKeyValue => {
                assert_eq!(
                    map.last_key_value(),
                    btree_map.last_key_value().map(|(k, v)| (k, v))
                );
            }
            Command::Len => {
                assert_eq!(map.len(), btree_map.len());
            }
            Command::PopFirst => {
                assert_eq!(map.pop_first(), btree_map.pop_first());
            }
            Command::PopIndex { index } => {
                if index < map.len() {
                    let map_val = map.pop_index(index);
                    let btree_key = btree_map.keys().nth(index).cloned();
                    assert_eq!(Some(map_val.0), btree_key);
                    assert_eq!(Some(map_val.1), btree_map.remove(&btree_key.unwrap()));
                }
            }
            Command::PopLast => {
                assert_eq!(map.pop_last(), btree_map.pop_last());
            }
            Command::Range { start, end } => {
                let start: Bound<u8> = start.into();
                let end: Bound<u8> = end.into();
                if !validate_bounds(start, end) {continue;}
                
                let map_range: Vec<_> = map.range((start, end)).collect();
                let btree_range: Vec<_> = btree_map.range((start, end)).collect();
                assert_eq!(map_range, btree_range);
            }
            Command::RangeMut { start, end } => {
                let start: Bound<u8> = start.into();
                let end = end.into();
                if !validate_bounds(start, end) {continue;}
                
                map.range_mut((start, end)).for_each(|(_, v)| *v = v.wrapping_add(1));
                btree_map.range_mut((start, end)).for_each(|(_, v)| *v = v.wrapping_add(1));
            }
            Command::RangeMutIdx { start, end } => {
                if start <= end && end <= map.len() {
                    map.range_mut_idx(start..end).for_each(|(_, v)| *v = v.wrapping_add(1));
                    btree_map.iter_mut().skip(start).take(end - start).for_each(|(_, v)| *v = v.wrapping_add(1));
                    
                    assert!(map.range_mut_idx(start..end).eq(btree_map.iter_mut().skip(start).take(end - start)));
                }
            }
            Command::Remove { key } => {
                assert_eq!(map.remove(&key), btree_map.remove(&key));
            }
            Command::RemoveEntry { key } => {
                assert_eq!(map.remove_entry(&key), btree_map.remove_entry(&key));
            }
            Command::Retain { threshold } => {
                map.retain(|k, _| *k <= threshold);
                btree_map.retain(|k, _| *k <= threshold);
            }
            Command::RetainVal { threshold } => {
                map.retain(|_, v| *v <= threshold);
                btree_map.retain(|_, v| *v <= threshold);
            }
            Command::SplitOff { key } => {
                let map_split = map.split_off(&key);
                let btree_split = btree_map.split_off(&key);
                assert!(map.iter().all(|(k, _)| *k < key));
                assert!(btree_map.iter().all(|(k, _)| *k < key));
                assert!(map_split.iter().all(|(k, _)| *k >= key));
                assert!(btree_split.iter().all(|(k, _)| *k >= key));
            }
            Command::Values => {
                assert!(map.values().eq(btree_map.values()));
            }
            Command::ValuesMut => {
                map.values_mut().for_each(|v| *v = v.wrapping_add(1));
                btree_map.values_mut().for_each(|v| *v = v.wrapping_add(1));
            }
            Command::Entry { key, value } => {
                map.entry(key).or_insert(value);
                btree_map.entry(key).or_insert(value);
            }
            Command::FirstEntry => {
                if let Some(entry) = map.first_entry() {
                    let key = *entry.key();
                    entry.remove();
                    btree_map.remove(&key);
                }
            }
            Command::LastEntry => {
                if let Some(entry) = map.last_entry() {
                    let key = *entry.key();
                    entry.remove();
                    btree_map.remove(&key);
                }
            }
            Command::LowerBound { bound } => {
                let bound = (&bound).into();
                let cursor = map.lower_bound(bound);
                let btree_bound = match bound {
                    Bound::Included(x) => btree_map.range(x..).next(),
                    Bound::Excluded(x) => x.checked_add(1).and_then(|x| btree_map.range(x..).next()),
                    Bound::Unbounded => btree_map.iter().next(),
                };
                assert_eq!(cursor.key(), btree_bound.map(|(k, _)| k));
            }
            Command::Rank { key } => {
                let rank = map.rank(&key);
                let btree_rank = btree_map.range(..key).count();
                assert_eq!(rank, btree_rank);
            }
            Command::Clone => {
                cloned_map = map.clone();
                let cloned_btree = btree_map.clone();
                assert!(cloned_map.iter().eq(map.iter()));
                assert!(cloned_btree.iter().eq(btree_map.iter()));
            }
            Command::CloneFrom { source } => {
                let mut source_map = TestMap::new();
                let mut source_btree = BeeMap::new();
                for (k, v) in source {
                    source_map.insert(k, v);
                    source_btree.insert(k, v);
                }
                map.clone_from(&source_map);
                btree_map.clone_from(&source_btree);
            }
        }

        // Final consistency check
        let map_contents: Vec<_> = map.iter().collect();
        let btree_contents: Vec<_> = btree_map.iter().collect();
        assert_eq!(map_contents, btree_contents);
    }
});

