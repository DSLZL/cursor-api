#![allow(missing_docs)]

use super::{Equivalent, HashMap};
use crate::hash_table::{
    HashTable, LockedBucket,
    bucket::{EntryPtr, MAP},
};
use core::{
    hash::{BuildHasher, Hash},
    mem::replace,
};
use sdd::Guard;

impl<K, V, S> HashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Creates a raw-entry builder for advanced hash-based entry operations.
    ///
    /// Methods ending with `_hashed_nocheck` trust the caller-provided hash.
    /// Prefer using [`RawEntryBuilder::hash`] to compute the hash value.
    #[inline]
    pub fn raw_entry(&self) -> RawEntryBuilder<'_, K, V, S> { RawEntryBuilder { map: self } }
}

/// A builder for creating a raw entry in a `HashMap`.
pub struct RawEntryBuilder<'a, K, V, S>
where S: BuildHasher
{
    map: &'a HashMap<K, V, S>,
}

impl<'a, K, V, S> RawEntryBuilder<'a, K, V, S>
where S: BuildHasher
{
    /// Computes a key hash using the map's build hasher.
    ///
    /// This is intended to be paired with `_hashed_nocheck` methods.
    #[inline]
    pub fn hash<Q>(&self, k: &Q) -> u64
    where Q: Hash + ?Sized {
        self.map.build_hasher.hash_one(k)
    }

    /// Access an entry by key.
    #[inline]
    pub async fn from_key_async<Q>(self, k: &Q) -> RawEntry<'a, K, V, S>
    where
        K: Hash + Eq,
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash(k);
        self.from_key_hashed_nocheck_async(hash, k).await
    }

    /// Access an entry by key.
    #[inline]
    pub fn from_key_sync<Q>(self, k: &Q) -> RawEntry<'a, K, V, S>
    where
        K: Hash + Eq,
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash(k);
        self.from_key_hashed_nocheck_sync(hash, k)
    }

    /// Access an entry by key.
    #[inline]
    pub fn try_from_key<Q>(self, k: &Q) -> Option<RawEntry<'a, K, V, S>>
    where
        K: Hash + Eq,
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash(k);
        self.try_from_key_hashed_nocheck(hash, k)
    }

    /// Accesses an entry by a pre-computed hash and key.
    ///
    /// # Contract
    ///
    /// `hash` must be computed for `k` with [`Self::hash`] (or an equivalent
    /// hasher state), otherwise behavior is logically unspecified.
    #[inline]
    pub async fn from_key_hashed_nocheck_async<Q>(self, hash: u64, k: &Q) -> RawEntry<'a, K, V, S>
    where
        K: Hash + Eq,
        Q: Equivalent<K> + ?Sized,
    {
        let locked_bucket = self.map.writer_async(hash).await;
        let entry_ptr = locked_bucket.search(k, hash);
        if entry_ptr.is_valid() {
            RawEntry::Occupied(RawOccupiedEntry { hashmap: self.map, locked_bucket, entry_ptr })
        } else {
            RawEntry::Vacant(RawVacantEntry { hashmap: self.map, locked_bucket })
        }
    }

    /// Accesses an entry by a pre-computed hash and key.
    ///
    /// # Contract
    ///
    /// `hash` must be computed for `k` with [`Self::hash`] (or an equivalent
    /// hasher state), otherwise behavior is logically unspecified.
    #[inline]
    pub fn from_key_hashed_nocheck_sync<Q>(self, hash: u64, k: &Q) -> RawEntry<'a, K, V, S>
    where
        K: Hash + Eq,
        Q: Equivalent<K> + ?Sized,
    {
        let locked_bucket = self.map.writer_sync(hash);
        let entry_ptr = locked_bucket.search(k, hash);
        if entry_ptr.is_valid() {
            RawEntry::Occupied(RawOccupiedEntry { hashmap: self.map, locked_bucket, entry_ptr })
        } else {
            RawEntry::Vacant(RawVacantEntry { hashmap: self.map, locked_bucket })
        }
    }

    /// Tries to access an entry by a pre-computed hash and key.
    ///
    /// Returns `None` if the target bucket cannot be locked.
    ///
    /// # Contract
    ///
    /// `hash` must be computed for `k` with [`Self::hash`] (or an equivalent
    /// hasher state), otherwise behavior is logically unspecified.
    #[inline]
    pub fn try_from_key_hashed_nocheck<Q>(self, hash: u64, k: &Q) -> Option<RawEntry<'a, K, V, S>>
    where
        K: Hash + Eq,
        Q: Equivalent<K> + ?Sized,
    {
        let guard = Guard::new();
        let locked_bucket = self.map.try_reserve_bucket(hash, &guard)?;
        let entry_ptr = locked_bucket.search(k, hash);
        if entry_ptr.is_valid() {
            Some(RawEntry::Occupied(RawOccupiedEntry {
                hashmap: self.map,
                locked_bucket,
                entry_ptr,
            }))
        } else {
            Some(RawEntry::Vacant(RawVacantEntry { hashmap: self.map, locked_bucket }))
        }
    }
}

/// A raw entry in the map.
pub enum RawEntry<'a, K, V, S>
where S: BuildHasher
{
    Occupied(RawOccupiedEntry<'a, K, V, S>),
    Vacant(RawVacantEntry<'a, K, V, S>),
}

impl<'a, K, V, S> RawEntry<'a, K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    // /// Sets the value of the entry, and returns an OccupiedEntry.
    // #[inline]
    // pub fn insert(self, key: K, value: V) -> RawOccupiedEntry<'a, K, V, S>
    // where
    //     K: Hash,
    //     S: BuildHasher,
    // {
    //     match self {
    //         RawEntry::Occupied(mut entry) => {
    //             entry.insert(value);
    //             entry
    //         }
    //         RawEntry::Vacant(entry) => entry.insert_entry(key, value),
    //     }
    // }

    // #[inline]
    // pub fn or_insert(self, default_key: K, default_val: V)
    // where
    //     K: Hash,
    //     S: BuildHasher,
    // {
    //     match self {
    //         RawEntry::Occupied(_entry) => {},
    //         RawEntry::Vacant(entry) => entry.insert(default_key, default_val),
    //     }
    // }

    // #[inline]
    // pub fn or_insert_with<F>(self, default: F)
    // where
    //     F: FnOnce() -> (K, V),
    //     K: Hash,
    //     S: BuildHasher,
    // {
    //     match self {
    //         RawEntry::Occupied(_entry) => {},
    //         RawEntry::Vacant(entry) => {
    //             let (k, v) = default();
    //             entry.insert(k, v)
    //         }
    //     }
    // }

    #[inline]
    pub fn and_modify<F>(mut self, f: F) -> Self
    where F: FnOnce(&K, &mut V) {
        if let RawEntry::Occupied(entry) = &mut self {
            let (k, v) = entry.get_key_value_mut();
            f(k, v);
        }
        self
    }
}

pub struct RawOccupiedEntry<'a, K, V, S>
where S: BuildHasher
{
    hashmap: &'a HashMap<K, V, S>,
    locked_bucket: LockedBucket<K, V, (), MAP>,
    entry_ptr: EntryPtr<K, V, MAP>,
}

impl<'a, K, V, S> RawOccupiedEntry<'a, K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    #[inline]
    #[must_use]
    pub fn key(&self) -> &K { &self.locked_bucket.entry(&self.entry_ptr).0 }

    // #[inline]
    // #[must_use]
    // pub fn key_mut(&mut self) -> &mut K { &mut self.locked_bucket.entry_mut(&mut self.entry_ptr).0 }

    // #[inline]
    // #[must_use]
    // pub fn into_key(mut self) -> &'a mut K {
    //     &mut self.locked_bucket.entry_mut(&mut self.entry_ptr).0
    // }

    #[inline]
    #[must_use]
    pub fn get(&self) -> &V { &self.locked_bucket.entry(&self.entry_ptr).1 }

    // #[inline]
    // #[must_use]
    // pub fn into_mut(mut self) -> &'a mut V {
    //     &mut self.locked_bucket.entry_mut(&mut self.entry_ptr).1
    // }

    #[inline]
    #[must_use]
    pub fn get_mut(&mut self) -> &mut V { self.locked_bucket.entry_mut(&mut self.entry_ptr).1 }

    #[inline]
    #[must_use]
    pub fn get_key_value(&self) -> (&K, &V) {
        let (k, v) = self.locked_bucket.entry(&self.entry_ptr);
        (k, v)
    }

    #[inline]
    #[must_use]
    pub fn get_key_value_mut(&mut self) -> (&K, &mut V) {
        let (k, v) = self.locked_bucket.entry_mut(&mut self.entry_ptr);
        (k, v)
    }

    // #[inline]
    // #[must_use]
    // pub fn into_key_value(mut self) -> (&'a mut K, &'a mut V) {
    //     let (k, v) = self.locked_bucket.entry_mut(&mut self.entry_ptr);
    //     (k, v)
    // }

    #[inline]
    pub fn insert(&mut self, value: V) -> V { replace(self.get_mut(), value) }

    // #[inline]
    // pub fn insert_key(&mut self, key: K) -> K { replace(self.key_mut(), key) }

    #[inline]
    pub fn remove(self) -> V { self.remove_entry().1 }

    #[inline]
    #[must_use]
    pub fn remove_entry(mut self) -> (K, V) {
        self.locked_bucket.remove(self.hashmap, &mut self.entry_ptr)
    }

    // #[inline]
    // pub fn replace_entry_with<F>(self, f: F) -> RawEntry<'a, K, V, S>
    // where F: FnOnce(&K, V) -> Option<V> {
    //     let proxy: hashbrown::hash_map::RawOccupiedEntry<'a, K, V, S> = unsafe {
    //         let (bucket, table): (
    //             core::ptr::NonNull<(K, V)>,
    //             &'a mut hash_table::HashTable<(K, V)>,
    //         ) = core::mem::transmute(self.entry);
    //         core::mem::transmute((bucket, table, self.hash_builder))
    //     };
    //     let result = proxy.replace_entry_with(f);
    //     match result {
    //         hashbrown::hash_map::RawEntry::Occupied(entry) => {
    //             let (bucket, table, hash_builder): (
    //                 core::ptr::NonNull<(K, V)>,
    //                 &'a mut hash_table::HashTable<(K, V)>,
    //                 &'a S,
    //             ) = unsafe { core::mem::transmute(entry) };
    //             RawEntry::Occupied(RawOccupiedEntry {
    //                 shard: self.shard,
    //                 entry: unsafe { core::mem::transmute((bucket, table)) },
    //                 hash_builder,
    //             })
    //         }
    //         hashbrown::hash_map::RawEntry::Vacant(entry) => {
    //             let (table, hash_builder) = unsafe { core::mem::transmute(entry) };
    //             RawEntry::Vacant(RawVacantEntry { shard: self.shard, table, hash_builder })
    //         }
    //     }
    // }
}

pub struct RawVacantEntry<'a, K, V, S>
where S: BuildHasher
{
    hashmap: &'a HashMap<K, V, S>,
    locked_bucket: LockedBucket<K, V, (), MAP>,
}

impl<'a, K, V, S> RawVacantEntry<'a, K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Inserts an entry by hashing `key` with the map hasher.
    ///
    /// This method is kept for compatibility.
    #[deprecated = "Use insert_hashed_nocheck to keep hash/key pairing explicit."]
    #[inline]
    pub fn insert(self, key: K, value: V) {
        let hash = self.hashmap.hash(&key);
        self.insert_hashed_nocheck(hash, key, value)
    }

    /// Inserts an entry using a caller-supplied hash.
    ///
    /// # Contract
    ///
    /// `hash` must be computed for `key` with [`RawEntryBuilder::hash`] (or an
    /// equivalent hasher state), otherwise behavior is logically unspecified.
    #[inline]
    #[allow(clippy::shadow_unrelated)]
    pub fn insert_hashed_nocheck(self, hash: u64, key: K, value: V) {
        self.locked_bucket.insert(hash, (key, value));
    }

    // #[inline]
    // pub fn insert_with_hasher<H>(
    //     self,
    //     hash: u64,
    //     key: K,
    //     value: V,
    //     hasher: H,
    // ) -> (&'a mut K, &'a mut V)
    // where
    //     H: Fn(&K) -> u64,
    // {
    //     let &mut (ref mut k, ref mut v) =
    //         self.table.insert_unique(hash, (key, value), |x| hasher(&x.0)).into_mut();
    //     (k, v)
    // }

    // /// Helper to match hashbrown behavior for API compatibility.
    // #[inline]
    // pub fn insert_entry(self, key: K, value: V) -> RawOccupiedEntry<'a, K, V, S> {
    //     let hash = self.hashmap.hash(&key);
    //     let entry_ptr = self.locked_bucket.insert(hash, (key, value));
    //     RawOccupiedEntry { hashmap: self.hashmap, locked_bucket: self.locked_bucket, entry_ptr }
    // }
}
