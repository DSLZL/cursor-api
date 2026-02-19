//! [`DataBlock`] defines the way key-value pairs are stored in memory.

use std::cell::UnsafeCell;
use std::mem::{MaybeUninit, needs_drop};

/// Defines the way key-value pairs are stored in memory.
pub(crate) struct DataBlock<K, V, const LEN: usize> {
    /// Array for storing keys.
    keys: UnsafeCell<[MaybeUninit<K>; LEN]>,
    /// Array for storing values.
    vals: UnsafeCell<[MaybeUninit<V>; LEN]>,
}

impl<K, V, const LEN: usize> DataBlock<K, V, LEN> {
    /// Creates a new [`DataBlock`].
    #[inline]
    pub(crate) const fn new() -> DataBlock<K, V, LEN> {
        DataBlock {
            keys: UnsafeCell::new([const { MaybeUninit::uninit() }; LEN]),
            vals: UnsafeCell::new([const { MaybeUninit::uninit() }; LEN]),
        }
    }

    /// Returns a pointer to the key at the specified position.
    #[inline]
    pub(crate) const fn key_ptr(&self, pos: usize) -> *const K {
        unsafe { self.keys.get().cast::<K>().add(pos) }
    }

    /// Returns a pointer to the value at the specified position.
    #[inline]
    pub(crate) const fn val_ptr(&self, pos: usize) -> *const V {
        unsafe { self.vals.get().cast::<V>().add(pos) }
    }

    /// Reads the key-value at the specified position.
    #[inline]
    pub(crate) const fn read(&self, pos: usize) -> (K, V) {
        unsafe { (self.key_ptr(pos).read(), self.val_ptr(pos).read()) }
    }

    /// Writes the key-value at the specified position.
    #[inline]
    pub(crate) const fn write(&self, pos: usize, key: K, val: V) {
        unsafe {
            self.key_ptr(pos).cast_mut().write(key);
            self.val_ptr(pos).cast_mut().write(val);
        }
    }

    /// Drops the key-value at the specified position.
    #[inline]
    pub(crate) fn drop_in_place(&self, pos: usize) {
        unsafe {
            if needs_drop::<K>() {
                self.key_ptr(pos).cast_mut().drop_in_place();
            }
            if needs_drop::<V>() {
                self.val_ptr(pos).cast_mut().drop_in_place();
            }
        }
    }
}

unsafe impl<K: Send, V: Send, const LEN: usize> Send for DataBlock<K, V, LEN> {}
unsafe impl<K: Send + Sync, V: Send + Sync, const LEN: usize> Sync for DataBlock<K, V, LEN> {}
