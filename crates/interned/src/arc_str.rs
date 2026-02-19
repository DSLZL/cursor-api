//! 引用计数的不可变字符串，支持全局字符串池复用
//!
//! # 核心设计理念
//!
//! `ArcStr` 通过全局字符串池实现内存去重，相同内容的字符串共享同一份内存。
//! 这在大量重复字符串的场景下能显著降低内存使用，同时保持字符串操作的高性能。
//!
//! # 架构概览
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        用户 API 层                               │
//! │  ArcStr::new() │ as_str() │ clone() │ Drop │ PartialEq...       │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                      全局字符串池                                 │
//! │               HashMap<ThreadSafePtr, ()>                        │
//! │               双重检查锁定 + 原子引用计数                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                    底层内存布局                                   │
//! │  [hash:u64][count:AtomicUsize][len:usize][string_data...]       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # 性能特征
//!
//! | 操作 | 时间复杂度 | 说明 |
//! |------|-----------|------|
//! | new() - 首次 | O(1) + 池插入 | 堆分配 + HashMap 插入 |
//! | new() - 命中 | O(1) | HashMap 查找 + 原子递增 |
//! | clone() | O(1) | 仅原子递增 |
//! | drop() | O(1) | 使用预存哈希快速删除 |
//! | as_str() | O(1) | 直接内存访问 |

use core::{
    alloc::Layout,
    borrow::Borrow,
    cmp::Ordering,
    fmt,
    hash::{BuildHasherDefault, Hash, Hasher},
    hint,
    marker::PhantomData,
    ptr::NonNull,
    str,
    sync::atomic::{
        AtomicUsize,
        Ordering::{Relaxed, Release},
    },
};
use manually_init::ManuallyInit;
use scc::{Equivalent, HashMap};

// ═══════════════════════════════════════════════════════════════════════════
//                          第一层：公共API与核心接口
// ═══════════════════════════════════════════════════════════════════════════

/// 引用计数的不可变字符串，支持全局字符串池复用
///
/// # 设计目标
///
/// - **内存去重**：相同内容的字符串共享同一内存地址
/// - **零拷贝克隆**：`clone()` 只涉及原子递增操作
/// - **线程安全**：支持多线程环境下的安全使用
/// - **高性能查找**：使用预计算哈希值优化池查找
///
/// # 使用示例
///
/// ```rust
/// use interned::ArcStr;
///
/// let s1 = ArcStr::new("hello");
/// let s2 = ArcStr::new("hello");
///
/// // 相同内容的字符串共享同一内存
/// assert_eq!(s1.as_ptr(), s2.as_ptr());
/// assert_eq!(s1.ref_count(), 2);
///
/// // 零成本的字符串访问
/// println!("{}", s1.as_str()); // "hello"
/// ```
///
/// # 内存安全
///
/// `ArcStr` 内部使用原子引用计数确保内存安全，无需担心悬挂指针或数据竞争。
/// 当最后一个引用被释放时，字符串将自动从全局池中移除并释放内存。
#[repr(transparent)]
pub struct ArcStr {
    /// 指向 `ArcStrInner` 的非空指针
    ///
    /// # 不变量
    /// - 指针始终有效，指向正确初始化的 `ArcStrInner`
    /// - 引用计数至少为 1（在 drop 开始前）
    /// - 字符串数据始终是有效的 UTF-8
    ptr: NonNull<ArcStrInner>,

    /// 零大小标记，确保 `ArcStr` 拥有数据的所有权语义
    _marker: PhantomData<ArcStrInner>,
}

// SAFETY: ArcStr 使用原子引用计数，可以安全地跨线程传递和访问
unsafe impl Send for ArcStr {}
unsafe impl Sync for ArcStr {}

impl ArcStr {
    /// 创建或复用字符串实例
    ///
    /// 如果全局池中已存在相同内容的字符串，则复用现有实例并增加引用计数；
    /// 否则创建新实例并加入池中。
    ///
    /// # 并发策略
    ///
    /// 使用双重检查锁定模式来平衡性能和正确性：
    /// 1. **读锁快速路径**：大多数情况下只需要读锁即可找到现有字符串
    /// 2. **写锁创建路径**：仅在确实需要创建新字符串时获取写锁  
    /// 3. **双重验证**：获取写锁后再次检查，防止并发创建重复实例
    ///
    /// # 性能特征
    ///
    /// - **池命中**：O(1) `HashMap` 查找 + 原子递增
    /// - **池缺失**：O(1) 内存分配 + O(1) `HashMap` 插入
    /// - **哈希计算**：使用 ahash 的高性能哈希算法
    ///
    /// # Examples
    ///
    /// ```rust
    /// let s1 = ArcStr::new("shared_content");
    /// let s2 = ArcStr::new("shared_content"); // 复用 s1 的内存
    /// assert_eq!(s1.as_ptr(), s2.as_ptr());
    /// ```
    pub fn new<S: AsRef<str>>(s: S) -> Self {
        let string = s.as_ref();

        // 阶段 0：预计算内容哈希
        //
        // 这个哈希值在整个生命周期中会被多次使用：
        // - 池查找时作为 HashMap 的键
        // - 存储在 ArcStrInner 中用于后续 drop 优化
        let hash = CONTENT_HASHER.hash_one(string);

        // ===== 阶段 1：读锁快速路径 =====
        // 大部分情况下字符串已经在池中，这个路径是最常见的
        {
            let pool = ARC_STR_POOL.get();
            if let Some(existing) = Self::try_find_existing(pool, hash, string) {
                return existing;
            }
            // 读锁自动释放
        }

        // ===== 阶段 2：写锁创建路径 =====
        // 进入这里说明需要创建新的字符串实例
        let pool = ARC_STR_POOL.get();

        match pool.raw_entry().from_key_hashed_nocheck_sync(hash, string) {
            scc::hash_map::RawEntry::Occupied(entry) => {
                // 双重检查：在获取写锁的过程中，其他线程可能已经创建了相同的字符串
                let ptr = entry.key().0;

                // 找到匹配的字符串，增加其引用计数
                // SAFETY: 池中的指针始终有效，且引用计数操作是原子的
                unsafe { ptr.as_ref().inc_strong() };

                Self { ptr, _marker: PhantomData }
            }
            scc::hash_map::RawEntry::Vacant(entry) => {
                // 确认需要创建新实例：分配内存并初始化
                let layout = ArcStrInner::layout_for_string(string.len());

                // SAFETY: layout_for_string 确保布局有效且大小合理
                let ptr = unsafe {
                    let alloc: *mut ArcStrInner = alloc::alloc::alloc(layout).cast();

                    if alloc.is_null() {
                        hint::cold_path();
                        alloc::alloc::handle_alloc_error(layout);
                    }

                    let ptr = NonNull::new_unchecked(alloc);
                    ArcStrInner::write_with_string(ptr, string, hash);
                    ptr
                };

                // 将新创建的字符串加入全局池
                // 使用 from_key_hashed_nocheck 避免重复计算哈希
                entry.insert_hashed_nocheck(hash, ThreadSafePtr(ptr), ());

                Self { ptr, _marker: PhantomData }
            }
        }
    }

    /// 获取字符串切片（零成本操作）
    ///
    /// 直接访问底层字符串数据，无任何额外开销。
    ///
    /// # 性能
    ///
    /// 这是一个 `const fn`，在编译时就能确定偏移量，
    /// 运行时仅需要一次内存解引用。
    #[must_use]
    #[inline]
    pub const fn as_str(&self) -> &str {
        // SAFETY: ptr 在 ArcStr 生命周期内始终指向有效的 ArcStrInner，
        // 且字符串数据保证是有效的 UTF-8
        unsafe { self.ptr.as_ref().as_str() }
    }

    /// 获取字符串的字节切片
    ///
    /// 提供对底层字节数据的直接访问。
    #[must_use]
    #[inline]
    pub const fn as_bytes(&self) -> &[u8] {
        // SAFETY: ptr 始终指向有效的 ArcStrInner
        unsafe { self.ptr.as_ref().as_bytes() }
    }

    /// 获取字符串长度（字节数）
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        // SAFETY: ptr 始终指向有效的 ArcStrInner
        unsafe { self.ptr.as_ref().string_len }
    }

    /// 检查字符串是否为空
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool { self.len() == 0 }

    /// 获取当前引用计数
    ///
    /// 注意：由于并发访问，返回的值可能在返回后立即发生变化。
    /// 此方法主要用于调试和测试。
    #[must_use]
    #[inline]
    pub fn ref_count(&self) -> usize {
        // SAFETY: ptr 始终指向有效的 ArcStrInner
        unsafe { self.ptr.as_ref().strong_count() }
    }

    /// 获取字符串数据的内存地址（用于调试和测试）
    ///
    /// 返回字符串内容的起始地址，可用于验证字符串是否共享内存。
    #[must_use]
    #[inline]
    pub const fn as_ptr(&self) -> *const u8 {
        // SAFETY: ptr 始终指向有效的 ArcStrInner
        unsafe { self.ptr.as_ref().string_ptr() }
    }

    /// 内部辅助函数：在池中查找已存在的字符串
    ///
    /// 这个函数被提取出来以消除读锁路径和写锁路径中的重复代码。
    /// 使用 hashbrown 的优化API来避免重复哈希计算。
    ///
    /// # 参数
    ///
    /// - `pool`: 字符串池的引用
    /// - `hash`: 预计算的字符串哈希值
    /// - `string`: 要查找的字符串内容
    ///
    /// # 返回值
    ///
    /// 如果找到匹配的字符串，返回增加引用计数后的 `ArcStr`；否则返回 `None`。
    #[must_use]
    #[inline]
    fn try_find_existing(pool: &PtrMap, hash: u64, string: &str) -> Option<Self> {
        // 使用 hashbrown 的 from_key_hashed_nocheck API
        // 这利用了 Equivalent trait 来进行高效比较
        use scc::hash_map::RawEntry;
        let RawEntry::Occupied(entry) = pool.raw_entry().from_key_hashed_nocheck_sync(hash, string)
        else {
            return None;
        };
        let ThreadSafePtr(ptr) = *entry.key();

        // 找到匹配的字符串，增加其引用计数
        // SAFETY: 池中的指针始终有效，且引用计数操作是原子的
        unsafe { ptr.as_ref().inc_strong() };

        Some(Self { ptr, _marker: PhantomData })
    }
}

impl Clone for ArcStr {
    /// 克隆字符串引用（仅增加引用计数）
    ///
    /// 这是一个极其轻量的操作，只涉及一次原子递增。
    /// 不会复制字符串内容，新的 `ArcStr` 与原实例共享相同的底层内存。
    ///
    /// # 性能
    ///
    /// 时间复杂度：O(1) - 单次原子操作
    /// 空间复杂度：O(1) - 无额外内存分配
    #[inline]
    fn clone(&self) -> Self {
        // SAFETY: ptr 在当前 ArcStr 生命周期内有效
        unsafe { self.ptr.as_ref().inc_strong() }
        Self { ptr: self.ptr, _marker: PhantomData }
    }
}

impl Drop for ArcStr {
    /// 释放字符串引用
    ///
    /// 递减引用计数，如果这是最后一个引用，则从全局池中移除并释放内存。
    ///
    /// # 并发处理
    ///
    /// 由于多个线程可能同时释放同一字符串的引用，这里使用了谨慎的双重检查：
    /// 1. 原子递减引用计数
    /// 2. 如果计数变为0，获取池的写锁
    /// 3. 再次检查引用计数（防止并发的clone操作）
    /// 4. 确认后从池中移除并释放内存
    ///
    /// # 性能优化
    ///
    /// 使用预存储的哈希值进行 O(1) 的池查找和删除，避免重新计算哈希。
    fn drop(&mut self) {
        // SAFETY: ptr 在 drop 开始时仍然有效
        unsafe {
            let inner = self.ptr.as_ref();

            // 原子递减引用计数
            if !inner.dec_strong() {
                // 不是最后一个引用，直接返回
                return;
            }

            // 这是最后一个引用，需要清理资源
            let pool = ARC_STR_POOL.get();
            // 使用指针相等比较，这是绝对的 O(1) 操作
            let entry =
                pool.raw_entry().from_key_hashed_nocheck_sync(inner.hash, &ThreadSafePtr(self.ptr));

            if let scc::hash_map::RawEntry::Occupied(e) = entry {
                // 双重检查引用计数
                // 在获取写锁期间，其他线程可能clone了这个字符串
                if inner.strong_count() != 0 {
                    return;
                }

                // 确认是最后一个引用，执行清理
                e.remove();

                // 释放底层内存
                let layout = ArcStrInner::layout_for_string_unchecked(inner.string_len);
                alloc::alloc::dealloc(self.ptr.cast().as_ptr(), layout);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//                          第二层：标准库集成
// ═══════════════════════════════════════════════════════════════════════════

// # 基础 Trait 实现
//
// 这些实现确保 `ArcStr` 能够与 Rust 的标准库类型无缝集成，
// 提供符合直觉的比较、格式化和访问接口。

impl PartialEq for ArcStr {
    /// 基于指针的快速相等比较
    ///
    /// # 优化原理
    ///
    /// 由于字符串池保证相同内容的字符串具有相同的内存地址，
    /// 我们可以通过比较指针来快速判断字符串是否相等，
    /// 避免逐字节的内容比较。
    ///
    /// 这使得相等比较成为 O(1) 操作，而不是 O(n)。
    #[inline]
    fn eq(&self, other: &Self) -> bool { self.ptr == other.ptr }
}

impl Eq for ArcStr {}

impl PartialOrd for ArcStr {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl Ord for ArcStr {
    /// 基于字符串内容的字典序比较
    ///
    /// 注意：这里必须比较内容而不是指针，因为指针地址与字典序无关。
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering { self.as_str().cmp(other.as_str()) }
}

impl Hash for ArcStr {
    /// 基于字符串内容的哈希
    ///
    /// 虽然内部存储了预计算的哈希值，但这里重新计算以确保
    /// 与 `&str` 和 `String` 的哈希值保持一致。
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) { self.as_str().hash(state) }
}

impl fmt::Display for ArcStr {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { fmt::Display::fmt(self.as_str(), f) }
}

impl fmt::Debug for ArcStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Debug::fmt(self.as_str(), f) }
}

impl const AsRef<str> for ArcStr {
    #[inline]
    fn as_ref(&self) -> &str { self.as_str() }
}

impl const AsRef<[u8]> for ArcStr {
    #[inline]
    fn as_ref(&self) -> &[u8] { self.as_bytes() }
}

impl const Borrow<str> for ArcStr {
    #[inline]
    fn borrow(&self) -> &str { self.as_str() }
}

impl const core::ops::Deref for ArcStr {
    type Target = str;

    #[inline]
    fn deref(&self) -> &Self::Target { self.as_str() }
}

// # 与其他字符串类型的互操作性
//
// 这些实现使得 `ArcStr` 可以与 Rust 生态系统中的各种字符串类型
// 进行直接比较，提供良好的开发体验。

impl const PartialEq<str> for ArcStr {
    #[inline]
    fn eq(&self, other: &str) -> bool { self.as_str() == other }
}

impl const PartialEq<&str> for ArcStr {
    #[inline]
    fn eq(&self, other: &&str) -> bool { self.as_str() == *other }
}

impl const PartialEq<ArcStr> for str {
    #[inline]
    fn eq(&self, other: &ArcStr) -> bool { self == other.as_str() }
}

impl const PartialEq<ArcStr> for &str {
    #[inline]
    fn eq(&self, other: &ArcStr) -> bool { *self == other.as_str() }
}

impl const PartialEq<String> for ArcStr {
    #[inline]
    fn eq(&self, other: &String) -> bool { self.as_str() == other.as_str() }
}

impl const PartialEq<ArcStr> for String {
    #[inline]
    fn eq(&self, other: &ArcStr) -> bool { self.as_str() == other.as_str() }
}

impl PartialOrd<str> for ArcStr {
    #[inline]
    fn partial_cmp(&self, other: &str) -> Option<Ordering> { Some(self.as_str().cmp(other)) }
}

impl PartialOrd<String> for ArcStr {
    #[inline]
    fn partial_cmp(&self, other: &String) -> Option<Ordering> {
        Some(self.as_str().cmp(other.as_str()))
    }
}

// # 类型转换实现
//
// 提供从各种字符串类型到 `ArcStr` 的便捷转换，
// 以及从 `ArcStr` 到其他类型的转换。

impl<'a> From<&'a str> for ArcStr {
    #[inline]
    fn from(s: &'a str) -> Self { Self::new(s) }
}

impl<'a> From<&'a String> for ArcStr {
    #[inline]
    fn from(s: &'a String) -> Self { Self::new(s) }
}

impl From<String> for ArcStr {
    #[inline]
    fn from(s: String) -> Self { Self::new(s) }
}

impl<'a> From<alloc::borrow::Cow<'a, str>> for ArcStr {
    #[inline]
    fn from(cow: alloc::borrow::Cow<'a, str>) -> Self { Self::new(cow) }
}

impl From<alloc::boxed::Box<str>> for ArcStr {
    #[inline]
    fn from(s: alloc::boxed::Box<str>) -> Self { Self::new(s) }
}

impl From<ArcStr> for String {
    #[inline]
    fn from(s: ArcStr) -> Self { s.as_str().to_owned() }
}

impl From<ArcStr> for alloc::boxed::Box<str> {
    #[inline]
    fn from(s: ArcStr) -> Self { s.as_str().into() }
}

impl str::FromStr for ArcStr {
    type Err = core::convert::Infallible;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> { Ok(Self::new(s)) }
}

/// # Serde 序列化支持
///
/// 条件编译的 Serde 支持，使 `ArcStr` 可以参与序列化/反序列化流程。
/// 序列化时输出字符串内容，反序列化时重新建立池化引用。
#[cfg(feature = "serde")]
mod serde_impls {
    use super::ArcStr;
    use serde_core::{Deserialize, Deserializer, Serialize, Serializer};

    impl Serialize for ArcStr {
        #[inline]
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer {
            self.as_str().serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for ArcStr {
        #[inline]
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de> {
            String::deserialize(deserializer).map(ArcStr::new)
        }
    }
}

/// # Rkyv 序列化支持
///
/// 条件编译的 Rkyv 支持，使 `ArcStr` 可以参与序列化/反序列化流程。
/// 序列化时输出字符串内容，反序列化时重新建立池化引用。
#[cfg(feature = "rkyv")]
mod rkyv_impls {
    use super::ArcStr;
    use core::cmp::Ordering;
    use rkyv::{
        Archive, Deserialize, DeserializeUnsized, Place, Serialize, SerializeUnsized,
        rancor::{Fallible, Source},
        string::{ArchivedString, StringResolver},
    };

    impl Archive for ArcStr {
        type Archived = ArchivedString;
        type Resolver = StringResolver;

        #[inline]
        fn resolve(&self, resolver: Self::Resolver, out: Place<Self::Archived>) {
            ArchivedString::resolve_from_str(self.as_str(), resolver, out);
        }
    }

    impl<S: Fallible + ?Sized> Serialize<S> for ArcStr
    where
        S::Error: Source,
        str: SerializeUnsized<S>,
    {
        fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
            ArchivedString::serialize_from_str(self.as_str(), serializer)
        }
    }

    impl<D: Fallible + ?Sized> Deserialize<ArcStr, D> for ArchivedString
    where str: DeserializeUnsized<str, D>
    {
        fn deserialize(&self, _: &mut D) -> Result<ArcStr, D::Error> {
            Ok(ArcStr::new(self.as_str()))
        }
    }

    impl PartialEq<ArcStr> for ArchivedString {
        #[inline]
        fn eq(&self, other: &ArcStr) -> bool { PartialEq::eq(self.as_str(), other.as_str()) }
    }

    impl PartialEq<ArchivedString> for ArcStr {
        #[inline]
        fn eq(&self, other: &ArchivedString) -> bool {
            PartialEq::eq(other.as_str(), self.as_str())
        }
    }

    impl PartialOrd<ArcStr> for ArchivedString {
        #[inline]
        fn partial_cmp(&self, other: &ArcStr) -> Option<Ordering> {
            self.as_str().partial_cmp(other.as_str())
        }
    }

    impl PartialOrd<ArchivedString> for ArcStr {
        #[inline]
        fn partial_cmp(&self, other: &ArchivedString) -> Option<Ordering> {
            self.as_str().partial_cmp(other.as_str())
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//                          第三层：核心实现机制
// ═══════════════════════════════════════════════════════════════════════════

// # 内存布局与数据结构设计
//
// 这个模块包含了 `ArcStr` 的底层数据结构定义和内存布局管理。
// 理解这部分有助于深入了解性能优化的原理。

/// 字符串内容的内部表示（DST 头部）
///
/// # 内存布局设计
///
/// 使用 `#[repr(C)]` 确保内存布局稳定，字符串数据紧跟在结构体后面：
///
/// ```text
/// 64位系统内存布局：
/// ┌────────────────────┬──────────────────────────────────────────┐
/// │ 字段                │ 大小与对齐                                │
/// ├────────────────────┼──────────────────────────────────────────┤
/// │ hash: u64          │ 8字节, 8字节对齐 (offset: 0)               │
/// │ count: AtomicUsize │ 8字节, 8字节对齐 (offset: 8)               │
/// │ string_len: usize  │ 8字节, 8字节对齐 (offset: 16)              │
/// ├────────────────────┼──────────────────────────────────────────┤
/// │ [字符串数据]         │ string_len字节, 1字节对齐 (offset: 24)     │
/// └────────────────────┴──────────────────────────────────────────┘
/// 总头部大小：24字节
///
/// 32位系统内存布局：
/// ┌────────────────────┬──────────────────────────────────────────┐
/// │ hash: u64          │ 8字节, 8字节对齐 (offset: 0)               │
/// │ count: AtomicUsize │ 4字节, 4字节对齐 (offset: 8)               │
/// │ string_len: usize  │ 4字节, 4字节对齐 (offset: 12)              │
/// ├────────────────────┼──────────────────────────────────────────┤
/// │ [字符串数据]         │ string_len字节, 1字节对齐 (offset: 16)     │
/// └────────────────────┴──────────────────────────────────────────┘
/// 总头部大小：16字节
/// ```
///
/// # 设计考量
///
/// 1. **哈希值前置**：将 `hash` 放在首位确保在32位系统上的正确对齐
/// 2. **原子计数器**：使用 `AtomicUsize` 保证并发安全的引用计数
/// 3. **长度缓存**：预存字符串长度避免重复计算
/// 4. **DST布局**：字符串数据直接跟随结构体，减少间接访问
#[repr(C)]
struct ArcStrInner {
    /// 预计算的内容哈希值
    ///
    /// 这个哈希值在多个场景中被复用：
    /// - 全局池的 `HashMap` 键
    /// - `Drop` 时的快速查找
    /// - 避免重复哈希计算的性能优化
    hash: u64,

    /// 原子引用计数
    ///
    /// 使用原生原子类型确保最佳性能。
    /// 计数范围：[1, `isize::MAX`]，超出时触发abort。
    count: AtomicUsize,

    /// 字符串的字节长度（UTF-8编码）
    ///
    /// 预存长度避免在每次访问时扫描字符串。
    /// 不包含NUL终止符。
    string_len: usize,
    // 注意：字符串数据紧跟在这个结构体后面，
    // 通过 layout_for_string() 计算的布局来确保正确的内存分配
}

impl ArcStrInner {
    /// 字符串长度的上限
    ///
    /// 计算公式：`isize::MAX - sizeof(ArcStrInner)`
    /// 这确保总分配大小不会溢出有符号整数范围。
    const MAX_LEN: usize = isize::MAX as usize - core::mem::size_of::<Self>();

    /// 获取字符串数据的起始地址
    ///
    /// # Safety
    ///
    /// - `self` 必须是指向有效 `ArcStrInner` 的指针
    /// - 必须确保字符串数据已经被正确初始化
    /// - 调用者负责确保返回的指针在使用期间保持有效
    #[must_use]
    #[inline]
    const unsafe fn string_ptr(&self) -> *const u8 {
        // SAFETY: repr(C) 保证字符串数据位于结构体末尾的固定偏移处
        core::ptr::from_ref(self).add(1).cast()
    }

    /// 获取字符串的字节切片
    ///
    /// # Safety
    ///
    /// - `self` 必须是指向有效 `ArcStrInner` 的指针
    /// - 字符串数据必须已经被正确初始化
    /// - `string_len` 必须准确反映实际字符串长度
    /// - 字符串数据必须在返回的切片生命周期内保持有效
    #[must_use]
    #[inline]
    const unsafe fn as_bytes(&self) -> &[u8] {
        let ptr = self.string_ptr();
        // SAFETY: 调用者保证 ptr 指向有效的 string_len 字节数据
        core::slice::from_raw_parts(ptr, self.string_len)
    }

    /// 获取字符串切片引用
    ///
    /// # Safety
    ///
    /// - `self` 必须是指向有效 `ArcStrInner` 的指针
    /// - 字符串数据必须是有效的 UTF-8 编码
    /// - `string_len` 必须准确反映实际字符串长度
    /// - 字符串数据必须在返回的切片生命周期内保持有效
    #[must_use]
    #[inline]
    const unsafe fn as_str(&self) -> &str {
        // SAFETY: 调用者保证字符串数据是有效的 UTF-8
        core::str::from_utf8_unchecked(self.as_bytes())
    }

    /// 计算存储指定长度字符串所需的内存布局
    ///
    /// 这个函数计算出正确的内存大小和对齐要求，
    /// 确保结构体和字符串数据都能正确对齐。
    ///
    /// # Panics
    ///
    /// 如果 `string_len > Self::MAX_LEN`，函数会panic。
    /// 这是为了防止整数溢出和无效的内存布局。
    ///
    /// # Examples
    ///
    /// ```rust
    /// let layout = ArcStrInner::layout_for_string(5); // "hello"
    /// assert!(layout.size() >= 24 + 5); // 64位系统
    /// ```
    fn layout_for_string(string_len: usize) -> Layout {
        if string_len > Self::MAX_LEN {
            hint::cold_path();
            panic!("字符串过长: {} 字节 (最大支持: {})", string_len, Self::MAX_LEN);
        }

        // SAFETY: 长度检查通过，布局计算是安全的
        unsafe { Self::layout_for_string_unchecked(string_len) }
    }

    /// 计算存储指定长度字符串所需的内存布局（不检查长度）
    ///
    /// # Safety
    ///
    /// 调用者必须保证 `string_len <= Self::MAX_LEN`
    const unsafe fn layout_for_string_unchecked(string_len: usize) -> Layout {
        let header = Layout::new::<Self>();
        let string_data = Layout::from_size_align_unchecked(string_len, 1);
        // SAFETY: 长度已经过检查，布局计算不会溢出
        let (combined, _offset) = header.extend(string_data).unwrap_unchecked();
        combined.pad_to_align()
    }

    /// 在指定内存位置初始化 `ArcStrInner` 并写入字符串数据
    ///
    /// 这是一个低级函数，负责设置完整的DST结构：
    /// 1. 初始化头部字段
    /// 2. 复制字符串数据到紧邻的内存
    ///
    /// # Safety
    ///
    /// - `ptr` 必须指向通过 `layout_for_string(string.len())` 分配的有效内存
    /// - 内存必须正确对齐且大小足够
    /// - `string` 必须是有效的 UTF-8 字符串
    /// - 调用者负责最终释放这块内存
    /// - 在调用此函数后，调用者必须确保引用计数正确管理
    const unsafe fn write_with_string(ptr: NonNull<Self>, string: &str, hash: u64) {
        let inner = ptr.as_ptr();

        // 第一步：初始化头部结构体
        // SAFETY: ptr 指向有效的已分配内存，大小足够容纳 Self
        core::ptr::write(
            inner,
            Self { hash, count: AtomicUsize::new(1), string_len: string.len() },
        );

        // 第二步：复制字符串数据到紧邻头部后的内存
        // SAFETY:
        // - string_ptr() 计算出的地址位于已分配内存范围内
        // - string.len() 与分配时的长度一致
        // - string.as_ptr() 指向有效的 UTF-8 数据
        let string_ptr = (*inner).string_ptr().cast_mut();
        core::ptr::copy_nonoverlapping(string.as_ptr(), string_ptr, string.len());
    }

    /// 原子递增引用计数
    ///
    /// # 溢出处理
    ///
    /// 如果引用计数超过 `isize::MAX`，函数会立即abort程序。
    /// 这是一个极端情况，在正常使用中几乎不可能发生。
    ///
    /// # Safety
    ///
    /// - `self` 必须指向有效的 `ArcStrInner`
    /// - 当前引用计数必须至少为 1（即存在有效引用）
    #[inline]
    unsafe fn inc_strong(&self) {
        let old_count = self.count.fetch_add(1, Relaxed);

        // 防止引用计数溢出 - 这是一个安全检查
        if old_count > isize::MAX as usize {
            hint::cold_path();
            // 溢出是内存安全问题，必须立即终止程序
            core::intrinsics::abort();
        }
    }

    /// 原子递减引用计数
    ///
    /// 使用 Release 内存序确保所有之前的修改对后续的操作可见。
    /// 这对于安全的内存回收至关重要。
    ///
    /// # Safety
    ///
    /// - `self` 必须指向有效的 `ArcStrInner`
    /// - 当前引用计数必须至少为 1
    ///
    /// # 返回值
    ///
    /// 如果这是最后一个引用（计数变为 0），返回 `true`；否则返回 `false`。
    #[inline]
    unsafe fn dec_strong(&self) -> bool {
        // Release ordering: 确保之前的所有修改对后续的内存释放操作可见
        self.count.fetch_sub(1, Release) == 1
    }

    /// 获取当前引用计数的快照
    ///
    /// 注意：由于并发性，返回值可能在返回后立即过时。
    /// 此方法主要用于调试和测试目的。
    #[inline]
    fn strong_count(&self) -> usize { self.count.load(Relaxed) }
}

// # 全局字符串池的设计与实现
//
// 全局池是整个系统的核心，负责去重和生命周期管理。

/// 线程安全的内部指针包装
///
/// 这个类型解决了在 `HashMap` 中存储 `NonNull<ArcStrInner>` 的问题：
/// - 提供必要的 trait 实现（ `Hash`, `PartialEq`, `Send`, `Sync` ）
/// - 封装指针的线程安全语义
/// - 支持基于内容的查找（通过 Equivalent trait）
///
/// # 线程安全性
///
/// 虽然包装了裸指针，但 `ThreadSafePtr` 是线程安全的，因为：
/// - 指向的 `ArcStrInner` 是不可变的（除了原子引用计数）
/// - 引用计数使用原子操作
/// - 生命周期由全局池管理，确保指针有效性
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
struct ThreadSafePtr(NonNull<ArcStrInner>);

// SAFETY: ArcStrInner 内容不可变且使用原子引用计数，可以安全地跨线程访问
unsafe impl Send for ThreadSafePtr {}
unsafe impl Sync for ThreadSafePtr {}

impl const core::ops::Deref for ThreadSafePtr {
    type Target = NonNull<ArcStrInner>;

    #[inline]
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl Hash for ThreadSafePtr {
    /// 使用预存储的哈希值
    ///
    /// 这是一个关键优化：我们不重新计算字符串内容的哈希，
    /// 而是直接使用存储在 `ArcStrInner` 中的预计算值。
    /// 配合 `IdentityHasher` 使用，避免任何额外的哈希计算。
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // SAFETY: ThreadSafePtr 保证指针在池生命周期内始终有效
        unsafe {
            let inner = self.0.as_ref();
            state.write_u64(inner.hash);
        }
    }
}

impl PartialEq for ThreadSafePtr {
    /// 基于指针相等的比较
    ///
    /// 这是池去重机制的核心：只有指向同一内存地址的指针
    /// 才被认为是"相同"的池条目。内容相同但地址不同的字符串
    /// 在池中是不应该同时存在的。
    #[inline]
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

impl Eq for ThreadSafePtr {}

impl Equivalent<ThreadSafePtr> for str {
    /// 支持用 `&str` 在 `HashSet<ThreadSafePtr>` 中查找
    ///
    /// 这个实现使得我们可以用字符串内容来查找池中的条目，
    /// 而不需要先构造一个 `ThreadSafePtr`。
    ///
    /// # 性能优化
    ///
    /// 先比较字符串长度（单个 usize 比较），只有长度相等时
    /// 才进行内容比较（潜在的 memcmp）。这避免了在长度不等时
    /// 构造 fat pointer 的开销。
    #[inline]
    fn equivalent(&self, key: &ThreadSafePtr) -> bool {
        // SAFETY: 池中的 ThreadSafePtr 保证指向有效的 ArcStrInner
        unsafe {
            let inner = key.0.as_ref();

            // 优化：先比较长度（O(1)），避免不必要的内容比较
            if inner.string_len != self.len() {
                return false;
            }

            // 长度相等时进行内容比较
            inner.as_str() == self
        }
    }
}

// # 哈希算法选择与池类型定义

/// 透传哈希器，用于全局池内部
///
/// 由于我们在 `ArcStrInner` 中预存了哈希值，池内部的 `HashMap`
/// 不需要重新计算哈希。`IdentityHasher` 直接透传 u64 值。
///
/// # 工作原理
///
/// 1. `ThreadSafePtr::hash()` 调用 `hasher.write_u64(stored_hash)`
/// 2. `IdentityHasher::write_u64()` 直接存储这个值
/// 3. `IdentityHasher::finish()` 返回存储的值
/// 4. `HashMap` 使用这个哈希值进行桶分配和查找
///
/// 这避免了重复的哈希计算，将池操作的哈希开销降到最低。
#[derive(Default, Clone, Copy)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn write(&mut self, _: &[u8]) {
        unreachable!("IdentityHasher usage error");
    }

    #[inline]
    fn write_u64(&mut self, id: u64) { self.0 = id }

    #[inline]
    fn finish(&self) -> u64 { self.0 }
}

/// 池的类型别名，简化代码
type PoolHasher = BuildHasherDefault<IdentityHasher>;
type PtrMap = HashMap<ThreadSafePtr, (), PoolHasher>;

/// 内容哈希计算器
///
/// 使用 ahash 的高性能随机哈希算法来计算字符串内容的哈希值。
/// 这个哈希值会被存储在 `ArcStrInner` 中，用于整个生命周期。
///
/// # 为什么使用 ahash？
///
/// - 高性能：比标准库的 `DefaultHasher` 更快
/// - 安全性：抗哈希洪水攻击
/// - 质量：分布均匀，减少哈希冲突
static CONTENT_HASHER: ManuallyInit<ahash::RandomState> = ManuallyInit::new();

/// 全局字符串池
///
/// 使用 `HashMap` 实现高并发的字符串池：
/// - **读锁**：多个线程可以同时查找现有字符串
/// - **写锁**：创建新字符串时需要独占访问
/// - **容量预分配**：避免初期的频繁扩容
///
/// # 并发模式
///
/// ```text
/// 并发读取（常见情况）:
/// Thread A: read_lock() -> 查找 "hello" -> 找到 -> 返回
/// Thread B: read_lock() -> 查找 "world" -> 找到 -> 返回  
/// Thread C: read_lock() -> 查找 "hello" -> 找到 -> 返回
///
/// 并发写入（偶尔发生）:
/// Thread D: write_lock() -> 查找 "new" -> 未找到 -> 创建 -> 插入 -> 返回
/// ```
static ARC_STR_POOL: ManuallyInit<PtrMap> = ManuallyInit::new();

/// 初始化全局字符串池
///
/// 这个函数必须在使用 `ArcStr` 之前调用，通常在程序启动时完成。
/// 初始化过程包括：
/// 1. 创建内容哈希计算器
/// 2. 创建空的字符串池（预分配128个条目的容量）
///
/// # 线程安全性
///
/// 虽然这个函数本身不是线程安全的，但它应该在单线程环境下
/// （如 main 函数开始或静态初始化时）被调用一次。
#[inline(always)]
// 只调用一次
#[allow(clippy::inline_always)]
pub(crate) fn __init() {
    CONTENT_HASHER.init(ahash::RandomState::new());
    ARC_STR_POOL.init(PtrMap::with_capacity_and_hasher(128, PoolHasher::default()));
}

// ═══════════════════════════════════════════════════════════════════════════
//                          第四层：性能优化实现
// ═══════════════════════════════════════════════════════════════════════════

/// # 内存管理优化策略
///
/// 这个模块包含了各种底层的性能优化实现，
/// 包括内存布局计算、分配策略和并发优化。

// （这里是性能关键的内部函数实现，已经在上面的代码中体现了）

/// # 并发控制优化
///
/// 双重检查锁定模式的详细实现分析：
///
/// ```text
/// 时间线示例：
/// T1: Thread A 调用 ArcStr::new("test")
/// T2: Thread A 获取读锁，查找池，未找到
/// T3: Thread A 释放读锁
/// T4: Thread B 调用 ArcStr::new("test")  
/// T5: Thread B 获取读锁，查找池，未找到
/// T6: Thread B 释放读锁
/// T7: Thread A 获取写锁
/// T8: Thread A 再次查找（双重检查），确认未找到
/// T9: Thread A 创建新实例，插入池
/// T10: Thread A 释放写锁
/// T11: Thread B 等待写锁...
/// T12: Thread B 获取写锁
/// T13: Thread B 再次查找（双重检查），找到！
/// T14: Thread B 增加引用计数，释放写锁
/// ```

// ═══════════════════════════════════════════════════════════════════════════
//                          第五层：测试与工具
// ═══════════════════════════════════════════════════════════════════════════

/// # 测试辅助工具
///
/// 这些函数仅在测试环境中可用，用于检查池的内部状态
/// 和进行隔离测试。

#[cfg(test)]
pub(crate) fn pool_stats() -> (usize, usize) {
    let pool = ARC_STR_POOL.get();
    (pool.len(), pool.capacity())
}

#[cfg(test)]
pub(crate) fn clear_pool_for_test() {
    use std::{thread, time::Duration};
    // 短暂等待确保其他线程完成操作
    thread::sleep(Duration::from_millis(10));
    ARC_STR_POOL.get().clear_sync();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{thread, time::Duration};

    /// 运行隔离的测试，确保测试间不会相互影响
    fn run_isolated_test<F: FnOnce()>(f: F) {
        clear_pool_for_test();
        f();
        clear_pool_for_test();
    }

    #[test]
    fn test_basic_functionality() {
        run_isolated_test(|| {
            let s1 = ArcStr::new("hello");
            let s2 = ArcStr::new("hello");
            let s3 = ArcStr::new("world");

            // 验证相等性和指针共享
            assert_eq!(s1, s2);
            assert_ne!(s1, s3);
            assert_eq!(s1.ptr, s2.ptr); // 相同内容共享内存
            assert_ne!(s1.ptr, s3.ptr); // 不同内容不同内存

            // 验证基础操作
            assert_eq!(s1.as_str(), "hello");
            assert_eq!(s1.len(), 5);
            assert!(!s1.is_empty());

            // 验证池状态
            let (count, _) = pool_stats();
            assert_eq!(count, 2); // "hello" 和 "world"
        });
    }

    #[test]
    fn test_reference_counting() {
        run_isolated_test(|| {
            let s1 = ArcStr::new("test");
            assert_eq!(s1.ref_count(), 1);

            let s2 = s1.clone();
            assert_eq!(s1.ref_count(), 2);
            assert_eq!(s2.ref_count(), 2);
            assert_eq!(s1.ptr, s2.ptr);

            drop(s2);
            assert_eq!(s1.ref_count(), 1);

            drop(s1);
            // 等待 drop 完成
            thread::sleep(Duration::from_millis(5));
            assert_eq!(pool_stats().0, 0);
        });
    }

    #[test]
    fn test_pool_reuse() {
        run_isolated_test(|| {
            let s1 = ArcStr::new("reuse_test");
            let s2 = ArcStr::new("reuse_test");

            assert_eq!(s1.ptr, s2.ptr);
            assert_eq!(s1.ref_count(), 2);
            assert_eq!(pool_stats().0, 1); // 只有一个池条目
        });
    }

    #[test]
    fn test_thread_safety() {
        run_isolated_test(|| {
            let s = ArcStr::new("shared");
            let handles: Vec<_> = (0..10)
                .map(|_| {
                    let s_clone = ArcStr::clone(&s);
                    thread::spawn(move || {
                        let local = ArcStr::new("shared");
                        assert_eq!(*s_clone, local);
                        assert_eq!(s_clone.ptr, local.ptr);
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    #[test]
    fn test_empty_string() {
        run_isolated_test(|| {
            let empty = ArcStr::new("");
            assert!(empty.is_empty());
            assert_eq!(empty.len(), 0);
            assert_eq!(empty.as_str(), "");
        });
    }

    #[test]
    fn test_from_implementations() {
        run_isolated_test(|| {
            use alloc::borrow::Cow;

            let s1 = ArcStr::from("from_str");
            let s2 = ArcStr::from(String::from("from_string"));
            let s3 = ArcStr::from(Cow::Borrowed("from_cow"));

            assert_eq!(s1.as_str(), "from_str");
            assert_eq!(s2.as_str(), "from_string");
            assert_eq!(s3.as_str(), "from_cow");
        });
    }
}
