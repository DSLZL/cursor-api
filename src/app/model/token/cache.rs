#![allow(unsafe_op_in_unsafe_fn)]

use super::{Randomness, RawToken, UserId};
use crate::common::utils::{from_base64, to_base64};
use alloc::alloc::{alloc, dealloc, handle_alloc_error};
use core::{
    alloc::Layout,
    hash::Hasher,
    marker::PhantomData,
    mem::SizedTypeProperties as _,
    ptr::{NonNull, copy_nonoverlapping},
    slice::from_raw_parts,
    str::from_utf8_unchecked,
    sync::atomic::{AtomicUsize, Ordering},
};
use manually_init::ManuallyInit;
use scc::HashMap;

/// Token 的唯一标识键
///
/// 由用户ID和随机数组成，用于在全局缓存中查找对应的 Token
#[derive(
    Debug, PartialEq, Eq, Hash, Clone, Copy, ::rkyv::Archive, ::rkyv::Serialize, ::rkyv::Deserialize,
)]
#[rkyv(derive(PartialEq, Eq, Hash))]
pub struct TokenKey {
    /// 用户唯一标识
    pub user_id: UserId,
    /// 随机数部分，用于保证 Token 的唯一性
    pub randomness: Randomness,
}

impl TokenKey {
    /// 将 TokenKey 序列化为 base64 字符串
    ///
    /// 格式：24字节（16字节 user_id + 8字节 randomness）编码为 32 字符的 base64
    #[allow(clippy::inherent_to_string)]
    #[inline]
    pub fn to_string(self) -> String {
        let mut bytes = [0u8; 24];
        unsafe {
            copy_nonoverlapping(self.user_id.to_bytes().as_ptr(), bytes.as_mut_ptr(), 16);
            copy_nonoverlapping(self.randomness.to_bytes().as_ptr(), bytes.as_mut_ptr().add(16), 8);
        }
        to_base64(&bytes)
    }

    /// 将 TokenKey 序列化为可读字符串
    ///
    /// 格式：`<user_id>-<randomness>`
    #[inline]
    pub fn to_string2(self) -> String {
        let mut buffer = itoa::Buffer::new();
        let mut string = String::with_capacity(60);
        string.push_str(buffer.format(self.user_id.as_u128()));
        string.push('-');
        string.push_str(buffer.format(self.randomness.as_u64()));
        string
    }

    /// 从字符串解析 TokenKey
    ///
    /// 支持两种格式：
    /// 1. 32字符的 base64 编码
    /// 2. `<user_id>-<randomness>` 格式
    pub fn from_string(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();

        if bytes.len() > 60 {
            return None;
        }

        // base64 格式
        if bytes.len() == 32 {
            let decoded: [u8; 24] = __unwrap!(from_base64(s)?.try_into());
            let user_id = UserId::from_bytes(__unwrap!(decoded[0..16].try_into()));
            let randomness = Randomness::from_bytes(__unwrap!(decoded[16..24].try_into()));
            return Some(Self { user_id, randomness });
        }

        // 分隔符格式
        let mut iter = bytes.iter().enumerate();
        let mut first_num_end = None;
        let mut second_num_start = None;

        // 第一次循环：找到第一个非数字字符
        for (i, b) in iter.by_ref() {
            if !b.is_ascii_digit() {
                first_num_end = Some(i);
                break;
            }
        }

        let first_num_end = first_num_end?;

        // 第二次循环：从上次停止的地方继续，找到下一个数字字符
        for (i, b) in iter {
            if b.is_ascii_digit() {
                second_num_start = Some(i);
                break;
            }
        }

        let second_num_start = second_num_start?;

        let first_part = unsafe { from_utf8_unchecked(bytes.get_unchecked(..first_num_end)) };
        let second_part = unsafe { from_utf8_unchecked(bytes.get_unchecked(second_num_start..)) };

        let user_id_val = first_part.parse().ok()?;
        let randomness_val = second_part.parse().ok()?;

        Some(Self {
            user_id: UserId::from_u128(user_id_val),
            randomness: Randomness::from_u64(randomness_val),
        })
    }
}

/// Token 的内部表示
///
/// # Memory Layout
/// ```text
/// +----------------------+
/// | raw: RawToken        | 原始 token 数据
/// | count: AtomicUsize   | 引用计数
/// | string_len: usize    | 字符串长度
/// +----------------------+
/// | string data...       | UTF-8 字符串表示
/// +----------------------+
/// ```
struct TokenInner {
    /// 原始 token 数据
    raw: RawToken,
    /// 原子引用计数
    count: AtomicUsize,
    /// 字符串表示的长度
    string_len: usize,

    // /// fallback链表（很罕见）的上一个
    // prev: Option<NonNull<TokenInner>>,
    /// fallback链表（很罕见）的下一个
    next: Option<NonNull<TokenInner>>,
}

impl TokenInner {
    const STRING_MAX_LEN: usize = isize::MAX as usize + 1 - Self::ALIGN - Self::SIZE;

    /// 获取字符串数据的起始地址
    #[inline(always)]
    const unsafe fn string_ptr(&self) -> *const u8 { (self as *const Self).add(1) as *const u8 }

    /// 获取字符串切片
    #[inline(always)]
    const unsafe fn as_str(&self) -> &str {
        let ptr = self.string_ptr();
        let slice = from_raw_parts(ptr, self.string_len);
        from_utf8_unchecked(slice)
    }

    /// 计算存储指定长度字符串所需的内存布局
    fn layout_for_string(string_len: usize) -> Layout {
        if string_len > Self::STRING_MAX_LEN {
            __cold_path!();
            panic!("string is too long");
        }
        unsafe {
            Layout::new::<Self>()
                .extend(Layout::from_size_align_unchecked(string_len, 1))
                .unwrap_unchecked()
                .0
                .pad_to_align()
        }
    }

    /// 在指定内存位置写入结构体和字符串数据
    unsafe fn write_with_string(ptr: NonNull<Self>, raw: RawToken, string: &str) {
        let inner = ptr.as_ptr();

        // 写入结构体字段
        (*inner).raw = raw;
        (*inner).count = AtomicUsize::new(1);
        (*inner).string_len = string.len();
        // (*inner).prev = None;
        (*inner).next = None;

        // 复制字符串数据
        let string_ptr = (*inner).string_ptr() as *mut u8;
        copy_nonoverlapping(string.as_ptr(), string_ptr, string.len());
    }

    fn find_last_slot(ptr: NonNull<Self>) -> NonNull<Self> {
        let mut last = ptr;
        while let Some(next) = unsafe { last.as_ref().next } {
            last = next;
        }
        last
    }

    fn find(mut ptr: NonNull<Self>, raw: &RawToken) -> Option<NonNull<Self>> {
        while let Some(next) = unsafe { ptr.as_ref().next } {
            if unsafe { ptr.as_ref().raw.eq(raw) } {
                return Some(ptr);
            }
            ptr = next;
        }
        None
    }

    fn find_prev(root: NonNull<Self>, node: NonNull<Self>) -> Option<NonNull<Self>> {
        let mut current = root;
        while let Some(next) = unsafe { current.as_ref().next } {
            if next == node {
                return Some(current);
            }
            current = next;
        }
        None
    }
}

/// 引用计数的 Token，支持全局缓存复用
///
/// Token 是不可变的，线程安全的，并且会自动进行缓存管理。
/// 相同的 TokenKey 会复用同一个底层实例。
#[repr(transparent)]
pub struct Token {
    ptr: NonNull<TokenInner>,
    _pd: PhantomData<TokenInner>,
}

// Safety: Token 使用原子引用计数，可以安全地在线程间传递
unsafe impl Send for Token {}
unsafe impl Sync for Token {}

impl Clone for Token {
    #[inline]
    fn clone(&self) -> Self {
        unsafe {
            let count = self.ptr.as_ref().count.fetch_add(1, Ordering::Relaxed);
            if count > isize::MAX as usize {
                __cold_path!();
                std::process::abort();
            }
        }

        Self { ptr: self.ptr, _pd: PhantomData }
    }
}

/// 线程安全的内部指针包装
#[derive(Clone, Copy)]
#[repr(transparent)]
struct ThreadSafePtr(NonNull<TokenInner>);

unsafe impl Send for ThreadSafePtr {}
unsafe impl Sync for ThreadSafePtr {}

/// 全局 Token 缓存池
static TOKEN_MAP: ManuallyInit<HashMap<TokenKey, ThreadSafePtr, ahash::RandomState>> =
    ManuallyInit::new();

#[inline(always)]
pub fn __init() { TOKEN_MAP.init(HashMap::with_capacity_and_hasher(64, ahash::RandomState::new())) }

impl Token {
    /// 创建或复用 Token 实例
    ///
    /// 如果缓存中已存在相同的 TokenKey 且 RawToken 相同，则复用；
    /// 否则创建新实例（可能会覆盖旧的）。
    ///
    /// # 并发安全性
    /// - 使用 read-write lock 保护全局缓存
    /// - 快速路径（read lock）：尝试复用已有实例
    /// - 慢速路径（write lock）：双重检查后创建新实例，防止竞态条件
    pub fn new(raw: RawToken, string: Option<String>) -> Self {
        use scc::hash_map::RawEntry;

        let key = raw.key();
        let hash;

        // 快速路径：尝试从缓存中查找并增加引用计数
        {
            let cache = TOKEN_MAP.get();
            let builder = cache.raw_entry();
            hash = builder.hash(&key);
            if let RawEntry::Occupied(entry) = builder.from_key_hashed_nocheck_sync(hash, &key) {
                let &ThreadSafePtr(ptr) = entry.get();
                unsafe {
                    let inner = ptr.as_ref();
                    // 验证 RawToken 是否完全匹配（key 相同不代表 raw 相同）
                    if inner.raw == raw {
                        let count = inner.count.fetch_add(1, Ordering::Relaxed);
                        // 防止引用计数溢出（理论上不可能，但作为安全检查）
                        if count > isize::MAX as usize {
                            __cold_path!();
                            std::process::abort();
                        }
                        return Self { ptr, _pd: PhantomData };
                    } else {
                        __cold_path!();
                        if let Some(ptr) = TokenInner::find(ptr, &raw) {
                            let inner = ptr.as_ref();
                            let count = inner.count.fetch_add(1, Ordering::Relaxed);
                            // 防止引用计数溢出（理论上不可能，但作为安全检查）
                            if count > isize::MAX as usize {
                                __cold_path!();
                                std::process::abort();
                            }
                            return Self { ptr, _pd: PhantomData };
                        }
                    }
                }
            }
        }

        // 慢速路径：创建新实例（需要独占访问缓存）
        let cache = TOKEN_MAP.get();

        match cache.raw_entry().from_key_hashed_nocheck_sync(hash, &key) {
            RawEntry::Occupied(entry) => {
                // 双重检查：防止在获取 write lock 前，其他线程已经创建了相同的 Token
                let &ThreadSafePtr(mut ptr) = entry.get();
                unsafe {
                    let inner = ptr.as_ref();
                    if inner.raw == raw {
                        let count = inner.count.fetch_add(1, Ordering::Relaxed);
                        if count > isize::MAX as usize {
                            __cold_path!();
                            std::process::abort();
                        }
                    } else {
                        __cold_path!();
                        crate::debug!("{} != {}", inner.raw, raw);
                        let mut last = TokenInner::find_last_slot(ptr);
                        // 分配并初始化新实例（使用自定义 DST 布局）
                        ptr = Self::__new(raw, string);
                        // ptr.as_mut().prev = Some(last);
                        last.as_mut().next = Some(ptr);
                    }
                }

                Self { ptr, _pd: PhantomData }
            }
            RawEntry::Vacant(entry) => {
                // 分配并初始化新实例（使用自定义 DST 布局）
                let ptr = Self::__new(raw, string);

                // 将新实例插入缓存（持有 write lock，保证线程安全）
                entry.insert_hashed_nocheck(hash, key, ThreadSafePtr(ptr));

                Self { ptr, _pd: PhantomData }
            }
        }
    }

    fn __new(raw: RawToken, string: Option<String>) -> NonNull<TokenInner> {
        unsafe {
            // 准备字符串表示（在堆上分配之前）
            let string = string.unwrap_or_else(|| raw.to_string());
            let layout = TokenInner::layout_for_string(string.len());

            let alloc = alloc(layout) as *mut TokenInner;
            if alloc.is_null() {
                handle_alloc_error(layout);
            }
            let ptr = NonNull::new_unchecked(alloc);
            TokenInner::write_with_string(ptr, raw, &string);

            ptr
        }
    }

    /// 获取原始 token 数据
    #[inline(always)]
    pub const fn raw(&self) -> &RawToken { unsafe { &self.ptr.as_ref().raw } }

    /// 获取字符串表示
    #[inline(always)]
    pub const fn as_str(&self) -> &str { unsafe { self.ptr.as_ref().as_str() } }

    /// 获取 token 的键
    #[inline(always)]
    pub const fn key(&self) -> TokenKey { self.raw().key() }

    /// 检查是否为网页 token
    #[inline(always)]
    pub const fn is_web(&self) -> bool { self.raw().is_web() }

    /// 检查是否为会话 token
    #[inline(always)]
    pub const fn is_session(&self) -> bool { self.raw().is_session() }
}

impl Drop for Token {
    fn drop(&mut self) {
        unsafe {
            let inner = self.ptr.as_ref();

            // 递减引用计数，使用 Release ordering 确保之前的所有修改对后续操作可见
            if inner.count.fetch_sub(1, Ordering::Release) != 1 {
                // 不是最后一个引用，直接返回
                return;
            }

            // 最后一个引用：需要清理资源
            // 获取 write lock 以保护缓存操作，同时防止并发的 new() 操作干扰
            let cache = TOKEN_MAP.get();

            let key = inner.raw.key();
            if let scc::hash_map::RawEntry::Occupied(mut e) = cache.raw_entry().from_key_sync(&key)
            {
                // 双重检查引用计数：防止在等待 write lock 期间，其他线程通过 new() 增加了引用
                // 例如：
                //   Thread A: fetch_sub 返回 1
                //   Thread B: 在 new() 中找到此 token，fetch_add 增加计数
                //   Thread A: 获取 write lock
                // 此时必须重新检查，否则会错误地释放正在使用的内存
                if inner.count.load(Ordering::Relaxed) != 0 {
                    // 有新的引用产生，取消释放操作
                    return;
                }

                let &ThreadSafePtr(ptr) = e.get();

                if ptr == self.ptr {
                    if let Some(next) = inner.next {
                        __cold_path!();
                        let ThreadSafePtr(v) = e.get_mut();
                        *v = next;
                    } else {
                        // 确认是最后一个引用，执行清理：
                        // 1. 从缓存中移除（防止后续 new() 找到已释放的指针）
                        e.remove();
                    }
                } else {
                    __cold_path!();
                    if let Some(mut prev) = TokenInner::find_prev(ptr, self.ptr) {
                        prev.as_mut().next = inner.next;
                        // if let Some(mut next) = inner.next {
                        //     next.as_mut().prev = prev;
                        // }
                    } else {
                        __cold_path!();
                        std::process::abort();
                    }
                }

                // 2. 释放堆内存（包括 TokenInner 和内联的字符串数据）
                let layout = TokenInner::layout_for_string(inner.string_len);
                dealloc(self.ptr.cast().as_ptr(), layout);
            }
        }
    }
}

// ===== Trait 实现 =====

impl PartialEq for Token {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool { self.ptr == other.ptr }
}

impl Eq for Token {}

impl core::hash::Hash for Token {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) { self.key().hash(state); }
}

impl core::fmt::Display for Token {
    #[inline(always)]
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result { f.write_str(self.as_str()) }
}

// ===== Serde 实现 =====

mod serde_impls {
    use super::*;
    use ::serde::{Deserialize, Deserializer, Serialize, Serializer};

    impl Serialize for Token {
        #[inline]
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer {
            self.as_str().serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for Token {
        #[inline]
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de> {
            let s = String::deserialize(deserializer)?;
            let raw_token = s.parse().map_err(::serde::de::Error::custom)?;
            Ok(Token::new(raw_token, Some(s)))
        }
    }
}
