#![allow(unsafe_op_in_unsafe_fn)]

use core::fmt;
use core::str::FromStr;

use super::hash::{Hash, HashError};
use super::timestamp_header;
use crate::common::utils::hex::HEX_TABLE;

#[derive(Debug)]
pub enum ChecksumError {
    InvalidFormat,
    HashError(HashError),
}

impl fmt::Display for ChecksumError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFormat => f.write_str("Invalid Checksum format"),
            Self::HashError(e) => write!(f, "Hash error: {e}"),
        }
    }
}

impl ::core::error::Error for ChecksumError {}

impl From<HashError> for ChecksumError {
    #[inline]
    fn from(err: HashError) -> Self { Self::HashError(err) }
}

#[derive(
    Clone,
    Copy,
    ::serde::Serialize,
    ::serde::Deserialize,
    ::rkyv::Archive,
    ::rkyv::Serialize,
    ::rkyv::Deserialize,
)]
pub struct Checksum {
    first: Hash,
    second: Hash,
}

impl Default for Checksum {
    #[inline(always)]
    fn default() -> Self { Self::random() }
}

impl Checksum {
    #[inline]
    pub fn new(first: Hash, second: Hash) -> Self { Self { first, second } }

    #[inline]
    pub fn random() -> Self { Self { first: Hash::random(), second: Hash::random() } }

    pub fn repair(s: &str) -> Self {
        let bytes = s.as_bytes();

        match bytes.len() {
            72 => Self::repair_short(unsafe { &*(bytes.as_ptr() as *const [u8; 72]) }),
            129 => Self::repair_normal(unsafe { &*(bytes.as_ptr() as *const [u8; 129]) }),
            137 => Self::repair_full(unsafe { &*(bytes.as_ptr() as *const [u8; 137]) }),
            _ => Self::random(),
        }
    }

    // 处理 72 字节格式：时间戳(8) + 设备哈希(64)
    #[inline]
    fn repair_short(bytes: &[u8; 72]) -> Self {
        // 验证时间戳部分
        if !is_valid_timestamp(unsafe { &*(bytes.as_ptr() as *const [u8; 8]) }) {
            return Self::random();
        }

        // 解码设备哈希
        let first = match decode_hex_hash(unsafe { &*(bytes.as_ptr().add(8) as *const [u8; 64]) }) {
            Some(hash) => hash,
            None => return Self::random(),
        };

        Self { first, second: Hash::random() }
    }

    // 处理 129 字节格式：设备哈希(64) + '/' + MAC哈希(64)
    #[inline]
    fn repair_normal(bytes: &[u8; 129]) -> Self {
        // 验证分隔符
        if bytes[64] != b'/' {
            return Self::default();
        }

        // 解码两个哈希
        let first = match decode_hex_hash(unsafe { &*(bytes.as_ptr() as *const [u8; 64]) }) {
            Some(hash) => hash,
            None => return Self::random(),
        };

        let second = match decode_hex_hash(unsafe { &*(bytes.as_ptr().add(65) as *const [u8; 64]) })
        {
            Some(hash) => hash,
            None => return Self::random(),
        };

        Self { first, second }
    }

    // 处理 137 字节格式：时间戳(8) + 设备哈希(64) + '/' + MAC哈希(64)
    #[inline]
    fn repair_full(bytes: &[u8; 137]) -> Self {
        // 验证时间戳
        if !is_valid_timestamp(unsafe { &*(bytes.as_ptr() as *const [u8; 8]) }) {
            return Self::random();
        }

        // 验证分隔符
        if bytes[72] != b'/' {
            return Self::random();
        }

        // 解码两个哈希
        let first = match decode_hex_hash(unsafe { &*(bytes.as_ptr().add(8) as *const [u8; 64]) }) {
            Some(hash) => hash,
            None => return Self::random(),
        };

        let second = match decode_hex_hash(unsafe { &*(bytes.as_ptr().add(73) as *const [u8; 64]) })
        {
            Some(hash) => hash,
            None => return Self::random(),
        };

        Self { first, second }
    }

    #[inline]
    pub const fn from_bytes(bytes: [u8; 64]) -> Self {
        unsafe {
            let ptr = bytes.as_ptr() as *const Hash;

            Self { first: ptr.read(), second: ptr.add(1).read() }
        }
    }

    #[inline]
    pub const fn into_bytes(self) -> [u8; 64] {
        unsafe { ::core::intrinsics::transmute::<[Hash; 2], [u8; 64]>([self.first, self.second]) }
    }

    #[allow(clippy::wrong_self_convention)]
    #[inline]
    pub fn to_str<'buf>(&self, buf: &'buf mut [u8; 137]) -> &'buf mut str {
        let dst = buf.as_mut_ptr();

        // SAFETY: `buf` is guaranteed to be at least `LEN` bytes
        // SAFETY: The encoded buffer is ASCII encoded
        unsafe {
            ::core::ptr::write(buf.as_mut_ptr().cast(), timestamp_header::read());

            self.first.to_str(&mut *(dst.add(8) as *mut [u8; 64]));
            *dst.add(72) = b'/';
            self.second.to_str(&mut *(dst.add(73) as *mut [u8; 64]));

            ::core::str::from_utf8_unchecked_mut(buf)
        }
    }
}

// 验证时间戳格式（允许字母数字、'-'、'_'）
#[inline]
const fn is_valid_timestamp(bytes: &[u8; 8]) -> bool {
    let mut i = 0;
    while i < 8 {
        let b = bytes[i];
        if !(b.is_ascii_alphanumeric() || b == b'-' || b == b'_') {
            return false;
        }
        i += 1;
    }
    true
}

// 解码 64 字符的十六进制字符串为 Hash
#[inline]
const fn decode_hex_hash(hex_bytes: &[u8; 64]) -> Option<Hash> {
    let mut result = [0u8; 32];
    let mut i = 0;

    while i < 32 {
        let pos = i * 2;
        let hi = hex_bytes[pos];
        let lo = hex_bytes[pos + 1];

        let high = HEX_TABLE[hi as usize];
        let low = HEX_TABLE[lo as usize];

        // 检查是否为有效的十六进制字符
        if high == 0xFF || low == 0xFF {
            return None;
        }

        result[i] = (high << 4) | low;
        i += 1;
    }

    Some(Hash::from_bytes(result))
}

impl fmt::Display for Checksum {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.to_str(&mut [0; 137]))
    }
}

impl FromStr for Checksum {
    type Err = ChecksumError;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.as_bytes();
        unsafe {
            let bytes = &*(match s.len() {
                129 if *s.get_unchecked(64) == b'/' => s,
                137 if *s.get_unchecked(72) == b'/' => s.get_unchecked(8..),
                _ => return Err(ChecksumError::InvalidFormat),
            }
            .as_ptr() as *const [u8; 129]);

            let first = Hash::from_str(::core::str::from_utf8_unchecked(&bytes[..64]))?;
            let second = Hash::from_str(::core::str::from_utf8_unchecked(&bytes[65..]))?;

            Ok(Self::new(first, second))
        }
    }
}

// impl ::serde::Serialize for Checksum {
//     #[inline]
//     fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//     where
//         S: ::serde::Serializer,
//     {
//         serializer.serialize_str(&self.to_string())
//     }
// }

// impl<'de> ::serde::Deserialize<'de> for Checksum {
//     #[inline]
//     fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
//     where
//         D: ::serde::Deserializer<'de>,
//     {
//         let s = <String as ::serde::Deserialize>::deserialize(deserializer)?;
//         Self::from_str(&s).map_err(::serde::de::Error::custom)
//     }
// }

// impl Distribution<Checksum> for StandardUniform {
//     fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Checksum {
//         let first = StandardUniform.sample(rng);
//         let second = StandardUniform.sample(rng);
//         Checksum::new(first, second)
//     }
// }
