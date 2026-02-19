use ::core::{fmt, str::FromStr};
use ::rand::{
    Rng as _,
    distr::{Distribution, StandardUniform},
};
use ::sha2::Digest as _;

use crate::common::utils::hex::HEX_CHARS;

static mut SAFE_HASH: bool = false;

pub(super) fn init() {
    unsafe { SAFE_HASH = crate::common::utils::parse_from_env("SAFE_HASH", true) }
}

#[derive(Debug)]
pub enum HashError {
    InvalidLength,
    InvalidUtf8,
    InvalidHex,
}

impl fmt::Display for HashError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLength => "Invalid Hash length",
            Self::InvalidUtf8 => "Invalid UTF-8 sequence",
            Self::InvalidHex => "Invalid hex value",
        })
    }
}

impl ::core::error::Error for HashError {}

#[derive(Clone, Copy, PartialEq, ::rkyv::Archive, ::rkyv::Serialize, ::rkyv::Deserialize)]
#[repr(transparent)]
pub struct Hash(pub(super) [u8; 32]);

impl Hash {
    const MIN: [u8; 32] = [0; 32];

    #[inline]
    pub fn random() -> Self {
        let mut bytes = [0u8; 32];
        rand::rng().fill_bytes(&mut bytes);
        if unsafe { SAFE_HASH } {
            bytes = sha2::Sha256::new().chain_update(bytes).finalize().into();
        }
        Self(bytes)
    }

    #[inline]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self { Self(bytes) }

    #[inline]
    pub const fn into_bytes(self) -> [u8; 32] { self.0 }

    #[inline]
    pub const fn nil() -> Self { Self(Self::MIN) }

    #[inline]
    pub fn is_nil(&self) -> bool { self.0 == Self::MIN }

    #[allow(clippy::wrong_self_convention)]
    #[inline]
    pub fn to_str<'buf>(&self, buf: &'buf mut [u8; 64]) -> &'buf mut str {
        for (i, &byte) in self.0.iter().enumerate() {
            buf[i * 2] = HEX_CHARS[(byte >> 4) as usize];
            buf[i * 2 + 1] = HEX_CHARS[(byte & 0x0f) as usize];
        }

        // SAFETY: 输出都是有效的 ASCII 字符
        unsafe { ::core::str::from_utf8_unchecked_mut(buf) }
    }
}

impl const Default for Hash {
    #[inline(always)]
    fn default() -> Self { Self::nil() }
}

impl fmt::Display for Hash {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [0u8; 64];
        f.write_str(self.to_str(&mut buf))
    }
}

impl FromStr for Hash {
    type Err = HashError;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let hex_array: &[u8; 64] = s
            .as_bytes()
            .try_into()
            .map_err(|_| HashError::InvalidLength)?;

        let hex_pairs = unsafe { hex_array.as_chunks_unchecked::<2>() };
        let mut result = [0u8; 32];

        for (dst, &[hi, lo]) in result.iter_mut().zip(hex_pairs) {
            *dst = crate::common::utils::hex::hex_to_byte(hi, lo).ok_or(HashError::InvalidHex)?;
        }

        Ok(Self(result))
    }
}

impl ::serde::Serialize for Hash {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: ::serde::Serializer,
    {
        serializer.serialize_str(self.to_str(&mut [0u8; 64]))
    }
}

impl<'de> ::serde::Deserialize<'de> for Hash {
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: ::serde::Deserializer<'de>,
    {
        let s = <&str as ::serde::Deserialize>::deserialize(deserializer)?;
        Self::from_str(s).map_err(::serde::de::Error::custom)
    }
}

impl Distribution<Hash> for StandardUniform {
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Hash {
        let mut bytes = [0u8; 32];
        rng.fill_bytes(&mut bytes);
        Hash(bytes)
    }
}

const _: [u8; 32] = [0; ::core::mem::size_of::<Hash>()];
const _: () = assert!(::core::mem::align_of::<Hash>() == 1);
