mod cache;
mod provider;

use crate::{
    app::constant::HEADER_B64,
    common::{
        model::token::TokenPayload,
        utils::{hex::HEX_CHARS, hex_to_byte, ulid},
    },
};
use base64_simd::{Out, URL_SAFE_NO_PAD};
pub(super) use cache::__init;
pub use cache::{Token, TokenKey};
use core::{fmt, mem::MaybeUninit};
use proto_value::stringify::Stringify;
pub use provider::{Provider, parse_providers};
use std::{io, str::FromStr};

mod randomness {
    /// 字节在格式化字符串中的起始位置（格式：XXXXXXXX-XXXX-XXXX）
    pub(super) static BYTE_OFFSETS: [usize; 8] = [0, 2, 4, 6, 9, 11, 14, 16];
}

#[derive(Debug)]
pub enum RandomnessError {
    InvalidLength,
    InvalidFormat,
}

impl fmt::Display for RandomnessError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength => write!(f, "Invalid Randomness length"),
            Self::InvalidFormat => write!(f, "Invalid format"),
        }
    }
}

impl ::core::error::Error for RandomnessError {}

#[derive(
    Clone, Copy, PartialEq, Eq, Hash, ::rkyv::Archive, ::rkyv::Deserialize, ::rkyv::Serialize,
)]
#[rkyv(derive(PartialEq, Eq, Hash))]
#[repr(transparent)]
pub struct Randomness(u64);

impl Randomness {
    #[inline]
    pub const fn from_u64(value: u64) -> Self { Self(value) }

    #[inline]
    pub const fn as_u64(self) -> u64 { self.0 }

    #[inline]
    pub const fn from_bytes(bytes: [u8; 8]) -> Self { Self(u64::from_ne_bytes(bytes)) }

    #[inline]
    pub const fn to_bytes(self) -> [u8; 8] { self.0.to_ne_bytes() }

    #[allow(clippy::wrong_self_convention)]
    #[inline]
    pub fn to_str<'buf>(&self, buf: &'buf mut [u8; 18]) -> &'buf mut str {
        let bytes: [u8; 8] = self.0.to_ne_bytes();

        for (&byte, pos) in bytes.iter().zip(randomness::BYTE_OFFSETS) {
            buf[pos] = HEX_CHARS[(byte >> 4) as usize];
            buf[pos + 1] = HEX_CHARS[(byte & 0x0F) as usize];
        }

        // 插入分隔符
        buf[8] = b'-';
        buf[13] = b'-';

        // SAFETY: buf 只包含有效的 ASCII 字符
        unsafe { core::str::from_utf8_unchecked_mut(buf) }
    }
}

impl const Default for Randomness {
    #[inline(always)]
    fn default() -> Self { Self(0) }
}

impl core::fmt::Debug for Randomness {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut buf = [0u8; 18];
        let s = self.to_str(&mut buf);
        f.debug_tuple("Randomness").field(&s).finish()
    }
}

impl fmt::Display for Randomness {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.to_str(&mut [0u8; 18]))
    }
}

impl FromStr for Randomness {
    type Err = RandomnessError;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 18 {
            return Err(RandomnessError::InvalidLength);
        }
        let bytes = s.as_bytes();

        if bytes[8] != b'-' || bytes[13] != b'-' {
            return Err(RandomnessError::InvalidFormat);
        }

        let mut result = [0u8; 8];

        for (result_byte, pos) in result.iter_mut().zip(randomness::BYTE_OFFSETS) {
            *result_byte =
                hex_to_byte(bytes[pos], bytes[pos + 1]).ok_or(RandomnessError::InvalidFormat)?;
        }

        Ok(Self(u64::from_ne_bytes(result)))
    }
}

impl ::serde::Serialize for Randomness {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: ::serde::Serializer {
        serializer.serialize_str(self.to_str(&mut [0u8; 18]))
    }
}

impl<'de> ::serde::Deserialize<'de> for Randomness {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: ::serde::Deserializer<'de> {
        struct RandomnessVisitor;

        impl ::serde::de::Visitor<'_> for RandomnessVisitor {
            type Value = Randomness;

            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                formatter.write_str("a string in the format XXXXXXXX-XXXX-XXXX")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where E: ::serde::de::Error {
                value.parse().map_err(E::custom)
            }
        }

        deserializer.deserialize_str(RandomnessVisitor)
    }
}

const _: [u8; 8] = [0; core::mem::size_of::<Randomness>()];
const _: () = assert!(core::mem::align_of::<Randomness>() == 8);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Subject {
    pub provider: Provider,
    pub id: UserId,
}

impl Subject {
    #[inline]
    fn to_helper(self) -> SubjectHelper {
        SubjectHelper { provider: self.provider.to_helper(), id: self.id.to_bytes() }
    }

    #[inline]
    fn from_str(s: &str) -> Result<Self, SubjectError> {
        let (provider, id_str) = s.split_once("|").ok_or(SubjectError::InvalidFormat)?;

        if provider.is_empty() {
            return Err(SubjectError::MissingProvider);
        }

        if id_str.is_empty() {
            return Err(SubjectError::MissingUserId);
        }

        let provider = Provider::from_str(provider)?;
        let id = id_str.parse().map_err(|_| SubjectError::InvalidUlid)?;

        Ok(Self { provider, id })
    }
}

impl fmt::Display for Subject {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.provider.as_str())?;
        f.write_str("|")?;
        f.write_str(self.id.to_str(&mut [0; 31]))
    }
}

#[derive(Debug)]
pub enum SubjectError {
    MissingProvider,
    MissingUserId,
    InvalidFormat,
    InvalidUlid,
    InvalidHex,
    UnsupportedProvider,
}

impl fmt::Display for SubjectError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::MissingProvider => "Missing provider",
            Self::MissingUserId => "Missing user_id",
            Self::InvalidFormat => "Invalid user_id format",
            Self::InvalidUlid => "Invalid ULID",
            Self::InvalidHex => "Invalid HEX",
            Self::UnsupportedProvider => "Unsupported provider",
        })
    }
}

impl ::core::error::Error for SubjectError {}

impl ::serde::Serialize for Subject {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: ::serde::Serializer {
        serializer.collect_str(self)
    }
}

impl<'de> ::serde::Deserialize<'de> for Subject {
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: ::serde::Deserializer<'de> {
        let s = String::deserialize(deserializer)?;
        Self::from_str(&s).map_err(::serde::de::Error::custom)
    }
}

/// 用户标识符，支持两种格式的高效ID系统
///
/// 采用向前兼容设计，通过检查高32位区分格式：
/// - 旧格式：24字符十六进制，高32位为0
/// - 新格式：`user_` + 26字符ULID，充分利用128位空间
///
/// ULID时间戳特性确保新格式高32位非零，实现无歧义格式识别。
#[derive(
    Clone, Copy, PartialEq, Eq, Hash, ::rkyv::Archive, ::rkyv::Serialize, ::rkyv::Deserialize,
)]
#[rkyv(derive(PartialEq, Eq, Hash))]
#[repr(align(8))]
pub struct UserId([u8; 16]);

impl UserId {
    const PREFIX: &'static str = "user_";

    // ==================== 公开API：构造与转换 ====================

    /// 从 u128 构造
    #[inline]
    pub const fn from_u128(value: u128) -> Self { Self(value.to_ne_bytes()) }

    /// 转换为 u128
    #[inline]
    pub const fn as_u128(self) -> u128 { u128::from_ne_bytes(self.0) }

    /// 从字节数组构造
    #[inline]
    pub const fn from_bytes(bytes: [u8; 16]) -> Self { Self(bytes) }

    /// 转换为字节数组
    #[inline]
    pub const fn to_bytes(self) -> [u8; 16] { self.0 }

    #[inline]
    pub const fn as_bytes(&self) -> &[u8; 16] { &self.0 }

    // ==================== 格式检测与字符串转换 ====================

    /// 检查是否为旧格式ID（高32位为0）
    #[inline]
    pub const fn is_legacy(&self) -> bool {
        // Memory layout (little-endian): [低32位][次低32位][次高32位][最高32位]
        //                     index:         [0]      [1]       [2]       [3]
        // Memory layout (big-endian):    [最高32位][次高32位][次低32位][低32位]
        //                     index:         [0]       [1]       [2]      [3]
        let parts = unsafe { self.0.as_chunks_unchecked::<4>() };

        #[cfg(target_endian = "little")]
        const HIGH_INDEX: usize = 3;
        #[cfg(target_endian = "big")]
        const HIGH_INDEX: usize = 0;

        u32::from_ne_bytes(parts[HIGH_INDEX]) == 0
    }

    /// 高性能字符串转换，旧格式24字符，新格式31字符
    #[allow(clippy::wrong_self_convention)]
    #[inline]
    pub fn to_str<'buf>(&self, buf: &'buf mut [u8; 31]) -> &'buf mut str {
        if self.is_legacy() {
            // 旧格式：24字符 hex，从 bytes[4..16] 编码
            for (i, &byte) in self.0[4..].iter().enumerate() {
                buf[i * 2] = HEX_CHARS[(byte >> 4) as usize];
                buf[i * 2 + 1] = HEX_CHARS[(byte & 0x0f) as usize];
            }

            // SAFETY: HEX_CHARS 确保输出是有效 ASCII
            unsafe { core::str::from_utf8_unchecked_mut(&mut buf[..24]) }
        } else {
            // 新格式：user_ + 26字符 ULID
            unsafe {
                core::ptr::copy_nonoverlapping(Self::PREFIX.as_ptr(), buf.as_mut_ptr(), 5);
                ulid::to_str(self.as_u128(), &mut *(buf.as_mut_ptr().add(5) as *mut [u8; 26]));
                core::str::from_utf8_unchecked_mut(buf)
            }
        }
    }
}

impl core::fmt::Debug for UserId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut buf = [0u8; 31];
        let s = self.to_str(&mut buf);
        f.debug_tuple("UserId").field(&s).finish()
    }
}

impl core::fmt::Display for UserId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.to_str(&mut [0; 31]))
    }
}

impl core::str::FromStr for UserId {
    type Err = SubjectError;

    fn from_str(s: &str) -> core::result::Result<Self, Self::Err> {
        match s.len() {
            31 => {
                let id_str = s.strip_prefix(Self::PREFIX).ok_or(SubjectError::InvalidFormat)?;
                let id_array = unsafe { id_str.as_bytes().as_array().unwrap_unchecked() };
                let id = ulid::from_bytes(id_array).map_err(|_| SubjectError::InvalidUlid)?;
                Ok(Self::from_u128(id))
            }
            24 => {
                let hex_array: &[u8; 24] = unsafe { s.as_bytes().as_array().unwrap_unchecked() };
                let hex_pairs = unsafe { hex_array.as_chunks_unchecked::<2>() };
                let mut result = [0u8; 16];

                for (dst, &[hi, lo]) in result[4..].iter_mut().zip(hex_pairs) {
                    *dst = hex_to_byte(hi, lo).ok_or(SubjectError::InvalidHex)?;
                }

                Ok(Self::from_bytes(result))
            }
            _ => Err(SubjectError::MissingUserId),
        }
    }
}

impl ::serde::Serialize for UserId {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> core::result::Result<S::Ok, S::Error>
    where S: ::serde::Serializer {
        serializer.serialize_str(self.to_str(&mut [0; 31]))
    }
}

impl<'de> ::serde::Deserialize<'de> for UserId {
    #[inline]
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where D: ::serde::Deserializer<'de> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(::serde::de::Error::custom)
    }
}

const _: [u8; 16] = [0; core::mem::size_of::<UserId>()];
const _: () = assert!(core::mem::align_of::<UserId>() <= 8);

#[derive(Debug)]
pub enum SessionIdError {
    MissingSessionId,
    InvalidFormat,
    InvalidUlid,
}

impl fmt::Display for SessionIdError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::MissingSessionId => "Missing session_id",
            Self::InvalidFormat => "Invalid session_id format",
            Self::InvalidUlid => "Invalid ULID",
        })
    }
}

impl ::core::error::Error for SessionIdError {}

#[derive(
    Clone, Copy, PartialEq, Eq, Hash, ::rkyv::Archive, ::rkyv::Serialize, ::rkyv::Deserialize,
)]
#[rkyv(derive(PartialEq, Eq, Hash))]
#[repr(align(8))]
pub struct SessionId([u8; 16]);

impl SessionId {
    const PREFIX: &'static str = "session_";

    #[inline]
    pub const fn empty() -> Self { Self([0; 16]) }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        let parts = unsafe { self.0.as_chunks_unchecked::<8>() };

        #[cfg(target_endian = "little")]
        const HIGH_INDEX: usize = 1;
        #[cfg(target_endian = "big")]
        const HIGH_INDEX: usize = 0;

        u64::from_ne_bytes(parts[HIGH_INDEX]) == 0
    }

    /// 从 u128 构造
    #[inline]
    pub const fn from_u128(value: u128) -> Self { Self(value.to_ne_bytes()) }

    /// 转换为 u128
    #[inline]
    pub const fn as_u128(self) -> u128 { u128::from_ne_bytes(self.0) }

    /// 从字节数组构造
    #[inline]
    pub const fn from_bytes(bytes: [u8; 16]) -> Self { Self(bytes) }

    /// 转换为字节数组
    #[inline]
    pub const fn to_bytes(self) -> [u8; 16] { self.0 }

    #[inline]
    pub const fn as_bytes(&self) -> &[u8; 16] { &self.0 }

    /// 高性能字符串转换，34字符
    #[allow(clippy::wrong_self_convention)]
    #[inline]
    pub fn to_str<'buf>(&self, buf: &'buf mut [u8; 34]) -> &'buf mut str {
        unsafe {
            core::ptr::copy_nonoverlapping(Self::PREFIX.as_ptr(), buf.as_mut_ptr(), 8);
            ulid::to_str(self.as_u128(), &mut *(buf.as_mut_ptr().add(8) as *mut [u8; 26]));
            core::str::from_utf8_unchecked_mut(buf)
        }
    }
}

impl core::fmt::Debug for SessionId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut buf = [0u8; 34];
        let s = self.to_str(&mut buf);
        f.debug_tuple("SessionId").field(&s).finish()
    }
}

impl core::fmt::Display for SessionId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.to_str(&mut [0; 34]))
    }
}

impl core::str::FromStr for SessionId {
    type Err = SessionIdError;

    fn from_str(s: &str) -> core::result::Result<Self, Self::Err> {
        match s.len() {
            34 => {
                let id_str = s.strip_prefix(Self::PREFIX).ok_or(SessionIdError::InvalidFormat)?;
                let id_array = unsafe { id_str.as_bytes().as_array().unwrap_unchecked() };
                let id = ulid::from_bytes(id_array).map_err(|_| SessionIdError::InvalidUlid)?;
                Ok(Self::from_u128(id))
            }
            _ => Err(SessionIdError::MissingSessionId),
        }
    }
}

impl ::serde::Serialize for SessionId {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> core::result::Result<S::Ok, S::Error>
    where S: ::serde::Serializer {
        serializer.serialize_str(self.to_str(&mut [0; 34]))
    }
}

impl<'de> ::serde::Deserialize<'de> for SessionId {
    #[inline]
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where D: ::serde::Deserializer<'de> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(::serde::de::Error::custom)
    }
}

const _: [u8; 16] = [0; core::mem::size_of::<SessionId>()];
const _: () = assert!(core::mem::align_of::<SessionId>() <= 8);

#[derive(
    Clone, Copy, PartialEq, Eq, Hash, ::rkyv::Archive, ::rkyv::Deserialize, ::rkyv::Serialize,
)]
pub struct Duration {
    pub start: i64,
    pub end: i64,
}

// impl Duration {
//     #[inline(always)]
//     pub const fn validity(&self) -> u32 {
//         (self.end - self.start) as u32
//     }

//     #[inline]
//     pub fn is_short(&self) -> bool {
//         TOKEN_VALIDITY_RANGE.is_short(self.validity())
//     }

//     #[inline]
//     pub fn is_long(&self) -> bool {
//         TOKEN_VALIDITY_RANGE.is_long(self.validity())
//     }
// }

#[derive(Debug)]
pub enum TokenError {
    InvalidHeader,
    InvalidFormat,
    InvalidBase64,
    InvalidJson(io::Error),
    InvalidSubject(SubjectError),
    InvalidRandomness(RandomnessError),
    InvalidSignatureLength,
}

impl ::core::error::Error for TokenError {}

impl fmt::Display for TokenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHeader => f.write_str("Invalid token header"),
            Self::InvalidFormat => f.write_str("Invalid token format"),
            Self::InvalidBase64 => write!(f, "Invalid base64 data"),
            Self::InvalidJson(e) => write!(f, "Invalid JSON: {e}"),
            Self::InvalidSubject(e) => write!(f, "Invalid subject: {e}"),
            Self::InvalidRandomness(e) => write!(f, "Invalid randomness: {e}"),
            Self::InvalidSignatureLength => f.write_str("Invalid signature length"),
        }
    }
}

#[derive(Clone, Copy)]
pub struct RawToken {
    /// 用户标识符
    pub subject: Subject,
    /// 签名
    pub signature: [u8; 32],
    /// 持续时间
    pub duration: Duration,
    /// 随机字符串
    pub randomness: Randomness,
    /// 会话
    pub is_session: bool,
    /// 会话ID
    pub workos_session_id: SessionId,
}

impl PartialEq for RawToken {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.signature != other.signature {
            return false;
        };
        core::intrinsics::likely(
            self.subject == other.subject
                && self.duration == other.duration
                && self.randomness == other.randomness
                && self.is_session == other.is_session
                && self.workos_session_id == other.workos_session_id,
        )
    }
}

impl Eq for RawToken {}

impl ::core::hash::Hash for RawToken {
    #[inline]
    fn hash<H: ::core::hash::Hasher>(&self, state: &mut H) {
        self.signature.hash(state);
        self.subject.hash(state);
        self.duration.hash(state);
        self.randomness.hash(state);
        self.is_session.hash(state);
    }
}

impl RawToken {
    #[inline(always)]
    fn into_token_payload(self) -> TokenPayload {
        TokenPayload {
            sub: self.subject,
            time: Stringify(self.duration.start),
            exp: self.duration.end,
            randomness: self.randomness,
            is_session: self.is_session,
            workos_session_id: self.workos_session_id,
        }
    }

    #[inline(always)]
    pub(super) fn to_helper(self) -> RawTokenHelper {
        RawTokenHelper {
            subject: self.subject.to_helper(),
            duration: self.duration,
            randomness: self.randomness,
            is_session: self.is_session,
            signature: self.signature,
            workos_session_id: self.workos_session_id,
        }
    }

    #[inline(always)]
    pub const fn key(&self) -> TokenKey {
        TokenKey { user_id: self.subject.id, randomness: self.randomness }
    }

    #[inline(always)]
    pub const fn is_web(&self) -> bool { !self.is_session }

    #[inline(always)]
    pub const fn is_session(&self) -> bool { self.is_session }
}

impl fmt::Debug for RawToken {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawToken")
            .field("p", &self.subject.provider.as_str())
            .field("i", &self.subject.id.as_u128())
            .field("r", &core::ops::Range { start: self.duration.start, end: self.duration.end })
            .field("n", &self.randomness.0)
            .field("w", &self.is_web())
            .field("s", &self.signature)
            .field("wsi", &self.workos_session_id.as_u128())
            .finish()
    }
}

impl fmt::Display for RawToken {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{HEADER_B64}{}.{}",
            URL_SAFE_NO_PAD
                .encode_to_string(__unwrap!(serde_json::to_vec(&self.into_token_payload()))),
            URL_SAFE_NO_PAD
                .encode_as_str(&self.signature, base64_simd::Out::from_slice(&mut [0; 43]))
        )
    }
}

impl FromStr for RawToken {
    type Err = TokenError;

    fn from_str(token: &str) -> Result<Self, Self::Err> {
        // 1. 分割并验证token格式
        let parts = token.strip_prefix(HEADER_B64).ok_or(TokenError::InvalidHeader)?;

        let (payload_b64, signature_b64) =
            parts.split_once('.').ok_or(TokenError::InvalidFormat)?;

        if signature_b64.len() != 43 {
            return Err(TokenError::InvalidSignatureLength);
        }

        // 2. 解码payload和signature
        let payload =
            URL_SAFE_NO_PAD.decode_to_vec(payload_b64).map_err(|_| TokenError::InvalidBase64)?;

        let mut signature = MaybeUninit::<[u8; 32]>::uninit();
        URL_SAFE_NO_PAD
            .decode(signature_b64.as_bytes(), Out::from_uninit_slice(signature.as_bytes_mut()))
            .map_err(|_| TokenError::InvalidBase64)?;

        // 3. 解析payload
        let payload: TokenPayload = serde_json::from_slice(&payload).map_err(|e| {
            let e: io::Error = e.into();
            match e.downcast::<SubjectError>() {
                Ok(e) => TokenError::InvalidSubject(e),
                Err(e) => match e.downcast::<RandomnessError>() {
                    Ok(e) => TokenError::InvalidRandomness(e),
                    Err(e) => TokenError::InvalidJson(e),
                },
            }
        })?;

        // 4. 构造RawToken
        Ok(Self {
            subject: payload.sub,
            duration: Duration { start: payload.time.0, end: payload.exp },
            randomness: payload.randomness,
            is_session: payload.is_session,
            signature: unsafe { signature.assume_init() },
            workos_session_id: payload.workos_session_id,
        })
    }
}

impl<'de> ::serde::Deserialize<'de> for RawToken {
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: ::serde::Deserializer<'de> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(::serde::de::Error::custom)
    }
}

#[derive(::rkyv::Archive, ::rkyv::Deserialize, ::rkyv::Serialize)]
#[repr(u8)]
pub enum ProviderHelper {
    Auth0,
    Github,
    Google,
    // Workos,
    Other(String) = u8::MAX,
}

impl ProviderHelper {
    #[inline]
    fn try_extract(self) -> Result<Provider, SubjectError> {
        match self {
            Self::Auth0 => Provider::from_str(provider::AUTH0),
            Self::Github => Provider::from_str(provider::GITHUB),
            Self::Google => Provider::from_str(provider::GOOGLE_OAUTH2),
            Self::Other(s) => Provider::from_str(&s),
        }
    }
}

#[derive(::rkyv::Archive, ::rkyv::Deserialize, ::rkyv::Serialize)]
pub struct SubjectHelper {
    provider: ProviderHelper,
    id: [u8; 16],
}

impl SubjectHelper {
    #[inline]
    fn try_extract(self) -> Result<Subject, SubjectError> {
        Ok(Subject { provider: self.provider.try_extract()?, id: UserId::from_bytes(self.id) })
    }
}

#[derive(::rkyv::Archive, ::rkyv::Deserialize, ::rkyv::Serialize)]
pub struct RawTokenHelper {
    pub subject: SubjectHelper,
    pub signature: [u8; 32],
    pub duration: Duration,
    pub randomness: Randomness,
    pub is_session: bool,
    pub workos_session_id: SessionId,
}

impl RawTokenHelper {
    #[inline]
    pub(super) fn extract(self) -> RawToken {
        RawToken {
            subject: __unwrap_panic!(self.subject.try_extract()),
            duration: self.duration,
            randomness: self.randomness,
            is_session: self.is_session,
            signature: self.signature,
            workos_session_id: self.workos_session_id,
        }
    }
}
