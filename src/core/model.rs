pub mod anthropic;
pub mod openai;
mod resolver;

// use crate::app::constant::TOOLU01_PREFIX;
use super::constant::Models;
use crate::{app::route::InfallibleSerialize, common::model::raw_json::RawJson};
pub(crate) use resolver::{ExtModel, init_resolver};
use serde::{Serialize, ser::SerializeStruct as _};

pub(super) type IndexMap<K, V> = indexmap::IndexMap<K, V, ahash::RandomState>;

#[derive(
    ::serde::Serialize,
    ::serde::Deserialize,
    ::rkyv::Archive,
    ::rkyv::Serialize,
    ::rkyv::Deserialize,
    Debug,
    Clone,
    Copy,
    PartialEq,
)]
#[repr(u8)]
pub enum Role {
    #[serde(rename = "system", alias = "developer")]
    System = 0u8,
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
}

// 模型定义
#[derive(Debug, Clone, Copy)]
pub struct Model {
    pub server_id: &'static str,
    pub client_id: &'static str,
    pub id: &'static str,
    pub owned_by: &'static str,
    pub is_thinking: bool,
    pub is_image: bool,
    pub is_max: bool,
    pub is_non_max: bool,
}

impl Model {
    #[inline]
    pub(super) fn id(&self) -> &'static str {
        use crate::app::model::ModelIdSource;
        match *super::constant::MODEL_ID_SOURCE {
            ModelIdSource::Id => self.id,
            ModelIdSource::ClientId => self.client_id,
            ModelIdSource::ServerId => self.server_id,
        }
    }
}

impl Serialize for Model {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        const MODEL_OBJECT: &str = "model";
        const CREATED: &i64 = &1706659200;
        const CREATED_AT: &str = "2024-01-31T00:00:00Z";

        let mut state = serializer.serialize_struct(MODEL_OBJECT, 11)?;

        state.serialize_field("id", self.id())?;
        state.serialize_field("display_name", self.client_id)?;
        state.serialize_field("created", CREATED)?;
        state.serialize_field("created_at", CREATED_AT)?;
        state.serialize_field("object", MODEL_OBJECT)?;
        state.serialize_field("type", MODEL_OBJECT)?;
        state.serialize_field("owned_by", self.owned_by)?;
        state.serialize_field("supports_thinking", &self.is_thinking)?;
        state.serialize_field("supports_images", &self.is_image)?;
        state.serialize_field("supports_max_mode", &self.is_max)?;
        state.serialize_field("supports_non_max_mode", &self.is_non_max)?;

        state.end()
    }
}

impl PartialEq for Model {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.server_id == other.server_id
            && self.client_id == other.client_id
    }
}

#[repr(transparent)]
pub struct ModelsResponse(pub Vec<Model>);

impl core::ops::Deref for ModelsResponse {
    type Target = Vec<Model>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl core::ops::DerefMut for ModelsResponse {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl Serialize for ModelsResponse {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        let mut state = serializer.serialize_struct("ModelsResponse", 2)?;

        state.serialize_field("object", "list")?;
        state.serialize_field("data", &self.0)?;

        state.end()
    }
}

#[repr(transparent)]
pub struct RawModelsResponse(pub(super) RawJson);

impl Serialize for RawModelsResponse {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        let mut state = serializer.serialize_struct("RawModelsResponse", 3)?;

        state.serialize_field("raw", &self.0)?;
        state.serialize_field("dur", &Models::last_update_elapsed())?;
        state.serialize_field("now", &crate::app::model::DateTime::now())?;

        state.end()
    }
}

unsafe impl InfallibleSerialize for RawModelsResponse {}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct MessageId([u8; 16]);

impl MessageId {
    pub const fn new(v: &[u8; 16]) -> Self { Self(*v) }

    #[allow(clippy::wrong_self_convention)]
    #[inline(always)]
    pub fn to_str<'buf>(&self, buf: &'buf mut [u8; 22]) -> &'buf mut str {
        crate::common::utils::base62::encode_fixed(u128::from_ne_bytes(self.0), buf);
        unsafe { ::core::str::from_utf8_unchecked_mut(buf) }
    }
}

impl ::core::fmt::Display for MessageId {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        f.write_str(self.to_str(&mut [0; 22]))
    }
}

// #[derive(Clone, Copy)]
// #[repr(transparent)]
// pub struct ToolUseId(u128);

// impl ToolUseId {
//     /// 从字符串解析 ToolUseId
//     /// 支持两种格式：
//     /// - "tool_xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx" (40字符)
//     /// - "toolu_01xxxxxxxxxxxxxxxxxxxx" (30字符)
//     pub fn parse_str(s: &str) -> Option<Self> {
//         use crate::common::utils::hex::{HEX_TABLE, SHL4_TABLE};

//         let input = s.as_bytes();
//         match (input.len(), input) {
//             // UUID格式：tool_xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx
//             (40, [b't', b'o', b'o', b'l', b'_', s @ ..]) => {
//                 // 验证连字符位置
//                 if !matches!([s[8], s[13], s[18], s[23]], [b'-', b'-', b'-', b'-']) {
//                     return None;
//                 }

//                 let mut buf = [0u8; 16];

//                 // 处理前7个完整的字节对（14字节）
//                 const POSITIONS: [u8; 7] = [0, 4, 9, 14, 19, 24, 28];
//                 for (j, &pos) in POSITIONS.iter().enumerate() {
//                     let i = pos as usize;

//                     let h1 = HEX_TABLE[s[i] as usize];
//                     let h2 = HEX_TABLE[s[i + 1] as usize];
//                     let h3 = HEX_TABLE[s[i + 2] as usize];
//                     let h4 = HEX_TABLE[s[i + 3] as usize];

//                     if h1 | h2 | h3 | h4 == 0xff {
//                         return None;
//                     }

//                     buf[j * 2] = SHL4_TABLE[h1 as usize] | h2;
//                     buf[j * 2 + 1] = SHL4_TABLE[h3 as usize] | h4;
//                 }

//                 // 处理最后3个十六进制字符（1.5字节）
//                 let h1 = HEX_TABLE[s[32] as usize];
//                 let h2 = HEX_TABLE[s[33] as usize];
//                 let h3 = HEX_TABLE[s[34] as usize];

//                 if h1 | h2 | h3 == 0xff {
//                     return None;
//                 }

//                 buf[14] = SHL4_TABLE[h1 as usize] | h2;
//                 buf[15] = SHL4_TABLE[h3 as usize] | 0x01; // 低4位设为1

//                 Some(Self(u128::from_ne_bytes(buf)))
//             }

//             // Base62格式：toolu_01xxxxxxxxxxxxxxxxxxxx
//             (30, [b't', b'o', b'o', b'l', b'u', b'_', b'0', b'1', s @ ..]) => {
//                 crate::common::utils::base62::decode_fixed(unsafe { &*s.as_ptr().cast() })
//                     .ok()
//                     .map(Self)
//             }

//             _ => None,
//         }
//     }

//     /// 转换为Base62格式字符串
//     #[allow(clippy::wrong_self_convention)]
//     #[inline(always)]
//     pub fn to_str<'buf>(&self, buf: &'buf mut [u8; 30]) -> &'buf mut str {
//         unsafe {
//             // 复制前缀 "toolu_01"
//             ::core::ptr::copy_nonoverlapping(TOOLU01_PREFIX.as_ptr(), buf.as_mut_ptr(), 8);

//             // 编码后续的Base62部分
//             crate::common::utils::base62::encode_fixed(
//                 self.0,
//                 &mut *buf.as_mut_ptr().add(8).cast(),
//             );

//             ::core::str::from_utf8_unchecked_mut(buf)
//         }
//     }

//     /// 转换为UUID格式的ByteStr
//     pub fn to_byte_str(self) -> prost::ByteStr {
//         let mut v = Vec::with_capacity(40);
//         v.extend_from_slice(b"tool_");
//         v.extend(Self::format_hyphenated(self.0.to_ne_bytes()));

//         unsafe { prost::ByteStr::from_utf8_unchecked(bytes::Bytes::from(v)) }
//     }

//     /// 格式化为UUID样式的字符串（不含前缀）
//     #[inline]
//     const fn format_hyphenated(src: [u8; 16]) -> [u8; 35] {
//         const HEX_LUT: &[u8; 16] = b"0123456789abcdef";

//         let mut dst = [0u8; 35];
//         let groups = [(0, 8), (9, 13), (14, 18), (19, 23)];

//         let mut src_idx = 0;

//         // 处理前4组，每组后面都有连字符
//         let mut group_idx = 0;
//         while group_idx < 4 {
//             let (start, end) = groups[group_idx];
//             let mut dst_idx = start;

//             while dst_idx < end {
//                 let byte = src[src_idx];
//                 src_idx += 1;

//                 dst[dst_idx] = HEX_LUT[(byte >> 4) as usize];
//                 dst[dst_idx + 1] = HEX_LUT[(byte & 0x0f) as usize];
//                 dst_idx += 2;
//             }

//             dst[end] = b'-';
//             group_idx += 1;
//         }

//         // 处理第5组的前6个字符（3个完整字节）
//         let mut dst_idx = 24;
//         while src_idx < 15 {
//             let byte = src[src_idx];
//             src_idx += 1;

//             dst[dst_idx] = HEX_LUT[(byte >> 4) as usize];
//             dst[dst_idx + 1] = HEX_LUT[(byte & 0x0f) as usize];
//             dst_idx += 2;
//         }

//         // 处理最后一个字节的高4位
//         dst[34] = HEX_LUT[(src[15] >> 4) as usize];

//         dst
//     }
// }

// impl ::core::fmt::Display for ToolUseId {
//     #[inline]
//     fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
//         f.write_str(self.to_str(&mut [0; 30]))
//     }
// }
