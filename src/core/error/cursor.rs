//! 这是一个处理来自Cursor API错误的模块，无需使用的字段已注释

use super::CanonicalError;
use crate::core::aiserver;

#[derive(Debug, ::serde::Deserialize)]
pub struct CursorError {
    error: Error,
}

#[derive(Debug, ::serde::Deserialize)]
struct Error {
    code: String,
    // message: String, // always: Error
    #[serde(default)]
    details: Vec<Detail>,
}

#[derive(Debug, ::serde::Deserialize)]
struct Detail {
    // r#type: String, // always: aiserver.v1.ErrorDetails
    // debug: ErrorDebug,
    #[serde(deserialize_with = "Detail::decode_base64_error_details")]
    value: aiserver::v1::ErrorDetails,
}

// #[derive(::serde::Deserialize)]
// #[serde(rename_all = "camelCase")]
// pub struct ErrorDebug {
//     error: String,
//     details: ErrorDetails,
//     is_expected: Option<bool>,
// }

// #[derive(::serde::Deserialize)]
// #[serde(rename_all = "camelCase")]
// pub struct ErrorDetails {
//     title: String,
//     detail: String,
//     is_retryable: Option<bool>,
// }

impl Detail {
    #[inline]
    fn decode_base64_error_details<'de, D>(
        deserializer: D,
    ) -> Result<aiserver::v1::ErrorDetails, D::Error>
    where D: ::serde::Deserializer<'de> {
        use prost::Message as _;
        let s = <String as ::serde::Deserialize>::deserialize(deserializer)?;
        match base64_simd::STANDARD_NO_PAD.decode_to_vec(s) {
            Ok(buf) => aiserver::v1::ErrorDetails::decode(&buf[..]).map_err(|e| {
                __cold_path!();
                serde::de::Error::custom(format_args!(
                    "failed to decode from Base64-decoded bytes: {e}"
                ))
            }),
            Err(e) => {
                __cold_path!();
                Err(serde::de::Error::custom(format_args!("invalid Base64 string: {e}")))
            }
        }
    }
}

impl CursorError {
    #[inline(always)]
    pub(super) fn code(&self) -> &str { &self.error.code }

    pub fn error(&self) -> Option<crate::core::aiserver::v1::error_details::Error> {
        self.error.details.first()?.value.error.get().try_into().ok()
    }

    #[inline]
    pub fn canonical(self) -> CanonicalError {
        let e = match self.error.details.len() {
            1 => unsafe {
                let mut vec = self.error.details;
                vec.set_len(0);
                core::ptr::read(vec.as_ptr())
            }
            .value
            .into(),
            0 => CanonicalError::unknown(),
            n => {
                eprintln!("收到少见错误数: {n}，请尝试联系开发者以获取支持");
                crate::debug!("错误({n}): {:?}", self.error);
                self.error.details.into_iter().map(|detail| detail.value.into()).sum()
            }
        };
        e.with_code(self.error.code)
    }

    #[inline]
    pub fn from_slice(v: &[u8]) -> Result<Self, serde_json::Error> { serde_json::from_slice(v) }
}

impl From<CursorError> for CanonicalError {
    #[inline]
    fn from(error: CursorError) -> Self { error.canonical() }
}
