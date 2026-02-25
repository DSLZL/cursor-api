use crate::{app::route::InfallibleSerialize, core::error::ErrorTriple};
use alloc::{borrow::Cow, string::String};
use core::fmt;
use http::StatusCode;
use prost::Message;

/// Prost message that also implements `Default`.
pub(crate) trait ProtobufMessage: Message + Default {}

macro_rules! impl_protobuf_message {
    ($($t:ty),*$(,)?) => {
        $(impl ProtobufMessage for $t {})*
    };
}

impl_protobuf_message!(
    crate::core::aiserver::v1::CppConfigRequest,
    crate::core::aiserver::v1::CppConfigResponse,
    crate::core::aiserver::v1::StreamCppRequest,
    crate::core::aiserver::v1::StreamCppResponse,
    crate::core::aiserver::v1::AvailableCppModelsResponse,
    crate::core::aiserver::v1::FsSyncFileRequest,
    crate::core::aiserver::v1::FsUploadFileRequest,
    crate::core::aiserver::v1::FsSyncFileResponse,
    crate::core::aiserver::v1::FsUploadFileResponse,
);

unsafe impl InfallibleSerialize for crate::core::aiserver::v1::CppConfigResponse {}
unsafe impl InfallibleSerialize for crate::core::aiserver::v1::AvailableCppModelsResponse {}
unsafe impl InfallibleSerialize for crate::core::aiserver::v1::FsSyncFileResponse {}
unsafe impl InfallibleSerialize for crate::core::aiserver::v1::FsUploadFileResponse {}

#[derive(Debug)]
pub(crate) enum DecoderError {
    /// Content-Type header is missing.
    MissingContentType,
    /// Content-Type header is not valid UTF-8.
    InvalidContentType,
    /// Content-Type is present but not supported.
    UnsupportedContentType { actual: String },

    /// Content-Encoding header is not valid UTF-8.
    InvalidContentEncoding,
    /// Content-Encoding is present but not supported.
    UnsupportedContentEncoding { actual: String },

    /// Gzip decompression failed.
    GzipDecompressionFailed,

    /// JSON payload cannot be parsed into the expected error type.
    CursorErrorJsonDecodeFailed { source: serde_json::Error },

    /// Protobuf payload cannot be decoded.
    ProtobufDecodeFailed { source: prost::DecodeError },
}

impl fmt::Display for DecoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingContentType => f.write_str("upstream response is missing Content-Type"),
            Self::InvalidContentType => f.write_str("upstream Content-Type is not valid UTF-8"),
            Self::UnsupportedContentType { actual } => {
                write!(f, "unsupported upstream Content-Type `{actual}`")
            }

            Self::InvalidContentEncoding => {
                f.write_str("upstream Content-Encoding is not valid UTF-8")
            }
            Self::UnsupportedContentEncoding { actual } => {
                write!(f, "unsupported upstream Content-Encoding `{actual}`")
            }

            Self::GzipDecompressionFailed => f.write_str("gzip decompression failed"),

            Self::CursorErrorJsonDecodeFailed { source, .. } => {
                write!(f, "failed to decode upstream JSON error payload: {source}")
            }

            Self::ProtobufDecodeFailed { source } => {
                write!(f, "failed to decode upstream protobuf payload: {source}")
            }
        }
    }
}

impl core::error::Error for DecoderError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::CursorErrorJsonDecodeFailed { source, .. } => Some(source),
            Self::ProtobufDecodeFailed { source } => Some(source),
            _ => None,
        }
    }
}

impl ErrorTriple for DecoderError {
    #[inline]
    fn triple(&self) -> (StatusCode, &'static str, Cow<'static, str>) {
        let status = StatusCode::BAD_GATEWAY;

        match self {
            Self::MissingContentType => (
                status,
                "upstream_missing_content_type",
                Cow::Borrowed("Upstream response is missing Content-Type header"),
            ),
            Self::InvalidContentType => (
                status,
                "upstream_invalid_content_type",
                Cow::Borrowed("Upstream response has an invalid Content-Type header"),
            ),
            Self::UnsupportedContentType { actual } => (
                status,
                "upstream_unsupported_content_type",
                Cow::Owned(format!("Unsupported upstream Content-Type `{actual}`")),
            ),

            Self::InvalidContentEncoding => (
                status,
                "upstream_invalid_content_encoding",
                Cow::Borrowed("Upstream response has an invalid Content-Encoding header"),
            ),
            Self::UnsupportedContentEncoding { actual } => (
                status,
                "upstream_unsupported_content_encoding",
                Cow::Owned(format!("Unsupported upstream Content-Encoding `{actual}`")),
            ),

            Self::GzipDecompressionFailed => (
                status,
                "upstream_gzip_decompression_failed",
                Cow::Borrowed("Failed to decompress upstream gzip payload"),
            ),

            Self::CursorErrorJsonDecodeFailed { source, .. } => (
                status,
                "upstream_error_json_decode_failed",
                Cow::Owned(format!("Failed to decode upstream JSON error payload: {source}")),
            ),

            Self::ProtobufDecodeFailed { source } => (
                status,
                "upstream_protobuf_decode_failed",
                Cow::Owned(format!("Failed to decode upstream protobuf payload: {source}")),
            ),
        }
    }
}
