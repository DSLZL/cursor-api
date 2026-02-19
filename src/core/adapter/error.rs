use crate::{
    app::model::ErrorInfo as LogErrorInfo,
    common::{
        model::{ApiStatus, GenericError},
        utils::proto_encode::ExceedSizeLimit,
    },
    core::model::{
        anthropic::{AnthropicError, AnthropicErrorInner},
        openai::{OpenAiError, OpenAiErrorInner},
    },
};
use alloc::borrow::Cow;
use interned::Str;

crate::define_typed_constants! {
    &'static str => {
        /// 图片功能禁用错误消息
        ERR_VISION_DISABLED = "Vision feature is disabled",
        /// Base64 图片限制错误消息
        ERR_BASE64_ONLY = "Only base64 encoded images are supported",
        /// Base64 解码失败错误消息
        ERR_BASE64_DECODE_FAILED = "Invalid base64 encoded image",
        /// HTTP 请求失败错误消息
        ERR_REQUEST_FAILED = "Cannot access the image URL",
        /// 响应读取失败错误消息
        ERR_RESPONSE_READ_FAILED = "Failed to download image from URL",
        /// 不支持的图片格式错误消息
        ERR_UNSUPPORTED_IMAGE_FORMAT = "Unsupported image format, only PNG, JPEG, WebP and non-animated GIF are supported",
        /// 不支持动态 GIF
        ERR_UNSUPPORTED_ANIMATED_GIF = "Animated GIF is not supported",
        /// 消息超过大小限制错误消息
        ERR_EXCEED_SIZE_LIMIT = ExceedSizeLimit::message(),
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Error {
    /// Vision feature is disabled
    VisionDisabled,

    /// Only base64 encoded images are supported
    Base64Only,

    /// Failed to decode base64 data
    Base64DecodeFailed,

    /// Failed to parse HTTP request URL
    UrlParseFailed,

    /// Failed to send HTTP request (network error, DNS failure, timeout, etc.)
    RequestFailed,

    /// Failed to read response body (connection dropped, incomplete data, etc.)
    ResponseReadFailed,

    /// Image format is not supported (must be PNG, JPEG, WebP, or static GIF)
    UnsupportedImageFormat,

    /// Animated GIFs are not supported
    UnsupportedAnimatedGif,

    /// Message exceeds 4 MiB size limit
    ExceedSizeLimit,
}

impl Error {
    /// Returns (status_code, error_code, error_message) tuple
    #[inline]
    pub const fn to_parts(self) -> (http::StatusCode, &'static str, &'static str) {
        crate::define_typed_constants! {
            &'static str => {
                PERMISSION_DENIED = "permission_denied",
                INVALID_ARGUMENT = "invalid_argument",
                UNAVAILABLE = "unavailable",
                RESOURCE_EXHAUSTED = "resource_exhausted",
            }
        }
        match self {
            Self::VisionDisabled => {
                (http::StatusCode::FORBIDDEN, PERMISSION_DENIED, ERR_VISION_DISABLED)
            }
            Self::Base64Only => (http::StatusCode::BAD_REQUEST, INVALID_ARGUMENT, ERR_BASE64_ONLY),
            Self::Base64DecodeFailed => {
                (http::StatusCode::BAD_REQUEST, INVALID_ARGUMENT, ERR_BASE64_DECODE_FAILED)
            }
            Self::UrlParseFailed => {
                (http::StatusCode::BAD_REQUEST, INVALID_ARGUMENT, ERR_BASE64_DECODE_FAILED)
            }
            Self::RequestFailed => (http::StatusCode::BAD_GATEWAY, UNAVAILABLE, ERR_REQUEST_FAILED),
            Self::ResponseReadFailed => {
                (http::StatusCode::BAD_GATEWAY, UNAVAILABLE, ERR_RESPONSE_READ_FAILED)
            }
            Self::UnsupportedImageFormat => {
                (http::StatusCode::BAD_REQUEST, INVALID_ARGUMENT, ERR_UNSUPPORTED_IMAGE_FORMAT)
            }
            Self::UnsupportedAnimatedGif => {
                (http::StatusCode::BAD_REQUEST, INVALID_ARGUMENT, ERR_UNSUPPORTED_ANIMATED_GIF)
            }
            Self::ExceedSizeLimit => {
                (http::StatusCode::PAYLOAD_TOO_LARGE, RESOURCE_EXHAUSTED, ERR_EXCEED_SIZE_LIMIT)
            }
        }
    }

    /// Converts to LogErrorInfo format
    pub const fn to_log_error(self) -> LogErrorInfo {
        let (_, error, message) = self.to_parts();
        LogErrorInfo::Detailed {
            error: Str::from_static(error),
            details: Str::from_static(message),
        }
    }

    // /// Converts to GenericError format
    // #[inline]
    // pub const fn into_generic(self) -> GenericError {
    //     let (status_code, error, message) = self.to_parts();

    //     GenericError {
    //         status: ApiStatus::Error,
    //         code: Some(status_code),
    //         error: Some(Cow::Borrowed(error)),
    //         message: Some(Cow::Borrowed(message)),
    //     }
    // }

    /// Converts to HTTP response tuple
    #[inline]
    pub const fn into_response_tuple(self) -> (http::StatusCode, axum::Json<GenericError>) {
        let (status_code, error, message) = self.to_parts();
        (
            status_code,
            axum::Json(GenericError {
                status: ApiStatus::Error,
                code: Some(status_code),
                error: Some(Cow::Borrowed(error)),
                message: Some(Cow::Borrowed(message)),
            }),
        )
    }

    /// Converts to OpenAI error format
    #[inline]
    pub const fn into_openai_tuple(self) -> (http::StatusCode, axum::Json<OpenAiError>) {
        let (status_code, code, message) = self.to_parts();
        (
            status_code,
            axum::Json(
                OpenAiErrorInner {
                    code: Some(Cow::Borrowed(code)),
                    message: Cow::Borrowed(message),
                }
                .wrapped(),
            ),
        )
    }

    /// Converts to Anthropic error format
    #[inline]
    pub const fn into_anthropic_tuple(self) -> (http::StatusCode, axum::Json<AnthropicError>) {
        let (status_code, code, message) = self.to_parts();
        (
            status_code,
            axum::Json(
                AnthropicErrorInner { r#type: code, message: Cow::Borrowed(message) }.wrapped(),
            ),
        )
    }
}

impl core::fmt::Display for Error {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.to_parts().2)
    }
}

impl ::core::error::Error for Error {}

impl axum::response::IntoResponse for Error {
    #[inline]
    fn into_response(self) -> axum::response::Response {
        self.into_response_tuple().into_response()
    }
}

impl From<ExceedSizeLimit> for Error {
    #[inline]
    fn from(_: ExceedSizeLimit) -> Self { Self::ExceedSizeLimit }
}
