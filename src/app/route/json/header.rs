use crate::core::error::ErrorTriple;
use alloc::borrow::Cow;
use http::{HeaderMap, StatusCode, header};

#[derive(Debug)]
pub(super) enum JsonContentTypeError<'a> {
    /// Content-Type header is missing
    Missing,
    /// Content-Type is present but not `application/json`
    Mismatch { actual: &'a str },
}

impl ErrorTriple for JsonContentTypeError<'_> {
    #[inline]
    fn triple(&self) -> (StatusCode, &'static str, Cow<'static, str>) {
        match self {
            Self::Missing => (
                StatusCode::BAD_REQUEST,
                "missing_content_type",
                Cow::Borrowed("Content-Type header is missing"),
            ),
            Self::Mismatch { actual } => (
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "content_type_mismatch",
                Cow::Owned(format!("Expected Content-Type `application/json`, got `{actual}`")),
            ),
        }
    }
}

pub(super) fn json_content_type(headers: &HeaderMap) -> Result<(), JsonContentTypeError<'_>> {
    let content_type = headers.get(header::CONTENT_TYPE).ok_or(JsonContentTypeError::Missing)?;
    if content_type == "application/json" {
        Ok(())
    } else {
        Err(JsonContentTypeError::Mismatch {
            actual: content_type.to_str().unwrap_or("<invalid UTF-8>"),
        })
    }
}
