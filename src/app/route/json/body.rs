use crate::core::error::ErrorTriple;
use alloc::borrow::Cow;
use http::StatusCode;

#[derive(Debug)]
pub(super) enum JsonBodyError {
    /// JSON syntax error (malformed JSON, unexpected EOF)
    Syntax { message: String },
    /// JSON data error (type mismatch, missing field, etc.)
    Data { message: String },
}

impl JsonBodyError {
    fn from_serde(err: serde_path_to_error::Error<serde_json::Error>) -> Self {
        let message = err.to_string();
        match err.inner().classify() {
            serde_json::error::Category::Data => Self::Data { message },
            serde_json::error::Category::Syntax | serde_json::error::Category::Eof => {
                Self::Syntax { message }
            }
            serde_json::error::Category::Io => {
                // SAFETY: we deserialize from &[u8] via from_slice,
                // IO errors are impossible without a Reader
                unsafe { core::hint::unreachable_unchecked() }
            }
        }
    }
}

impl ErrorTriple for JsonBodyError {
    #[inline]
    fn triple(&self) -> (StatusCode, &'static str, Cow<'static, str>) {
        match self {
            Self::Syntax { message } => {
                (StatusCode::BAD_REQUEST, "json_syntax_error", Cow::Owned(message.clone()))
            }
            Self::Data { message } => {
                (StatusCode::UNPROCESSABLE_ENTITY, "json_data_error", Cow::Owned(message.clone()))
            }
        }
    }
}

pub(super) fn from_bytes<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, JsonBodyError> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);

    serde_path_to_error::deserialize(&mut deserializer).map_err(JsonBodyError::from_serde).and_then(
        |value| {
            deserializer
                .end()
                .map(|()| value)
                .map_err(|err| JsonBodyError::Syntax { message: err.to_string() })
        },
    )
}
