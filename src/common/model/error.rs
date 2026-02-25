use super::GenericError;
use crate::{
    app::route::InfallibleJson,
    core::{
        error::ErrorExt,
        model::{anthropic, openai},
    },
};
use alloc::borrow::Cow;
use http::StatusCode;

pub enum ChatError {
    ModelNotSupported(String),
    EmptyMessages(StatusCode),
    RequestFailed(StatusCode, Cow<'static, str>),
}

impl ChatError {
    #[inline]
    pub fn error_type(&self) -> &'static str {
        match self {
            Self::ModelNotSupported(_) => "model_not_supported",
            Self::EmptyMessages(_) => "empty_messages",
            Self::RequestFailed(_, _) => "request_failed",
        }
    }
}

impl core::fmt::Display for ChatError {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ModelNotSupported(model) => write!(f, "Model '{model}' is not supported"),
            Self::EmptyMessages(_) => write!(f, "Message array cannot be empty"),
            Self::RequestFailed(_, err) => write!(f, "Request failed: {err}"),
        }
    }
}

impl ChatError {
    #[inline]
    fn status_code(&self) -> StatusCode {
        match *self {
            Self::ModelNotSupported(_) => StatusCode::BAD_REQUEST,
            Self::EmptyMessages(sc) => sc,
            Self::RequestFailed(sc, _) => sc,
        }
    }

    #[inline]
    fn to_generic(&self) -> GenericError {
        GenericError {
            status: super::ApiStatus::Error,
            code: None,
            error: Some(Cow::Borrowed(self.error_type())),
            message: Some(Cow::Owned(self.to_string())),
        }
    }

    #[inline]
    fn to_openai(&self) -> openai::OpenAiError {
        openai::OpenAiErrorInner {
            code: Some(Cow::Borrowed(self.error_type())),
            message: Cow::Owned(self.to_string()),
        }
        .wrapped()
    }

    #[inline]
    fn to_anthropic(&self) -> anthropic::AnthropicError {
        anthropic::AnthropicErrorInner {
            r#type: self.error_type(),
            message: Cow::Owned(self.to_string()),
        }
        .wrapped()
    }
}

impl ErrorExt for ChatError {
    #[inline]
    fn into_generic_tuple(self) -> (http::StatusCode, InfallibleJson<GenericError>) {
        (self.status_code(), InfallibleJson(self.to_generic()))
    }
    #[inline]
    fn into_openai_tuple(self) -> (http::StatusCode, InfallibleJson<openai::OpenAiError>) {
        (self.status_code(), InfallibleJson(self.to_openai()))
    }
    #[inline]
    fn into_anthropic_tuple(
        self,
    ) -> (http::StatusCode, InfallibleJson<anthropic::AnthropicError>) {
        (self.status_code(), InfallibleJson(self.to_anthropic()))
    }
}
