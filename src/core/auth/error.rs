use crate::{
    app::route::InfallibleJson,
    common::model::{ApiStatus, GenericError},
    core::{
        error::ErrorExt,
        model::{anthropic, openai},
    },
};
use alloc::borrow::Cow;
use axum::response::{IntoResponse, Response};
use http::StatusCode;

/// Authentication and authorization errors
#[derive(Clone, Copy)]
pub enum AuthError {
    /// Authentication failed (invalid token, missing token, etc.)
    Unauthorized,

    /// No available tokens in the queue
    NoAvailableTokens,

    /// Token alias not found (admin tokens only)
    AliasNotFound,
}

impl AuthError {
    /// Returns the HTTP status code for this error
    #[inline]
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::NoAvailableTokens => StatusCode::SERVICE_UNAVAILABLE,
            Self::AliasNotFound => StatusCode::NOT_FOUND,
        }
    }

    /// Returns the error type identifier
    #[inline]
    pub fn error_type(&self) -> &'static str {
        match self {
            Self::Unauthorized => "unauthorized",
            Self::NoAvailableTokens => "no_available_tokens",
            Self::AliasNotFound => "alias_not_found",
        }
    }

    /// Returns the error message
    #[inline]
    fn message(&self) -> &'static str {
        match self {
            Self::Unauthorized => "Invalid authorization token",
            Self::NoAvailableTokens => "No available tokens in queue",
            Self::AliasNotFound => "Token alias not found",
        }
    }
}

impl AuthError {
    /// Converts to Generic error format
    #[inline]
    pub fn into_generic(self) -> GenericError {
        GenericError {
            status: ApiStatus::Error,
            code: Some(self.status_code()),
            error: Some(Cow::Borrowed(self.error_type())),
            message: Some(Cow::Borrowed(self.message())),
        }
    }

    /// Converts to OpenAI error format
    #[inline]
    pub fn into_openai(self) -> openai::OpenAiError {
        openai::OpenAiErrorInner {
            code: Some(Cow::Borrowed(self.error_type())),
            message: Cow::Borrowed(self.message()),
        }
        .wrapped()
    }

    /// Converts to Anthropic error format
    #[inline]
    pub fn into_anthropic(self) -> anthropic::AnthropicError {
        anthropic::AnthropicErrorInner {
            r#type: self.error_type(),
            message: Cow::Borrowed(self.message()),
        }
        .wrapped()
    }
}

impl ErrorExt for AuthError {
    /// Converts to Generic error format
    #[inline]
    fn into_generic_tuple(self) -> (StatusCode, InfallibleJson<GenericError>) {
        (self.status_code(), InfallibleJson(self.into_generic()))
    }

    /// Converts to OpenAI error format
    #[inline]
    fn into_openai_tuple(self) -> (StatusCode, InfallibleJson<openai::OpenAiError>) {
        (self.status_code(), InfallibleJson(self.into_openai()))
    }

    /// Converts to Anthropic error format
    #[inline]
    fn into_anthropic_tuple(self) -> (StatusCode, InfallibleJson<anthropic::AnthropicError>) {
        (self.status_code(), InfallibleJson(self.into_anthropic()))
    }
}

impl IntoResponse for AuthError {
    #[inline]
    fn into_response(self) -> Response { self.into_generic_tuple().into_response() }
}
