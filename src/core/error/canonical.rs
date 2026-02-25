use crate::{
    app::{constant::UNKNOWN, model::ErrorInfo, route::InfallibleJson},
    common::model::{ApiStatus, GenericError},
    core::{
        aiserver::v1::{CustomErrorDetails, ErrorDetails},
        model::{anthropic, openai},
    },
};
use alloc::borrow::Cow;
use core::num::NonZeroU16;
use interned::Str;

pub struct CanonicalError {
    pub code: Option<String>,
    pub details: Option<CustomErrorDetails>,
    pub status_code: NonZeroU16,
    pub r#type: &'static str,
}

impl From<ErrorDetails> for CanonicalError {
    #[inline]
    fn from(error: ErrorDetails) -> Self {
        Self {
            code: None,
            details: error.details,
            status_code: ErrorDetails::status_code(error.error.get()),
            r#type: ErrorDetails::r#type(error.error.get()),
        }
    }
}

impl CanonicalError {
    #[inline]
    pub const fn unknown() -> Self {
        Self {
            code: None,
            details: None,
            status_code: unsafe { NonZeroU16::new_unchecked(500) },
            r#type: UNKNOWN,
        }
    }

    #[inline]
    pub fn with_code(mut self, code: String) -> Self {
        self.code = Some(code);
        self
    }
}

impl ::core::iter::Sum for CanonicalError {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut code: Option<String> = None;
        let mut details: Option<CustomErrorDetails> = None;
        let mut status_code = unsafe { NonZeroU16::new_unchecked(100) };
        let mut r#type = UNKNOWN;

        for e in iter {
            if let Some(acode) = e.code {
                if let Some(code) = code.as_mut() {
                    code.push_str(&acode);
                } else {
                    code = Some(acode);
                }
            }
            if let Some(adetails) = e.details {
                if let Some(details) = details.as_mut() {
                    details.add(adetails);
                } else {
                    details = Some(adetails);
                }
            }
            if status_code < e.status_code {
                status_code = e.status_code;
                r#type = e.r#type;
            }
        }

        Self { code, details, status_code, r#type }
    }
}

impl CanonicalError {
    #[inline]
    pub fn title(&self) -> Option<String> {
        match &self.details {
            Some(details) => Some(details.title.clone()),
            None => self.code.as_ref().map(|s| s.replace("_", " ")),
        }
    }

    #[inline]
    pub fn detail(&self) -> Option<&str> {
        self.details.as_ref().map(|details| details.detail.as_str())
    }

    #[inline]
    pub fn status_code(&self) -> http::StatusCode {
        unsafe { ::core::intrinsics::transmute(self.status_code) }
    }

    #[inline]
    pub fn into_generic(self) -> GenericError {
        let code = Some(unsafe { core::intrinsics::transmute(self.status_code) });
        let anthropic::AnthropicErrorInner { r#type, message } = self.into_anthropic();

        GenericError {
            status: ApiStatus::Error,
            code,
            error: Some(Cow::Borrowed(r#type)),
            message: Some(message),
        }
    }

    #[inline]
    pub fn into_openai(self) -> openai::OpenAiErrorInner {
        let message = if let Some(details) = self.details {
            Cow::Owned(__unwrap!(serde_json::to_string(&details)))
        } else {
            Cow::Borrowed(UNKNOWN)
        };

        openai::OpenAiErrorInner { code: self.code.map(Cow::Owned), message }
    }

    #[inline]
    pub fn into_anthropic(self) -> anthropic::AnthropicErrorInner {
        let code = match self.code {
            Some(code) => Cow::Owned(code),
            None => Cow::Borrowed(UNKNOWN),
        };

        let message = if let Some(details) = self.details {
            #[derive(::serde::Serialize)]
            struct Message {
                code: Cow<'static, str>,
                details: CustomErrorDetails,
            }

            Cow::Owned(__unwrap!(serde_json::to_string(&Message { code, details })))
        } else {
            code
        };

        anthropic::AnthropicErrorInner { r#type: self.r#type, message }
    }

    #[inline]
    pub fn to_error_info(&self) -> ErrorInfo {
        ErrorInfo::new(
            self.title().map(Str::new).unwrap_or(Str::from_static(UNKNOWN)),
            self.detail().map(Str::new),
        )
    }
}

impl super::ErrorExt for CanonicalError {
    fn into_generic_tuple(self) -> (http::StatusCode, InfallibleJson<GenericError>) {
        (self.status_code(), InfallibleJson(self.into_generic()))
    }
    fn into_openai_tuple(self) -> (http::StatusCode, InfallibleJson<openai::OpenAiError>) {
        (self.status_code(), InfallibleJson(self.into_openai().wrapped()))
    }
    fn into_anthropic_tuple(self) -> (http::StatusCode, InfallibleJson<anthropic::AnthropicError>) {
        (self.status_code(), InfallibleJson(self.into_anthropic().wrapped()))
    }
}
