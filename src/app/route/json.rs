mod body;
mod header;

use crate::{
    app::constant::header::JSON,
    common::model::GenericError,
    core::{
        error::ErrorExt,
        model::{anthropic::AnthropicError, openai::OpenAiError},
    },
};
use axum::{
    extract::{FromRequest, OptionalFromRequest, Request},
    response::{IntoResponse, Response},
};
use body::from_bytes;
use bytes::Bytes;
use header::{JsonContentTypeError, json_content_type};
use http::{StatusCode, header::CONTENT_TYPE};
use serde::{Serialize, de::DeserializeOwned};

/// # Safety
///
/// Implementors must guarantee that `serde::Serialize` for `T`
/// **never fails** (no custom error paths, no map keys that fail, etc.).
/// This allows `serde_json::to_vec` to be called with `unwrap_unchecked`.
pub unsafe trait InfallibleSerialize: Serialize {}

#[derive(Debug, Clone, Copy, Default)]
#[must_use]
pub struct InfallibleJson<T>(pub T);

#[derive(Debug, Clone, Copy, Default)]
#[must_use]
pub struct GenericJson<T>(pub T);

#[derive(Debug, Clone, Copy, Default)]
#[must_use]
pub struct OpenAiJson<T>(pub T);

#[derive(Debug, Clone, Copy, Default)]
#[must_use]
pub struct AnthropicJson<T>(pub T);

impl<T: InfallibleSerialize> IntoResponse for InfallibleJson<T> {
    fn into_response(self) -> Response {
        fn make_response(buf: Vec<u8>) -> Response {
            ([(CONTENT_TYPE, JSON)], Bytes::from(buf)).into_response()
        }

        // SAFETY: T: InfallibleSerialize guarantees serialization cannot fail
        let buf = unsafe { serde_json::to_vec(&self.0).unwrap_unchecked() };
        make_response(buf)
    }
}

macro_rules! impl_json_from_request {
    ($json_type:ident, $error_method:ident, $error_type:ty) => {
        impl<T, S> FromRequest<S> for $json_type<T>
        where
            T: DeserializeOwned,
            S: Send + Sync,
        {
            type Rejection = (StatusCode, InfallibleJson<$error_type>);

            async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
                json_content_type(req.headers()).map_err(ErrorExt::$error_method)?;
                let bytes =
                    Bytes::from_request(req, state).await.map_err(ErrorExt::$error_method)?;
                from_bytes(&bytes).map($json_type).map_err(ErrorExt::$error_method)
            }
        }

        impl<T, S> OptionalFromRequest<S> for $json_type<T>
        where
            T: DeserializeOwned,
            S: Send + Sync,
        {
            type Rejection = (StatusCode, InfallibleJson<$error_type>);

            async fn from_request(
                req: Request,
                state: &S,
            ) -> Result<Option<Self>, Self::Rejection> {
                match json_content_type(req.headers()) {
                    Ok(()) => {
                        let bytes = Bytes::from_request(req, state)
                            .await
                            .map_err(ErrorExt::$error_method)?;
                        Ok(Some(
                            from_bytes(&bytes).map($json_type).map_err(ErrorExt::$error_method)?,
                        ))
                    }
                    Err(JsonContentTypeError::Missing) => Ok(None),
                    Err(e) => Err(ErrorExt::$error_method(e)),
                }
            }
        }
    };
}

impl_json_from_request!(GenericJson, into_generic_tuple, GenericError);
impl_json_from_request!(OpenAiJson, into_openai_tuple, OpenAiError);
impl_json_from_request!(AnthropicJson, into_anthropic_tuple, AnthropicError);

// Mark error types as infallible for serialization.
// SAFETY: GenericError, OpenAiError, AnthropicError only contain
// strings, numbers, options thereof — serde_json serialization cannot fail.
unsafe impl InfallibleSerialize for GenericError {}
unsafe impl InfallibleSerialize for OpenAiError {}
unsafe impl InfallibleSerialize for AnthropicError {}
