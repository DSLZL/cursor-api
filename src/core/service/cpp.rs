use crate::{
    app::{
        constant::header::{
            CHUNKED, CLIENT_KEY, EVENT_STREAM, JSON, KEEP_ALIVE, NO_CACHE_REVALIDATE,
        },
        lazy::{cpp_config_url, cpp_models_url},
        model::CppService,
        route::{GenericJson, InfallibleJson},
    },
    common::{
        client::{AiServiceRequest, build_client_request},
        model::{GenericError, error::ChatError},
        utils::{CollectBytesParts, encode_message, encode_message_framed, new_uuid_v4},
    },
    core::{
        aiserver::v1::{
            AvailableCppModelsResponse, CppConfigRequest, CppConfigResponse, FsSyncFileRequest,
            FsSyncFileResponse, FsUploadFileRequest, FsUploadFileResponse, StreamCppRequest,
        },
        auth::TokenBundle,
        error::ErrorExt as _,
        stream::decoder::{
            cpp::{StreamDecoder, StreamMessage},
            direct,
        },
    },
};
use alloc::borrow::Cow;
use axum::{
    body::Body,
    response::{IntoResponse as _, Response},
};
use bytes::Bytes;
use core::convert::Infallible;
use futures_util::StreamExt as _;
use http::{
    Extensions, HeaderMap, StatusCode,
    header::{
        ACCESS_CONTROL_ALLOW_CREDENTIALS, ACCESS_CONTROL_ALLOW_HEADERS, CACHE_CONTROL, CONNECTION,
        CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE, COOKIE, TRANSFER_ENCODING, VARY,
    },
};

const TO_REMOVE_HEADERS: [http::HeaderName; 7] = [
    CONTENT_TYPE,
    CONTENT_LENGTH,
    CONTENT_ENCODING,
    TRANSFER_ENCODING,
    VARY,
    ACCESS_CONTROL_ALLOW_CREDENTIALS,
    ACCESS_CONTROL_ALLOW_HEADERS,
];

fn json_response_with_upstream_parts(parts: http::response::Parts, body: Vec<u8>) -> Response {
    let mut builder = Response::builder().status(parts.status).version(parts.version);

    for (name, value) in parts.headers.iter() {
        if TO_REMOVE_HEADERS.contains(name) {
            continue;
        }
        builder = builder.header(name, value);
    }

    let mut res = __unwrap!(
        builder
            .header(CONTENT_TYPE, JSON)
            .header(CONTENT_LENGTH, body.len())
            .body(Body::from(body))
    );
    *res.extensions_mut() = parts.extensions;
    res
}

pub async fn handle_cpp_config(
    mut headers: HeaderMap,
    mut extensions: Extensions,
    GenericJson(request): GenericJson<CppConfigRequest>,
) -> Result<InfallibleJson<CppConfigResponse>, (StatusCode, InfallibleJson<GenericError>)> {
    let (ext_token, use_pri) = __unwrap!(extensions.remove::<TokenBundle>());

    let (data, compressed) = match encode_message(&request) {
        Ok(o) => o,
        Err(e) => return Err(e.into_response_tuple()),
    };

    let req = build_client_request(AiServiceRequest {
        ext_token: &ext_token,
        fs_client_key: headers.remove(CLIENT_KEY),
        url: cpp_config_url(use_pri),
        stream: false,
        compressed,
        trace_id: new_uuid_v4(),
        use_pri,
        cookie: headers.remove(COOKIE),
        exact_length: Some(data.len()),
    });

    match CollectBytesParts(req.body(data)).await {
        Ok((parts, bytes)) => match direct::decode::<CppConfigResponse>(&parts.headers, &bytes) {
            Ok(Ok(data)) => Ok(InfallibleJson(data)),
            Ok(Err(cursor_err)) => Err(cursor_err.canonical().into_generic_tuple()),
            Err(e) => Err(e.into_generic_tuple()),
        },
        Err(e) => {
            let e = e.without_url();
            Err(ChatError::RequestFailed(
                if e.is_timeout() {
                    StatusCode::GATEWAY_TIMEOUT
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                },
                Cow::Owned(e.to_string()),
            )
            .into_generic_tuple())
        }
    }
}

pub async fn handle_cpp_models(
    mut headers: HeaderMap,
    mut extensions: Extensions,
) -> Result<InfallibleJson<AvailableCppModelsResponse>, (StatusCode, InfallibleJson<GenericError>)>
{
    let (ext_token, use_pri) = __unwrap!(extensions.remove::<TokenBundle>());

    let req = build_client_request(AiServiceRequest {
        ext_token: &ext_token,
        fs_client_key: headers.remove(CLIENT_KEY),
        url: cpp_models_url(use_pri),
        stream: false,
        compressed: false,
        trace_id: new_uuid_v4(),
        use_pri,
        cookie: headers.remove(COOKIE),
        exact_length: Some(0),
    });

    match CollectBytesParts(req).await {
        Ok((parts, bytes)) => {
            match direct::decode::<AvailableCppModelsResponse>(&parts.headers, &bytes) {
                Ok(Ok(data)) => Ok(InfallibleJson(data)),
                Ok(Err(cursor_err)) => Err(cursor_err.canonical().into_generic_tuple()),
                Err(e) => Err(e.into_generic_tuple()),
            }
        }
        Err(e) => {
            let e = e.without_url();
            Err(ChatError::RequestFailed(
                if e.is_timeout() {
                    StatusCode::GATEWAY_TIMEOUT
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                },
                Cow::Owned(e.to_string()),
            )
            .into_generic_tuple())
        }
    }
}

pub async fn handle_upload_file(
    mut headers: HeaderMap,
    mut extensions: Extensions,
    GenericJson(request): GenericJson<FsUploadFileRequest>,
) -> Result<Response, Response> {
    let (ext_token, use_pri) = __unwrap!(extensions.remove::<TokenBundle>());

    let (data, compressed) = match encode_message(&request) {
        Ok(o) => o,
        Err(e) => return Err(e.into_response()),
    };

    let req = build_client_request(AiServiceRequest {
        ext_token: &ext_token,
        fs_client_key: headers.remove(CLIENT_KEY),
        url: ext_token.gcpp_host().get_url(CppService::FSUploadFile, use_pri),
        stream: false,
        compressed,
        trace_id: new_uuid_v4(),
        use_pri,
        cookie: headers.remove(COOKIE),
        exact_length: Some(data.len()),
    });

    match CollectBytesParts(req.body(data)).await {
        Ok((parts, bytes)) => {
            match direct::decode::<FsUploadFileResponse>(&parts.headers, &bytes) {
                Ok(Ok(data)) => {
                    let body = __unwrap!(serde_json::to_vec(&data));
                    Ok(json_response_with_upstream_parts(parts, body))
                }
                Ok(Err(cursor_err)) => {
                    let body =
                        __unwrap!(serde_json::to_vec(&cursor_err.canonical().into_generic()));
                    Err(json_response_with_upstream_parts(parts, body))
                }
                Err(e) => Err(e.into_generic_tuple().into_response()),
            }
        }
        Err(e) => {
            let e = e.without_url();
            Err(ChatError::RequestFailed(
                if e.is_timeout() {
                    StatusCode::GATEWAY_TIMEOUT
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                },
                Cow::Owned(e.to_string()),
            )
            .into_generic_tuple()
            .into_response())
        }
    }
}

pub async fn handle_sync_file(
    mut headers: HeaderMap,
    mut extensions: Extensions,
    GenericJson(request): GenericJson<FsSyncFileRequest>,
) -> Result<Response, Response> {
    let (ext_token, use_pri) = __unwrap!(extensions.remove::<TokenBundle>());

    let (data, compressed) = match encode_message(&request) {
        Ok(o) => o,
        Err(e) => return Err(e.into_response()),
    };

    let req = build_client_request(AiServiceRequest {
        ext_token: &ext_token,
        fs_client_key: headers.remove(CLIENT_KEY),
        url: ext_token.gcpp_host().get_url(CppService::FSSyncFile, use_pri),
        stream: false,
        compressed,
        trace_id: new_uuid_v4(),
        use_pri,
        cookie: headers.remove(COOKIE),
        exact_length: Some(data.len()),
    });

    match CollectBytesParts(req.body(data)).await {
        Ok((parts, bytes)) => match direct::decode::<FsSyncFileResponse>(&parts.headers, &bytes) {
            Ok(Ok(data)) => {
                let body = __unwrap!(serde_json::to_vec(&data));
                Ok(json_response_with_upstream_parts(parts, body))
            }
            Ok(Err(cursor_err)) => {
                let body = __unwrap!(serde_json::to_vec(&cursor_err.canonical().into_generic()));
                Err(json_response_with_upstream_parts(parts, body))
            }
            Err(e) => Err(e.into_generic_tuple().into_response()),
        },
        Err(e) => {
            let e = e.without_url();
            Err(ChatError::RequestFailed(
                if e.is_timeout() {
                    StatusCode::GATEWAY_TIMEOUT
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                },
                Cow::Owned(e.to_string()),
            )
            .into_generic_tuple()
            .into_response())
        }
    }
}

pub async fn handle_stream_cpp(
    mut headers: HeaderMap,
    mut extensions: Extensions,
    GenericJson(request): GenericJson<StreamCppRequest>,
) -> Result<Response, (StatusCode, InfallibleJson<GenericError>)> {
    let (ext_token, use_pri) = __unwrap!(extensions.remove::<TokenBundle>());

    let data = match encode_message_framed(&request) {
        Ok(o) => o,
        Err(e) => return Err(e.into_response_tuple()),
    };

    let req = build_client_request(AiServiceRequest {
        ext_token: &ext_token,
        fs_client_key: headers.remove(CLIENT_KEY),
        url: ext_token.gcpp_host().get_url(CppService::StreamCpp, use_pri),
        stream: true,
        compressed: true,
        trace_id: new_uuid_v4(),
        use_pri,
        cookie: headers.remove(COOKIE),
        exact_length: Some(data.len()),
    });

    let res = match req.body(data).send().await {
        Ok(r) => r,
        Err(e) => {
            let e = e.without_url();

            return Err(ChatError::RequestFailed(
                if e.is_timeout() {
                    StatusCode::GATEWAY_TIMEOUT
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                },
                Cow::Owned(e.to_string()),
            )
            .into_generic_tuple());
        }
    };

    // Format SSE events.
    #[inline]
    fn format_sse_event(vector: &mut Vec<u8>, message: &StreamMessage) {
        vector.extend_from_slice(b"event: ");
        vector.extend_from_slice(message.type_name().as_bytes());
        vector.extend_from_slice(b"\ndata: ");
        let vector = {
            let mut ser = serde_json::Serializer::new(vector);
            __unwrap!(serde::Serialize::serialize(message, &mut ser));
            ser.into_inner()
        };
        vector.extend_from_slice(b"\n\n");
    }

    fn process_messages<I>(messages: impl IntoIterator<Item = I::Item, IntoIter = I>) -> Vec<u8>
    where I: Iterator<Item = StreamMessage> {
        let mut response_data = Vec::with_capacity(128);
        for message in messages {
            format_sse_event(&mut response_data, &message);
        }
        response_data
    }

    let mut decoder = StreamDecoder::new();

    let stream = res.bytes_stream().map(move |chunk| {
        let chunk = match chunk {
            Ok(c) => c,
            Err(_) => return Ok::<_, Infallible>(Bytes::new()),
        };

        let messages = match decoder.decode(&chunk) {
            Ok(msgs) => msgs,
            Err(()) => {
                let count = decoder.get_empty_stream_count();
                if count > 1 {
                    eprintln!("[警告] 连续空流: {count} 次");
                    return Ok(Bytes::from_static(
                        b"event: error\ndata: {\"type\":\"error\",\"error\":{\"code\":533,\"type\":\"unknown\",\"details\":{\"title\":\"Empty\",\"detail\":\"Empty stream\"}}}\n\n",
                    ));
                }
                return Ok(Bytes::new());
            }
        };

        if messages.is_empty() {
            return Ok(Bytes::new());
        }

        Ok(Bytes::from(process_messages(messages)))
    });

    Ok(__unwrap!(
        Response::builder()
            .header(CACHE_CONTROL, NO_CACHE_REVALIDATE)
            .header(CONNECTION, KEEP_ALIVE)
            .header(CONTENT_TYPE, EVENT_STREAM)
            .header(TRANSFER_ENCODING, CHUNKED)
            .body(Body::from_stream(stream))
    ))
}
