use crate::app::{
    constant::header::HEADER_VALUE_TEXT_PLAIN_UTF8,
    model::{Checksum, Hash, timestamp_header},
    route::{InfallibleJson, InfallibleSerialize},
};
use axum::{
    body::Body,
    http::{HeaderMap, header::CONTENT_TYPE},
    response::{IntoResponse as _, Response},
};
use std::sync::LazyLock;

static HEADERS_TEXT_PLAIN: LazyLock<HeaderMap> =
    LazyLock::new(|| HeaderMap::from_iter([(CONTENT_TYPE, HEADER_VALUE_TEXT_PLAIN_UTF8)]));

pub async fn handle_gen_uuid() -> Response {
    let mut buf = vec![0u8; 36];
    ::uuid::Uuid::new_v4().hyphenated().encode_lower(&mut buf);
    let body = Body::from(buf);

    (HEADERS_TEXT_PLAIN.clone(), body).into_response()
}

pub async fn handle_gen_hash() -> Response {
    let mut buf = vec![0u8; 64];
    Hash::random().to_str(unsafe { &mut *buf.as_mut_ptr().cast() });
    let body = Body::from(buf);

    (HEADERS_TEXT_PLAIN.clone(), body).into_response()
}

pub async fn handle_gen_checksum() -> Response {
    let mut buf = vec![0u8; 137];
    Checksum::random().to_str(unsafe { &mut *buf.as_mut_ptr().cast() });
    let body = Body::from(buf);

    (HEADERS_TEXT_PLAIN.clone(), body).into_response()
}

pub async fn handle_get_checksum_header() -> Response {
    let body = Body::from(timestamp_header::read().to_vec());
    (HEADERS_TEXT_PLAIN.clone(), body).into_response()
}

pub enum NtpSyncResult {
    Delta(String),
    Error(String),
}

impl serde::Serialize for NtpSyncResult {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        match *self {
            NtpSyncResult::Delta(ref value) => {
                serializer.serialize_newtype_variant("Result", 0, "delta", value)
            }
            NtpSyncResult::Error(ref value) => {
                serializer.serialize_newtype_variant("Result", 1, "error", value)
            }
        }
    }
}

unsafe impl InfallibleSerialize for NtpSyncResult {}

pub async fn handle_ntp_sync_once() -> InfallibleJson<NtpSyncResult> {
    InfallibleJson(match crate::common::model::ntp::sync_once().await {
        Ok(delta_nanos) => {
            crate::common::model::ntp::DELTA
                .store(delta_nanos, core::sync::atomic::Ordering::Relaxed);
            NtpSyncResult::Delta(format!("{}ms", delta_nanos / 1_000_000))
        }
        Err(e) => NtpSyncResult::Error(format!("NTP同步失败: {e}")),
    })
}
