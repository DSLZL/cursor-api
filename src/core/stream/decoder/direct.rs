use super::{
    decompress_gzip,
    types::{DecoderError, ProtobufMessage},
};
use crate::core::error::CursorError;
use alloc::borrow::Cow;
use http::{
    HeaderMap,
    header::{CONTENT_ENCODING, CONTENT_TYPE},
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct UnaryDecoder {
    compression: Compression,
    content_type: UnaryContentType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Compression {
    Identity,
    Gzip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnaryContentType {
    Protobuf,
    Json,
}

impl UnaryDecoder {
    /// Parse Connect unary response metadata from headers.
    pub(crate) fn new(headers: &HeaderMap) -> Result<Self, DecoderError> {
        Ok(Self {
            compression: parse_compression(headers)?,
            content_type: parse_content_type(headers)?,
        })
    }

    /// Decode a Connect unary response payload.
    pub(crate) fn decode<T: ProtobufMessage>(
        &self,
        data: &[u8],
    ) -> Result<Result<T, CursorError>, DecoderError> {
        let payload = match self.compression {
            Compression::Identity => Cow::Borrowed(data),
            Compression::Gzip => {
                let decompressed =
                    decompress_gzip(data).ok_or(DecoderError::GzipDecompressionFailed)?;
                Cow::Owned(decompressed)
            }
        };

        match self.content_type {
            UnaryContentType::Json => {
                let cursor_err = serde_json::from_slice::<CursorError>(payload.as_ref())
                    .map_err(|source| DecoderError::CursorErrorJsonDecodeFailed { source })?;
                Ok(Err(cursor_err))
            }
            UnaryContentType::Protobuf => {
                let msg = T::decode(payload.as_ref())
                    .map_err(|source| DecoderError::ProtobufDecodeFailed { source })?;
                Ok(Ok(msg))
            }
        }
    }
}

/// Decode a Connect unary response using response headers + body.
pub(crate) fn decode<T: ProtobufMessage>(
    headers: &HeaderMap,
    data: &[u8],
) -> Result<Result<T, CursorError>, DecoderError> {
    UnaryDecoder::new(headers)?.decode::<T>(data)
}

fn parse_compression(headers: &HeaderMap) -> Result<Compression, DecoderError> {
    let Some(value) = headers.get(CONTENT_ENCODING) else {
        return Ok(Compression::Identity);
    };

    let raw = value.to_str().map_err(|_| DecoderError::InvalidContentEncoding)?;
    let mut parts = raw.split(',').map(|s| s.trim()).filter(|s| !s.is_empty());

    let first = parts.next().unwrap_or("");
    if parts.next().is_some() {
        return Err(DecoderError::UnsupportedContentEncoding { actual: raw.into() });
    }

    if first.eq_ignore_ascii_case("gzip") {
        Ok(Compression::Gzip)
    } else if first.is_empty() || first.eq_ignore_ascii_case("identity") {
        Ok(Compression::Identity)
    } else {
        Err(DecoderError::UnsupportedContentEncoding { actual: raw.into() })
    }
}

fn parse_content_type(headers: &HeaderMap) -> Result<UnaryContentType, DecoderError> {
    let value = headers.get(CONTENT_TYPE).ok_or(DecoderError::MissingContentType)?;
    let raw = value.to_str().map_err(|_| DecoderError::InvalidContentType)?;

    let mime = raw.split(';').next().unwrap_or("").trim();

    if mime.eq_ignore_ascii_case("application/json") {
        Ok(UnaryContentType::Json)
    } else if mime.eq_ignore_ascii_case("application/proto") {
        Ok(UnaryContentType::Protobuf)
    } else {
        Err(DecoderError::UnsupportedContentType { actual: raw.into() })
    }
}
