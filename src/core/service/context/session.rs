use bytes::Bytes;
use core::{
    pin::Pin,
    result::Result,
    task::{Context, Poll},
};
use futures_core::Stream;
use reqwest::{DataStream, Decoder};
use tokio::sync::mpsc::{Sender, error::SendError};

type BoxError = Box<dyn ::core::error::Error + Send + Sync>;

pub struct SessionStream(DataStream<Decoder>);

impl Stream for SessionStream {
    type Item = Result<Bytes, reqwest::Error>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut Pin::into_inner(self).0).poll_next(cx)
    }
    fn size_hint(&self) -> (usize, Option<usize>) { self.0.size_hint() }
}

pub struct SessionSink(Sender<Result<Bytes, BoxError>>);

impl SessionSink {
    pub async fn send(&self, item: Bytes) -> Result<(), SendError<Bytes>> {
        self.0.send(Ok(item)).await.map_err(|e| {
            let Ok(item) = e.0 else { __unreachable!() };
            SendError(item)
        })
    }
}

impl SessionStream {
    pub fn new(res: reqwest::Response) -> Self { Self(DataStream(res.into_response().into_body())) }
}

impl SessionSink {
    pub fn new() -> (Self, reqwest::Body) {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let body = reqwest::Body::wrap_stream(stream);
        (Self(tx), body)
    }
}

pub struct Session {
    pub stream: SessionStream,
    pub sink: SessionSink,
}
