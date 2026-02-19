use super::model::anthropic::ToolResultBlockParam;
use byte_str::ByteStr;
use bytes::Bytes;
use core::{
    pin::Pin,
    task::{Context, Poll},
};
use futures::Stream;
use manually_init::ManuallyInit;
use parking_lot::RwLock;
use reqwest::{DataStream, Decoder};
use tokio::sync::mpsc;

type HashMap<K, V> = hashbrown::HashMap<K, V, ahash::RandomState>;
type BoxError = Box<dyn ::core::error::Error + Send + Sync>;
type SessionTx = mpsc::Sender<ToolResultBlockParam>;

pub struct Session {
    stream: DataStream<Decoder>,
    sender: mpsc::Sender<Result<Bytes, BoxError>>,
}

impl Stream for Session {
    type Item = Result<Bytes, reqwest::Error>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.stream).poll_next(cx)
    }
}

pub struct Registry {
    inner: RwLock<HashMap<ByteStr, SessionTx>>,
}

impl Registry {
    pub fn init() {
        REGISTRY.init(Registry {
            inner: RwLock::new(HashMap::with_capacity_and_hasher(16, ahash::RandomState::new())),
        })
    }
}

static REGISTRY: ManuallyInit<Registry> = ManuallyInit::new();
