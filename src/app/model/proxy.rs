use super::{
    ApiStatus, DeleteResponseExpectation,
    proxy_pool::{Proxies, SingleProxy},
};
use crate::app::route::InfallibleSerialize;
use interned::Str;
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, sync::Arc};

type HashMap<K, V> = hashbrown::HashMap<K, V, ahash::RandomState>;
type HashSet<K> = hashbrown::HashSet<K, ahash::RandomState>;

// 代理信息响应
#[derive(Serialize)]
pub struct ProxyInfoResponse {
    pub status: ApiStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proxies: Option<Arc<HashMap<Str, SingleProxy>>>,
    pub proxies_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub general_proxy: Option<Arc<Str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<Cow<'static, str>>,
}

unsafe impl InfallibleSerialize for ProxyInfoResponse {}

// 更新代理配置请求
pub type ProxyUpdateRequest = Proxies;

// 添加代理请求
#[derive(Deserialize)]
pub struct ProxyAddRequest {
    pub proxies: HashMap<String, SingleProxy>,
}

// 删除代理请求
#[derive(Deserialize)]
pub struct ProxiesDeleteRequest {
    #[serde(default)]
    pub names: HashSet<String>,
    #[serde(default)]
    pub expectation: DeleteResponseExpectation,
}

// 删除代理响应
#[derive(Serialize)]
pub struct ProxiesDeleteResponse {
    pub status: ApiStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_proxies: Option<Arc<HashMap<Str, SingleProxy>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_names: Option<Vec<String>>,
}

unsafe impl InfallibleSerialize for ProxiesDeleteResponse {}

// 设置通用代理请求
#[derive(Deserialize)]
pub struct SetGeneralProxyRequest {
    pub name: String,
}

// 设置获取图像代理请求
// #[derive(Deserialize)]
// pub struct SetFetchImageProxyRequest {
//     pub name: Option<String>,
// }
