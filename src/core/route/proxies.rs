use crate::{
    app::{
        model::{
            CommonResponse, ProxiesDeleteRequest, ProxiesDeleteResponse, ProxyAddRequest,
            ProxyInfoResponse, ProxyUpdateRequest, SetGeneralProxyRequest,
            proxy_pool::{self, Proxies},
        },
        route::{GenericJson, InfallibleJson},
    },
    common::model::{ApiStatus, GenericError},
};
use alloc::{borrow::Cow, sync::Arc};
use http::StatusCode;
use interned::Str;

type HashMap<K, V> = hashbrown::HashMap<K, V, ahash::RandomState>;

crate::define_typed_constants! {
    &'static str => {
        // ERROR_SAVE_PROXY_CONFIG = "Failed to save proxy configuration: ",
        MESSAGE_SAVE_PROXY_CONFIG_FAILED = "无法保存代理配置",
        ERROR_PROXY_NAME_NOT_FOUND = "Proxy name not found",
        MESSAGE_PROXY_NAME_NOT_FOUND = "代理名称不存在",
        MESSAGE_GENERAL_PROXY_SET = "通用代理已设置",
        MESSAGE_PROXY_CONFIG_UPDATED = "代理配置已更新",
        MESSAGE_NO_NEW_PROXY_ADDED = "没有添加新代理",
        MESSAGE_ADDED_PREFIX = "已添加 ",
        MESSAGE_ADDED_SUFFIX = " 个新代理",
    }
}

fn format_save_proxy_config_error(
    e: Box<dyn core::error::Error + Send + Sync + 'static>,
) -> String {
    format!("Failed to save proxy configuration: {e}")
}

// 获取所有代理配置
pub async fn handle_get_proxies() -> InfallibleJson<ProxyInfoResponse> {
    // 获取代理配置并立即释放锁
    let proxies = proxy_pool::proxies().load_full();

    let proxies_count = proxies.len();
    let general_proxy = proxy_pool::general_name().load_full();

    InfallibleJson(ProxyInfoResponse {
        status: ApiStatus::Success,
        proxies: Some(proxies),
        proxies_count,
        general_proxy: Some(general_proxy),
        message: None,
    })
}

// 更新代理配置
pub async fn handle_set_proxies(
    GenericJson(proxies): GenericJson<ProxyUpdateRequest>,
) -> Result<InfallibleJson<ProxyInfoResponse>, (StatusCode, InfallibleJson<GenericError>)> {
    // 更新全局代理池并保存配置
    proxies.update_global();
    if let Err(e) = Proxies::update_and_save().await {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Owned(format_save_proxy_config_error(e))),
                message: Some(Cow::Borrowed(MESSAGE_SAVE_PROXY_CONFIG_FAILED)),
            }),
        ));
    }

    // 获取通用代理信息（在更新应用状态前）
    let proxies_count = proxy_pool::proxies().load().len();

    Ok(InfallibleJson(ProxyInfoResponse {
        status: ApiStatus::Success,
        proxies: None,
        proxies_count,
        general_proxy: None,
        message: Some(Cow::Borrowed(MESSAGE_PROXY_CONFIG_UPDATED)),
    }))
}

// 添加新的代理
pub async fn handle_add_proxy(
    GenericJson(request): GenericJson<ProxyAddRequest>,
) -> Result<InfallibleJson<ProxyInfoResponse>, (StatusCode, InfallibleJson<GenericError>)> {
    // 获取当前的代理配置
    let current = proxy_pool::proxies().load_full();
    let proxies = request
        .proxies
        .into_iter()
        .filter(|(name, _)| !current.contains_key(name.as_str()))
        .collect::<HashMap<_, _>>();

    if proxies.is_empty() {
        // 如果没有新代理，返回当前状态
        let proxies_count = current.len();

        return Ok(InfallibleJson(ProxyInfoResponse {
            status: ApiStatus::Success,
            proxies: Some(current),
            proxies_count,
            general_proxy: None,
            message: Some(Cow::Borrowed(MESSAGE_NO_NEW_PROXY_ADDED)),
        }));
    }

    let mut current = (*current).clone();

    // 处理新的代理
    let mut added_count = 0;

    for (name, proxy) in proxies {
        // 直接添加新的代理
        current.insert(name.into(), proxy);
        added_count += 1;
    }

    // 更新全局代理池并保存配置
    proxy_pool::proxies().store(Arc::new(current));
    if let Err(e) = Proxies::update_and_save().await {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Owned(format_save_proxy_config_error(e))),
                message: Some(Cow::Borrowed(MESSAGE_SAVE_PROXY_CONFIG_FAILED)),
            }),
        ));
    }

    // 获取更新后的信息
    let proxies_count = proxy_pool::proxies().load().len();

    Ok(InfallibleJson(ProxyInfoResponse {
        status: ApiStatus::Success,
        proxies: None,
        proxies_count,
        general_proxy: None,
        message: Some(Cow::Owned(
            [MESSAGE_ADDED_PREFIX, itoa::Buffer::new().format(added_count), MESSAGE_ADDED_SUFFIX]
                .concat(),
        )),
    }))
}

// 删除指定的代理
pub async fn handle_delete_proxies(
    GenericJson(request): GenericJson<ProxiesDeleteRequest>,
) -> Result<InfallibleJson<ProxiesDeleteResponse>, (StatusCode, InfallibleJson<GenericError>)> {
    let names = request.names;

    // 获取当前的代理配置并计算失败的代理名称
    let current = proxy_pool::proxies().load_full();

    // 计算失败的代理名称
    let capacity = (names.len() * 3) >> 2;
    let mut processing_names: Vec<String> = Vec::with_capacity(capacity);
    let mut failed_names: Vec<String> = Vec::with_capacity(capacity);
    for name in names {
        if current.contains_key(name.as_str()) {
            processing_names.push(name);
        } else {
            failed_names.push(name);
        }
    }

    // 删除指定的代理
    if !processing_names.is_empty() {
        let mut map = current
            .iter()
            .filter_map(|(name, value)| {
                if !processing_names.iter().any(|s| *s == *name) {
                    Some((name.clone(), value.clone()))
                } else {
                    None
                }
            })
            .collect::<HashMap<_, _>>();
        if map.is_empty() {
            map = crate::app::model::proxy_pool::default_proxies();
        }
        proxy_pool::proxies().store(Arc::new(map));
    }

    // 更新全局代理池并保存配置
    if let Err(e) = Proxies::update_and_save().await {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Owned(format_save_proxy_config_error(e))),
                message: Some(Cow::Borrowed(MESSAGE_SAVE_PROXY_CONFIG_FAILED)),
            }),
        ));
    }

    // 根据expectation返回不同的结果
    let updated_proxies = if request.expectation.needs_updated_tokens() {
        Some(proxy_pool::proxies().load_full())
    } else {
        None
    };

    Ok(InfallibleJson(ProxiesDeleteResponse {
        status: ApiStatus::Success,
        updated_proxies,
        failed_names: if request.expectation.needs_failed_tokens() && !failed_names.is_empty() {
            Some(failed_names)
        } else {
            None
        },
    }))
}

// 设置通用代理
pub async fn handle_set_general_proxy(
    GenericJson(request): GenericJson<SetGeneralProxyRequest>,
) -> Result<InfallibleJson<CommonResponse>, (StatusCode, InfallibleJson<GenericError>)> {
    // 检查代理名称是否存在
    if !proxy_pool::proxies().load().contains_key(request.name.as_str()) {
        return Err((
            StatusCode::BAD_REQUEST,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Borrowed(ERROR_PROXY_NAME_NOT_FOUND)),
                message: Some(Cow::Borrowed(MESSAGE_PROXY_NAME_NOT_FOUND)),
            }),
        ));
    }

    // 设置通用代理
    proxy_pool::general_name().store(Arc::new(Str::from(request.name)));

    // 更新全局代理池并保存配置
    if let Err(e) = Proxies::update_and_save().await {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Owned(format_save_proxy_config_error(e))),
                message: Some(Cow::Borrowed(MESSAGE_SAVE_PROXY_CONFIG_FAILED)),
            }),
        ));
    }

    Ok(InfallibleJson(CommonResponse {
        status: ApiStatus::Success,
        message: Cow::Borrowed(MESSAGE_GENERAL_PROXY_SET),
    }))
}
