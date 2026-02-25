use crate::{
    app::{
        constant::UNNAMED,
        model::{
            AppState, Checksum, CommonResponse, ExtToken, GcppHost, Hash, RawToken, Token,
            TokenError, TokenHealth, TokenInfo, TokenManager, TokenStatus, TokensAddRequest,
            TokensAddResponse, TokensAliasSetRequest, TokensDeleteRequest, TokensDeleteResponse,
            TokensGetResponse, TokensMergeRequest, TokensProxySetRequest, TokensStatusSetRequest,
            TokensTimezoneSetRequest, TokensUpdateRequest,
        },
        route::{GenericJson, InfallibleJson},
    },
    common::model::{ApiStatus, GenericError},
};
use alloc::{borrow::Cow, sync::Arc};
use axum::extract::State;
use core::str::FromStr as _;
use http::StatusCode;
use interned::ArcStr;

type HashSet<K> = hashbrown::HashSet<K, ahash::RandomState>;

crate::define_typed_constants! {
    &'static str => {
        SET_SUCCESS = "已设置",
        SET_FAILURE_COUNT = "个令牌设置失败",
        UPDATE_SUCCESS = "已更新",
        UPDATE_FAILURE_COUNT = "个令牌更新失败",
        ERROR_SAVE_TOKEN_DATA = "Failed to save token data",
        MESSAGE_SAVE_TOKEN_DATA_FAILED = "无法保存令牌数据",
        ERROR_NO_TOKENS_PROVIDED = "No tokens provided",
        MESSAGE_NO_TOKENS_PROVIDED = "未提供任何令牌",
        ERROR_SAVE_TOKEN_PROFILES = "Failed to save token profiles",
        ERROR_SAVE_TOKENS = "Failed to save tokens",
        ERROR_SAVE_TOKEN_STATUSES = "Failed to save token statuses",
        ERROR_SAVE_TOKEN_ALIASES = "Failed to save token aliases",
        ERROR_SAVE_TOKEN_PROXIES = "Failed to save token proxies",
        ERROR_SAVE_TOKEN_TIMEZONES = "Failed to save token timezones",
        MESSAGE_SAVE_TOKEN_PROFILE_FAILED = "无法保存令牌配置数据",
        MESSAGE_SAVE_TOKEN_CONFIG_VERSION_FAILED = "无法保存令牌配置版本数据",
        MESSAGE_SAVE_TOKEN_STATUS_FAILED = "无法保存令牌状态数据",
        MESSAGE_SAVE_TOKEN_PROXY_FAILED = "无法保存令牌代理数据",
        MESSAGE_SAVE_TOKEN_TIMEZONE_FAILED = "无法保存令牌时区数据",
    }
}

pub async fn handle_get_tokens(
    State(state): State<Arc<AppState>>,
) -> InfallibleJson<TokensGetResponse> {
    let tokens = state.token_manager_read().await.list();

    InfallibleJson(TokensGetResponse { tokens })
}

pub async fn handle_set_tokens(
    State(state): State<Arc<AppState>>,
    GenericJson(tokens): GenericJson<TokensUpdateRequest>,
) -> Result<InfallibleJson<TokensAddResponse>, StatusCode> {
    // 获取写锁并更新token manager
    let mut token_manager = state.token_manager_write().await;
    *token_manager = TokenManager::new(tokens.len());
    for (alias, token_info) in tokens {
        let _ = token_manager.add(token_info, alias);
    }
    let tokens_count = token_manager.tokens().len();

    // 保存到文件
    token_manager.save().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(InfallibleJson(TokensAddResponse {
        tokens_count,
        message: "Token files have been updated and reloaded",
    }))
}

pub async fn handle_add_tokens(
    State(state): State<Arc<AppState>>,
    GenericJson(request): GenericJson<TokensAddRequest>,
) -> Result<InfallibleJson<TokensAddResponse>, (StatusCode, InfallibleJson<GenericError>)> {
    // 获取token manager的写锁
    let mut token_manager = state.token_manager_write().await;

    // 创建现有token的集合
    let existing_tokens: HashSet<_> = token_manager
        .tokens()
        .iter()
        .flatten()
        .map(|info| info.bundle.primary_token.raw())
        .collect();

    // 处理新的tokens
    let mut new_tokens = Vec::with_capacity(request.tokens.len());
    for token_info in request.tokens {
        if let Ok(raw) = <RawToken as ::core::str::FromStr>::from_str(&token_info.token)
            && !existing_tokens.contains(&raw)
        {
            new_tokens.push((
                TokenInfo {
                    bundle: ExtToken {
                        primary_token: Token::new(raw, Some(token_info.token)),
                        secondary_token: None,
                        checksum: token_info
                            .checksum
                            .as_deref()
                            .map(Checksum::repair)
                            .unwrap_or_default(),
                        client_key: token_info
                            .client_key
                            .and_then(|s| Hash::from_str(&s).ok())
                            .unwrap_or_else(Hash::random),
                        session_id: token_info
                            .session_id
                            .and_then(|s| uuid::Uuid::parse_str(&s).ok())
                            .unwrap_or_else(uuid::Uuid::new_v4),
                        config_version: token_info
                            .config_version
                            .and_then(|s| uuid::Uuid::parse_str(&s).ok()),
                        proxy: token_info.proxy.map(ArcStr::new),
                        timezone: token_info
                            .timezone
                            .and_then(|s| chrono_tz::Tz::from_str(&s).ok()),
                        gcpp_host: token_info.gcpp_host.and_then(|s| GcppHost::from_str(&s)),
                    },
                    status: TokenStatus { enabled: request.enabled, health: TokenHealth::new() },
                    usage: None,
                    user: None,
                    stripe: None,
                    sessions: vec![],
                },
                token_info
                    .alias
                    .filter(|s| s.split_whitespace().next().is_some())
                    .map(Cow::Owned)
                    .unwrap_or(Cow::Borrowed(UNNAMED)),
            ));
        }
    }

    // 如果有新tokens才进行后续操作
    if !new_tokens.is_empty() {
        // 添加新tokens
        for (token_info, alias) in new_tokens {
            let _ = token_manager.add(token_info, alias);
        }
        let tokens_count = token_manager.tokens().len();

        // 保存到文件
        token_manager.save().await.map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                InfallibleJson(GenericError {
                    status: ApiStatus::Error,
                    code: None,
                    error: Some(Cow::Borrowed(ERROR_SAVE_TOKEN_DATA)),
                    message: Some(Cow::Borrowed(MESSAGE_SAVE_TOKEN_DATA_FAILED)),
                }),
            )
        })?;

        Ok(InfallibleJson(TokensAddResponse {
            tokens_count,
            message: "New tokens have been added and reloaded",
        }))
    } else {
        // 如果没有新tokens，返回当前状态
        let tokens_count = token_manager.tokens().len();

        Ok(InfallibleJson(TokensAddResponse { tokens_count, message: "No new tokens were added" }))
    }
}

pub async fn handle_delete_tokens(
    State(state): State<Arc<AppState>>,
    GenericJson(request): GenericJson<TokensDeleteRequest>,
) -> Result<InfallibleJson<TokensDeleteResponse>, (StatusCode, InfallibleJson<GenericError>)> {
    let mut token_manager = state.token_manager_write().await;

    // 一次遍历完成删除和失败记录
    let (has_updates, failed_tokens) = {
        let mut has_updates = false;
        let mut failed_tokens = if request.include_failed_tokens { Some(Vec::new()) } else { None };

        for alias in request.aliases {
            match token_manager.alias_map().get(alias.as_str()) {
                Some(&id) => {
                    let _ = token_manager.remove(id);
                    has_updates = true;
                }
                None => {
                    if let Some(ref mut failed) = failed_tokens {
                        failed.push(alias);
                    }
                }
            }
        }

        (has_updates, failed_tokens)
    };

    // 如果有更新则保存
    if has_updates {
        token_manager.save().await.map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                InfallibleJson(GenericError {
                    status: ApiStatus::Success,
                    code: None,
                    error: Some(Cow::Borrowed(ERROR_SAVE_TOKEN_DATA)),
                    message: Some(Cow::Borrowed(MESSAGE_SAVE_TOKEN_DATA_FAILED)),
                }),
            )
        })?;
    }

    Ok(InfallibleJson(TokensDeleteResponse { status: ApiStatus::Success, failed_tokens }))
}

pub async fn handle_update_tokens_profile(
    State(state): State<Arc<AppState>>,
    GenericJson(aliases): GenericJson<HashSet<String>>,
) -> Result<InfallibleJson<CommonResponse>, (StatusCode, InfallibleJson<GenericError>)> {
    // 验证请求
    if aliases.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Borrowed(ERROR_NO_TOKENS_PROVIDED)),
                message: Some(Cow::Borrowed(MESSAGE_NO_TOKENS_PROVIDED)),
            }),
        ));
    }

    // 获取当前的 token_manager
    let mut token_manager = state.token_manager_write().await;

    // 批量设置tokens的profile
    let mut updated_count = 0u32;
    let mut failed_count = 0u32;

    let mut alias_updaters: Vec<(usize, String)> = Vec::with_capacity(aliases.len());

    for alias in &aliases {
        // 验证token是否在token_manager中存在
        if let Some(id) = token_manager.alias_map().get(alias.as_str()).copied() {
            let alias_is_unnamed = unsafe {
                token_manager
                    .id_to_alias()
                    .get_unchecked(id)
                    .as_ref()
                    .unwrap_unchecked()
                    .is_unnamed()
            };
            let token_info = unsafe { token_manager.tokens_mut().get_unchecked_mut(id) };

            // 获取profile
            let (usage, stripe, user, sessions) = crate::common::utils::get_token_profile(
                token_info.bundle.get_client(),
                token_info.bundle.as_unext(),
                true,
                true,
            )
            .await;

            // 设置profile
            if alias_is_unnamed
                && let Some(ref user) = user
                && let Some(alias) = user.alias()
            {
                // Safety: capacity == aliases.len && token_info.len <= aliases.len
                unsafe {
                    let len = alias_updaters.len();
                    let end = alias_updaters.as_mut_ptr().add(len);
                    ::core::ptr::write(end, (id, alias.clone()));
                    alias_updaters.set_len(len + 1);
                }
            }
            token_info.usage = usage;
            token_info.user = user;
            token_info.stripe = stripe;
            if let Some(sessions) = sessions {
                token_info.sessions = sessions;
            }
            updated_count += 1;
        } else {
            failed_count += 1;
        }
    }

    for (id, alias) in alias_updaters {
        let _ = token_manager.set_alias(id, alias);
    }

    // 保存更改
    if updated_count > 0 && token_manager.save().await.is_err() {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Borrowed(ERROR_SAVE_TOKEN_PROFILES)),
                message: Some(Cow::Borrowed(MESSAGE_SAVE_TOKEN_PROFILE_FAILED)),
            }),
        ));
    }

    Ok(InfallibleJson(CommonResponse {
        status: ApiStatus::Success,
        message: Cow::Owned(
            [
                UPDATE_SUCCESS,
                itoa::Buffer::new().format(updated_count),
                "个令牌配置, ",
                itoa::Buffer::new().format(failed_count),
                UPDATE_FAILURE_COUNT,
            ]
            .concat(),
        ),
    }))
}

pub async fn handle_update_tokens_config_version(
    State(state): State<Arc<AppState>>,
    GenericJson(aliases): GenericJson<HashSet<String>>,
) -> Result<InfallibleJson<CommonResponse>, (StatusCode, InfallibleJson<GenericError>)> {
    if aliases.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Borrowed(ERROR_NO_TOKENS_PROVIDED)),
                message: Some(Cow::Borrowed(MESSAGE_NO_TOKENS_PROVIDED)),
            }),
        ));
    }

    let mut token_manager = state.token_manager_write().await;

    let mut updated_count = 0u32;
    let mut failed_count = 0u32;
    let mut short_token_count = 0u32;

    for alias in aliases {
        if let Some(info) = token_manager
            .alias_map()
            .get(alias.as_str())
            .copied()
            .map(|id| unsafe { token_manager.tokens_mut().get_unchecked_mut(id) })
        {
            if info.bundle.primary_token.is_web() {
                short_token_count += 1;
                failed_count += 1;
            } else if let Some(config_version) = {
                crate::common::utils::get_server_config(
                    info.bundle.clone_without_config_version(),
                    true,
                )
                .await
            } {
                info.bundle.config_version = Some(config_version);
                updated_count += 1;
            } else {
                failed_count += 1;
            }
        } else {
            failed_count += 1;
        }
    }

    // 保存更改
    if updated_count > 0 && token_manager.save().await.is_err() {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Borrowed(ERROR_SAVE_TOKEN_PROFILES)),
                message: Some(Cow::Borrowed(MESSAGE_SAVE_TOKEN_CONFIG_VERSION_FAILED)),
            }),
        ));
    }

    let message = if short_token_count > 0 {
        [
            UPDATE_SUCCESS,
            itoa::Buffer::new().format(updated_count),
            "个令牌配置版本；",
            itoa::Buffer::new().format(failed_count),
            "个令牌更新失败，其中有",
            itoa::Buffer::new().format(short_token_count),
            "个令牌是非会话令牌",
        ]
        .concat()
    } else {
        [
            UPDATE_SUCCESS,
            itoa::Buffer::new().format(updated_count),
            "个令牌配置版本；",
            itoa::Buffer::new().format(failed_count),
            UPDATE_FAILURE_COUNT,
        ]
        .concat()
    };

    Ok(InfallibleJson(CommonResponse { status: ApiStatus::Success, message: Cow::Owned(message) }))
}

pub async fn handle_refresh_tokens(
    State(state): State<Arc<AppState>>,
    GenericJson(aliases): GenericJson<HashSet<String>>,
) -> Result<InfallibleJson<CommonResponse>, (StatusCode, InfallibleJson<GenericError>)> {
    if aliases.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Borrowed(ERROR_NO_TOKENS_PROVIDED)),
                message: Some(Cow::Borrowed(MESSAGE_NO_TOKENS_PROVIDED)),
            }),
        ));
    }

    let mut token_manager = state.token_manager_write().await;

    let mut updated_count = 0u32;
    let mut failed_count = 0u32;

    for alias in aliases {
        if let Some(writer) = token_manager
            .alias_map()
            .get(alias.as_str())
            .copied()
            .map(|id| unsafe { token_manager.tokens_mut().into_token_writer(id) })
            && crate::common::utils::get_new_token(writer, true).await
        {
            updated_count += 1;
        } else {
            failed_count += 1;
        }
    }

    // 保存更改
    if updated_count > 0 && token_manager.save().await.is_err() {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Borrowed(ERROR_SAVE_TOKENS)),
                message: Some(Cow::Borrowed(MESSAGE_SAVE_TOKEN_DATA_FAILED)),
            }),
        ));
    }

    Ok(InfallibleJson(CommonResponse {
        status: ApiStatus::Success,
        message: Cow::Owned(
            [
                "已刷新",
                itoa::Buffer::new().format(updated_count),
                "个令牌, ",
                itoa::Buffer::new().format(failed_count),
                "个令牌刷新失败",
            ]
            .concat(),
        ),
    }))
}

pub async fn handle_set_tokens_status(
    State(state): State<Arc<AppState>>,
    GenericJson(request): GenericJson<TokensStatusSetRequest>,
) -> Result<InfallibleJson<CommonResponse>, (StatusCode, InfallibleJson<GenericError>)> {
    // 验证请求
    if request.aliases.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Borrowed(ERROR_NO_TOKENS_PROVIDED)),
                message: Some(Cow::Borrowed(MESSAGE_NO_TOKENS_PROVIDED)),
            }),
        ));
    }

    // 获取当前的 token_manager
    let mut token_manager = state.token_manager_write().await;

    // 批量设置tokens的profile
    let mut updated_count = 0u32;
    let mut failed_count = 0u32;

    for alias in request.aliases {
        // 验证token是否在token_manager中存在
        if let Some(info) = token_manager
            .alias_map()
            .get(alias.as_str())
            .copied()
            .map(|id| unsafe { token_manager.tokens_mut().get_unchecked_mut(id) })
        {
            info.status.enabled = request.enabled;
            updated_count += 1;
        } else {
            failed_count += 1;
        }
    }

    // 保存更改
    if updated_count > 0 && token_manager.save().await.is_err() {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Borrowed(ERROR_SAVE_TOKEN_STATUSES)),
                message: Some(Cow::Borrowed(MESSAGE_SAVE_TOKEN_STATUS_FAILED)),
            }),
        ));
    }

    Ok(InfallibleJson(CommonResponse {
        status: ApiStatus::Success,
        message: Cow::Owned(
            [
                SET_SUCCESS,
                itoa::Buffer::new().format(updated_count),
                "个令牌状态, ",
                itoa::Buffer::new().format(failed_count),
                SET_FAILURE_COUNT,
            ]
            .concat(),
        ),
    }))
}

pub async fn handle_set_tokens_alias(
    State(state): State<Arc<AppState>>,
    GenericJson(request): GenericJson<TokensAliasSetRequest>,
) -> Result<InfallibleJson<CommonResponse>, (StatusCode, InfallibleJson<GenericError>)> {
    // 验证请求
    if request.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Borrowed(ERROR_NO_TOKENS_PROVIDED)),
                message: Some(Cow::Borrowed(MESSAGE_NO_TOKENS_PROVIDED)),
            }),
        ));
    }

    let mut token_manager = state.token_manager_write().await;
    let mut updated_count = 0u32;
    let mut failed_count = 0u32;

    for (old_alias, new_alias) in request {
        // 通过旧别名查找token ID
        match token_manager.alias_map().get(old_alias.as_str()).copied() {
            Some(token_id) => {
                // 使用set_alias方法更新别名
                match token_manager.set_alias(token_id, new_alias) {
                    Ok(()) => updated_count += 1,
                    Err(TokenError::AliasExists) => {
                        // 新别名已存在
                        failed_count += 1;
                    }
                    Err(TokenError::InvalidId) => {
                        // 理论上不应该发生，因为ID是从alias_map获取的
                        failed_count += 1;
                    }
                }
            }
            None => {
                // 找不到对应的旧别名
                failed_count += 1;
            }
        }
    }

    // 保存更改
    if updated_count > 0
        && let Err(e) = token_manager.save().await
    {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Borrowed(ERROR_SAVE_TOKEN_ALIASES)),
                message: Some(Cow::Owned(e.to_string())),
            }),
        ));
    }

    Ok(InfallibleJson(CommonResponse {
        status: ApiStatus::Success,
        message: Cow::Owned(
            [
                SET_SUCCESS,
                itoa::Buffer::new().format(updated_count),
                "个令牌别名, ",
                itoa::Buffer::new().format(failed_count),
                SET_FAILURE_COUNT,
            ]
            .concat(),
        ),
    }))
}

pub async fn handle_set_tokens_proxy(
    State(state): State<Arc<AppState>>,
    GenericJson(request): GenericJson<TokensProxySetRequest>,
) -> Result<InfallibleJson<CommonResponse>, (StatusCode, InfallibleJson<GenericError>)> {
    // 验证请求
    if request.aliases.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Borrowed(ERROR_NO_TOKENS_PROVIDED)),
                message: Some(Cow::Borrowed(MESSAGE_NO_TOKENS_PROVIDED)),
            }),
        ));
    }

    // 获取当前的 token_manager
    let mut token_manager = state.token_manager_write().await;

    // 批量设置tokens的proxy
    let mut updated_count = 0u32;
    let mut failed_count = 0u32;

    for alias in request.aliases {
        // 验证token是否在token_manager中存在
        if let Some(info) = token_manager
            .alias_map()
            .get(alias.as_str())
            .copied()
            .map(|id| unsafe { token_manager.tokens_mut().get_unchecked_mut(id) })
        {
            info.bundle.proxy = request.proxy.clone();
            updated_count += 1;
        } else {
            failed_count += 1;
        }
    }

    // 保存更改
    if updated_count > 0 && token_manager.save().await.is_err() {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Borrowed(ERROR_SAVE_TOKEN_PROXIES)),
                message: Some(Cow::Borrowed(MESSAGE_SAVE_TOKEN_PROXY_FAILED)),
            }),
        ));
    }

    Ok(InfallibleJson(CommonResponse {
        status: ApiStatus::Success,
        message: Cow::Owned(
            [
                SET_SUCCESS,
                itoa::Buffer::new().format(updated_count),
                "个令牌代理, ",
                itoa::Buffer::new().format(failed_count),
                SET_FAILURE_COUNT,
            ]
            .concat(),
        ),
    }))
}

pub async fn handle_set_tokens_timezone(
    State(state): State<Arc<AppState>>,
    GenericJson(request): GenericJson<TokensTimezoneSetRequest>,
) -> Result<InfallibleJson<CommonResponse>, (StatusCode, InfallibleJson<GenericError>)> {
    // 验证请求
    if request.aliases.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Borrowed(ERROR_NO_TOKENS_PROVIDED)),
                message: Some(Cow::Borrowed(MESSAGE_NO_TOKENS_PROVIDED)),
            }),
        ));
    }

    // 获取当前的 token_manager
    let mut token_manager = state.token_manager_write().await;

    // 批量设置tokens的timezone
    let mut updated_count = 0u32;
    let mut failed_count = 0u32;

    for alias in request.aliases {
        // 验证token是否在token_manager中存在
        if let Some(info) = token_manager
            .alias_map()
            .get(alias.as_str())
            .copied()
            .map(|id| unsafe { token_manager.tokens_mut().get_unchecked_mut(id) })
        {
            info.bundle.timezone = request.timezone;
            updated_count += 1;
        } else {
            failed_count += 1;
        }
    }

    // 保存更改
    if updated_count > 0 && token_manager.save().await.is_err() {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Borrowed(ERROR_SAVE_TOKEN_TIMEZONES)),
                message: Some(Cow::Borrowed(MESSAGE_SAVE_TOKEN_TIMEZONE_FAILED)),
            }),
        ));
    }

    Ok(InfallibleJson(CommonResponse {
        status: ApiStatus::Success,
        message: Cow::Owned(
            [
                SET_SUCCESS,
                itoa::Buffer::new().format(updated_count),
                "个令牌时区, ",
                itoa::Buffer::new().format(failed_count),
                SET_FAILURE_COUNT,
            ]
            .concat(),
        ),
    }))
}

pub async fn handle_merge_tokens(
    State(state): State<Arc<AppState>>,
    GenericJson(request): GenericJson<TokensMergeRequest>,
) -> Result<InfallibleJson<CommonResponse>, (StatusCode, InfallibleJson<GenericError>)> {
    // 验证请求
    if request.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Borrowed(ERROR_NO_TOKENS_PROVIDED)),
                message: Some(Cow::Borrowed(MESSAGE_NO_TOKENS_PROVIDED)),
            }),
        ));
    }

    // 获取token manager的写锁
    let mut token_manager = state.token_manager_write().await;

    // 应用merge
    let mut updated_count = 0u32;
    let mut failed_count = 0u32;

    for (alias, token_info) in request {
        // 验证token是否在token_manager中存在
        if token_info.has_some()
            && let Some(mut writer) = token_manager
                .alias_map()
                .get(alias.as_str())
                .copied()
                .map(|id| unsafe { token_manager.tokens_mut().into_token_writer(id) })
        {
            let bundle = &mut **writer;
            if let Some(token) = token_info.primary_token {
                bundle.primary_token = token;
            }
            if let Some(token) = token_info.secondary_token {
                bundle.secondary_token = Some(token);
            }
            if let Some(checksum) = token_info.checksum {
                bundle.checksum = checksum;
            }
            if let Some(client_key) = token_info.client_key {
                bundle.client_key = client_key;
            }
            if let Some(config_version) = token_info.config_version {
                bundle.config_version = Some(config_version);
            }
            if let Some(session_id) = token_info.session_id {
                bundle.session_id = session_id;
            }
            if let Some(proxy) = token_info.proxy {
                bundle.proxy = Some(proxy);
            }
            if let Some(timezone) = token_info.timezone {
                bundle.timezone = Some(timezone);
            }
            if let Some(gcpp_host) = token_info.gcpp_host {
                bundle.gcpp_host = Some(gcpp_host);
            }
            updated_count += 1;
        } else {
            failed_count += 1;
        }
    }

    // 保存更改
    if updated_count > 0 && token_manager.save().await.is_err() {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            InfallibleJson(GenericError {
                status: ApiStatus::Error,
                code: None,
                error: Some(Cow::Borrowed(ERROR_SAVE_TOKENS)),
                message: Some(Cow::Borrowed(MESSAGE_SAVE_TOKEN_DATA_FAILED)),
            }),
        ));
    }

    Ok(InfallibleJson(CommonResponse {
        status: ApiStatus::Success,
        message: Cow::Owned(
            [
                "已合并",
                itoa::Buffer::new().format(updated_count),
                "个令牌, ",
                itoa::Buffer::new().format(failed_count),
                "个令牌合并失败",
            ]
            .concat(),
        ),
    }))
}
