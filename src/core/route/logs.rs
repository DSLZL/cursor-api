use crate::{
    app::{
        constant::AUTHORIZATION_BEARER_PREFIX,
        lazy::AUTH_TOKEN,
        model::{
            AppState, DateTime, ExtToken, GetLogsParams, LogStatus, RequestLog, TokenKey,
            log_manager,
        },
        route::{GenericJson, InfallibleJson, InfallibleSerialize},
    },
    common::model::{ApiStatus, userinfo::MembershipType},
    core::config::parse_dynamic_token,
};
use alloc::sync::Arc;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode, header::AUTHORIZATION},
};
use core::sync::atomic::Ordering;

type HashMap<K, V> = hashbrown::HashMap<K, V, ahash::RandomState>;
type HashSet<K> = hashbrown::HashSet<K, ahash::RandomState>;

#[derive(serde::Deserialize, Default)]
pub struct LogsQueryParams {
    // 分页与排序控制
    pub limit: Option<usize>,  // 返回记录数量限制
    pub offset: Option<usize>, // 起始位置偏移量
    pub reverse: Option<bool>, // 反向排序，默认false（从旧到新）

    // 时间范围过滤
    pub from_date: Option<DateTime>, // 开始日期时间
    pub to_date: Option<DateTime>,   // 结束日期时间

    // 用户标识过滤
    pub user_id: Option<String>,         // 按用户ID精确匹配
    pub email: Option<String>,           // 按用户邮箱过滤（部分匹配）
    pub membership_type: Option<String>, // 按会员类型过滤

    // 核心业务过滤
    pub status: Option<String>,              // 按状态过滤
    pub model: Option<String>,               // 按模型名称过滤（部分匹配）
    pub include_models: Option<Vec<String>>, // 包含特定模型
    pub exclude_models: Option<Vec<String>>, // 排除特定模型

    // 请求特征过滤
    pub stream: Option<bool>,    // 是否为流式请求
    pub has_chain: Option<bool>, // 是否包含对话链

    // 错误相关过滤
    pub has_error: Option<bool>, // 是否包含错误
    pub error: Option<String>,   // 按错误过滤（部分匹配）

    // 性能指标过滤
    pub min_total_time: Option<f64>, // 最小总耗时（秒）
    pub max_total_time: Option<f64>, // 最大总耗时（秒）
    pub min_tokens: Option<i32>,     // 最小token数
    pub max_tokens: Option<i32>,     // 最大token数
}

#[derive(::serde::Deserialize)]
pub struct LogsRequest {
    #[serde(default)]
    pub query: LogsQueryParams,
}

pub async fn handle_get_logs(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    GenericJson(request): GenericJson<LogsRequest>,
) -> Result<InfallibleJson<LogsResponse>, StatusCode> {
    let auth_token = headers
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix(AUTHORIZATION_BEARER_PREFIX))
        .ok_or(StatusCode::UNAUTHORIZED)?;

    let user_token = if auth_token != *AUTH_TOKEN {
        Some(if let Some(token_key) = TokenKey::from_string(auth_token) {
            token_key
        } else {
            parse_dynamic_token(auth_token)
                .and_then(|key_config| key_config.into_tuple())
                .and_then(|t| t.0.token.validate(t.1))
                .ok_or(StatusCode::UNAUTHORIZED)?
                .key()
        })
    } else {
        None
    };

    let status_enum = if let Some(status) = &request.query.status {
        match LogStatus::from_str_name(status) {
            Some(s) => Some(s),
            None => {
                return Ok(InfallibleJson(LogsResponse {
                    status: ApiStatus::Success,
                    total: 0,
                    active: None,
                    error: None,
                    logs: Vec::new(),
                    timestamp: DateTime::now(),
                }));
            }
        }
    } else {
        None
    };

    let membership_enum = if let Some(membership_type) = &request.query.membership_type {
        match MembershipType::from_str(membership_type) {
            Some(m) => Some(m),
            None => {
                return Ok(InfallibleJson(LogsResponse {
                    status: ApiStatus::Success,
                    total: 0,
                    active: None,
                    error: None,
                    logs: Vec::new(),
                    timestamp: DateTime::now(),
                }));
            }
        }
    } else {
        None
    };

    let parsed_user_id = if let Some(user_id) = &request.query.user_id {
        match user_id.parse() {
            Ok(id) => Some(id),
            Err(_) => return Err(StatusCode::BAD_REQUEST),
        }
    } else {
        None
    };

    let (active, error) = if user_token.is_some() {
        (None, None)
    } else {
        (
            Some(state.active_requests.load(Ordering::Relaxed)),
            Some(state.error_requests.load(Ordering::Relaxed)),
        )
    };

    let params = GetLogsParams {
        token_key: user_token,
        log_status: status_enum,
        membership_type: membership_enum,
        user_id: parsed_user_id,
        from_date: request.query.from_date,
        to_date: request.query.to_date,
        email: request.query.email,
        model: request.query.model,
        include_models: request.query.include_models,
        exclude_models: request.query.exclude_models,
        stream: request.query.stream,
        has_chain: request.query.has_chain,
        has_error: request.query.has_error,
        error: request.query.error,
        min_total_time: request.query.min_total_time,
        max_total_time: request.query.max_total_time,
        min_tokens: request.query.min_tokens,
        max_tokens: request.query.max_tokens,
        reverse: request.query.reverse.unwrap_or(false),
        offset: request.query.offset.unwrap_or(0),
        limit: request.query.limit.unwrap_or(usize::MAX),
    };

    let (total, logs) = log_manager::get_logs(params).await;

    Ok(InfallibleJson(LogsResponse {
        status: ApiStatus::Success,
        total,
        active,
        error,
        logs,
        timestamp: DateTime::now(),
    }))
}

#[derive(serde::Serialize)]
pub struct LogsResponse {
    pub status: ApiStatus,
    pub total: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<u64>,
    pub logs: Vec<RequestLog>,
    pub timestamp: DateTime,
}

unsafe impl InfallibleSerialize for LogsResponse {}

pub async fn handle_get_logs_tokens(
    headers: HeaderMap,
    GenericJson(keys): GenericJson<HashSet<String>>,
) -> Result<InfallibleJson<LogsTokensResponse>, StatusCode> {
    // 获取认证头
    let auth_token = headers
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix(AUTHORIZATION_BEARER_PREFIX))
        .ok_or(StatusCode::UNAUTHORIZED)?;

    if auth_token == *AUTH_TOKEN {
        let keys: Vec<_> = keys
            .into_iter()
            .filter_map(|s| TokenKey::from_string(&s).map(|key| (s, key)))
            .collect();
        let len = keys.len();
        let map = log_manager::get_tokens(keys).await;
        Ok(InfallibleJson(LogsTokensResponse {
            status: ApiStatus::Success,
            tokens: map,
            total: len as u64,
            timestamp: DateTime::now(),
        }))
    } else {
        let token_key = if let Some(token_key) = TokenKey::from_string(auth_token) {
            token_key
        } else {
            parse_dynamic_token(auth_token)
                .and_then(|key_config| key_config.into_tuple())
                .and_then(|t| t.0.token.validate(t.1))
                .ok_or(StatusCode::UNAUTHORIZED)?
                .key()
        };
        let mut iter = keys.into_iter();
        let key = iter.next();
        if let Some(key_str) = key
            && iter.next().is_none()
        {
            match TokenKey::from_string(&key_str) {
                Some(key) if key == token_key => {
                    let result = log_manager::get_token(token_key).await;
                    Ok(InfallibleJson(LogsTokensResponse {
                        status: ApiStatus::Success,
                        tokens: HashMap::from_iter([(key_str, result)]),
                        total: 1,
                        timestamp: DateTime::now(),
                    }))
                }
                Some(_) => Err(StatusCode::UNAUTHORIZED),
                None => Err(StatusCode::BAD_REQUEST),
            }
        } else {
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

#[derive(::serde::Serialize)]
pub struct LogsTokensResponse {
    pub status: ApiStatus,
    pub tokens: HashMap<String, Option<ExtToken>>,
    pub total: u64,
    pub timestamp: DateTime,
}

unsafe impl InfallibleSerialize for LogsTokensResponse {}
