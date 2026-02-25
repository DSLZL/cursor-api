use crate::{
    app::{
        constant::AUTHORIZATION_BEARER_PREFIX,
        lazy::{AUTH_TOKEN, KEY_PREFIX},
        model::{
            AppConfig, BuildKeyRequest, BuildKeyResponse, ExtToken, GetConfigVersionRequest,
            GetConfigVersionResponse, Token, UnextTokenRef, UsageCheckModelType,
            dynamic_key::get_hash, proxy_pool::get_client_or_general,
        },
        route::{GenericJson, InfallibleJson, InfallibleSerialize},
    },
    common::{
        model::userinfo::{Session, StripeProfile, UsageProfile, UserProfile},
        utils::{to_base64, token_to_tokeninfo},
    },
    core::config::{ConfiguredKey, configured_key},
};
use axum::http::{HeaderMap, StatusCode, header::AUTHORIZATION};
use interned::ArcStr;

// 常量定义
const ERROR_UNAUTHORIZED: &str = "Unauthorized";
const ERROR_INVALID_SESSION_TOKEN: &str =
    "Invalid parameter: session_token must be a session token, not a web token";
const ERROR_INVALID_WEB_TOKEN: &str =
    "Invalid parameter: web_token must be a web token, not a session token";

// 验证认证令牌
fn verify_auth_token(headers: &HeaderMap) -> bool {
    headers
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix(AUTHORIZATION_BEARER_PREFIX))
        .is_none_or(|h| !AppConfig::share_token_eq(h) && h != *AUTH_TOKEN)
}

pub async fn handle_build_key(
    headers: HeaderMap,
    GenericJson(request): GenericJson<BuildKeyRequest>,
) -> (StatusCode, InfallibleJson<BuildKeyResponse>) {
    if verify_auth_token(&headers) {
        return (
            StatusCode::UNAUTHORIZED,
            InfallibleJson(BuildKeyResponse::Error(ERROR_UNAUTHORIZED)),
        );
    }

    let token_key = request.token.key();
    let secret = get_hash(&request.token);
    let token_info = token_to_tokeninfo(
        request.token,
        request.checksum,
        request.client_key,
        request.config_version,
        request.session_id,
        request.proxy_name,
        request.timezone,
        request.gcpp_host,
    );

    // 构建 proto 消息
    let key_config = ConfiguredKey {
        token_info: Some(token_info),
        secret: Some(secret),
        disable_vision: request.disable_vision,
        enable_slow_pool: request.enable_slow_pool,
        include_web_references: request.include_web_references,
        usage_check_models: if let Some(usage_check_models) = request.usage_check_models {
            Some(configured_key::UsageCheckModel {
                r#type: usage_check_models.model_type,
                model_ids: if matches!(usage_check_models.model_type, UsageCheckModelType::Custom) {
                    usage_check_models.model_ids.iter().map(|s| s.to_string()).collect()
                } else {
                    Vec::new()
                },
            })
        } else {
            None
        },
    };

    // 序列化
    let mut encoder = ::minicbor::Encoder::new(Vec::with_capacity(::minicbor::len(&key_config)));
    let _ = encoder.encode(key_config);

    let key = [&**KEY_PREFIX, to_base64(&encoder.into_writer()).as_str()].concat();

    (
        StatusCode::OK,
        InfallibleJson(BuildKeyResponse::Keys([
            key,
            token_key.to_string(),
            token_key.to_string2(),
        ])),
    )
}

pub async fn handle_get_config_version(
    headers: HeaderMap,
    GenericJson(request): GenericJson<GetConfigVersionRequest>,
) -> (StatusCode, InfallibleJson<GetConfigVersionResponse>) {
    if verify_auth_token(&headers) {
        return (
            StatusCode::UNAUTHORIZED,
            InfallibleJson(GetConfigVersionResponse::Error(ERROR_UNAUTHORIZED)),
        );
    }

    let token = ExtToken {
        primary_token: Token::new(request.token, None),
        secondary_token: None,
        checksum: request.checksum,
        client_key: request.client_key,
        config_version: None,
        session_id: request.session_id,
        proxy: request.proxy_name.map(ArcStr::new),
        timezone: request.timezone.and_then(|s| {
            use ::core::str::FromStr as _;
            chrono_tz::Tz::from_str(&s).ok()
        }),
        gcpp_host: request.gcpp_host,
    };

    match crate::common::utils::get_server_config(token, false).await {
        Some(cv) => (StatusCode::OK, InfallibleJson(GetConfigVersionResponse::ConfigVersion(cv))),
        None => (StatusCode::FORBIDDEN, InfallibleJson(GetConfigVersionResponse::Error("No data"))),
    }
}

#[derive(serde::Deserialize)]
pub struct GetTokenProfileRequest {
    session_token: Token,
    web_token: Token,
    proxy_name: Option<String>,
    include_sessions: bool,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GetTokenProfileResponse {
    TokenProfile(
        (Option<UsageProfile>, Option<StripeProfile>, Option<UserProfile>, Option<Vec<Session>>),
    ),
    Error(&'static str),
}

unsafe impl InfallibleSerialize for GetTokenProfileResponse {}

pub async fn handle_get_token_profile(
    headers: HeaderMap,
    GenericJson(request): GenericJson<GetTokenProfileRequest>,
) -> (StatusCode, InfallibleJson<GetTokenProfileResponse>) {
    if verify_auth_token(&headers) {
        return (
            StatusCode::UNAUTHORIZED,
            InfallibleJson(GetTokenProfileResponse::Error(ERROR_UNAUTHORIZED)),
        );
    }

    if request.session_token.is_web() {
        return (
            StatusCode::BAD_REQUEST,
            InfallibleJson(GetTokenProfileResponse::Error(ERROR_INVALID_SESSION_TOKEN)),
        );
    }

    if request.web_token.is_session() {
        return (
            StatusCode::BAD_REQUEST,
            InfallibleJson(GetTokenProfileResponse::Error(ERROR_INVALID_WEB_TOKEN)),
        );
    }

    let unext = UnextTokenRef {
        primary_token: &request.session_token,
        secondary_token: Some(&request.web_token),
    };

    (
        StatusCode::OK,
        InfallibleJson(GetTokenProfileResponse::TokenProfile(
            crate::common::utils::get_token_profile(
                get_client_or_general(request.proxy_name.as_deref()),
                unext,
                false,
                request.include_sessions,
            )
            .await,
        )),
    )
}
