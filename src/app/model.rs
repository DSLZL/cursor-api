mod alias;
mod build_key;
mod checksum;
mod config;
mod context_fill_mode;
mod cpp;
pub mod cursor_version;
mod default_instructions;
pub mod dynamic_key;
mod exchange_map;
mod fetch_model;
mod hash;
mod id_source;
mod log;
pub mod platform;
mod proxy;
pub mod proxy_pool;
mod state;
pub mod timestamp_header;
mod token;
mod tz;
mod usage_check;
pub mod version;
mod vision_ability;

use super::constant::{
    AUTHORIZATION_BEARER_PREFIX, EMPTY_STRING, STATUS_FAILURE, STATUS_PENDING, STATUS_SUCCESS,
};
use crate::{
    app::route::InfallibleSerialize,
    common::model::{
        ApiStatus,
        userinfo::{Session, StripeProfile, UsageProfile, UserProfile},
    },
};
pub use alias::Alias;
use alloc::borrow::Cow;
pub use build_key::{
    BuildKeyRequest, BuildKeyResponse, GetConfigVersionRequest, GetConfigVersionResponse,
    UsageCheckModelType,
};
pub use checksum::Checksum;
pub use config::AppConfig;
pub use context_fill_mode::create_explicit_context;
pub use cpp::{CppService, GcppHost};
pub use default_instructions::{DEFAULT_INSTRUCTIONS, DefaultInstructions};
pub use exchange_map::ExchangeMap;
pub use fetch_model::FetchMode;
pub use hash::Hash;
pub use id_source::ModelIdSource;
use interned::{ArcStr, Str};
pub use log::{GetLogsParams, LogUpdate, manager as log_manager};
pub use proxy::{
    ProxiesDeleteRequest, ProxiesDeleteResponse, ProxyAddRequest, ProxyInfoResponse,
    ProxyUpdateRequest, SetGeneralProxyRequest,
};
use proxy_pool::get_client_or_general;
use reqwest::Client;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
pub use state::{AppState, QueueType, TokenError, TokenHealth, TokenManager, TokenWriter};
pub use token::{
    Duration as TokenDuration, Randomness, RawToken, RawTokenHelper, SessionId, Subject, Token,
    TokenKey, UserId,
};
pub use tz::DateTime;
pub use usage_check::UsageCheck;
pub use version::Version;
pub use vision_ability::VisionAbility;

type HashMap<K, V> = hashbrown::HashMap<K, V, ahash::RandomState>;

#[derive(Clone, Copy, PartialEq, Archive, RkyvDeserialize, RkyvSerialize)]
#[repr(u8)]
pub enum LogStatus {
    Pending,
    Success,
    Failure,
}

impl Serialize for LogStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        serializer.serialize_str(self.as_str_name())
    }
}

impl<'de> Deserialize<'de> for LogStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        let s = <String as Deserialize>::deserialize(deserializer)?;
        Self::from_str_name(&s).ok_or_else(|| {
            serde::de::Error::custom("invalid status, expected 'pending', 'success', or 'failure'")
        })
    }
}

impl LogStatus {
    pub fn as_str_name(&self) -> &'static str {
        match self {
            Self::Pending => STATUS_PENDING,
            Self::Success => STATUS_SUCCESS,
            Self::Failure => STATUS_FAILURE,
        }
    }

    pub fn from_str_name(s: &str) -> Option<Self> {
        match s {
            STATUS_PENDING => Some(Self::Pending),
            STATUS_SUCCESS => Some(Self::Success),
            STATUS_FAILURE => Some(Self::Failure),
            _ => None,
        }
    }
}

// 请求日志
#[derive(Serialize, Clone)]
pub struct RequestLog {
    pub id: u64,
    pub timestamp: DateTime,
    pub model: &'static str,
    pub token_info: LogTokenInfo,
    pub chain: Chain,
    pub timing: TimingInfo,
    pub stream: bool,
    pub status: LogStatus,
    pub error: ErrorInfo,
}

impl RequestLog {
    #[inline(always)]
    pub fn token_key(&self) -> TokenKey { self.token_info.key }
}

#[derive(Serialize, Clone, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct Chain {
    // #[serde(skip_serializing_if = "Prompt::is_none")]
    // pub prompt: Prompt,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delays: Option<(String, Vec<(u32, f32)>)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ChainUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<String>,
}

impl Chain {
    #[inline]
    pub fn has_some(&self) -> bool {
        !(self.delays.is_none() && self.usage.is_none() && self.think.is_none())
    }
}

#[derive(Serialize, Clone, Copy, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct ChainUsage {
    pub input: i32,
    pub output: i32,
    pub cache_write: i32,
    pub cache_read: i32,
    pub cents: f32,
}

impl ChainUsage {
    pub fn total(&self) -> i32 { self.input + self.output + self.cache_read + self.cache_write }

    pub fn into_openai(self) -> crate::core::model::openai::Usage {
        use crate::core::model::openai;
        crate::core::model::openai::Usage {
            prompt_tokens: self.input,
            completion_tokens: self.output,
            total_tokens: self.input + self.output,
            prompt_tokens_details: openai::PromptTokensDetails { cached_tokens: self.cache_read },
            // completion_tokens_details: openai::CompletionTokensDetails { reasoning_tokens: 0 },
        }
    }

    pub fn into_anthropic(self) -> crate::core::model::anthropic::Usage {
        use crate::core::model::anthropic;
        anthropic::Usage {
            input_tokens: self.input,
            output_tokens: self.output,
            cache_creation_input_tokens: self.cache_write,
            cache_read_input_tokens: self.cache_read,
        }
    }

    pub fn into_anthropic_delta(self) -> crate::core::model::anthropic::MessageDeltaUsage {
        use crate::core::model::anthropic;
        anthropic::MessageDeltaUsage {
            input_tokens: self.input,
            output_tokens: self.output,
            cache_creation_input_tokens: self.cache_write,
            cache_read_input_tokens: self.cache_read,
        }
    }
}

impl From<crate::common::model::userinfo::TokenUsage> for ChainUsage {
    #[inline]
    fn from(uasge: crate::common::model::userinfo::TokenUsage) -> Self {
        Self {
            input: uasge.input_tokens,
            output: uasge.output_tokens,
            cache_write: uasge.cache_write_tokens,
            cache_read: uasge.cache_read_tokens,
            cents: uasge.total_cents,
        }
    }
}

// #[derive(Serialize, Clone)]
// #[serde(untagged)]
// pub enum Prompt {
//     None,
//     Origin(String),
//     Parsed(Vec<PromptMessage>),
// }

// #[derive(Serialize, Clone)]
// pub struct PromptMessage {
//     role: Role,
//     content: PromptContent,
// }

// #[derive(Clone)]
// #[repr(transparent)]
// pub struct PromptContent(crate::leak::ArcStr);

// impl Serialize for PromptContent {
//     fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//     where
//         S: serde::Serializer,
//     {
//         serializer.serialize_str(&self.0)
//     }
// }

// impl PromptContent {
//     #[inline]
//     pub fn into_owned(self) -> String {
//         self.0.as_str().to_owned()
//     }
// }

// impl Prompt {
//     pub fn new(input: String) -> Self {
//         let mut messages = Vec::new();
//         let mut remaining = input.as_str();

//         while !remaining.is_empty() {
//             // 检查是否以任一开始标记开头，并确定相应的结束标记
//             let (role, end_tag, content) =
//                 if let Some(r) = remaining.strip_prefix("<|BEGIN_SYSTEM|>\n") {
//                     (Role::System, "\n<|END_SYSTEM|>\n", r)
//                 } else if let Some(r) = remaining.strip_prefix("<|BEGIN_USER|>\n") {
//                     (Role::User, "\n<|END_USER|>\n", r)
//                 } else if let Some(r) = remaining.strip_prefix("<|BEGIN_ASSISTANT|>\n") {
//                     (Role::Assistant, "\n<|END_ASSISTANT|>\n", r)
//                 } else {
//                     return Self::Origin(input);
//                 };

//             // 更新remaining为去除前缀后的内容
//             remaining = content;

//             // 查找结束标记
//             if let Some((content_part, after_end)) = remaining.split_once(end_tag) {
//                 // 提取内容
//                 let content =
//                     PromptContent(crate::leak::intern_arc(content_part.trim_leading_newlines()));
//                 messages.push(PromptMessage { role, content });

//                 // 移动到结束标记之后
//                 remaining = after_end;

//                 // 跳过消息之间的额外换行符
//                 if remaining.as_bytes().first().copied() == Some(b'\n') {
//                     remaining = unsafe { remaining.get_unchecked(1..) };
//                 }
//             } else {
//                 return Self::Origin(input);
//             }
//         }

//         Self::Parsed(messages)
//     }

//     #[inline(always)]
//     pub const fn is_none(&self) -> bool {
//         matches!(*self, Self::None)
//     }

//     #[inline(always)]
//     pub const fn is_some(&self) -> bool {
//         !self.is_none()
//     }
// }

#[derive(Serialize, Clone, Copy, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct TimingInfo {
    pub total: f64, // 总用时(秒)
}

#[derive(Serialize, Clone)]
#[serde(untagged)]
pub enum ErrorInfo {
    Empty,
    Simple(Str),
    Detailed { error: Str, details: Str },
}

impl ErrorInfo {
    #[inline]
    pub fn new(error: Str, details: Option<Str>) -> Self {
        if let Some(details) = details {
            Self::Detailed { error, details }
        } else {
            Self::Simple(error)
        }
    }

    // #[inline]
    // pub fn set_detail(&mut self, detail: Str) {
    //     match self {
    //         ErrorInfo::Empty => {
    //             *self = Self::Detailed { error: Str::from_static(EMPTY_STRING), details: detail }
    //         }
    //         ErrorInfo::Simple(error) => {
    //             *self = Self::Detailed { error: core::mem::take(error), details: detail }
    //         }
    //         ErrorInfo::Detailed { details, .. } => {
    //             *details = detail;
    //         }
    //     }
    // }

    pub fn contains(&self, pat: &str) -> bool {
        match self {
            Self::Empty => false,
            Self::Simple(error) => error.contains(pat),
            Self::Detailed { error, details } => error.contains(pat) || details.contains(pat),
        }
    }

    #[inline(always)]
    pub const fn is_none(&self) -> bool { matches!(*self, Self::Empty) }

    #[inline(always)]
    pub const fn is_some(&self) -> bool { !self.is_none() }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ExtToken {
    /// 主token - 可以是client或web token
    pub primary_token: Token,
    /// 次要token - 如果存在，必定是web token
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secondary_token: Option<Token>,
    pub checksum: Checksum,
    #[serde(skip_serializing_if = "Hash::is_nil", default = "Hash::random")]
    pub client_key: Hash,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_version: Option<uuid::Uuid>,
    #[serde(skip_serializing_if = "uuid::Uuid::is_nil")]
    pub session_id: uuid::Uuid,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proxy: Option<ArcStr>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<chrono_tz::Tz>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gcpp_host: Option<GcppHost>,
}

impl ExtToken {
    #[inline]
    pub fn clone(&self) -> Self {
        Self {
            primary_token: self.primary_token.clone(),
            secondary_token: None,
            checksum: self.checksum,
            client_key: self.client_key,
            config_version: self.config_version,
            session_id: self.session_id,
            proxy: self.proxy.clone(),
            timezone: self.timezone,
            gcpp_host: self.gcpp_host,
        }
    }

    #[inline]
    pub fn clone_without_config_version(&self) -> Self {
        Self {
            primary_token: self.primary_token.clone(),
            secondary_token: None,
            checksum: self.checksum,
            client_key: self.client_key,
            config_version: None,
            session_id: self.session_id,
            proxy: self.proxy.clone(),
            timezone: self.timezone,
            gcpp_host: self.gcpp_host,
        }
    }

    /// 获取适用于此 token 的 HTTP 客户端
    #[inline]
    pub fn get_client(&self) -> Client { get_client_or_general(self.proxy.as_deref()) }

    #[inline]
    pub fn get_client_lazy(&self) -> impl FnOnce<(), Output = Client> + Send + 'static {
        let proxy = self.proxy.clone();
        move || get_client_or_general(proxy.as_deref())
    }

    /// 获取此 token 关联的时区
    #[inline]
    fn timezone(&self) -> chrono_tz::Tz {
        if let Some(tz) = self.timezone { tz } else { *super::lazy::GENERAL_TIMEZONE }
    }

    #[inline]
    pub fn gcpp_host(&self) -> GcppHost {
        if let Some(gh) = self.gcpp_host { gh } else { *super::lazy::GENERAL_GCPP_HOST }
    }

    /// 返回关联的时区名称
    #[inline]
    pub fn timezone_name(&self) -> &'static str { self.timezone().name() }

    /// 返回关联的时区名称的头部值
    #[inline]
    pub fn timezone_as_header_value(&self) -> http::HeaderValue {
        unsafe { crate::common::model::HeaderValue::from_static(self.timezone_name()).into() }
    }

    /// 获取当前时区的当前时间
    #[inline]
    pub fn now(&self) -> chrono::DateTime<chrono_tz::Tz> {
        use chrono::TimeZone as _;
        self.timezone().from_utc_datetime(&DateTime::naive_now())
    }

    #[inline]
    pub fn store_unext(&self) -> UnextToken {
        UnextToken {
            primary_token: self.primary_token.clone(),
            secondary_token: self.secondary_token.clone(),
        }
    }

    #[inline]
    pub fn as_unext(&'_ self) -> UnextTokenRef<'_> {
        UnextTokenRef {
            primary_token: &self.primary_token,
            secondary_token: self.secondary_token.as_ref(),
        }
    }
}

pub struct UnextToken {
    /// 主token - 可以是client或web token
    pub primary_token: Token,
    /// 次要token - 如果存在，必定是web token
    pub secondary_token: Option<Token>,
}

impl UnextToken {
    #[inline]
    pub fn as_ref(&'_ self) -> UnextTokenRef<'_> {
        UnextTokenRef {
            primary_token: &self.primary_token,
            secondary_token: self.secondary_token.as_ref(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct UnextTokenRef<'a> {
    /// 主token - 可以是client或web token
    pub primary_token: &'a Token,
    /// 次要token - 如果存在，必定是web token
    pub secondary_token: Option<&'a Token>,
}

impl<'a> UnextTokenRef<'a> {
    #[inline]
    pub const fn web_token(self) -> &'a Token {
        match self.secondary_token {
            Some(t) => t,
            None => self.primary_token,
        }
    }

    #[inline]
    pub fn session_token(self) -> &'a Token { self.primary_token }

    #[inline]
    pub fn format_workos_cursor_session_token(self) -> http::HeaderValue {
        crate::define_typed_constants! {
            &'static str => {
                TOKEN_PREFIX = "WorkosCursorSessionToken=",
                SEPARATOR = "%3A%3A",
            }
            usize => {
                USER_ID_LEN = 31,
                PREFIX_AND_USER_ID_AND_SEPARATOR = TOKEN_PREFIX.len() + USER_ID_LEN + SEPARATOR.len(),
            }
        }

        let token = self.web_token();
        let token_str = token.as_str();

        // 预分配足够的空间: TOKEN_PREFIX + user_id + SEPARATOR + token_str
        let mut result = String::with_capacity(PREFIX_AND_USER_ID_AND_SEPARATOR + token_str.len());

        result.push_str(TOKEN_PREFIX);
        result.push_str(token.raw().subject.id.to_str(&mut [0; USER_ID_LEN]));
        result.push_str(SEPARATOR);
        result.push_str(token_str);

        let bytes = result.into_bytes().into_boxed_slice().into();

        unsafe { crate::common::model::HeaderValue { inner: bytes, is_sensitive: true }.into() }
    }

    #[inline]
    pub fn format_bearer_token(self) -> http::HeaderValue {
        let token = self.session_token().as_str();

        let mut result = Vec::with_capacity(AUTHORIZATION_BEARER_PREFIX.len() + token.len());

        result.extend_from_slice(AUTHORIZATION_BEARER_PREFIX.as_bytes());
        result.extend_from_slice(token.as_bytes());

        let bytes = result.into_boxed_slice().into();

        unsafe { crate::common::model::HeaderValue { inner: bytes, is_sensitive: true }.into() }
    }
}

// 用于存储 token 信息
#[derive(Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    #[serde(flatten)]
    pub bundle: ExtToken,
    #[serde(default)]
    pub status: TokenStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<UserProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stripe: Option<StripeProfile>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub sessions: Vec<Session>,
}

#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
struct ExtTokenHelper {
    primary_token: RawTokenHelper,
    secondary_token: Option<RawTokenHelper>,
    checksum: Checksum,
    client_key: Hash,
    config_version: Option<uuid::Uuid>,
    session_id: uuid::Uuid,
    proxy: Option<String>,
    timezone: Option<String>,
    gcpp_host: Option<GcppHost>,
}

impl ExtTokenHelper {
    #[inline]
    fn new(token_info: &ExtToken) -> Self {
        Self {
            primary_token: token_info.primary_token.raw().to_helper(),
            secondary_token: token_info.secondary_token.as_ref().map(|t| t.raw().to_helper()),
            checksum: token_info.checksum,
            client_key: token_info.client_key,
            config_version: token_info.config_version,
            session_id: token_info.session_id,
            proxy: token_info.proxy.as_deref().map(ToString::to_string),
            timezone: token_info.timezone.as_ref().map(ToString::to_string),
            gcpp_host: token_info.gcpp_host,
        }
    }

    #[inline]
    fn extract(self) -> ExtToken {
        ExtToken {
            primary_token: Token::new(self.primary_token.extract(), None),
            secondary_token: self.secondary_token.map(|h| Token::new(h.extract(), None)),
            checksum: self.checksum,
            client_key: self.client_key,
            config_version: self.config_version,
            session_id: self.session_id,
            proxy: self.proxy.map(ArcStr::new),
            timezone: self.timezone.map(|s| __unwrap_panic!(s.parse())),
            gcpp_host: self.gcpp_host,
        }
    }
}

#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
struct TokenInfoHelper {
    alias: String,
    bundle: ExtTokenHelper,
    status: TokenStatus,
    usage: Option<UsageProfile>,
    user: Option<UserProfile>,
    stripe: Option<StripeProfile>,
    sessions: Vec<Session>,
}

impl TokenInfoHelper {
    #[inline]
    fn new(token_info: &TokenInfo, alias: String) -> Self {
        Self {
            alias,
            bundle: ExtTokenHelper::new(&token_info.bundle),
            status: token_info.status,
            usage: token_info.usage,
            user: token_info.user.clone(),
            stripe: token_info.stripe,
            sessions: token_info.sessions.clone(),
        }
    }

    #[inline]
    fn extract(self) -> (TokenInfo, String) {
        (
            TokenInfo {
                bundle: self.bundle.extract(),
                status: self.status,
                usage: self.usage,
                user: self.user,
                stripe: self.stripe,
                sessions: self.sessions,
            },
            self.alias,
        )
    }
}

#[derive(Clone, Serialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct LogTokenInfo {
    #[serde(serialize_with = "serialize_token_key")]
    pub key: TokenKey,
    pub usage: Option<UsageProfile>,
    pub user: Option<UserProfile>,
    pub stripe: Option<StripeProfile>,
}

fn serialize_token_key<S>(key: &TokenKey, serializer: S) -> Result<S::Ok, S::Error>
where S: ::serde::Serializer {
    // use ::serde::ser::SerializeStruct as _;
    // let mut state = serializer.serialize_struct("TokenKey", 2)?;
    // state.serialize_field("user_id", &key.user_id.as_u128())?;
    // state.serialize_field("id", &key.randomness.as_u64())?;
    // state.end()
    serializer.serialize_str(&key.to_string())
}

#[derive(Clone, Copy, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct TokenStatus {
    #[serde(default = "crate::common::utils::r#true")]
    pub enabled: bool,
    #[serde(default)]
    pub health: TokenHealth,
}

impl const Default for TokenStatus {
    #[inline]
    fn default() -> Self { Self { enabled: true, health: TokenHealth::new() } }
}

impl TokenInfo {
    #[inline(always)]
    pub fn is_enabled(&self) -> bool { self.status.enabled }
}

// pub struct TokenValidityRange {
//     short: ValidityRange,
//     long: ValidityRange,
// }

// impl TokenValidityRange {
//     #[inline]
//     pub(super) fn new(short: ValidityRange, long: ValidityRange) -> Self {
//         Self { short, long }
//     }

//     #[inline]
//     pub fn is_short(&self, val: u32) -> bool {
//         self.short.is_valid(val)
//     }

//     #[inline]
//     pub fn is_long(&self, val: u32) -> bool {
//         self.long.is_valid(val)
//     }
// }

pub struct TokensGetResponse {
    pub tokens: Vec<(usize, Alias, TokenInfo)>,
}

impl Serialize for TokensGetResponse {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        use serde::ser::SerializeStruct as _;
        let mut state = serializer.serialize_struct("TokensGetResponse", 3)?;
        state.serialize_field("status", &ApiStatus::Success)?;
        state.serialize_field("tokens", &self.tokens)?;
        state.serialize_field("tokens_count", &self.tokens.len())?;
        state.end()
    }
}

unsafe impl InfallibleSerialize for TokensGetResponse {}

pub struct TokensAddResponse {
    pub tokens_count: usize,
    pub message: &'static str,
}

impl Serialize for TokensAddResponse {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        use serde::ser::SerializeStruct as _;
        let mut state = serializer.serialize_struct("TokensAddResponse", 3)?;
        state.serialize_field("status", &ApiStatus::Success)?;
        state.serialize_field("tokens_count", &self.tokens_count)?;
        state.serialize_field("message", self.message)?;
        state.end()
    }
}

unsafe impl InfallibleSerialize for TokensAddResponse {}

// TokensUpdateRequest 结构体
pub type TokensUpdateRequest = Vec<(String, TokenInfo)>;

#[derive(Deserialize)]
pub struct TokensAddRequest {
    pub tokens: Vec<TokensAddRequestTokenInfo>,
    #[serde(default)]
    pub enabled: bool,
}

#[derive(Deserialize)]
pub struct TokensAddRequestTokenInfo {
    pub alias: Option<String>,
    pub token: String,
    pub checksum: Option<String>,
    pub client_key: Option<String>,
    pub config_version: Option<String>,
    pub session_id: Option<String>,
    pub proxy: Option<String>,
    pub timezone: Option<String>,
    pub gcpp_host: Option<String>,
}

pub type TokensMergeRequest = HashMap<String, TokensMergeRequestTokenInfo>;

#[derive(Deserialize)]
pub struct TokensMergeRequestTokenInfo {
    pub primary_token: Option<Token>,
    pub secondary_token: Option<Token>,
    pub checksum: Option<Checksum>,
    pub client_key: Option<Hash>,
    pub config_version: Option<uuid::Uuid>,
    pub session_id: Option<uuid::Uuid>,
    pub proxy: Option<ArcStr>,
    pub timezone: Option<chrono_tz::Tz>,
    pub gcpp_host: Option<GcppHost>,
}

impl TokensMergeRequestTokenInfo {
    #[inline]
    pub fn has_some(&self) -> bool {
        !(self.primary_token.is_none()
            && self.secondary_token.is_none()
            && self.checksum.is_none()
            && self.client_key.is_none()
            && self.config_version.is_none()
            && self.session_id.is_none()
            && self.proxy.is_none()
            && self.timezone.is_none()
            && self.gcpp_host.is_none())
    }
}

// TokensDeleteRequest 结构体
#[derive(Deserialize)]
pub struct TokensDeleteRequest {
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub include_failed_tokens: bool,
}

// TokensDeleteResponse 结构体
#[derive(Serialize)]
pub struct TokensDeleteResponse {
    pub status: ApiStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_tokens: Option<Vec<String>>,
}

unsafe impl InfallibleSerialize for TokensDeleteResponse {}

#[derive(Serialize)]
pub struct CommonResponse {
    pub status: ApiStatus,
    pub message: Cow<'static, str>,
}

unsafe impl InfallibleSerialize for CommonResponse {}

#[derive(Deserialize)]
pub struct TokensStatusSetRequest {
    pub aliases: Vec<String>,
    #[serde(default = "crate::common::utils::r#true")]
    pub enabled: bool,
}

pub type TokensAliasSetRequest = HashMap<String, String>;

#[derive(Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DeleteResponseExpectation {
    #[default]
    Simple,
    UpdatedTokens,
    FailedTokens,
    Detailed,
}

impl DeleteResponseExpectation {
    #[inline]
    pub fn needs_updated_tokens(&self) -> bool {
        matches!(
            self,
            DeleteResponseExpectation::UpdatedTokens | DeleteResponseExpectation::Detailed
        )
    }

    #[inline]
    pub fn needs_failed_tokens(&self) -> bool {
        matches!(
            self,
            DeleteResponseExpectation::FailedTokens | DeleteResponseExpectation::Detailed
        )
    }
}

#[derive(Deserialize)]
pub struct TokensProxySetRequest {
    pub aliases: Vec<String>,
    pub proxy: Option<ArcStr>,
}

#[derive(Deserialize)]
pub struct TokensTimezoneSetRequest {
    pub aliases: Vec<String>,
    pub timezone: Option<chrono_tz::Tz>,
}
