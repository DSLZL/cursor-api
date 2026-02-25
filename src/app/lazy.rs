pub mod log;
mod path;

use super::{
    constant::{
        CURSOR_API2_HOST, CURSOR_API4_HOST, CURSOR_GCPP_ASIA_HOST, CURSOR_GCPP_EU_HOST,
        CURSOR_GCPP_US_HOST, CURSOR_HOST, EMPTY_STRING, HTTPS_PREFIX,
    },
    model::{DateTime, GcppHost},
};
use crate::common::{
    model::{
        HeaderValue,
        niche_types::{NonNegativeI8, NonNegativeI16},
    },
    utils::{ParseFromEnv, parse_from_env},
};
use alloc::borrow::Cow;
use manually_init::ManuallyInit;
pub use path::{
    CONFIG_FILE_PATH, DATA_DIR, LOGS_FILE_PATH, PROXIES_FILE_PATH, TOKENS_FILE_PATH,
    init as init_paths,
};
use std::sync::LazyLock;
use url::Url;

pub static AUTH_TOKEN: ManuallyInit<Cow<'static, str>> = ManuallyInit::new();

pub static START_TIME: ManuallyInit<DateTime> = ManuallyInit::new();

#[inline]
pub fn init_start_time() { START_TIME.init(DateTime::now()) }

#[inline]
#[rustfmt::skip]
pub fn init() {
    PRI_REVERSE_PROXY_HOST.init(parse_from_env("PRI_REVERSE_PROXY_HOST", EMPTY_STRING));
    PUB_REVERSE_PROXY_HOST.init(parse_from_env("PUB_REVERSE_PROXY_HOST", EMPTY_STRING));
    HeaderValue::validate(PRI_REVERSE_PROXY_HOST.as_bytes())
        .expect("HeaderValue::validate with invalid bytes");
    HeaderValue::validate(PUB_REVERSE_PROXY_HOST.as_bytes())
        .expect("HeaderValue::validate with invalid bytes");
    KEY_PREFIX.init(parse_from_env("KEY_PREFIX", DEFAULT_KEY_PREFIX));
    unsafe {
        USE_PRI_REVERSE_PROXY = !PRI_REVERSE_PROXY_HOST.is_empty();
        USE_PUB_REVERSE_PROXY = !PUB_REVERSE_PROXY_HOST.is_empty();
    }
    TCP_KEEPALIVE
        .init(ToDuration::parse_from_env("TCP_KEEPALIVE"));
    TCP_KEEPALIVE_INTERVAL
        .init(ToDuration::parse_from_env("TCP_KEEPALIVE_INTERVAL"));
    TCP_KEEPALIVE_RETRIES
        .init(ToCount::parse_from_env("TCP_KEEPALIVE_RETRIES"));
    HTTP2_ADAPTIVE_WINDOW.init(parse_from_env("HTTP2_ADAPTIVE_WINDOW", true));
    HTTP2_KEEP_ALIVE_INTERVAL
        .init(ToDuration::parse_from_env("HTTP2_KEEP_ALIVE_INTERVAL"));
    HTTP2_KEEP_ALIVE_TIMEOUT
        .init(ToDuration::parse_from_env("HTTP2_KEEP_ALIVE_TIMEOUT"));
    HTTP2_KEEP_ALIVE_WHILE_IDLE.init(parse_from_env("HTTP2_KEEP_ALIVE_WHILE_IDLE", true));
    SERVICE_TIMEOUT
        .init(parse_from_env("SERVICE_TIMEOUT", DEFAULT_SERVICE_TIMEOUT).min(MAX_SERVICE_TIMEOUT));
    REAL_USAGE.init(parse_from_env("REAL_USAGE", true));
}

pub static GENERAL_TIMEZONE: LazyLock<chrono_tz::Tz> = LazyLock::new(|| {
    use std::str::FromStr as _;
    let tz = parse_from_env("GENERAL_TIMEZONE", EMPTY_STRING);
    if tz.is_empty() {
        __eprintln!(
            "未配置时区，请在环境变量GENERAL_TIMEZONE中设置，格式如'Asia/Shanghai'\n将使用默认时区: Asia/Shanghai"
        );
        chrono_tz::Tz::Asia__Shanghai
    } else {
        match chrono_tz::Tz::from_str(&tz) {
            Ok(tz) => tz,
            Err(e) => {
                eprintln!("无法解析时区 '{tz}': {e}\n将使用默认时区: Asia/Shanghai");
                chrono_tz::Tz::Asia__Shanghai
            }
        }
    }
});

pub static GENERAL_GCPP_HOST: LazyLock<GcppHost> = LazyLock::new(|| {
    let gcpp_host = parse_from_env("GENERAL_GCPP_HOST", EMPTY_STRING);
    if gcpp_host.is_empty() {
        __eprintln!(
            "未配置默认代码补全区域，请在环境变量GENERAL_GCPP_HOST中设置，格式如'Asia'\n将使用默认区域: Asia"
        );
        GcppHost::Asia
    } else {
        match GcppHost::from_str(&gcpp_host) {
            Some(gcpp_host) => gcpp_host,
            None => {
                eprintln!("无法解析区域 '{gcpp_host}'\n将使用默认区域: Asia");
                GcppHost::Asia
            }
        }
    }
});

pub static PRI_REVERSE_PROXY_HOST: ManuallyInit<Cow<'static, str>> = ManuallyInit::new();
pub static PUB_REVERSE_PROXY_HOST: ManuallyInit<Cow<'static, str>> = ManuallyInit::new();

pub fn pri_reverse_proxy_host() -> http::header::HeaderValue {
    unsafe { HeaderValue::from_bytes(PRI_REVERSE_PROXY_HOST.get().as_bytes()).into() }
}
pub fn pub_reverse_proxy_host() -> http::header::HeaderValue {
    unsafe { HeaderValue::from_bytes(PUB_REVERSE_PROXY_HOST.get().as_bytes()).into() }
}

const DEFAULT_KEY_PREFIX: &str = "sk-";
pub static KEY_PREFIX: ManuallyInit<Cow<'static, str>> = ManuallyInit::new();

pub static mut USE_PRI_REVERSE_PROXY: bool = false;
pub static mut USE_PUB_REVERSE_PROXY: bool = false;

macro_rules! def_cursor_api_url {
    (
        init_fn: $init_fn:ident,
        $(
            $group_name:ident => {
                host: $api_host:ident,
                apis: [
                    $( $name:ident => $path:expr ),+ $(,)?
                ]
            }
        ),+ $(,)?
    ) => {
        // 为每个API生成静态变量和getter函数
        $(
            $(
                paste::paste! {
                    static [<PRI_ $name:upper>]: ManuallyInit<Url> = ManuallyInit::new();
                    static [<PUB_ $name:upper>]: ManuallyInit<Url> = ManuallyInit::new();

                    #[inline(always)]
                    #[doc = $path]
                    pub fn $name(use_pri: bool) -> &'static Url {
                        if use_pri {
                            [<PRI_ $name:upper>].get()
                        } else {
                            [<PUB_ $name:upper>].get()
                        }
                    }
                }
            )+
        )+

        // 生成统一的初始化函数
        pub fn $init_fn() {
            $(
                $(
                    paste::paste! {
                        // 初始化私有URL
                        {
                            let host = if unsafe { USE_PRI_REVERSE_PROXY } {
                                &PRI_REVERSE_PROXY_HOST
                            } else {
                                $api_host
                            };
                            let mut url = String::with_capacity(HTTPS_PREFIX.len() + host.len() + $path.len());
                            url.push_str(HTTPS_PREFIX);
                            url.push_str(host);
                            url.push_str($path);
                            [<PRI_ $name:upper>].init(unsafe { Url::parse(&url).unwrap_unchecked() });
                        }

                        // 初始化公共URL
                        {
                            let host = if unsafe { USE_PUB_REVERSE_PROXY } {
                                &PUB_REVERSE_PROXY_HOST
                            } else {
                                $api_host
                            };
                            let mut url = String::with_capacity(HTTPS_PREFIX.len() + host.len() + $path.len());
                            url.push_str(HTTPS_PREFIX);
                            url.push_str(host);
                            url.push_str($path);
                            [<PUB_ $name:upper>].init(unsafe { Url::parse(&url).unwrap_unchecked() });
                        }
                    }
                )+
            )+
        }
    };
}

// 一次性定义所有API
def_cursor_api_url! {
    init_fn: init_all_cursor_urls,

    // API2 HOST 相关API
    api2_group => {
        host: CURSOR_API2_HOST,
        apis: [
            chat_url => "/aiserver.v1.ChatService/StreamUnifiedChatWithTools",
            chat_models_url => "/aiserver.v1.AiService/AvailableModels",
            stripe_url => "/auth/full_stripe_profile",
            token_poll_url => "/auth/poll",
            token_refresh_url => "/oauth/token",
            server_config_url => "/aiserver.v1.ServerConfigService/GetServerConfig",
            dry_chat_url => "/aiserver.v1.ChatService/GetPromptDryRun",
        ]
    },

    // CURSOR HOST 相关API
    cursor_group => {
        host: CURSOR_HOST,
        apis: [
            usage_api_url => "/api/usage-summary",
            user_api_url => "/api/dashboard/get-me",
            token_upgrade_url => "/api/auth/loginDeepCallbackControl",
            // teams_url => "/api/dashboard/teams",
            // aggregated_usage_events_url => "/api/dashboard/get-aggregated-usage-events",
            filtered_usage_events_url => "/api/dashboard/get-filtered-usage-events",
            sessions_url => "/api/auth/sessions",
            is_on_new_pricing_url => "/api/dashboard/is-on-new-pricing",
            get_privacy_mode_url => "/api/dashboard/get-user-privacy-mode",
        ]
    },

    // API4 HOST 相关API
    api4_group => {
        host: CURSOR_API4_HOST,
        apis: [
            cpp_config_url => "/aiserver.v1.AiService/CppConfig",
        ]
    },

    // API2 HOST CPP相关API
    api2_cpp_group => {
        host: CURSOR_API2_HOST,
        apis: [
            cpp_models_url => "/aiserver.v1.CppService/AvailableModels",
        ]
    },

    // GCPP ASIA HOST 相关API
    gcpp_asia_group => {
        host: CURSOR_GCPP_ASIA_HOST,
        apis: [
            asia_upload_file_url => "/aiserver.v1.FileSyncService/FSUploadFile",
            asia_sync_file_url => "/aiserver.v1.FileSyncService/FSSyncFile",
            asia_stream_cpp_url => "/aiserver.v1.AiService/StreamCpp",
            // asia_next_cursor_prediction_url => "/aiserver.v1.AiService/StreamNextCursorPrediction",
        ]
    },

    // GCPP EU HOST 相关API
    gcpp_eu_group => {
        host: CURSOR_GCPP_EU_HOST,
        apis: [
            eu_upload_file_url => "/aiserver.v1.FileSyncService/FSUploadFile",
            eu_sync_file_url => "/aiserver.v1.FileSyncService/FSSyncFile",
            eu_stream_cpp_url => "/aiserver.v1.AiService/StreamCpp",
            // eu_next_cursor_prediction_url => "/aiserver.v1.AiService/StreamNextCursorPrediction",
        ]
    },

    // GCPP US HOST 相关API
    gcpp_us_group => {
        host: CURSOR_GCPP_US_HOST,
        apis: [
            us_upload_file_url => "/aiserver.v1.FileSyncService/FSUploadFile",
            us_sync_file_url => "/aiserver.v1.FileSyncService/FSSyncFile",
            us_stream_cpp_url => "/aiserver.v1.AiService/StreamCpp",
            // us_next_cursor_prediction_url => "/aiserver.v1.AiService/StreamNextCursorPrediction",
        ]
    }
}

// TCP 和超时相关常量
const DEFAULT_TCP_KEEPALIVE: NonNegativeI16 = NonNegativeI16::new(60).unwrap();
const MAX_TCP_KEEPALIVE: NonNegativeI16 = NonNegativeI16::new(600).unwrap();
pub static TCP_KEEPALIVE: ManuallyInit<ToDuration<DEFAULT_TCP_KEEPALIVE, MAX_TCP_KEEPALIVE>> =
    ManuallyInit::new();

const DEFAULT_TCP_KEEPALIVE_INTERVAL: NonNegativeI16 = NonNegativeI16::new(15).unwrap();
const MAX_TCP_KEEPALIVE_INTERVAL: NonNegativeI16 = NonNegativeI16::new(600).unwrap();
pub static TCP_KEEPALIVE_INTERVAL: ManuallyInit<
    ToDuration<DEFAULT_TCP_KEEPALIVE_INTERVAL, MAX_TCP_KEEPALIVE_INTERVAL>,
> = ManuallyInit::new();

const DEFAULT_TCP_KEEPALIVE_RETRIES: NonNegativeI8 = NonNegativeI8::new(5).unwrap();
const MAX_TCP_KEEPALIVE_RETRIES: NonNegativeI8 = NonNegativeI8::new(20).unwrap();
pub static TCP_KEEPALIVE_RETRIES: ManuallyInit<
    ToCount<DEFAULT_TCP_KEEPALIVE_RETRIES, MAX_TCP_KEEPALIVE_RETRIES>,
> = ManuallyInit::new();

pub static HTTP2_ADAPTIVE_WINDOW: ManuallyInit<bool> = ManuallyInit::new();

const DEFAULT_HTTP2_KEEP_ALIVE_INTERVAL: NonNegativeI16 = NonNegativeI16::new(30).unwrap();
const MAX_HTTP2_KEEP_ALIVE_INTERVAL: NonNegativeI16 = NonNegativeI16::new(600).unwrap();
pub static HTTP2_KEEP_ALIVE_INTERVAL: ManuallyInit<
    ToDuration<DEFAULT_HTTP2_KEEP_ALIVE_INTERVAL, MAX_HTTP2_KEEP_ALIVE_INTERVAL>,
> = ManuallyInit::new();

const DEFAULT_HTTP2_KEEP_ALIVE_TIMEOUT: NonNegativeI16 = NonNegativeI16::new(20).unwrap();
const MAX_HTTP2_KEEP_ALIVE_TIMEOUT: NonNegativeI16 = NonNegativeI16::new(600).unwrap();
pub static HTTP2_KEEP_ALIVE_TIMEOUT: ManuallyInit<
    ToDuration<DEFAULT_HTTP2_KEEP_ALIVE_TIMEOUT, MAX_HTTP2_KEEP_ALIVE_TIMEOUT>,
> = ManuallyInit::new();

pub static HTTP2_KEEP_ALIVE_WHILE_IDLE: ManuallyInit<bool> = ManuallyInit::new();

const DEFAULT_SERVICE_TIMEOUT: u16 = 30;
const MAX_SERVICE_TIMEOUT: u16 = 600;
pub static SERVICE_TIMEOUT: ManuallyInit<u16> = ManuallyInit::new();

#[derive(Debug, Clone, Copy)]
pub struct ToDuration<const DEFAULT: NonNegativeI16, const MAX: NonNegativeI16>(
    Option<NonNegativeI16>,
);

impl<const DEFAULT: NonNegativeI16, const MAX: NonNegativeI16> ParseFromEnv
    for ToDuration<DEFAULT, MAX>
{
    fn parse_from_env(key: &str) -> Option<Self::Result> {
        let v = i32::parse_from_env(key)?;
        Some(if v < 0 {
            Self(None)
        } else {
            Self(NonNegativeI16::new(v.min(MAX.as_inner() as i32) as i16))
        })
    }
}

impl<const DEFAULT: NonNegativeI16, const MAX: NonNegativeI16> ToDuration<DEFAULT, MAX> {
    pub fn to_duration(self) -> Option<core::time::Duration> {
        self.0.map(|v| core::time::Duration::from_secs(v.as_inner() as _))
    }
    pub fn to_duration_or_default(self) -> core::time::Duration {
        self.to_duration()
            .unwrap_or(const { core::time::Duration::from_secs(DEFAULT.as_inner() as _) })
    }
    pub fn parse_from_env(key: &'static str) -> Self {
        <Self as ParseFromEnv>::parse_from_env(key).unwrap_or(Self(Some(DEFAULT)))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ToCount<const DEFAULT: NonNegativeI8, const MAX: NonNegativeI8>(Option<NonNegativeI8>);

impl<const DEFAULT: NonNegativeI8, const MAX: NonNegativeI8> ParseFromEnv
    for ToCount<DEFAULT, MAX>
{
    fn parse_from_env(key: &str) -> Option<Self::Result> {
        let v = i16::parse_from_env(key)?;
        Some(if v < 0 {
            Self(None)
        } else {
            Self(NonNegativeI8::new(v.min(MAX.as_inner() as i16) as i8))
        })
    }
}

impl<const DEFAULT: NonNegativeI8, const MAX: NonNegativeI8> ToCount<DEFAULT, MAX> {
    pub fn to_count(self) -> Option<u32> { self.0.map(|v| v.as_inner() as _) }
    // pub fn to_count_or_default(self) -> u32 {
    //     self.to_count().unwrap_or(const { DEFAULT.as_inner() as _ })
    // }
    pub fn parse_from_env(key: &'static str) -> Self {
        <Self as ParseFromEnv>::parse_from_env(key).unwrap_or(Self(Some(DEFAULT)))
    }
}

pub static REAL_USAGE: ManuallyInit<bool> = ManuallyInit::new();

// pub static TOKEN_VALIDITY_RANGE: ManuallyInit<TokenValidityRange> = ManuallyInit::new(|| {
//     let short = if let Ok(Ok(validity)) = std::env::var("TOKEN_SHORT_VALIDITY")
//         .as_deref()
//         .map(ValidityRange::from_str)
//     {
//         validity
//     } else {
//         ValidityRange::new(5400, 5400)
//     };
//     let long = if let Ok(Ok(validity)) = std::env::var("TOKEN_LONG_VALIDITY")
//         .as_deref()
//         .map(ValidityRange::from_str)
//     {
//         validity
//     } else {
//         ValidityRange::new(5184000, 5184000)
//     };
//     TokenValidityRange::new(short, long)
// });
