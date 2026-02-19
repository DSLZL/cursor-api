use minicbor::{CborLen, Decode, Encode};

/// 动态配置的 API KEY
#[derive(Clone, PartialEq, Encode, Decode, CborLen)]
pub struct ConfiguredKey {
    /// 认证令牌（必需）
    #[n(0)]
    pub token_info: Option<configured_key::TokenInfo>,
    /// 密码SHA256哈希值
    #[n(1)]
    pub secret: Option<[u8; 32]>,
    /// 是否禁用图片处理能力
    #[n(2)]
    pub disable_vision: Option<bool>,
    /// 是否启用慢速池
    #[n(3)]
    pub enable_slow_pool: Option<bool>,
    /// 包含网络引用
    #[n(4)]
    pub include_web_references: Option<bool>,
    /// 使用量检查模型规则
    #[n(5)]
    pub usage_check_models: Option<configured_key::UsageCheckModel>,
}

pub mod configured_key {
    use super::*;

    /// 认证令牌信息
    #[derive(Clone, PartialEq, Encode, Decode, CborLen)]
    pub struct TokenInfo {
        /// 令牌（必需）
        #[n(0)]
        pub token: token_info::Token,
        /// 校验和(\[u8; 64\])
        #[n(1)]
        pub checksum: [u8; 64],
        /// 客户端标识(\[u8; 32\])
        #[n(2)]
        pub client_key: [u8; 32],
        /// 配置版本
        #[n(3)]
        pub config_version: Option<[u8; 16]>,
        /// 会话ID
        #[n(4)]
        pub session_id: [u8; 16],
        /// 代理名称
        #[n(5)]
        pub proxy_name: Option<String>,
        /// 时区
        #[n(6)]
        pub timezone: Option<String>,
        /// 代码补全
        #[n(7)]
        pub gcpp_host: Option<u8>,
    }

    pub mod token_info {
        use super::*;

        #[derive(Clone, PartialEq, Encode, Decode, CborLen)]
        pub struct Token {
            #[n(0)]
            pub provider: String,
            /// 用户ID(\[u8; 16\])
            #[n(1)]
            pub sub_id: [u8; 16],
            /// 随机字符串(\[u8; 8\])
            #[n(2)]
            pub randomness: [u8; 8],
            /// 生成时间（Unix 时间戳）
            #[n(3)]
            pub start: i64,
            /// 过期时间（Unix 时间戳）
            #[n(4)]
            pub end: i64,
            /// 签名(\[u8; 32\])
            #[n(5)]
            pub signature: [u8; 32],
            /// 是否为会话令牌
            #[n(6)]
            pub is_session: bool,
            /// WorkOS会话ID(\[u8; 16\])
            #[n(7)]
            pub workos_session_id: [u8; 16],
        }
    }

    /// 使用量检查模型规则
    #[derive(Clone, PartialEq, Encode, Decode, CborLen)]
    pub struct UsageCheckModel {
        /// 检查类型
        #[n(0)]
        pub r#type: usage_check_model::Type,
        /// 模型 ID 列表，当 type 为 TYPE_CUSTOM 时生效
        #[n(1)]
        pub model_ids: Vec<String>,
    }

    pub mod usage_check_model {
        use super::*;

        /// 检查类型
        #[derive(::serde::Deserialize, Clone, Copy, PartialEq, Encode, Decode, CborLen)]
        #[serde(rename_all = "lowercase")]
        #[cbor(index_only)]
        pub enum Type {
            /// 未指定
            #[n(0)]
            Default = 0,
            /// 禁用
            #[n(1)]
            Disabled = 1,
            /// 全部
            #[n(2)]
            All = 2,
            /// 自定义列表
            #[n(3)]
            Custom = 3,
        }
    }
}
