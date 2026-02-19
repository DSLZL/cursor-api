use crate::{
    AppConfig,
    app::{
        lazy::KEY_PREFIX,
        model::{Randomness, RawToken, SessionId, Subject, TokenDuration, UserId, dynamic_key},
    },
    common::utils::from_base64,
};

// include!(concat!(env!("OUT_DIR"), "/key.rs"));
include!("config/key.rs");

impl ConfiguredKey {
    pub fn move_to_config_builder(&mut self, config: &mut KeyConfigBuilder) {
        if self.usage_check_models.is_some() {
            config.usage_check_models = self.usage_check_models.take();
        }
        if self.disable_vision.is_some() {
            config.disable_vision = self.disable_vision.take();
        }
        if self.enable_slow_pool.is_some() {
            config.enable_slow_pool = self.enable_slow_pool.take();
        }
        if self.include_web_references.is_some() {
            config.include_web_references = self.include_web_references.take();
        }
    }

    pub fn into_tuple(self) -> Option<(configured_key::TokenInfo, [u8; 32])> {
        self.token_info.zip(self.secret)
    }
}

impl configured_key::token_info::Token {
    #[inline]
    pub fn from_raw(raw: RawToken) -> Self {
        Self {
            provider: raw.subject.provider.to_string(),
            signature: raw.signature,
            sub_id: raw.subject.id.to_bytes(),
            randomness: raw.randomness.to_bytes(),
            start: raw.duration.start,
            end: raw.duration.end,
            is_session: raw.is_session,
            workos_session_id: raw.workos_session_id.to_bytes(),
        }
    }

    #[inline]
    pub fn validate(self, hash: [u8; 32]) -> Option<RawToken> {
        let raw = RawToken {
            subject: Subject {
                provider: self.provider.parse().ok()?,
                id: UserId::from_bytes(self.sub_id),
            },
            randomness: Randomness::from_bytes(self.randomness),
            signature: self.signature,
            duration: TokenDuration { start: self.start, end: self.end },
            is_session: self.is_session,
            workos_session_id: SessionId::from_bytes(self.workos_session_id),
        };
        if dynamic_key::get_hash(&raw) != hash {
            return None;
        }
        Some(raw)
    }
}

pub fn parse_dynamic_token(auth_token: &str) -> Option<ConfiguredKey> {
    auth_token.strip_prefix(&**KEY_PREFIX).and_then(from_base64).and_then(|decoded_bytes| {
        let mut decoder = ::minicbor::Decoder::new(&decoded_bytes);
        decoder.decode().ok()
    })
}

#[derive(Clone)]
pub struct KeyConfig {
    pub usage_check_models: Option<configured_key::UsageCheckModel>,
    pub disable_vision: bool,
    pub enable_slow_pool: bool,
    pub include_web_references: bool,
}

#[derive(Clone)]
pub struct KeyConfigBuilder {
    pub usage_check_models: Option<configured_key::UsageCheckModel>,
    pub disable_vision: Option<bool>,
    pub enable_slow_pool: Option<bool>,
    pub include_web_references: Option<bool>,
}

impl KeyConfigBuilder {
    pub const fn new() -> Self {
        Self {
            usage_check_models: None,
            disable_vision: None,
            enable_slow_pool: None,
            include_web_references: None,
        }
    }

    pub fn with_global(self) -> KeyConfig {
        let Self { usage_check_models, disable_vision, enable_slow_pool, include_web_references } =
            self;
        KeyConfig {
            usage_check_models,
            disable_vision: disable_vision.unwrap_or_else(|| AppConfig::vision_ability().is_none()),
            enable_slow_pool: enable_slow_pool.unwrap_or_else(AppConfig::is_slow_pool_enabled),
            include_web_references: include_web_references
                .unwrap_or_else(AppConfig::is_web_references_included),
        }
    }
}
