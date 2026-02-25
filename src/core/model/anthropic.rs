use super::{IndexMap, Role};
use crate::{
    app::constant::{ERROR, TYPE},
    common::utils::const_string::const_string,
};
use alloc::borrow::Cow;
use byte_str::ByteStr;
use core::fmt;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, MapAccess, Visitor},
    ser::SerializeStruct,
};

#[derive(Debug, Deserialize)]
pub struct MessageCreateParams {
    pub model: String,
    pub messages: Vec<MessageParam>,
    #[allow(dead_code)]
    pub max_tokens: usize,
    // #[serde(default)]
    // pub mcp_servers: Vec<McpServer>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub system: Option<SystemContent>,
    // #[serde(default)]
    // pub thinking: Option<ThinkingConfig>,
    #[serde(default)]
    pub tools: Vec<Tool>,
}

impl MessageCreateParams {
    #[inline(always)]
    pub fn strip(self) -> ((Vec<MessageParam>, Option<SystemContent>), Vec<Tool>) {
        ((self.messages, self.system), self.tools)
    }
}

#[derive(Debug, Deserialize)]
pub struct MessageParam {
    #[serde(deserialize_with = "deserialize_anthropic_role")]
    pub role: Role,
    pub content: MessageContent,
}

fn deserialize_anthropic_role<'de, D>(deserializer: D) -> Result<Role, D::Error>
where D: ::serde::Deserializer<'de> {
    let s = <String as ::serde::Deserialize>::deserialize(deserializer)?;
    match s.as_str() {
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        other => Err(serde::de::Error::custom(format_args!(
            "Invalid Anthropic role '{other}': only 'user' and 'assistant' are supported"
        ))),
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    String(String),
    Array(Vec<ContentBlockParam>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockParam {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    Document {
        source: DocumentSource,
    },
    Thinking {
        thinking: String,
        #[serde(default)]
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: ByteStr,
        name: ByteStr,
        input: IndexMap<String, serde_json::Value>,
    },
    ToolResult {
        tool_use_id: ByteStr,
        content: Option<ToolResultContent>,
        #[serde(default)]
        is_error: bool,
    },
    #[serde(skip)]
    Taked,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    String(String),
    Array(Vec<ToolResultContentBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContentBlock {
    Text { text: String },
    Image { source: ImageSource },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64(ImageSourceBase64),
    Url { url: String },
}

#[derive(Debug, Deserialize)]
pub struct ImageSourceBase64 {
    pub data: String,
    pub media_type: MediaType,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum MediaType {
    ImagePng,
    ImageJpeg,
    ImageGif,
    ImageWebp,
    // ApplicationPdf,
    // TextPlain,
}

impl MediaType {
    const IMAGE_JPEG: &str = "image/jpeg";
    const IMAGE_PNG: &str = "image/png";
    const IMAGE_GIF: &str = "image/gif";
    const IMAGE_WEBP: &str = "image/webp";
    // const APPLICATION_PDF: &str = "application/pdf";
    // const TEXT_PLAIN: &str = "text/plain";

    pub fn from_mime(mime: &str) -> Option<Self> {
        match mime {
            Self::IMAGE_JPEG => Some(Self::ImageJpeg),
            Self::IMAGE_PNG => Some(Self::ImagePng),
            Self::IMAGE_GIF => Some(Self::ImageGif),
            Self::IMAGE_WEBP => Some(Self::ImageWebp),
            // Self::APPLICATION_PDF => Some(Self::ApplicationPdf),
            // Self::TEXT_PLAIN => Some(Self::TextPlain),
            _ => None,
        }
    }

    pub fn as_mime(self) -> &'static str {
        match self {
            Self::ImageJpeg => Self::IMAGE_JPEG,
            Self::ImagePng => Self::IMAGE_PNG,
            Self::ImageGif => Self::IMAGE_GIF,
            Self::ImageWebp => Self::IMAGE_WEBP,
            // Self::ApplicationPdf => Self::APPLICATION_PDF,
            // Self::TextPlain => Self::TEXT_PLAIN,
        }
    }
}

impl<'de> Deserialize<'de> for MediaType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        let mime = String::deserialize(deserializer)?;
        Self::from_mime(&mime)
            .ok_or_else(|| serde::de::Error::custom(format_args!("Unsupported media type: {mime}")))
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    Base64 {
        data: String,
        #[allow(dead_code)]
        media_type: ApplicationPdf,
    },
    Url {
        url: String,
    },
}

const_string!(ApplicationPdf = "application/pdf");

// #[derive(Deserialize)]
// pub struct McpServer {
//     pub name: String,
//     #[serde(rename = "type")]
//     pub server_type: McpServerType,
//     pub url: String,
//     pub authorization_token: Option<String>,
//     #[serde(default)]
//     pub tool_configuration: Option<ToolConfiguration>,
// }

// #[derive(Deserialize)]
// #[serde(rename_all = "lowercase")]
// pub enum McpServerType {
//     Url,
// }

// #[derive(Deserialize)]
// pub struct ToolConfiguration {
//     #[serde(default)]
//     pub allowed_tools: Vec<String>,
//     #[serde(default)]
//     pub enabled: Option<bool>,
// }

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum SystemContent {
    String(String),
    Array(Vec<TextBlockParam>),
}

#[derive(Debug)]
pub struct TextBlockParam {
    pub text: String,
}

impl<'de> Deserialize<'de> for TextBlockParam {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        struct TextBlockParamVisitor;

        impl<'de> Visitor<'de> for TextBlockParamVisitor {
            type Value = TextBlockParam;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a TextBlockParam with type 'text'")
            }

            fn visit_map<V>(self, mut map: V) -> Result<TextBlockParam, V::Error>
            where V: MapAccess<'de> {
                let mut type_ = None;
                let mut text = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        TYPE => {
                            if type_.is_some() {
                                return Err(de::Error::duplicate_field(TYPE));
                            }
                            let value: String = map.next_value()?;
                            if value != "text" {
                                return Err(de::Error::custom(format_args!(
                                    "expected type to be 'text', found '{value}'"
                                )));
                            }
                            type_ = Some(value);
                        }
                        "text" => {
                            if text.is_some() {
                                return Err(de::Error::duplicate_field("text"));
                            }
                            text = Some(map.next_value()?);
                        }
                        _ => {
                            // 忽略未知字段
                            let _: serde::de::IgnoredAny = map.next_value()?;
                        }
                    }
                }

                let text =
                    text.ok_or_else(|| de::Error::custom(format_args!("missing field 'text'")))?;

                Ok(TextBlockParam { text })
            }
        }

        deserializer.deserialize_map(TextBlockParamVisitor)
    }
}

// #[derive(Debug, Deserialize)]
// #[serde(tag = "type", rename_all = "lowercase")]
// pub enum ThinkingConfig {
//     Enabled {
//         #[allow(dead_code)]
//         budget_tokens: i64,
//     },
//     Disabled,
// }

// #[derive(Debug, Serialize, Deserialize, Clone)]
// #[serde(rename_all = "camelCase")]
// pub struct ToolInputSchema {
//     pub r#type: ToolInputSchemaType,
//     #[serde(default)]
//     pub properties: alloc::collections::BTreeMap<String, PropertySchema>,
//     #[serde(default)]
//     pub required: Vec<String>,
//     #[serde(default)]
//     pub additional_properties: bool,
//     #[serde(rename = "$schema", skip_serializing_if = "Option::is_none")]
//     pub schema: Option<String>,
// }

// #[derive(Debug, Serialize, Deserialize, Clone, Copy)]
// #[serde(rename_all = "lowercase")]
// pub enum ToolInputSchemaType {
//     Object,
// }

// #[derive(Debug, Serialize, Deserialize, Clone)]
// pub struct PropertySchema {
//     pub r#type: String,
//     #[serde(default, skip_serializing_if = "Option::is_none")]
//     pub description: Option<String>,
// }

#[derive(Debug, Deserialize, Clone)]
pub struct Tool {
    pub input_schema: IndexMap<String, serde_json::Value>,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Serialize, Default)]
pub struct Usage {
    pub input_tokens: i32,
    pub output_tokens: i32,
    #[serde(default, skip_serializing_if = "i32_is_zero")]
    pub cache_creation_input_tokens: i32,
    #[serde(default, skip_serializing_if = "i32_is_zero")]
    pub cache_read_input_tokens: i32,
}

#[derive(Serialize, Default)]
pub struct MessageDeltaUsage {
    #[serde(skip_serializing_if = "i32_is_zero")]
    pub input_tokens: i32,
    pub output_tokens: i32,
    #[serde(skip_serializing_if = "i32_is_zero")]
    pub cache_creation_input_tokens: i32,
    #[serde(skip_serializing_if = "i32_is_zero")]
    pub cache_read_input_tokens: i32,
}

#[inline]
fn i32_is_zero(i: &i32) -> bool { *i == 0 }

pub struct Message<'a> {
    pub content: Vec<ContentBlock>,
    pub usage: Usage,
    pub id: &'a str,
    pub model: &'static str,
    pub stop_reason: Option<StopReason>,
}

impl Serialize for Message<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        let mut state = serializer.serialize_struct("Message", 8)?;
        state.serialize_field("id", self.id)?;
        state.serialize_field(TYPE, "message")?;
        state.serialize_field("role", "assistant")?;
        state.serialize_field("content", &self.content)?;
        state.serialize_field("model", self.model)?;
        state.serialize_field("stop_reason", &self.stop_reason)?;
        state.serialize_field("stop_sequence", &None::<bool>)?;
        state.serialize_field("usage", &self.usage)?;
        state.end()
    }
}

pub struct MessageDelta {
    pub stop_reason: StopReason,
}

impl Serialize for MessageDelta {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        let mut state = serializer.serialize_struct("MessageDelta", 2)?;
        state.serialize_field("stop_reason", &self.stop_reason)?;
        state.serialize_field("stop_sequence", &None::<bool>)?;
        state.end()
    }
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn = 0,
    // MaxTokens = 1,
    // StopSequence = 2,
    ToolUse = 3,
    // PauseTurn = 4,
    // Refusal = 5,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: ByteStr,
        name: ByteStr,
        input: IndexMap<String, serde_json::Value>,
    },
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RawContentBlockDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RawMessageStreamEvent<'a> {
    MessageStart { message: Message<'a> },
    MessageDelta { delta: MessageDelta, usage: MessageDeltaUsage },
    // MessageStop,
    ContentBlockStart { index: u32, content_block: ContentBlock },
    ContentBlockDelta { index: u32, delta: RawContentBlockDelta },
    ContentBlockStop { index: u32 },
    Ping,
    Error { error: AnthropicErrorInner },
}

impl<'a> RawMessageStreamEvent<'a> {
    #[inline(always)]
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::MessageStart { .. } => "message_start",
            Self::MessageDelta { .. } => "message_delta",
            // Self::MessageStop => "message_stop",
            Self::ContentBlockStart { .. } => "content_block_start",
            Self::ContentBlockDelta { .. } => "content_block_delta",
            Self::ContentBlockStop { .. } => "content_block_stop",
            Self::Ping => "ping",
            Self::Error { .. } => ERROR,
        }
    }
}

mod private {
    use super::*;

    #[derive(Serialize)]
    pub struct ErrorDetail {
        pub r#type: &'static str,
        pub message: Cow<'static, str>,
    }

    impl ErrorDetail {
        #[inline(always)]
        pub const fn wrapped(self) -> AnthropicError { AnthropicError(self) }
    }
}

pub type AnthropicErrorInner = private::ErrorDetail;

#[repr(transparent)]
pub struct AnthropicError(private::ErrorDetail);

impl Serialize for AnthropicError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        let mut state = serializer.serialize_struct("AnthropicError", 2)?;
        state.serialize_field(TYPE, ERROR)?;
        state.serialize_field(ERROR, &self.0)?;
        state.end()
    }
}

#[derive(Serialize)]
pub struct MessagesCountTokens {
    pub input_tokens: i32,
}
