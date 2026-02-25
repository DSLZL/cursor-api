use super::{IndexMap, Role};
use crate::{
    app::constant::{ERROR, TYPE},
    common::{
        model::tri::Tri,
        utils::{const_string::const_string, option_as_array},
    },
};
use alloc::borrow::Cow;
use byte_str::ByteStr;
use serde::{Deserialize, Deserializer, Serialize, Serializer, ser::SerializeStruct as _};

#[derive(Deserialize)]
pub struct ChatCompletionCreateParams {
    pub model: String,
    pub messages: Vec<ChatCompletionMessageParam>,
    // #[serde(default)]
    // pub reasoning_effort: ReasoningEffort,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: ChatCompletionStreamOptions,
    #[serde(default)]
    pub tools: Vec<ChatCompletionTool>,
}

impl ChatCompletionCreateParams {
    #[inline(always)]
    pub fn strip(
        self,
    ) -> (Vec<ChatCompletionMessageParam>, Vec<ChatCompletionTool>, bool, ChatCompletionStreamOptions)
    {
        (self.messages, self.tools, self.stream, self.stream_options)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum ChatCompletionMessageParam {
    #[serde(rename = "system", alias = "developer")]
    System { content: ChatCompletionContentText },
    #[serde(rename = "user")]
    User { content: ChatCompletionContent },
    #[serde(rename = "assistant")]
    Assistant {
        #[serde(default)]
        content: ChatCompletionContentText,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ChatCompletionMessageToolCall>,
    },
    #[serde(rename = "tool")]
    Tool { content: ChatCompletionContentText, tool_call_id: ByteStr },
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatCompletionContent {
    String(String),
    Array(Vec<ChatCompletionContentPart>),
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatCompletionContentText {
    String(String),
    #[serde(with = "chat_completion_content_part_text")]
    Array(Vec<String>),
}

impl const Default for ChatCompletionContentText {
    #[inline]
    fn default() -> Self { Self::String(String::new()) }
}

impl ChatCompletionContentText {
    pub fn text(self) -> String {
        match self {
            Self::String(string) => string,
            Self::Array(contents) => contents.join("\n"),
        }
    }
}

mod chat_completion_content_part_text {
    use super::{ChatCompletionContentPartText, Deserialize, Deserializer, Serialize, Serializer};
    pub(super) fn serialize<S>(_self: &Vec<String>, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        // SAFETY: ChatCompletionContentPartText and String have the same layout
        let this: &Vec<ChatCompletionContentPartText> = unsafe { core::mem::transmute(_self) };
        this.serialize(serializer)
    }
    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
    where D: Deserializer<'de> {
        let value = Vec::<ChatCompletionContentPartText>::deserialize(deserializer)?;
        // SAFETY: String and ChatCompletionContentPartText have the same layout
        Ok(unsafe { core::mem::transmute(value) })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[repr(transparent)]
pub enum ChatCompletionContentPartText {
    Text { text: String },
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionMessageToolCall {
    Function {
        id: ByteStr,
        function: chat_completion_message_tool_call::Function,
    },
    #[serde(skip)]
    Taked,
}

pub mod chat_completion_message_tool_call {
    use super::{ByteStr, Deserialize, Serialize};
    #[derive(Serialize, Deserialize)]
    pub struct Function {
        pub arguments: String,
        pub name: ByteStr,
    }
}

// #[derive(Deserialize, Default)]
// #[serde(rename_all = "lowercase")]
// pub enum ReasoningEffort {
//     Minimal,
//     Low,
//     #[default]
//     Medium,
//     High,
// }

#[derive(Deserialize, Default)]
pub struct ChatCompletionStreamOptions {
    pub include_usage: bool,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionTool {
    Function { function: FunctionDefinition },
}

#[derive(Debug, Deserialize, Clone)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: IndexMap<String, serde_json::Value>,
}

#[derive(Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Serialize)]
pub struct ChatCompletionMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    pub role: Assistant,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ChatCompletionMessageToolCall>,
}

#[derive(Serialize)]
pub struct ChatCompletion<'a> {
    pub id: &'a str,
    #[serde(serialize_with = "option_as_array::serialize")]
    pub choices: Option<chat_completion::Choice>,
    pub created: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<&'static str>,
    pub object: ObjectChatCompletion,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

pub mod chat_completion {
    use super::{ChatCompletionMessage, FinishReason, Serialize};
    #[derive(Serialize)]
    pub struct Choice {
        pub finish_reason: FinishReason,
        pub index: i32,
        pub logprobs: (),
        pub message: ChatCompletionMessage,
    }
}

#[derive(Serialize)]
pub struct ChatCompletionChunk<'a> {
    pub id: &'a str,
    #[serde(serialize_with = "option_as_array::serialize")]
    pub choices: Option<chat_completion_chunk::Choice>,
    pub created: i64,
    pub model: &'static str,
    pub object: ObjectChatCompletionChunk,
    #[serde(skip_serializing_if = "Tri::is_undefined")]
    pub usage: Tri<Usage>,
}

pub mod chat_completion_chunk {
    use super::{FinishReason, Serialize};
    #[derive(Serialize)]
    pub struct Choice {
        #[serde(serialize_with = "serialize_zero")]
        pub index: (),
        pub delta: Option<choice::Delta>,
        pub logprobs: (),
        pub finish_reason: Option<FinishReason>,
    }
    fn serialize_zero<S>(_: &(), serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        serializer.serialize_u32(0)
    }
    pub mod choice {
        use super::{
            super::{Role, option_as_array},
            Serialize,
        };
        #[derive(Serialize)]
        pub struct Delta {
            #[serde(skip_serializing_if = "Option::is_none")]
            pub content: Option<alloc::borrow::Cow<'static, str>>,
            #[serde(skip_serializing_if = "Option::is_none")]
            pub role: Option<Role>,
            #[serde(with = "option_as_array", skip_serializing_if = "Option::is_none")]
            pub tool_calls: Option<Box<delta::ToolCall>>,
        }
        pub mod delta {
            use super::{super::super::ByteStr, Serialize};
            #[derive(Serialize)]
            pub struct ToolCall {
                pub index: u32,
                #[serde(skip_serializing_if = "Option::is_none")]
                pub id: Option<ByteStr>,
                #[serde(skip_serializing_if = "Option::is_none")]
                pub function: Option<tool_call::Function>,
            }
            pub mod tool_call {
                use super::{ByteStr, Serialize};
                use crate::core::model::openai::EmptyString;
                #[derive(Serialize)]
                pub enum Function {
                    Start { name: ByteStr, arguments: EmptyString },
                    Partial { arguments: String },
                }
            }
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    // Length,
    ToolCalls,
    // ContentFilter,
    // FunctionCall,
}

#[derive(Default)]
pub struct PromptTokensDetails {
    pub cached_tokens: i32,
    // pub audio_tokens: i32,
}

impl PromptTokensDetails {
    #[inline]
    fn is_zero(&self) -> bool { self.cached_tokens == 0 }
}

impl Serialize for PromptTokensDetails {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        let mut state = serializer.serialize_struct("prompt_tokens_details", 1)?;
        state.serialize_field("cached_tokens", &self.cached_tokens)?;
        state.end()
    }
}

// #[derive(Default)]
// pub struct CompletionTokensDetails {
//   pub reasoning_tokens: i32,
//   // pub audio_tokens: i32,
//   // pub accepted_prediction_tokens: i32,
//   // pub rejected_prediction_tokens: i32,
// }

// impl Serialize for CompletionTokensDetails {
//   #[inline]
//   fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//   where
//     S: Serializer,
//   {
//     let mut state = serializer.serialize_struct("completion_tokens_details", 1)?;
//     state.serialize_field("reasoning_tokens", &self.reasoning_tokens)?;
//     state.end()
//   }
// }

#[derive(Serialize, Default)]
pub struct Usage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
    #[serde(default, skip_serializing_if = "PromptTokensDetails::is_zero")]
    pub prompt_tokens_details: PromptTokensDetails,
    // pub completion_tokens_details: CompletionTokensDetails,
}

const_string!(Assistant = "assistant");
const_string!(ObjectChatCompletion = "chat.completion");
const_string!(ObjectChatCompletionChunk = "chat.completion.chunk");
const_string!(EmptyString = "");

mod private {
    use super::*;

    pub struct ErrorDetail {
        pub code: Option<Cow<'static, str>>,
        pub message: Cow<'static, str>,
    }

    impl ErrorDetail {
        #[inline(always)]
        pub const fn wrapped(self) -> OpenAiError { OpenAiError(self) }
    }
}

pub type OpenAiErrorInner = private::ErrorDetail;

#[repr(transparent)]
pub struct OpenAiError(private::ErrorDetail);

impl Serialize for OpenAiError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        let mut state = serializer.serialize_struct("OpenAIError", 4)?;
        state.serialize_field(TYPE, ERROR)?;
        state.serialize_field("code", &self.0.code)?;
        state.serialize_field("message", &self.0.message)?;
        state.serialize_field("param", &None::<bool>)?;
        state.end()
    }
}
