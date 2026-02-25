use super::{
    AdapterError, BaseUuid, Messages, NEWLINE, extract_external_links, extract_web_references_info,
    traits::*,
    utils::{ToolId, ToolName, ToolResultBuilder},
};
use crate::{
    app::model::DEFAULT_INSTRUCTIONS,
    common::utils::proto_encode::encode_message_framed,
    core::{
        aiserver::v1::{
            ClientSideToolV2, ClientSideToolV2Call, ClientSideToolV2Result, ComposerExternalLink,
            ConversationMessage, EnvironmentInfo, McpParams, McpResult, conversation_message,
            mcp_params, stream_unified_chat_request,
        },
        model::{
            ExtModel, IndexMap,
            openai::{
                ChatCompletionContent, ChatCompletionContentPart, ChatCompletionContentText,
                ChatCompletionMessageParam, ChatCompletionMessageToolCall, ChatCompletionTool,
                FunctionDefinition, ImageUrl, chat_completion_message_tool_call::Function,
            },
        },
    },
};
use byte_str::ByteStr;
use uuid::Uuid;

struct Openai;

crate::define_typed_constants! {
    &'static str => {
        /// 支持的图片格式
        FORMAT_PNG = "png",
        FORMAT_JPEG = "jpeg",
        FORMAT_JPG = "jpg",
        FORMAT_WEBP = "webp",
        FORMAT_GIF = "gif",
        /// Data URL 前缀
        DATA_IMAGE_PREFIX = "data:image/",
        /// Base64 分隔符
        BASE64_SEPARATOR = ";base64,",
        /// 双换行符用于分隔指令
        DOUBLE_NEWLINE = "\n\n",
    }
}

impl ImageParams for ImageUrl {
    type Base64ImageParams = str;
    fn extract(&self) -> Result<&str, &str> {
        let url = &self.url;
        if let Some(url) = url.strip_prefix(DATA_IMAGE_PREFIX) { Ok(url) } else { Err(url) }
    }
}

impl ToolParam for FunctionDefinition {
    fn extract(self) -> (String, Option<String>, IndexMap<String, serde_json::Value>) {
        (self.name, self.description, self.parameters)
    }
}

impl ToolParam for ChatCompletionTool {
    fn extract(self) -> (String, Option<String>, IndexMap<String, serde_json::Value>) {
        let ChatCompletionTool::Function { function } = self;
        function.extract()
    }
}

impl ToolResult for ChatCompletionContentText {
    fn is_error(&self) -> bool {
        // 可能有副作用
        false
    }
    fn size_hint(&self) -> Option<usize> {
        Some(match self {
            ChatCompletionContentText::String(..) => 1,
            ChatCompletionContentText::Array(cs) => cs.len(),
        })
    }
    async fn add_to(self, builder: &mut ToolResultBuilder) -> Result<(), AdapterError> {
        match self {
            ChatCompletionContentText::String(text) => builder.add(text),
            ChatCompletionContentText::Array(cs) => cs.into_iter().for_each(|c| builder.add(c)),
        }
        Ok(())
    }
}

impl Adapter for Openai {
    type ImageParams = ImageUrl;
    type MessageParams = Vec<ChatCompletionMessageParam>;
    type ToolParam = ChatCompletionTool;
    type ToolResult = ChatCompletionContentText;
    fn _process_base64_image(url: &str) -> Result<(Vec<u8>, image::ImageFormat), AdapterError> {
        let (format, data) =
            url.split_once(BASE64_SEPARATOR).ok_or(AdapterError::Base64DecodeFailed)?;
        // 检查图片格式
        let format = match format {
            FORMAT_PNG => image::ImageFormat::Png,
            FORMAT_JPG | FORMAT_JPEG => image::ImageFormat::Jpeg,
            FORMAT_GIF => image::ImageFormat::Gif,
            FORMAT_WEBP => image::ImageFormat::WebP,
            _ => return Err(AdapterError::UnsupportedImageFormat),
        };
        let image_data = base64_simd::STANDARD
            .decode_to_vec(data)
            .map_err(|_| AdapterError::Base64DecodeFailed)?;
        Ok((image_data, format))
    }
    async fn process_message_params(
        messages: Vec<ChatCompletionMessageParam>,
        supported_tools: Vec<proto_value::Enum<ClientSideToolV2>>,
        now: chrono::DateTime<chrono_tz::Tz>,
        image_support: bool,
        is_agentic: bool,
    ) -> Result<(String, Messages, Vec<ComposerExternalLink>), AdapterError> {
        // 分别收集 system 指令和 user/assistant 对话
        let (system_messages, mut params): (Vec<_>, Vec<_>) = messages
            .into_iter()
            .partition(|param| matches!(param, ChatCompletionMessageParam::System { .. }));

        // 收集 system 指令
        let instructions = system_messages
            .into_iter()
            .map(|param| {
                let ChatCompletionMessageParam::System { content } = param else {
                    __unreachable!()
                };
                content.text()
            })
            .collect::<Vec<_>>()
            .join(DOUBLE_NEWLINE);

        // 使用默认指令或收集到的指令
        let instructions = if instructions.is_empty() {
            DEFAULT_INSTRUCTIONS.get().get(now)
        } else {
            instructions
        };

        // 处理空对话情况
        if params.is_empty() {
            return Ok((
                instructions,
                Messages::from_single(ConversationMessage {
                    r#type: conversation_message::MessageType::Human.into(),
                    bubble_id: Uuid::new_v4().to_byte_str(),
                    unified_mode: Some(stream_unified_chat_request::UnifiedMode::Chat.into()),
                    ..Default::default()
                }),
                vec![],
            ));
        }

        // 如果第一条是 assistant，插入空的 user 消息
        if params
            .first()
            .is_some_and(|param| matches!(param, ChatCompletionMessageParam::Assistant { .. }))
        {
            params.insert(
                0,
                ChatCompletionMessageParam::User {
                    content: ChatCompletionContent::String(String::new()),
                },
            );
        }

        // 确保最后一条是 user
        // if params
        //     .last()
        //     .is_some_and(|param| matches!(param, ChatCompletionMessageParam::Assistant { .. }))
        // {
        //     params.push(ChatCompletionMessageParam::User {
        //         content: ChatCompletionContent::String(String::new()),
        //     });
        // }

        // 转换为 proto messages
        let mut messages = Messages::with_capacity(params.len());
        let mut base_uuid = BaseUuid::new();
        let mut params = params.into_iter().peekable();

        while let Some(param) = params.next() {
            let atext;
            let mut images = Vec::new();
            let mut next = None;
            let is_user;
            let mut external_links = Vec::new();

            match param {
                ChatCompletionMessageParam::User { content } => {
                    is_user = true;
                    atext = match content {
                        ChatCompletionContent::String(text) => text,
                        ChatCompletionContent::Array(contents) => {
                            let mut text_parts = Vec::new();
                            for content in contents {
                                match content {
                                    ChatCompletionContentPart::Text { text } => {
                                        text_parts.push(text)
                                    }
                                    ChatCompletionContentPart::ImageUrl { image_url } => {
                                        if image_support {
                                            Self::process_image(
                                                image_url,
                                                &mut images,
                                                &mut base_uuid,
                                            )
                                            .await?;
                                        }
                                    }
                                }
                            }
                            text_parts.join(NEWLINE)
                        }
                    };
                }
                ChatCompletionMessageParam::Assistant { content, mut tool_calls } => {
                    is_user = false;
                    atext = match content {
                        ChatCompletionContentText::String(text) => text,
                        ChatCompletionContentText::Array(text_parts) => text_parts.join(NEWLINE),
                    };
                    while matches!(params.peek(), Some(ChatCompletionMessageParam::Tool { .. })) {
                        let Some(ChatCompletionMessageParam::Tool { content, tool_call_id }) =
                            params.next()
                        else {
                            __unreachable!()
                        };
                        if let Some(tc) = tool_calls.iter_mut().rfind(|tc| {
                            if let ChatCompletionMessageToolCall::Function { id, .. } = tc {
                                *id == tool_call_id
                            } else {
                                false
                            }
                        }) {
                            let ChatCompletionMessageToolCall::Function { id, function } =
                                core::mem::replace(tc, ChatCompletionMessageToolCall::Taked)
                            else {
                                __unreachable!()
                            };
                            let Function { arguments, name } = function;
                            let ToolName { tool_name, name, server_name } = ToolName::parse(name);
                            let result = content.result().await?;
                            let tool_id = ToolId::parse(id);
                            let result = Some(ClientSideToolV2Result {
                                tool: ClientSideToolV2::Mcp.into(),
                                tool_call_id: tool_id.tool_call_id.clone(),
                                model_call_id: tool_id.model_call_id.clone(),
                                tool_index: Some(1),
                                result: Some(Result::McpResult(McpResult {
                                    selected_tool: name.clone(),
                                    result,
                                })),
                            });
                            use crate::core::aiserver::v1::{
                                client_side_tool_v2_call::Params,
                                client_side_tool_v2_result::Result,
                                conversation_message::ToolResult,
                            };
                            let raw_args: ByteStr = arguments.into();
                            let tool_call = Some(ClientSideToolV2Call {
                                tool: ClientSideToolV2::Mcp.into(),
                                params: Some(Params::McpParams(McpParams {
                                    tools: vec![mcp_params::Tool {
                                        name,
                                        parameters: raw_args.clone(),
                                        server_name,
                                        ..Default::default()
                                    }],
                                })),
                                tool_call_id: tool_id.tool_call_id.clone(),
                                name: tool_name.clone(),
                                tool_index: Some(1),
                                model_call_id: tool_id.model_call_id.clone(),
                                ..Default::default()
                            });
                            let result = ToolResult {
                                tool_call_id: tool_id.tool_call_id,
                                tool_name,
                                tool_index: 1,
                                model_call_id: tool_id.model_call_id,
                                raw_args,
                                result,
                                tool_call,
                            };
                            next = Some(ConversationMessage {
                                r#type: conversation_message::MessageType::Ai.into(),
                                bubble_id: Uuid::new_v4().to_byte_str(),
                                server_bubble_id: Some(Uuid::new_v4().to_byte_str()),
                                tool_results: vec![result],
                                unified_mode: Some(
                                    if is_agentic {
                                        stream_unified_chat_request::UnifiedMode::Agent
                                    } else {
                                        stream_unified_chat_request::UnifiedMode::Chat
                                    }
                                    .into(),
                                ),
                                ..Default::default()
                            });
                        }
                    }
                }
                _ => continue,
            }

            // 处理消息内容和相关字段
            let (final_text, web_references, use_web) = if !is_user {
                let (text, web_refs, has_web) = extract_web_references_info(atext);
                (text, web_refs, has_web.to_opt())
            } else {
                extract_external_links(&atext, &mut external_links, &mut base_uuid);
                (atext, vec![], None)
            };

            messages.push(ConversationMessage {
                text: final_text,
                r#type: if is_user {
                    conversation_message::MessageType::Human
                } else {
                    conversation_message::MessageType::Ai
                }
                .into(),
                images,
                bubble_id: Uuid::new_v4().to_byte_str(),
                server_bubble_id: if is_user { None } else { Some(Uuid::new_v4().to_byte_str()) },
                is_agentic: false,
                // existed_subsequent_terminal_command: false,
                // existed_previous_terminal_command: false,
                web_references,
                // git_context: None,
                // cached_conversation_summary: None,
                // attached_human_changes: false,
                thinking: None,
                unified_mode: Some(
                    if is_agentic {
                        stream_unified_chat_request::UnifiedMode::Agent
                    } else {
                        stream_unified_chat_request::UnifiedMode::Chat
                    }
                    .into(),
                ),
                external_links,
                use_web,
                ..Default::default()
            });

            if let Some(next) = next {
                messages.push(next);
            }
        }

        // 获取最后一条用户消息的URLs
        let external_links = messages
            .last_mut()
            .map(|msg| {
                msg.supported_tools = supported_tools;
                msg.external_links.clone()
            })
            .unwrap_or_default();

        Ok((instructions, messages, external_links))
    }
}

pub async fn encode_create_params(
    params: Vec<ChatCompletionMessageParam>,
    tools: Vec<ChatCompletionTool>,
    now: chrono::DateTime<chrono_tz::Tz>,
    model: ExtModel,
    msg_id: Uuid,
    environment_info: EnvironmentInfo,
    disable_vision: bool,
    enable_slow_pool: bool,
) -> Result<Vec<u8>, AdapterError> {
    Openai::encode_create_params(
        params,
        tools,
        now,
        model,
        msg_id,
        environment_info,
        disable_vision,
        enable_slow_pool,
    )
    .await
    .and_then(|message| encode_message_framed(&message).map_err(Into::into))
}

pub async fn encode_tool_result(
    content: ChatCompletionContentText,
    tool_call_id: ByteStr,
    tool_name: ByteStr,
) -> Result<Vec<u8>, AdapterError> {
    let message = Openai::encode_tool_result(content, tool_call_id, tool_name).await?;
    encode_message_framed(&message).map_err(Into::into)
}
