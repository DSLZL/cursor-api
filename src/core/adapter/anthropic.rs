use super::{
    AdapterError, BaseUuid, Messages, NEWLINE, extract_external_links, extract_web_references_info,
    process_http_to_base64_image,
    traits::*,
    utils::{RawContent, ToolId, ToolName, ToolResultBuilder},
};
use crate::{
    app::{constant::EMPTY_STRING, model::DEFAULT_INSTRUCTIONS},
    common::utils::proto_encode::{encode_message, encode_message_framed},
    core::{
        aiserver::v1::{
            ClientSideToolV2, ClientSideToolV2Call, ClientSideToolV2Result, ComposerExternalLink,
            ConversationMessage, EnvironmentInfo, McpParams, McpResult, conversation_message,
            mcp_params, stream_unified_chat_request,
        },
        model::{
            ExtModel, IndexMap, Role,
            anthropic::{
                ContentBlockParam, DocumentSource, ImageSource, ImageSourceBase64, MediaType,
                MessageContent, MessageParam, SystemContent, Tool, ToolResultContent,
                ToolResultContentBlock,
            },
        },
    },
};
use byte_str::ByteStr;
use uuid::Uuid;

struct Anthropic;

impl ImageParams for ImageSource {
    type Base64ImageParams = ImageSourceBase64;
    fn extract(&self) -> Result<&ImageSourceBase64, &str> {
        match self {
            ImageSource::Base64(base64) => Ok(base64),
            ImageSource::Url { url } => Err(url),
        }
    }
}

impl ToolParam for Tool {
    fn extract(self) -> (String, Option<String>, IndexMap<String, serde_json::Value>) {
        (self.name, self.description, self.input_schema)
    }
}

impl ToolResult for (Option<ToolResultContent>, bool) {
    fn is_error(&self) -> bool { self.1 }
    fn size_hint(&self) -> Option<usize> {
        self.0.as_ref().map(|c| match c {
            ToolResultContent::String(..) => 1,
            ToolResultContent::Array(cs) => cs.len(),
        })
    }
    async fn add_to(self, builder: &mut ToolResultBuilder) -> Result<(), AdapterError> {
        if let Some(c) = self.0 {
            match c {
                ToolResultContent::String(text) => builder.add(text),
                ToolResultContent::Array(cs) => {
                    for c in cs {
                        match c {
                            ToolResultContentBlock::Text { text } => {
                                builder.add(RawContent::text(text))
                            }
                            ToolResultContentBlock::Image { source } => match source {
                                ImageSource::Base64(b) => {
                                    base64_simd::STANDARD
                                        .check(b.data.as_bytes())
                                        .map_err(|_| AdapterError::Base64DecodeFailed)?;
                                    builder.add(RawContent::image(b.data, b.media_type.as_mime()))
                                }
                                ImageSource::Url { url } => {
                                    let url = url::Url::parse(&url)
                                        .map_err(|_| AdapterError::UrlParseFailed)?;
                                    builder.add(process_http_to_base64_image(url).await?)
                                }
                            },
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl Adapter for Anthropic {
    type ImageParams = ImageSource;
    type MessageParams = (Vec<MessageParam>, Option<SystemContent>);
    type ToolParam = Tool;
    type ToolResult = (Option<ToolResultContent>, bool);
    fn _process_base64_image(
        params: &ImageSourceBase64,
    ) -> Result<(Vec<u8>, image::ImageFormat), AdapterError> {
        let ImageSourceBase64 { media_type, data } = params;
        let image_data = base64_simd::STANDARD
            .decode_to_vec(data)
            .map_err(|_| AdapterError::Base64DecodeFailed)?;
        let format = match media_type {
            MediaType::ImagePng => image::ImageFormat::Png,
            MediaType::ImageJpeg => image::ImageFormat::Jpeg,
            MediaType::ImageGif => image::ImageFormat::Gif,
            MediaType::ImageWebp => image::ImageFormat::WebP,
        };
        Ok((image_data, format))
    }
    async fn process_message_params(
        params: (Vec<MessageParam>, Option<SystemContent>),
        supported_tools: Vec<proto_value::Enum<ClientSideToolV2>>,
        now: chrono::DateTime<chrono_tz::Tz>,
        image_support: bool,
        is_agentic: bool,
    ) -> Result<(String, Messages, Vec<ComposerExternalLink>), AdapterError> {
        let (mut params, system) = params;

        // 收集 system 指令
        let instructions = system.map(|content| match content {
            SystemContent::String(text) => text,
            SystemContent::Array(contents) => {
                contents.into_iter().map(|c| c.text).collect::<Vec<_>>().join(NEWLINE)
            }
        });

        // 使用默认指令或收集到的指令
        let instructions = if let Some(instructions) = instructions {
            instructions
        } else {
            DEFAULT_INSTRUCTIONS.get().get(now)
        };

        // 处理空对话情况
        if params.is_empty() {
            return Ok((
                instructions,
                Messages::from_single(ConversationMessage {
                    r#type: conversation_message::MessageType::Human.into(),
                    bubble_id: Uuid::new_v4().to_byte_str(),
                    unified_mode: Some(stream_unified_chat_request::UnifiedMode::Chat.into()),
                    // is_simple_looping_message: Some(false),
                    ..Default::default()
                }),
                vec![],
            ));
        }

        // 如果第一条是 assistant，插入空的 user 消息
        if params.first().is_some_and(|input| input.role == Role::Assistant) {
            params.insert(
                0,
                MessageParam { role: Role::User, content: MessageContent::String(String::new()) },
            );
        }

        // 确保最后一条是 user
        // if params.last().is_some_and(|input| input.role == Role::Assistant) {
        //     params.push(MessageParam {
        //         role: Role::User,
        //         content: MessageContent::String(String::new()),
        //     });
        // }

        // 转换为 proto messages
        let mut messages = Messages::with_capacity(params.len());
        let mut base_uuid = BaseUuid::new();
        let mut params = params.into_iter().peekable();

        while let Some(param) = params.next() {
            let mut external_links = Vec::new();
            let (text, images, thinking, next) = match param.content {
                MessageContent::String(text) => (text, vec![], vec![], None),
                MessageContent::Array(contents) if param.role == Role::User => {
                    let text_parts_len = contents
                        .iter()
                        .filter(|c| matches!(**c, ContentBlockParam::Text { .. }))
                        .count();
                    let images_len =
                        if image_support { contents.len() - text_parts_len } else { 0 };
                    let mut text_parts = Vec::with_capacity(text_parts_len);
                    let mut images = Vec::with_capacity(images_len);

                    for content in contents {
                        match content {
                            ContentBlockParam::Text { text } => text_parts.push(text),
                            ContentBlockParam::Image { source } => {
                                if image_support {
                                    Self::process_image(source, &mut images, &mut base_uuid)
                                        .await?;
                                }
                            }
                            ContentBlockParam::Document { source } => match source {
                                DocumentSource::Base64 { data, .. } => {
                                    external_links.push(ComposerExternalLink {
                                        url: String::new(),
                                        uuid: base_uuid.add_and_to_string(),
                                        pdf_content: data,
                                        is_pdf: true,
                                        filename: ByteStr::from_static("document.pdf"),
                                    })
                                }
                                DocumentSource::Url { url } => {
                                    external_links.push(ComposerExternalLink {
                                        url,
                                        uuid: base_uuid.add_and_to_string(),
                                        ..Default::default()
                                    })
                                }
                            },
                            _ => {}
                        }
                    }

                    (text_parts.join(NEWLINE), images, vec![], None)
                }
                MessageContent::Array(mut contents) if param.role == Role::Assistant => {
                    let mut text_parts = Vec::new();
                    let mut all_thinking_blocks = Vec::new();
                    let mut next = None;

                    if let Some(param) = params.peek()
                        && param.role == Role::User
                        && let MessageContent::Array(ref cs) = param.content
                        && cs.iter().all(|c| matches!(*c, ContentBlockParam::ToolResult { .. }))
                    {
                        let Some(MessageContent::Array(cs)) = params.next().map(|p| p.content)
                        else {
                            __unreachable!()
                        };
                        for c in cs {
                            let ContentBlockParam::ToolResult { tool_use_id, content, is_error } =
                                c
                            else {
                                __unreachable!()
                            };
                            if let Some(c) = contents.iter_mut().rfind(|c| {
                                if let ContentBlockParam::ToolUse { id, .. } = c {
                                    *id == tool_use_id
                                } else {
                                    false
                                }
                            }) {
                                let ContentBlockParam::ToolUse { name, input, .. } =
                                    core::mem::replace(c, ContentBlockParam::Taked)
                                else {
                                    __unreachable!()
                                };
                                let ToolName { tool_name, name, server_name } =
                                    ToolName::parse(name);
                                let result = (content, is_error).result().await?;
                                let tool_id = ToolId::parse(tool_use_id);
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
                                let raw_args: ByteStr =
                                    __unwrap!(serde_json::to_string(&input)).into();
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

                    for content in contents {
                        match content {
                            ContentBlockParam::Text { text } => {
                                text_parts.push(text);
                            }
                            ContentBlockParam::Thinking { thinking, signature } => {
                                all_thinking_blocks.push(conversation_message::Thinking {
                                    text: thinking,
                                    signature,
                                    redacted_thinking: String::new(),
                                });
                            }
                            ContentBlockParam::RedactedThinking { data } => {
                                all_thinking_blocks.push(conversation_message::Thinking {
                                    text: String::new(),
                                    signature: String::new(),
                                    redacted_thinking: data,
                                });
                            }
                            _ => {}
                        }
                    }

                    (text_parts.join(NEWLINE), vec![], all_thinking_blocks, next)
                }
                _ => __unreachable!(),
            };

            // 处理消息内容和相关字段
            let (final_text, web_references, use_web) = match param.role {
                Role::Assistant => {
                    let (text, web_refs, has_web) = extract_web_references_info(text);
                    (text, web_refs, has_web.to_opt())
                }
                Role::User => {
                    extract_external_links(&text, &mut external_links, &mut base_uuid);
                    (text, vec![], None)
                }
                _ => __unreachable!(),
            };

            let is_user = param.role == Role::User;

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
                tool_results: vec![],
                is_agentic: is_agentic && is_user,
                web_references,
                thinking: match thinking.len() {
                    0 => None,
                    1 => thinking.into_iter().next(),
                    _ => Some(conversation_message::Thinking {
                        text: thinking
                            .into_iter()
                            .map(|t| t.text)
                            .filter(|s| !s.is_empty())
                            .collect::<Vec<_>>()
                            .join(EMPTY_STRING),
                        signature: String::new(),
                        redacted_thinking: String::new(),
                    }),
                },
                unified_mode: Some(
                    if is_agentic {
                        stream_unified_chat_request::UnifiedMode::Agent
                    } else {
                        stream_unified_chat_request::UnifiedMode::Chat
                    }
                    .into(),
                ),
                supported_tools: vec![],
                external_links,
                use_web,
                // is_simple_looping_message: Some(false),
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

pub mod non_stream {
    use super::*;
    pub async fn encode_create_params(
        params: (Vec<MessageParam>, Option<SystemContent>),
        tools: Vec<Tool>,
        now: chrono::DateTime<chrono_tz::Tz>,
        model: ExtModel,
        msg_id: Uuid,
        environment_info: EnvironmentInfo,
        disable_vision: bool,
        enable_slow_pool: bool,
    ) -> Result<(Vec<u8>, bool), AdapterError> {
        super::Anthropic::encode_create_params(
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
        .and_then(|message| encode_message(&message).map_err(Into::into))
    }
}

pub async fn encode_create_params(
    params: (Vec<MessageParam>, Option<SystemContent>),
    tools: Vec<Tool>,
    now: chrono::DateTime<chrono_tz::Tz>,
    model: ExtModel,
    msg_id: Uuid,
    environment_info: EnvironmentInfo,
    disable_vision: bool,
    enable_slow_pool: bool,
) -> Result<Vec<u8>, AdapterError> {
    Anthropic::encode_create_params(
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
    tool_result: (Option<ToolResultContent>, bool),
    tool_use_id: ByteStr,
    tool_name: ByteStr,
) -> Result<Vec<u8>, AdapterError> {
    let message = Anthropic::encode_tool_result(tool_result, tool_use_id, tool_name).await?;
    encode_message_framed(&message).map_err(Into::into)
}
