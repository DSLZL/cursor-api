use super::{
    super::{
        aiserver::v1::{
            AzureState, ClientSideToolV2, ClientSideToolV2Result, ComposerExternalLink,
            CurrentFileInfo, CursorPosition, CursorRange, EnvironmentInfo, ImageProto, McpResult,
            ModelDetails, StreamUnifiedChatRequest, StreamUnifiedChatRequestWithTools,
            client_side_tool_v2_result, image_proto, mcp_params, stream_unified_chat_request,
            stream_unified_chat_request_with_tools,
        },
        model::{ExtModel, IndexMap},
    },
    AGENT_MODE_NAME, ASK_MODE_NAME, AdapterError, BaseUuid, Messages, WEB_SEARCH_MODE,
    is_animated_gif, process_http_image,
    utils::{ToolId, ToolResultBuilder},
};
use crate::app::model::{AppConfig, VisionAbility, create_explicit_context};
use byte_str::ByteStr;
use url::Url;

pub(super) trait ToolParam: Sized + 'static {
    fn extract(self) -> (String, Option<String>, IndexMap<String, serde_json::Value>);
}

pub(super) trait ToolResult: Sized + 'static {
    fn size_hint(&self) -> Option<usize>;
    fn is_error(&self) -> bool;
    async fn add_to(self, builder: &mut ToolResultBuilder) -> Result<(), AdapterError>;
    async fn result(self) -> Result<ByteStr, AdapterError> {
        let mut builder = {
            let is_error = self.is_error();
            if let Some(capacity) = self.size_hint() {
                ToolResultBuilder::with_capacity(capacity, is_error)
            } else {
                ToolResultBuilder::new(is_error)
            }
        };
        self.add_to(&mut builder).await?;
        Ok(builder.build().into())
    }
}

pub(super) trait ImageParams: Sized + 'static {
    type Base64ImageParams: ?Sized;
    fn extract(&self) -> Result<&Self::Base64ImageParams, &str>;
}

pub(super) trait Adapter: Sized + 'static {
    type ImageParams: ImageParams;
    type MessageParams: Sized + 'static;
    type ToolParam: ToolParam;
    type ToolResult: ToolResult;
    async fn process_message_params(
        params: Self::MessageParams,
        supported_tools: Vec<proto_value::Enum<ClientSideToolV2>>,
        now: chrono::DateTime<chrono_tz::Tz>,
        image_support: bool,
        is_agentic: bool,
    ) -> Result<(String, Messages, Vec<ComposerExternalLink>), AdapterError>;
    fn _process_base64_image(
        params: &<Self::ImageParams as ImageParams>::Base64ImageParams,
    ) -> Result<(Vec<u8>, image::ImageFormat), AdapterError>;
    /// 处理 base64 编码的图片
    fn process_base64_image(
        params: &<Self::ImageParams as ImageParams>::Base64ImageParams,
    ) -> Result<(bytes::Bytes, Option<image_proto::Dimension>), AdapterError> {
        let (data, format) = Self::_process_base64_image(params)?;
        // 检查是否为动态 GIF
        if format == image::ImageFormat::Gif && is_animated_gif(&data) {
            return Err(AdapterError::UnsupportedAnimatedGif);
        }
        // 获取图片尺寸
        let dimensions = image::load_from_memory_with_format(&data, format)
            .ok()
            .and_then(|img| img.try_into().ok());
        Ok((data.into(), dimensions))
    }
    async fn process_image(
        params: Self::ImageParams,
        images: &mut Vec<ImageProto>,
        base_uuid: &mut BaseUuid,
    ) -> Result<(), AdapterError> {
        let res = match AppConfig::vision_ability() {
            VisionAbility::None => Err(AdapterError::VisionDisabled),
            va => match params.extract() {
                Ok(params) => Self::process_base64_image(params),
                Err(url) => {
                    if va == VisionAbility::All {
                        process_http_image(
                            Url::parse(url).map_err(|_| AdapterError::UrlParseFailed)?,
                        )
                        .await
                    } else {
                        Err(AdapterError::Base64Only)
                    }
                }
            },
        };
        match res {
            Ok((image_data, dimension)) => {
                images.push(ImageProto {
                    data: image_data,
                    dimension,
                    uuid: base_uuid.add_and_to_string(),
                    // task_specific_description: None,
                });
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
    async fn encode_create_params(
        params: Self::MessageParams,
        tools: Vec<Self::ToolParam>,
        now: chrono::DateTime<chrono_tz::Tz>,
        model: ExtModel,
        msg_id: uuid::Uuid,
        environment_info: EnvironmentInfo,
        disable_vision: bool,
        enable_slow_pool: bool,
    ) -> Result<StreamUnifiedChatRequestWithTools, AdapterError> {
        let is_chat = tools.is_empty();
        let is_agentic = !is_chat;
        let supported_tools = if is_agentic { vec![ClientSideToolV2::Mcp.into()] } else { vec![] };

        let (instructions, messages, external_links) = Self::process_message_params(
            params,
            supported_tools.clone(),
            now,
            !disable_vision && model.is_image,
            is_agentic,
        )
        .await?;

        let explicit_context = create_explicit_context(instructions.into());

        let long_context = AppConfig::is_long_context_enabled();

        let message = StreamUnifiedChatRequestWithTools {
            request: Some(
                stream_unified_chat_request_with_tools::Request::StreamUnifiedChatRequest(
                    Box::new(StreamUnifiedChatRequest {
                        conversation: messages.inner,
                        full_conversation_headers_only: messages.headers,
                        // allow_long_file_scan: Some(false),
                        explicit_context,
                        // can_handle_filenames_after_language_ids: Some(false),
                        model_details: Some(ModelDetails {
                            model_name: Some(model.id()),
                            azure_state: Some(AzureState::default()),
                            enable_slow_pool: enable_slow_pool.to_opt(),
                            max_mode: Some(model.max),
                        }),
                        use_web: if model.web {
                            Some(ByteStr::from_static(WEB_SEARCH_MODE))
                        } else {
                            None
                        },
                        external_links,
                        should_cache: Some(true),
                        current_file: Some(CurrentFileInfo {
                            contents_start_at_line: 1,
                            cursor_position: Some(CursorPosition::default()),
                            total_number_of_lines: 1,
                            selection: Some(CursorRange {
                                start_position: Some(CursorPosition::default()),
                                end_position: Some(CursorPosition::default()),
                            }),
                            ..Default::default()
                        }),
                        // use_reference_composer_diff_prompt: Some(false),
                        use_new_compression_scheme: Some(true),
                        is_chat,
                        conversation_id: msg_id.to_string(),
                        environment_info: Some(environment_info),
                        is_agentic,
                        supported_tools: supported_tools.clone(),
                        mcp_tools: tools
                            .into_iter()
                            .map(|tool| {
                                let (name, description, parameters) = tool.extract();
                                mcp_params::Tool {
                                    server_name: ByteStr::from_static("custom"),
                                    name: name.into(),
                                    description: description.unwrap_or_default(),
                                    parameters: __unwrap!(serde_json::to_string(&parameters))
                                        .into(),
                                }
                            })
                            .collect(),
                        use_full_inputs_context: long_context.to_opt(),
                        // is_resume: Some(false),
                        allow_model_fallbacks: Some(false),
                        // number_of_times_shown_fallback_model_warning: Some(0),
                        unified_mode: Some(
                            if is_agentic {
                                stream_unified_chat_request::UnifiedMode::Agent
                            } else {
                                stream_unified_chat_request::UnifiedMode::Chat
                            }
                            .into(),
                        ),
                        // tools_requiring_accepted_return: supported_tools,
                        should_disable_tools: Some(is_chat),
                        thinking_level: Some(
                            if model.is_thinking {
                                stream_unified_chat_request::ThinkingLevel::High
                            } else {
                                stream_unified_chat_request::ThinkingLevel::Unspecified
                            }
                            .into(),
                        ),
                        uses_rules: Some(false),
                        // mode_uses_auto_apply: Some(false),
                        unified_mode_name: Some(ByteStr::from_static(if is_chat {
                            ASK_MODE_NAME
                        } else {
                            AGENT_MODE_NAME
                        })),
                    }),
                ),
            ),
        };

        // crate::debug!("send: {message:#?}");

        Ok(message)
    }
    async fn encode_tool_result(
        tool_result: Self::ToolResult,
        tool_call_id: ByteStr,
        tool_name: ByteStr,
    ) -> Result<StreamUnifiedChatRequestWithTools, AdapterError> {
        let result = tool_result.result().await?;
        let tool_id = ToolId::parse(tool_call_id);
        Ok(StreamUnifiedChatRequestWithTools {
            request: Some(stream_unified_chat_request_with_tools::Request::ClientSideToolV2Result(
                Box::new(ClientSideToolV2Result {
                    tool: ClientSideToolV2::Mcp.into(),
                    tool_call_id: tool_id.tool_call_id,
                    model_call_id: tool_id.model_call_id,
                    tool_index: None,
                    result: Some(client_side_tool_v2_result::Result::McpResult(McpResult {
                        selected_tool: tool_name,
                        result,
                    })),
                }),
            )),
        })
    }
}

pub(super) trait ToOpt: Copy {
    fn to_opt(self) -> Option<Self>;
}

impl ToOpt for bool {
    #[inline(always)]
    fn to_opt(self) -> Option<Self> { if self { Some(true) } else { None } }
}

pub(super) trait ToByteStr: Sized {
    fn to_byte_str(&self) -> ByteStr;
}

impl ToByteStr for uuid::Uuid {
    #[inline(always)]
    fn to_byte_str(&self) -> ByteStr {
        let mut buffer = [0; 36];
        self.as_hyphenated().encode_lower(&mut buffer);
        unsafe { ByteStr::from_utf8_unchecked(bytes::Bytes::copy_from_slice(&buffer)) }
    }
}
