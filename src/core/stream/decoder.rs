pub mod cpp;
pub mod direct;
pub mod types;
mod utils;

// use core::sync::atomic::{AtomicU32, Ordering};
use crate::core::{
    adapter::ToolId,
    aiserver::v1::{StreamUnifiedChatResponseWithTools, WebReference},
    error::{CursorError, StreamError},
};
use alloc::borrow::Cow;
use byte_str::ByteStr;
use grpc_stream::{Buffer, RawMessage, decompress_gzip};
use prost::Message as _;
use std::time::Instant;

const INCREMENTAL_COPY_THRESHOLD: usize = 64;

pub trait InstantExt: Sized {
    fn duration_as_secs_f32(&mut self) -> f32;
}

impl InstantExt for Instant {
    #[inline]
    fn duration_as_secs_f32(&mut self) -> f32 {
        let now = Instant::now();
        let duration = now.duration_since(*self);
        *self = now;
        duration.as_secs_f32()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Thinking {
    Text(String),
    Signature(String),
    RedactedThinking(String),
}

#[derive(Debug, PartialEq, Clone)]
pub struct ToolCall {
    pub id: ByteStr,
    pub name: ByteStr,
    pub input: String,
    pub is_last: bool,
}

#[derive(Debug, PartialEq, Clone)]
pub enum StreamMessage {
    // 调试
    // Debug(String),
    // 网络引用
    WebReference(Vec<WebReference>),
    // 内容开始标志
    #[cfg(test)]
    ContentStart,
    // 思考
    Thinking(Thinking),
    // 消息内容
    Content(String),
    // 工具调用
    ToolCall(ToolCall),
    // 流结束标志
    StreamEnd,
}

impl StreamMessage {
    #[inline]
    fn convert_web_ref_to_content(self) -> Self {
        match self {
            StreamMessage::WebReference(refs) => {
                if refs.is_empty() {
                    return StreamMessage::Content(String::new());
                }

                use crate::common::utils::string_builder::StringBuilder;

                // 计算需要添加的字符串部分数量
                // 每个web引用需要8个部分：序号、"[", 标题、"](", URL、")<", chunk、换行符
                // 再加上头部"WebReferences:\n"和末尾的额外换行符，共两个部分
                // let parts_count = refs.len() * 8 + 2;

                let mut string = String::with_capacity(refs.len() * 48).append("WebReferences:\n");
                let mut buffer = itoa::Buffer::new();

                for (i, web_ref) in refs.iter().enumerate() {
                    string
                        .append_mut(buffer.format(i + 1))
                        .append_mut(". [")
                        .append_mut(&web_ref.title)
                        .append_mut("](")
                        .append_mut(&web_ref.url)
                        // .append_mut(")<")
                        // .append_mut(&web_ref.chunk)
                        // .append_mut(">\n");
                        .append_mut(")\n");
                }

                string.append_mut("\n");

                StreamMessage::Content(string)
            }
            other => other,
        }
    }
    #[inline]
    fn any_content(&self) -> bool {
        if let StreamMessage::Content(s)
        | StreamMessage::Thinking(Thinking::Text(s) | Thinking::Signature(s))
        | StreamMessage::ToolCall(ToolCall { input: s, .. }) = self
        {
            !s.is_empty()
        } else {
            false
        }
    }
}

#[derive(Default)]
struct Context {
    raw_args_len: usize,
    processed: u32,
    // 调试使用
    // counter: u32,
}

pub struct StreamDecoder {
    // 主要数据缓冲区 (32字节)
    buffer: Buffer,
    // 结果相关 (24字节 + 24字节 + 24字节)
    first_result: Option<Vec<StreamMessage>>,
    content_delays: Option<(String, Vec<(u32, f32)>)>,
    thinking_content: Option<String>,
    // 计数器和时间 (8字节 + 8字节)
    context: Context,
    empty_stream_count: usize,
    last_content_time: Instant,
    // 状态标志 (1字节 + 1字节 + 1字节)
    first_result_ready: bool,
    first_result_taken: bool,
    has_seen_content: bool,
}

impl StreamDecoder {
    pub fn new() -> Self {
        // static COUNTER: AtomicU32 = AtomicU32::new(0);
        Self {
            buffer: Buffer::with_capacity(64),
            first_result: None,
            content_delays: None,
            thinking_content: None,
            context: Context {
                raw_args_len: 0,
                processed: 0,
                // counter: COUNTER.fetch_add(1, Ordering::SeqCst),
            },
            empty_stream_count: 0,
            last_content_time: Instant::now(),
            first_result_ready: false,
            first_result_taken: false,
            has_seen_content: false,
        }
    }

    #[inline]
    pub fn get_empty_stream_count(&self) -> usize { self.empty_stream_count }

    #[inline]
    pub fn reset_empty_stream_count(&mut self) {
        if self.empty_stream_count > 0 {
            // crate::debug!("重置连续空流计数，之前的计数为: {}", self.empty_stream_count);
            self.empty_stream_count = 0;
        }
    }

    #[inline]
    pub fn take_first_result(&mut self) -> Option<Vec<StreamMessage>> {
        if !self.buffer.is_empty() {
            return None;
        }
        if self.first_result.is_some() {
            self.first_result_taken = true;
        }
        self.first_result.take()
    }

    #[cfg(test)]
    fn is_incomplete(&self) -> bool { !self.buffer.is_empty() }

    #[inline]
    pub fn is_first_result_ready(&self) -> bool { self.first_result_ready }

    #[inline]
    pub fn tool_processed(&self) -> u32 { self.context.processed }

    #[inline]
    pub fn take_content_delays(&mut self) -> Option<(String, Vec<(u32, f32)>)> {
        core::mem::take(&mut self.content_delays)
    }

    #[inline]
    pub fn take_thinking_content(&mut self) -> Option<String> {
        core::mem::take(&mut self.thinking_content)
    }

    #[inline]
    pub fn no_first_cache(mut self) -> Self {
        self.first_result_ready = true;
        self.first_result_taken = true;
        self
    }

    pub fn decode(
        &mut self,
        data: &[u8],
        convert_web_ref: bool,
    ) -> Result<Vec<StreamMessage>, StreamError> {
        if data.is_empty() || {
            self.buffer.extend_from_slice(data);
            self.buffer.len() < 5
        } {
            self.empty_stream_count += 1;
            let arg = if self.buffer.is_empty() {
                format_args!("为空")
            } else {
                __cold_path!();
                format_args!(": {}", hex::encode(&self.buffer))
            };
            crate::debug!("数据长度小于5字节，当前数据{arg}");
            return Err(StreamError::EmptyStream);
        }

        self.reset_empty_stream_count();

        let mut iter = (&self.buffer).into_iter();
        let count = iter.len();

        if let Some(content_delays) = self.content_delays.as_mut() {
            content_delays.1.reserve(count);
        } else {
            self.content_delays = Some((String::with_capacity(64), Vec::with_capacity(count)));
        }

        let mut messages = Vec::with_capacity(count);

        for raw_msg in iter.by_ref() {
            if raw_msg.data.is_empty() {
                #[cfg(test)]
                messages.push(StreamMessage::ContentStart);
                continue;
            }

            // if self.context.processed {
            //     let remaining = self.buffer.len() - iter.offset();
            //     crate::debug!("remaining: {remaining} bytes");
            //     if remaining != 0 {
            //         crate::debug!("type: {}, data: {}", raw_msg.r#type, hex::encode(raw_msg.data));
            //     }
            //     continue;
            // }

            let result = match Self::process_message(raw_msg, &mut self.context) {
                Ok(x) => x,
                Err(e) => {
                    if e.error()
                        == Some(crate::core::aiserver::v1::error_details::Error::UserAbortedRequest)
                    {
                        messages.push(StreamMessage::StreamEnd);
                        continue;
                    } else {
                        return Err(StreamError::Upstream(e));
                    }
                }
            };

            if let Some(msg) = result {
                if !self.has_seen_content && msg.any_content() {
                    self.has_seen_content = true;
                }
                if let StreamMessage::Content(ref content) = msg {
                    let delay = self.last_content_time.duration_as_secs_f32();
                    let content_delays = __unwrap!(self.content_delays.as_mut());
                    content_delays.0.push_str(content);
                    content_delays.1.push((content.chars().count() as u32, delay));
                } else if let StreamMessage::Thinking(Thinking::Text(ref text)) = msg {
                    if let Some(thinking_content) = self.thinking_content.as_mut() {
                        thinking_content.push_str(text);
                    } else {
                        self.thinking_content = Some(text.clone());
                    }
                }
                let msg = if convert_web_ref { msg.convert_web_ref_to_content() } else { msg };
                messages.push(msg);
            }
        }

        unsafe { self.buffer.advance_unchecked(iter.offset()) };

        if !self.first_result_taken && !messages.is_empty() {
            if self.first_result.is_none() {
                self.first_result = Some(::core::mem::take(&mut messages));
            } else if !self.first_result_ready
                && let Some(first_result) = &mut self.first_result
            {
                first_result.append(&mut messages);
            }
        }
        if !self.first_result_ready {
            self.first_result_ready =
                self.first_result.is_some() && !self.first_result_taken && self.has_seen_content;
        }
        Ok(messages)
    }

    #[inline]
    fn process_message(
        raw_msg: RawMessage<'_>,
        ctx: &mut Context,
    ) -> Result<Option<StreamMessage>, CursorError> {
        let is_compressed = raw_msg.r#type & 0b01 != 0;
        let is_protobuf = raw_msg.r#type & 0b10 == 0;
        let msg_data = if is_compressed {
            match decompress_gzip(raw_msg.data) {
                Some(data) => Cow::Owned(data),
                None => return Ok(None),
            }
        } else {
            Cow::Borrowed(raw_msg.data)
        };
        let msg_data = &*msg_data;
        if is_protobuf {
            Ok(Self::handle_text_message(msg_data, ctx))
        } else {
            Self::handle_json_message(msg_data)
        }
    }

    fn handle_text_message(msg_data: &[u8], ctx: &mut Context) -> Option<StreamMessage> {
        // let count = self.counter.fetch_add(1, Ordering::SeqCst);
        if let Ok(wrapper) = StreamUnifiedChatResponseWithTools::decode(msg_data) {
            // crate::debug!("StreamUnifiedChatResponseWithTools [hex: {}]: {:#?}", hex::encode(msg_data), response);
            // crate::debug!("{count}: {response:?}");
            if let Some(response) = wrapper.response {
                // crate::debug!("received: {response:#?}");
                use super::super::aiserver::v1::{
                    client_side_tool_v2_call::Params,
                    stream_unified_chat_response_with_tools::Response,
                };
                match response {
                    Response::ClientSideToolV2Call(response) => {
                        let mut result = None;
                        let mut finish = false;

                        // if !response.raw_args.is_empty() {
                        //     crate::debug!("detected: {:?}", response.raw_args);
                        // }

                        if response.is_streaming {
                            use core::cmp::Ordering;

                            if !utils::has_space_after_separator(response.raw_args.as_bytes()) {
                                return None;
                            }

                            let raw_args_len = response.raw_args.len();

                            match raw_args_len.cmp(&ctx.raw_args_len) {
                                Ordering::Greater => {
                                    // 有新增数据，提取增量部分
                                    let args = unsafe {
                                        response.raw_args.get_unchecked(ctx.raw_args_len..)
                                    };

                                    if args.len() > INCREMENTAL_COPY_THRESHOLD {
                                        __cold_path!();
                                        // 大块数据：原地移动避免重新分配
                                        let mut raw_args = response.raw_args;
                                        let count = raw_args_len - ctx.raw_args_len;

                                        unsafe {
                                            let v = raw_args.as_mut_vec();
                                            // SAFETY: the conditions for `ptr::copy` have all been checked above,
                                            // as have those for `ptr::add`.
                                            let ptr = v.as_mut_ptr();
                                            let src_ptr = ptr.add(ctx.raw_args_len);
                                            core::ptr::copy(src_ptr, ptr, count);
                                            v.set_len(count);
                                        }
                                        result = Some(raw_args);
                                    } else {
                                        // 小块数据：直接克隆更快
                                        result = Some(args.to_owned());
                                    }

                                    ctx.raw_args_len = raw_args_len;
                                }

                                Ordering::Equal => {
                                    // 无新数据，跳过此次处理
                                    return None;
                                }

                                Ordering::Less => {
                                    __cold_path!();
                                    eprintln!(
                                        "Warning: raw_args_len decreased: {} < {} (possible stream reset)",
                                        raw_args_len, ctx.raw_args_len
                                    );
                                    crate::debug!(
                                        "Streaming length regression detected: \
                                        raw_args={:?},
                                        tool_call_id={}, model_call_id={}, \
                                        expected_len={}, actual_len={}, \
                                        delta={}, is_last={}",
                                        response.raw_args,
                                        response.tool_call_id,
                                        response.model_call_id.as_deref().unwrap_or_default(),
                                        ctx.raw_args_len,
                                        raw_args_len,
                                        ctx.raw_args_len as i64 - raw_args_len as i64,
                                        response.is_last_message
                                    );
                                }
                            }

                            if response.is_last_message {
                                finish = true;
                            }
                        } else {
                            result = Some(response.raw_args);
                            finish = true;
                        }

                        if finish {
                            ctx.processed += 1;
                            ctx.raw_args_len = 0;
                        }

                        let id = ToolId::format(response.tool_call_id, response.model_call_id);
                        let name = response
                            .params
                            .and_then(|ps| {
                                let Params::McpParams(ps) = ps;
                                ps.tools.into_iter().next()
                            })
                            .map(|tool| {
                                if tool.server_name == *"custom" {
                                    tool.name
                                } else {
                                    format!("mcp__{}__{}", tool.server_name, tool.name).into()
                                }
                            })
                            .unwrap_or_default();

                        return result.map(|input| {
                            StreamMessage::ToolCall(ToolCall { id, name, input, is_last: finish })
                        });
                    }
                    Response::StreamUnifiedChatResponse(response) => {
                        if !response.text.is_empty() {
                            return Some(StreamMessage::Content(response.text));
                        } else if let Some(thinking) = response.thinking {
                            return Some(StreamMessage::Thinking(thinking.into()));
                        }
                        // else if let Some(filled_prompt) = response.filled_prompt {
                        //     return Some(StreamMessage::Debug(filled_prompt));
                        // }
                        else if let Some(web_citation) = response.web_citation {
                            return Some(StreamMessage::WebReference(web_citation.references));
                        }
                    }
                }
            }
        }
        // crate::debug!("{count}: {}", hex::encode(msg_data));
        None
    }

    fn handle_json_message(msg_data: &[u8]) -> Result<Option<StreamMessage>, CursorError> {
        if msg_data.len() == 2 {
            return Ok(Some(StreamMessage::StreamEnd));
        }
        // let count = self.counter.fetch_add(1, Ordering::SeqCst);
        // if let Some(text) = utils::string_from_utf8(msg_data) {
        // crate::debug!("JSON消息 [hex: {}]: {}", hex::encode(msg_data), text);
        // crate::debug!("{count}: {text:?}");
        if let Ok(error) = CursorError::from_slice(msg_data) {
            // crate::debug!("received: {error:#?}");
            return Err(error);
        } else {
            crate::debug!("[JSON error] {}", hex::encode(msg_data));
        }
        // }
        // crate::debug!("{count}: {}", hex::encode(msg_data));
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test() {
        unsafe {
            std::env::set_var("DEBUG_LOG_FILE", "debug1.log");
        }
        crate::app::lazy::log::init();
        let stream_data = include_str!("../../../tests/data/stream_data.txt");
        let bytes = hex::decode(stream_data).unwrap();
        let mut decoder = StreamDecoder::new().no_first_cache();

        if let Err(e) = decoder.decode(&bytes, false) {
            println!("解析错误: {e}");
        }
        if decoder.is_incomplete() {
            println!("数据不完整");
        }

        tokio::time::sleep(std::time::Duration::new(3, 5)).await
    }

    #[test]
    fn test_single_chunk() {
        let stream_data = include_str!("../../../tests/data/stream_data.txt");
        let bytes = hex::decode(stream_data).unwrap();
        let mut decoder = StreamDecoder::new().no_first_cache();

        match decoder.decode(&bytes, false) {
            Ok(messages) => {
                for message in messages {
                    match message {
                        StreamMessage::StreamEnd => {
                            println!("流结束");
                            break;
                        }
                        // StreamMessage::Usage(msg) => {
                        //     println!("额度uuid: {msg}");
                        // }
                        StreamMessage::Content(msg) => {
                            println!("消息内容: {msg}");
                        }
                        StreamMessage::Thinking(msg) => {
                            println!("思考: {msg:?}");
                        }
                        StreamMessage::ToolCall(msg) => {
                            println!("工具调用: {msg:?}");
                        }
                        StreamMessage::WebReference(refs) => {
                            println!("网页引用:");
                            for (i, web_ref) in refs.iter().enumerate() {
                                println!(
                                    "{}. {} - {} - {}",
                                    i, web_ref.url, web_ref.title, web_ref.chunk
                                );
                            }
                        }
                        // StreamMessage::Debug(prompt) => {
                        //     println!("调试信息: {prompt}");
                        // }
                        StreamMessage::ContentStart => {
                            println!("流开始");
                        }
                    }
                }
            }
            Err(e) => {
                println!("解析错误: {e}");
            }
        }
        if decoder.is_incomplete() {
            println!("数据不完整");
        }
    }

    #[test]
    fn test_multiple_chunks() {
        let stream_data = include_str!("../../../tests/data/stream_data.txt");
        let bytes = hex::decode(stream_data).unwrap();
        let mut decoder = StreamDecoder::new().no_first_cache();

        fn find_next_message_boundary(bytes: &[u8]) -> usize {
            if bytes.len() < 5 {
                return bytes.len();
            }
            let msg_len = u32::from_be_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;
            5 + msg_len
        }

        let mut offset = 0;
        let mut should_break = false;

        while offset < bytes.len() {
            let remaining_bytes = &bytes[offset..];
            let msg_boundary = find_next_message_boundary(remaining_bytes);
            let current_msg_bytes = &remaining_bytes[..msg_boundary];
            let hex_str = hex::encode(current_msg_bytes);

            match decoder.decode(current_msg_bytes, false) {
                Ok(messages) => {
                    for message in messages {
                        match message {
                            StreamMessage::StreamEnd => {
                                println!("流结束 [hex: {hex_str}]");
                                should_break = true;
                                break;
                            }
                            // StreamMessage::Usage(msg) => {
                            //     println!("额度uuid: {msg}");
                            // }
                            StreamMessage::Content(msg) => {
                                println!("消息内容 [hex: {hex_str}]: {msg}");
                            }
                            StreamMessage::Thinking(msg) => {
                                println!("思考: {msg:?}");
                            }
                            StreamMessage::ToolCall(msg) => {
                                println!("工具调用: {msg:?}");
                            }
                            StreamMessage::WebReference(refs) => {
                                println!("网页引用 [hex: {hex_str}]:");
                                for (i, web_ref) in refs.iter().enumerate() {
                                    println!(
                                        "{}. {} - {} - {}",
                                        i, web_ref.url, web_ref.title, web_ref.chunk
                                    );
                                }
                            }
                            // StreamMessage::Debug(prompt) => {
                            //     println!("调试信息 [hex: {hex_str}]: {prompt}");
                            // }
                            StreamMessage::ContentStart => {
                                println!("流开始 [hex: {hex_str}]");
                            }
                        }
                    }
                    if should_break {
                        break;
                    }
                    if decoder.is_incomplete() {
                        println!("数据不完整 [hex: {hex_str}]");
                        break;
                    }
                    offset += msg_boundary;
                }
                Err(e) => {
                    println!("解析错误 [hex: {hex_str}]: {e}");
                    break;
                }
            }
        }
    }
}
