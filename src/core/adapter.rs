use super::aiserver::v1::{
    ComposerExternalLink, ConversationMessage, ConversationMessageHeader, WebReference,
    image_proto::Dimension,
};
use crate::app::model::proxy_pool::get_fetch_image_client;

pub mod anthropic;
pub mod openai;

mod error;
mod traits;
mod utils;
pub use error::Error as AdapterError;
pub use utils::ToolId;

crate::define_typed_constants! {
    &'static str => {
        /// 换行符
        NEWLINE = "\n",
        /// Web 搜索模式
        WEB_SEARCH_MODE = "full_search",
        /// Ask 模式名称
        ASK_MODE_NAME = "Ask",
        /// Agent 模式名称
        AGENT_MODE_NAME = "Agent",
    }
}

#[inline]
fn parse_web_references(text: &str) -> Vec<WebReference> {
    let mut web_refs = Vec::new();
    let lines = text.lines().skip(1); // 跳过 "WebReferences:" 行

    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            break;
        }

        // 跳过序号和空格
        let mut chars = line.chars();
        for c in chars.by_ref() {
            if c == '.' {
                break;
            }
        }
        let remaining = chars.as_str().trim_start();

        // 解析 [title](url) 部分
        let mut chars = remaining.chars();
        if chars.next() != Some('[') {
            continue;
        }

        let mut title = String::with_capacity(64);
        let mut url = String::with_capacity(64);
        let mut chunk = String::with_capacity(64);
        let mut current = &mut title;
        let mut state = 0; // 0: title, 1: url, 2: chunk

        while let Some(c) = chars.next() {
            match (state, c) {
                (0, ']') => {
                    state = 1;
                    if chars.next() != Some('(') {
                        break;
                    }
                    current = &mut url;
                }
                (1, ')') => {
                    state = 2;
                    if chars.next() == Some('<') {
                        current = &mut chunk;
                    } else {
                        break;
                    }
                }
                (2, '>') => break,
                (_, c) => current.push(c),
            }
        }

        web_refs.push(WebReference { title, url, chunk });
    }

    web_refs
}

// 解析消息中的外部链接
#[inline]
fn extract_external_links(
    text: &str,
    external_links: &mut Vec<ComposerExternalLink>,
    base_uuid: &mut BaseUuid,
) {
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '@' {
            let mut url = String::new();
            while let Some(&next_char) = chars.peek() {
                if next_char.is_whitespace() {
                    break;
                }
                url.push(__unwrap!(chars.next()));
            }

            if !url.is_empty()
                && let Ok(parsed_url) = url::Url::parse(&url)
                && {
                    let scheme = parsed_url.scheme().as_bytes();
                    scheme == b"http" || scheme == b"https"
                }
            {
                external_links.push(ComposerExternalLink {
                    url,
                    uuid: base_uuid.add_and_to_string(),
                    ..Default::default()
                });
            }
        }
    }
}

// 检测并分离 WebReferences
#[inline]
fn extract_web_references_info(text: String) -> (String, Vec<WebReference>, bool) {
    if text.starts_with("WebReferences:") {
        if let Some((web_refs_text, content_text)) = text.split_once("\n\n") {
            let web_refs = parse_web_references(web_refs_text);
            let has_web_refs = !web_refs.is_empty();
            (content_text.to_string(), web_refs, has_web_refs)
        } else {
            (text.to_string(), vec![], false)
        }
    } else {
        (text.to_string(), vec![], false)
    }
}

struct BaseUuid {
    inner: u16,
    buffer: itoa::Buffer,
}

impl BaseUuid {
    #[inline]
    fn new() -> Self {
        Self {
            inner: rand::RngExt::random_range(&mut rand::rng(), 256u16..384),
            buffer: itoa::Buffer::new(),
        }
    }
    #[inline]
    fn add_and_to_string(&mut self) -> String {
        let s = self.buffer.format(self.inner).to_string();
        self.inner = self.inner.wrapping_add(1);
        s
    }
}

// #[inline]
// fn sanitize_tool_name(input: &str) -> String {
//     let mut result = String::with_capacity(input.len());

//     for c in input.chars() {
//         match c {
//             '.' => result.push('_'),
//             c if c.is_whitespace() => result.push('_'),
//             c if c.is_ascii_alphanumeric() || c == '_' || c == '-' => result.push(c),
//             _ => {} // 忽略其他字符
//         }
//     }

//     result
// }

// 处理 HTTP 图片 URL
async fn process_http_image(
    url: url::Url,
) -> Result<(bytes::Bytes, Option<Dimension>), AdapterError> {
    let response =
        get_fetch_image_client().get(url).send().await.map_err(|_| AdapterError::RequestFailed)?;
    let image_data = response.bytes().await.map_err(|_| AdapterError::ResponseReadFailed)?;

    // 检查图片格式
    let format = image::guess_format(&image_data);
    match format {
        Ok(image::ImageFormat::Png | image::ImageFormat::Jpeg | image::ImageFormat::WebP) => {
            // 这些格式都支持
        }
        Ok(image::ImageFormat::Gif) => {
            if is_animated_gif(&image_data) {
                return Err(AdapterError::UnsupportedAnimatedGif);
            }
        }
        _ => return Err(AdapterError::UnsupportedImageFormat),
    }
    let format = unsafe { format.unwrap_unchecked() };

    // 获取图片尺寸
    let dimensions = image::load_from_memory_with_format(&image_data, format)
        .ok()
        .and_then(|img| img.try_into().ok());

    Ok((image_data, dimensions))
}

fn is_animated_gif(data: &[u8]) -> bool {
    let mut options = gif::DecodeOptions::new();
    options.skip_frame_decoding(true);
    if let Ok(frames) = options.read_info(std::io::Cursor::new(data))
        && frames.into_iter().nth(1).is_some()
    {
        true
    } else {
        false
    }
}

async fn process_http_to_base64_image(
    url: url::Url,
) -> Result<(String, &'static str), AdapterError> {
    let response =
        get_fetch_image_client().get(url).send().await.map_err(|_| AdapterError::RequestFailed)?;
    let image_data = response.bytes().await.map_err(|_| AdapterError::ResponseReadFailed)?;

    // 检查图片格式
    let format = image::guess_format(&image_data);
    match format {
        Ok(image::ImageFormat::Png | image::ImageFormat::Jpeg | image::ImageFormat::WebP) => {
            // 这些格式都支持
        }
        Ok(image::ImageFormat::Gif) => {
            if is_animated_gif(&image_data) {
                return Err(AdapterError::UnsupportedAnimatedGif);
            }
        }
        _ => return Err(AdapterError::UnsupportedImageFormat),
    }
    let format = unsafe { format.unwrap_unchecked() };

    Ok((
        base64_simd::STANDARD.encode_to_string(&image_data[..]),
        match format {
            image::ImageFormat::Png => "image/png",
            image::ImageFormat::Jpeg => "image/jpeg",
            image::ImageFormat::Gif => "image/gif",
            image::ImageFormat::WebP => "image/webp",
            _ => __unreachable!(),
        },
    ))
}

struct Messages {
    inner: Vec<ConversationMessage>,
    headers: Vec<ConversationMessageHeader>,
}

impl Messages {
    #[inline]
    fn with_capacity(capacity: usize) -> Self {
        Self { inner: Vec::with_capacity(capacity), headers: Vec::with_capacity(capacity) }
    }
    #[inline]
    fn push(&mut self, message: ConversationMessage) {
        self.headers.push(ConversationMessageHeader {
            bubble_id: message.bubble_id.clone(),
            server_bubble_id: message.server_bubble_id.clone(),
            r#type: message.r#type,
        });
        self.inner.push(message);
    }
    #[inline]
    fn from_single(message: ConversationMessage) -> Self {
        let mut v = Self::with_capacity(1);
        v.push(message);
        v
    }
    #[inline]
    fn last_mut(&mut self) -> Option<&mut ConversationMessage> { self.inner.last_mut() }
}
