use alloc::borrow::Cow;

use crate::common::model::{ApiStatus, GenericError};

const SIZE_LIMIT_MSG: &str = "Message exceeds 4 MiB size limit";

#[derive(Debug)]
pub struct ExceedSizeLimit;

impl ExceedSizeLimit {
    #[inline]
    pub const fn message() -> &'static str { SIZE_LIMIT_MSG }

    #[inline]
    pub const fn into_generic(self) -> GenericError {
        GenericError {
            status: ApiStatus::Error,
            code: Some(http::StatusCode::PAYLOAD_TOO_LARGE),
            error: Some(Cow::Borrowed("resource_exhausted")),
            message: Some(Cow::Borrowed(SIZE_LIMIT_MSG)),
        }
    }

    #[inline]
    pub const fn into_response_tuple(self) -> (http::StatusCode, axum::Json<GenericError>) {
        (http::StatusCode::PAYLOAD_TOO_LARGE, axum::Json(self.into_generic()))
    }
}

impl core::fmt::Display for ExceedSizeLimit {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(SIZE_LIMIT_MSG)
    }
}

impl ::core::error::Error for ExceedSizeLimit {}

impl axum::response::IntoResponse for ExceedSizeLimit {
    #[inline]
    fn into_response(self) -> axum::response::Response {
        self.into_response_tuple().into_response()
    }
}

/// 尝试压缩数据，仅当压缩后体积更小时返回压缩结果
///
/// 压缩决策逻辑：
/// 1. 数据 ≤ 1KB → 不压缩（开销大于收益）
/// 2. 压缩后体积 ≥ 原始 → 不压缩（无效压缩）
/// 3. 否则返回压缩数据
#[inline]
fn try_compress_if_beneficial(data: &[u8]) -> Option<Vec<u8>> {
    const COMPRESSION_THRESHOLD: usize = 1024; // 1KB

    // 小数据不压缩
    if data.len() <= COMPRESSION_THRESHOLD {
        return None;
    }

    let compressed = grpc_stream::compress_gzip(data);

    // 仅当压缩有效时返回
    if compressed.len() < data.len() { Some(compressed) } else { None }
}

/// 编码protobuf消息，自动压缩优化
///
/// 根据消息大小和压缩效果自动选择最优编码方式：
/// - 小消息（≤1KB）：直接返回原始编码
/// - 大消息：尝试gzip压缩，仅当压缩有效时使用
///
/// # Arguments
/// * `message` - 实现了`prost::Message`的protobuf消息
///
/// # Returns
/// - `Ok((data, is_compressed))` - 编码成功
///   - `data`: 编码后的字节数据（可能已压缩）
///   - `is_compressed`: `true`表示返回的是压缩数据，`false`表示原始编码
/// - `Err(&str)` - 消息超过4MiB大小限制
///
/// # Errors
/// 当消息编码后长度超过`MAX_DECOMPRESSED_SIZE_BYTES`（4MiB）时返回错误
///
/// # Example
/// ```ignore
/// let msg = MyMessage { field: 42 };
/// let (data, compressed) = encode_message(&msg)?;
/// if compressed {
///     println!("使用压缩，节省空间");
/// }
/// ```
#[inline(always)]
pub fn encode_message(message: &impl ::prost::Message) -> Result<(Vec<u8>, bool), ExceedSizeLimit> {
    let estimated_size = message.encoded_len();

    // 检查消息大小是否超过限制
    if estimated_size > grpc_stream::MAX_DECOMPRESSED_SIZE_BYTES {
        __cold_path!();
        return Err(ExceedSizeLimit);
    }

    // 编码到Vec
    let mut encoded = Vec::with_capacity(estimated_size);
    message.encode_raw(&mut encoded);

    // 尝试压缩并返回最优结果
    if let Some(compressed) = try_compress_if_beneficial(&encoded) {
        Ok((compressed, true))
    } else {
        Ok((encoded, false))
    }
}

/// 编码protobuf消息为带协议头的帧格式
///
/// 生成包含元数据的完整协议帧，适用于流式传输场景。
///
/// # 协议格式
/// ```text
/// [压缩标志 1B][消息长度 4B BE][消息体/压缩数据]
/// ```
/// - **字节0**: 压缩标志 (`0x00`=未压缩, `0x01`=gzip压缩)
/// - **字节1-4**: 消息体长度，大端序u32
/// - **字节5+**: 实际消息数据
///
/// # 压缩策略
/// 与`encode_message`相同：小消息不压缩，压缩无效时回退到原始数据
///
/// # Arguments
/// * `message` - 实现了`prost::Message`的protobuf消息
///
/// # Returns
/// - `Ok(framed_data)` - 完整的协议帧数据
/// - `Err(&str)` - 消息超过4MiB大小限制
///
/// # Errors
/// 当消息编码后长度超过`MAX_DECOMPRESSED_SIZE_BYTES`（4MiB）时返回错误
///
/// # Safety
/// 内部使用`MaybeUninit`和unsafe代码优化性能，但保证内存安全：
/// - 所有写入操作在边界内
/// - 返回前确保所有数据已初始化
///
/// # Example
/// ```ignore
/// let msg = MyMessage { field: 42 };
/// let frame = encode_message_framed(&msg)?;
/// // frame可直接写入网络流
/// stream.write_all(&frame)?;
/// ```
#[inline(always)]
pub fn encode_message_framed(message: &impl ::prost::Message) -> Result<Vec<u8>, ExceedSizeLimit> {
    let estimated_size = message.encoded_len();

    // 检查消息大小是否超过限制（4MiB远小于u32::MAX-5，无需额外检查协议限制）
    if estimated_size > grpc_stream::MAX_DECOMPRESSED_SIZE_BYTES {
        __cold_path!();
        return Err(ExceedSizeLimit);
    }

    use ::core::mem::MaybeUninit;

    // 分配未初始化buffer：[5字节头部][消息体]
    // 使用MaybeUninit避免不必要的零初始化
    let mut buffer = Vec::<MaybeUninit<u8>>::with_capacity(5 + estimated_size);

    unsafe {
        // 预设长度（内容待初始化）
        buffer.set_len(5 + estimated_size);

        // 获取头部和消息体的指针
        let header_ptr: *mut u8 = buffer.as_mut_ptr().cast();
        let body_ptr = header_ptr.add(5);

        // 编码消息体到偏移5的位置
        message.encode_raw(&mut ::core::slice::from_raw_parts_mut(body_ptr, estimated_size));

        // 尝试压缩消息体
        let body_slice = ::core::slice::from_raw_parts(body_ptr, estimated_size);
        let (compression_flag, final_len) =
            if let Some(compressed) = try_compress_if_beneficial(body_slice) {
                let compressed_len = compressed.len();

                // 压缩成功时消息长度必然 < 原始长度 ≤ 4MiB
                ::core::hint::assert_unchecked(compressed_len < estimated_size);

                // 用压缩数据覆盖原始消息体
                ::core::ptr::copy_nonoverlapping(compressed.as_ptr(), body_ptr, compressed_len);

                // 截断buffer到实际使用的长度
                buffer.set_len(5 + compressed_len);

                (0x01, compressed_len)
            } else {
                // 压缩无效，使用原始数据
                (0x00, estimated_size)
            };

        // 写入协议头部
        // 字节0: 压缩标志
        *header_ptr = compression_flag;

        // 字节1-4: 消息长度（大端序）
        let len_bytes = (final_len as u32).to_be_bytes();
        ::core::ptr::copy_nonoverlapping(len_bytes.as_ptr(), header_ptr.add(1), 4);

        // 此时buffer所有数据已初始化，安全转换为Vec<u8>
        #[allow(clippy::missing_transmute_annotations)]
        Ok(::core::intrinsics::transmute(buffer))
    }
}
