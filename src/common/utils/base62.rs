#![allow(unsafe_op_in_unsafe_fn, unused)]

//! 高性能固定长度 base62 编解码器
//!
//! 本模块提供了针对 `u128` 类型优化的 base62 编解码功能。
//!
//! # 特性
//!
//! - 固定长度输出：所有编码结果恰好为 22 字节
//! - 高性能：使用魔术数除法避免昂贵的 u128 除法操作
//! - 零分配：不需要堆内存分配
//! - 前导零填充：数值较小时自动在前面补 '0'
//!
//! # 示例
//!
//! ```
//! use base62_u128::{encode_fixed, decode_fixed, BASE62_LEN};
//!
//! let mut buf = [0u8; BASE62_LEN];
//! encode_fixed(12345u128, &mut buf);
//! assert_eq!(&buf[..], b"00000000000000000003D7");
//!
//! let decoded = decode_fixed(&buf).unwrap();
//! assert_eq!(decoded, 12345u128);
//! ```

use core::fmt;

// ============================================================================
// 常量定义
// ============================================================================

/// Base62 的基数
const BASE: u64 = 62;

/// 编码输出的固定长度
pub const BASE62_LEN: usize = 22;

/// 62^10 - 用于将 u128 分解为可管理的块
///
/// 这个值是精心选择的，因为：
/// - 它足够大，可以高效地分解 u128
/// - 它足够小，可以放入 u64
const BASE_TO_10: u64 = 839_299_365_868_340_224;
const BASE_TO_10_U128: u128 = BASE_TO_10 as u128;

/// 快速除法的魔术数 - 用于计算 u128 / BASE_TO_10
///
/// 这些常量通过以下方式计算得出：
/// - MULTIPLY = ceil(2^(128 + SHIFT) / BASE_TO_10)
/// - SHIFT 选择为使结果精确的最小值
const DIV_BASE_TO_10_MULTIPLY: u128 = 233_718_071_534_448_225_491_982_379_416_108_680_074;
const DIV_BASE_TO_10_SHIFT: u8 = 59;

/// Base62 字符集（标准顺序：0-9, A-Z, a-z）
const CHARSET: &[u8; 62] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// 解码查找表 - 将 ASCII 字符映射到其 base62 值
///
/// - 有效字符映射到 0-61
/// - 无效字符映射到 0xFF
const DECODE_LUT: &[u8; 256] = &{
    let mut lut = [0xFF; 256];
    let mut i = 0;
    while i < 62 {
        lut[CHARSET[i] as usize] = i as u8;
        i += 1;
    }
    lut
};

// ============================================================================
// 错误类型
// ============================================================================

/// Base62 解码错误
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DecodeError {
    /// 解码结果超出 u128 范围
    ArithmeticOverflow,
    /// 遇到无效的 base62 字符
    InvalidCharacter {
        /// 无效的字节值
        byte: u8,
        /// 字节在输入中的位置
        position: usize,
    },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::ArithmeticOverflow => {
                write!(f, "decoded number would overflow u128")
            }
            DecodeError::InvalidCharacter { byte, position } => {
                write!(
                    f,
                    "invalid base62 character '{}' (0x{:02X}) at position {}",
                    byte.escape_ascii(),
                    byte,
                    position
                )
            }
        }
    }
}

impl ::core::error::Error for DecodeError {}

// ============================================================================
// 核心算法
// ============================================================================

/// 使用魔术数快速计算 u128 / BASE_TO_10
///
/// # 返回值
///
/// (商, 余数)
///
/// # 算法
///
/// 使用定点算术避免昂贵的 u128 除法：
/// - quotient = (num * MULTIPLY) >> (128 + SHIFT)
/// - remainder = num - quotient * BASE_TO_10
#[inline(always)]
fn fast_div_base_to_10(num: u128) -> (u128, u64) {
    let quotient = mulh(num, DIV_BASE_TO_10_MULTIPLY) >> DIV_BASE_TO_10_SHIFT;
    let remainder = num - quotient * BASE_TO_10_U128;
    (quotient, remainder as u64)
}

/// 计算两个 u128 相乘的高 128 位
///
/// # 算法
///
/// 将输入分解为 64 位块进行乘法：
/// ```text
/// x = x_hi * 2^64 + x_lo
/// y = y_hi * 2^64 + y_lo
/// x * y = x_hi * y_hi * 2^128 + (x_hi * y_lo + x_lo * y_hi) * 2^64 + x_lo * y_lo
/// ```
#[inline(always)]
const fn mulh(x: u128, y: u128) -> u128 {
    let x_lo = x as u64 as u128;
    let x_hi = x >> 64;
    let y_lo = y as u64 as u128;
    let y_hi = y >> 64;

    let z0 = x_lo * y_lo;
    let z1 = x_lo * y_hi;
    let z2 = x_hi * y_lo;
    let z3 = x_hi * y_hi;

    let carry = (z0 >> 64) + (z1 as u64 as u128) + (z2 as u64 as u128);
    z3 + (z1 >> 64) + (z2 >> 64) + (carry >> 64)
}

// ============================================================================
// 公共 API
// ============================================================================

/// 将 u128 编码为固定长度的 base62 字符串
///
/// # 参数
///
/// - `num`: 要编码的数值
/// - `buf`: 输出缓冲区，必须恰好为 [`BASE62_LEN`] 字节
///
/// # 性能
///
/// 此函数经过高度优化：
/// - 使用两次快速除法将 u128 分解为三个 u64 块
/// - 每个块使用原生 u64 运算进行编码
/// - 无分支预测失败，无内存分配
///
/// # 示例
///
/// ```
/// # use base62_u128::{encode_fixed, BASE62_LEN};
/// let mut buf = [0u8; BASE62_LEN];
/// encode_fixed(u128::MAX, &mut buf);
/// assert_eq!(&buf[..], b"7n42DGM5Tflk9n8mt7Fhc7");
/// ```
#[inline]
pub fn encode_fixed(num: u128, buf: &mut [u8; BASE62_LEN]) {
    // 将 u128 分解为三个块：
    // num = high * (62^10)^2 + mid * 62^10 + low
    let (quotient, low) = fast_div_base_to_10(num);
    let (high, mid) = fast_div_base_to_10(quotient);

    // 编码各个块
    // SAFETY: 所有索引都在编译时已知的范围内
    unsafe {
        // 低 10 位 -> buf[12..22]
        encode_u64_chunk(low, 10, buf.as_mut_ptr().add(12));
        // 中 10 位 -> buf[2..12]
        encode_u64_chunk(mid, 10, buf.as_mut_ptr().add(2));
        // 高 2 位 -> buf[0..2]
        encode_u64_chunk(high as u64, 2, buf.as_mut_ptr());
    }
}

/// 编码 u64 值到指定长度的 base62 字符串
///
/// # Safety
///
/// 调用者必须确保：
/// - `ptr` 指向至少 `len` 字节的有效内存
/// - `num` 编码后不会超过 `len` 个字符
#[inline(always)]
unsafe fn encode_u64_chunk(mut num: u64, len: usize, ptr: *mut u8) {
    for i in (0..len).rev() {
        let digit = (num % BASE) as usize;
        num /= BASE;
        *ptr.add(i) = *CHARSET.get_unchecked(digit);
    }
}

/// 将固定长度的 base62 字符串解码为 u128
///
/// # 参数
///
/// - `buf`: 输入缓冲区，必须恰好为 [`BASE62_LEN`] 字节
///
/// # 错误
///
/// - [`DecodeError::InvalidCharacter`]: 输入包含非 base62 字符
/// - [`DecodeError::ArithmeticOverflow`]: 解码结果超出 u128 范围
///
/// # 示例
///
/// ```
/// # use base62_u128::{decode_fixed, BASE62_LEN};
/// let input = b"7n42DGM5Tflk9n8mt7Fhc7";
/// let buf: [u8; BASE62_LEN] = input.try_into().unwrap();
/// let decoded = decode_fixed(&buf).unwrap();
/// assert_eq!(decoded, u128::MAX);
/// ```
pub fn decode_fixed(buf: &[u8; BASE62_LEN]) -> Result<u128, DecodeError> {
    let mut result = 0u128;

    for (position, &byte) in buf.iter().enumerate() {
        // 使用查找表快速获取字符值
        let value = DECODE_LUT[byte as usize];
        if value == 0xFF {
            return Err(DecodeError::InvalidCharacter { byte, position });
        }

        // 安全地累加结果，检查溢出
        result = result
            .checked_mul(BASE as u128)
            .and_then(|r| r.checked_add(value as u128))
            .ok_or(DecodeError::ArithmeticOverflow)?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let test_values = [0u128, 1, 61, 62, 3843, u64::MAX as u128, u128::MAX / 2, u128::MAX];

        for &value in &test_values {
            let mut buf = [0u8; BASE62_LEN];
            encode_fixed(value, &mut buf);
            let decoded = decode_fixed(&buf).unwrap();
            assert_eq!(value, decoded, "Failed for value: {}", value);
        }
    }

    #[test]
    fn test_invalid_decode() {
        let mut buf = [b'0'; BASE62_LEN];
        buf[0] = b'!'; // Invalid character

        match decode_fixed(&buf) {
            Err(DecodeError::InvalidCharacter { byte: b'!', position: 0 }) => {}
            other => panic!("Expected InvalidCharacter error, got: {:?}", other),
        }
    }
}
