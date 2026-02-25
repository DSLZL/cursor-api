// use alloc::borrow::Cow;

// #[allow(private_bounds)]
// #[inline]
// pub fn string_from_utf8<V: StringFrom>(v: V) -> Option<String> {
//     if byte_str::is_valid_utf8(v.as_bytes()) {
//         Some(unsafe { String::from_utf8_unchecked(v.into_vec()) })
//     } else {
//         None
//     }
// }

// trait StringFrom: Sized {
//     fn as_bytes(&self) -> &[u8];
//     fn into_vec(self) -> Vec<u8>;
// }

// impl StringFrom for &[u8] {
//     #[inline(always)]
//     fn as_bytes(&self) -> &[u8] { self }
//     #[inline(always)]
//     fn into_vec(self) -> Vec<u8> { self.to_vec() }
// }

// impl StringFrom for Cow<'_, [u8]> {
//     #[inline(always)]
//     fn as_bytes(&self) -> &[u8] { self }
//     #[inline(always)]
//     fn into_vec(self) -> Vec<u8> { self.into_owned() }
// }

// mod private {
//     pub trait Sealed: Sized {}

//     impl Sealed for &[u8] {}
//     impl Sealed for super::Cow<'_, [u8]> {}
// }

/// 检查JSON片段中第一个分隔符后是否有空格
///
/// # 规则
/// - 查找第一个不在字符串内的 `:` 或 `,`
/// - 检查其后是否紧跟空格 (0x20)
/// - 不验证JSON格式正确性
///
/// # Examples
/// ```
/// assert_eq!(has_space_after_separator(b"{\"a\": 1}"), true);
/// assert_eq!(has_space_after_separator(b"{\"a\":1}"), false);
/// assert_eq!(has_space_after_separator(b"\"no separator\""), false);
/// ```
pub(super) const fn has_space_after_separator(json: &[u8]) -> bool {
    let mut in_string = false;
    let mut i = 0;

    while i < json.len() {
        let byte = json[i];

        if in_string {
            if byte == b'\\' {
                // 跳过转义字符（避免 \" 误判为字符串结束）
                i += 2;
                continue;
            }
            if byte == b'"' {
                in_string = false;
            }
        } else {
            match byte {
                b'"' => in_string = true,
                b':' | b',' => {
                    // 找到分隔符，检查下一个字节
                    return i + 1 < json.len() && json[i + 1] == b' ';
                }
                _ => {}
            }
        }

        i += 1;
    }

    false
}
