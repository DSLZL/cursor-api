#![allow(private_bounds)]

//! 数字字符串化模块
//!
//! 用于在序列化/反序列化时将数字转换为字符串，
//! 特别适用于与 JavaScript 交互时避免精度损失的场景。

use core::{fmt, marker::PhantomData};
use serde_core::{Deserialize, Deserializer, Serialize, Serializer, de};

/// 密封特征，限制可以被字符串化的类型
mod private {
    use super::*;

    /// 在 JavaScript 必需以 BigInt 存储的类型
    pub trait BigInt: Sized + Copy {
        fn serialize_to<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error>;
        fn deserialize_from<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error>;
    }

    impl BigInt for i64 {
        #[inline(always)]
        fn serialize_to<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.serialize_str(itoa::Buffer::new().format(*self))
        }

        #[inline(always)]
        fn deserialize_from<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            deserializer.deserialize_any(BigIntVisitor::<Self>(PhantomData))
        }
    }

    impl BigInt for u64 {
        #[inline(always)]
        fn serialize_to<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.serialize_str(itoa::Buffer::new().format(*self))
        }

        #[inline(always)]
        fn deserialize_from<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            deserializer.deserialize_any(BigIntVisitor::<Self>(PhantomData))
        }
    }

    struct BigIntVisitor<T: BigInt>(PhantomData<T>);

    impl<'de> de::Visitor<'de> for BigIntVisitor<i64> {
        type Value = i64;

        #[inline]
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("an integer or a string containing an integer")
        }

        #[inline]
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> { Ok(v) }

        #[inline]
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
            i64::try_from(v).map_err(|_| E::custom(format_args!("integer {v} is out of range")))
        }

        #[inline]
        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            v.parse().map_err(|_| E::custom(format_args!("invalid integer: {v}")))
        }
    }

    impl<'de> de::Visitor<'de> for BigIntVisitor<u64> {
        type Value = u64;

        #[inline]
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("an unsigned integer or a string containing an unsigned integer")
        }

        #[inline]
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> { Ok(v) }

        #[inline]
        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            v.parse().map_err(|_| E::custom(format_args!("invalid unsigned integer: {v}")))
        }
    }
}

trait Item: Sized {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error>;
    fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error>;
}

impl<T> Item for T
where T: private::BigInt
{
    #[inline(always)]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        T::serialize_to(self, serializer)
    }

    #[inline(always)]
    fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        T::deserialize_from(deserializer)
    }
}

struct OptVisitor<T>(PhantomData<T>);

impl<'de, T> de::Visitor<'de> for OptVisitor<T>
where T: private::BigInt
{
    type Value = Option<T>;

    #[inline]
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("an optional integer or string containing an integer")
    }

    #[inline]
    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> { Ok(None) }

    #[inline]
    fn visit_some<DE: Deserializer<'de>>(self, deserializer: DE) -> Result<Self::Value, DE::Error> {
        T::deserialize_from(deserializer).map(Some)
    }
}

impl<T> Item for Option<T>
where T: private::BigInt
{
    #[inline]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Some(value) => value.serialize(serializer),
            None => serializer.serialize_none(),
        }
    }

    #[inline]
    fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_option(OptVisitor(PhantomData))
    }
}

#[cfg(feature = "alloc")]
struct SerializeItem<'a, T: Item>(&'a T);

#[cfg(feature = "alloc")]
impl<T: Item> Serialize for SerializeItem<'_, T> {
    #[inline(always)]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        self.0.serialize(serializer)
    }
}

#[cfg(feature = "alloc")]
struct RepeatVisitor<T>(PhantomData<T>);

#[cfg(feature = "alloc")]
impl<'de, T> de::Visitor<'de> for RepeatVisitor<T>
where T: private::BigInt
{
    type Value = ::alloc::vec::Vec<T>;

    #[inline]
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a sequence of integers or strings containing integers")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where A: de::SeqAccess<'de> {
        let mut values = ::alloc::vec::Vec::with_capacity(seq.size_hint().unwrap_or(0));
        while let Some(wrapper) = seq.next_element::<Stringify<T>>()? {
            values.push(wrapper.0);
        }
        Ok(values)
    }
}

#[cfg(feature = "alloc")]
impl<T> Item for ::alloc::vec::Vec<T>
where T: private::BigInt
{
    #[inline]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter().map(SerializeItem))
    }

    #[inline]
    fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(RepeatVisitor(PhantomData))
    }
}

/// 字符串化包装器
///
/// 用于将数字类型在序列化时转换为字符串，在反序列化时兼容数字和字符串。
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Stringify<T>(pub T);

impl<T> Stringify<T> {
    #[inline(always)]
    #[must_use]
    pub const fn new(value: T) -> Self { Self(value) }

    #[inline(always)]
    #[must_use]
    pub const fn inner(&self) -> &T { &self.0 }

    #[inline(always)]
    #[must_use]
    pub fn into_inner(self) -> T { self.0 }
}

impl<T: Item> Serialize for Stringify<T> {
    #[inline(always)]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de, T: Item> Deserialize<'de> for Stringify<T> {
    #[inline(always)]
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        T::deserialize(deserializer).map(Self)
    }
}

/// 用于 `#[serde(with = "stringify")]` 的序列化函数
#[inline(always)]
pub fn serialize<T: Item, S: Serializer>(value: &T, serializer: S) -> Result<S::Ok, S::Error> {
    value.serialize(serializer)
}

/// 用于 `#[serde(with = "stringify")]` 的反序列化函数
#[inline(always)]
pub fn deserialize<'de, T: Item, D: Deserializer<'de>>(deserializer: D) -> Result<T, D::Error> {
    T::deserialize(deserializer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use serde_json::{from_str, to_string};

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestStruct {
        #[serde(with = "super")]
        large_number: u64,

        #[serde(with = "super")]
        optional_number: Option<i64>,

        #[cfg(feature = "alloc")]
        #[serde(with = "super")]
        id_list: ::alloc::vec::Vec<u64>,
    }

    #[test]
    fn test_stringify_wrapper_api() {
        let value = Stringify::new(9007199254740993u64);
        assert_eq!(*value.inner(), 9007199254740993u64);
        assert_eq!(value.into_inner(), 9007199254740993u64);

        let json = to_string(&Stringify(123u64)).unwrap();
        assert_eq!(json, r#""123""#);

        let de: Stringify<u64> = from_str(r#""123""#).unwrap();
        assert_eq!(de.0, 123);
    }

    #[test]
    fn test_struct_serialization() {
        let test = TestStruct {
            large_number: 9007199254740993u64,
            optional_number: Some(-9007199254740993i64),
            #[cfg(feature = "alloc")]
            id_list: ::alloc::vec![1, 2],
        };

        let json = to_string(&test).unwrap();
        assert!(json.contains(r#""large_number":"9007199254740993""#));
        assert!(json.contains(r#""optional_number":"-9007199254740993""#));

        #[cfg(feature = "alloc")]
        assert!(json.contains(r#""id_list":["1","2"]"#));

        let deserialized: TestStruct = from_str(&json).unwrap();
        assert_eq!(deserialized, test);
    }

    #[test]
    fn test_empty_and_none() {
        let test = TestStruct {
            large_number: 123,
            optional_number: None,
            #[cfg(feature = "alloc")]
            id_list: ::alloc::vec![],
        };

        let json = to_string(&test).unwrap();
        assert!(json.contains(r#""optional_number":null"#));
        #[cfg(feature = "alloc")]
        assert!(json.contains(r#""id_list":[]"#));

        let deserialized: TestStruct = from_str(&json).unwrap();
        assert_eq!(deserialized.optional_number, None);
        #[cfg(feature = "alloc")]
        assert!(deserialized.id_list.is_empty());
    }

    /// 验证能同时接受字符串和数字输入
    #[test]
    fn test_deserialization_robustness() {
        let mixed_json = r#"{
            "large_number": "9007199254740993",
            "optional_number": -123,
            "id_list": ["100", 200, "300"]
        }"#;

        #[cfg(not(feature = "alloc"))]
        let mixed_json = r#"{
            "large_number": "9007199254740993",
            "optional_number": -123
        }"#;

        let deserialized: TestStruct = from_str(mixed_json).unwrap();
        assert_eq!(deserialized.large_number, 9007199254740993u64);
        assert_eq!(deserialized.optional_number, Some(-123i64));

        #[cfg(feature = "alloc")]
        assert_eq!(deserialized.id_list, ::alloc::vec![100, 200, 300]);
    }

    #[test]
    fn test_error_handling() {
        use ::alloc::string::ToString as _;

        let err_json =
            r#"{ "large_number": "not_a_number", "optional_number": null, "id_list": [] }"#;
        #[cfg(not(feature = "alloc"))]
        let err_json = r#"{ "large_number": "not_a_number", "optional_number": null }"#;

        let err = from_str::<TestStruct>(err_json).unwrap_err();
        assert!(err.to_string().contains("invalid unsigned integer"));

        // u64::MAX + 1
        let overflow_json =
            r#"{ "large_number": "18446744073709551616", "optional_number": null, "id_list": [] }"#;
        #[cfg(not(feature = "alloc"))]
        let overflow_json =
            r#"{ "large_number": "18446744073709551616", "optional_number": null }"#;

        let err = from_str::<TestStruct>(overflow_json).unwrap_err();
        assert!(err.to_string().contains("invalid unsigned integer"));
    }
}
