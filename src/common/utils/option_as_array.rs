use serde::{Serialize, Serializer, ser::SerializeTuple as _};

/// 将 Option<T> 序列化为 JSON 数组：
/// Some(v) -> [v]
/// None    -> []
pub fn serialize<S, T>(option: &Option<T>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    T: Serialize,
{
    // 计算元组长度：Some 为 1，None 为 0
    let len = option.is_some() as usize;
    let mut tup = serializer.serialize_tuple(len)?;
    match option {
        Some(value) => {
            // 序列化为单元素数组 [value]
            tup.serialize_element(value)?;
        }
        None => {
            // 序列化为空数组 []
        }
    }
    tup.end()
}

// /// 将 JSON 数组反序列化为 Option<T>：
// /// [v] -> Some(v)
// /// []  -> None
// pub fn deserialize<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
// where
//     D: Deserializer<'de>,
//     T: Deserialize<'de>,
// {
//     struct OptionArrayVisitor<T>(PhantomData<T>);
//     impl<'de, T> de::Visitor<'de> for OptionArrayVisitor<T>
//     where T: Deserialize<'de>
//     {
//         type Value = Option<T>;
//         fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
//             formatter.write_str("an empty array or an array with one element")
//         }
//         fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
//         where A: de::SeqAccess<'de> {
//             // 尝试读取数组的第一个元素
//             match seq.next_element()? {
//                 Some(value) => {
//                     // 读取到值，对应 [v] -> Some(value)
//                     Ok(Some(value))
//                 }
//                 None => {
//                     // 没有元素，对应 [] -> None
//                     Ok(None)
//                 }
//             }
//         }
//     }
//     deserializer.deserialize_seq(OptionArrayVisitor(PhantomData))
// }
