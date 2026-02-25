use crate::app::{
    constant::{AUDIENCE, ISSUER, SCOPE, TYPE_SESSION, TYPE_WEB},
    model::{Randomness, SessionId, Subject},
};
use proto_value::stringify::Stringify;

pub struct TokenPayload {
    pub sub: Subject,
    pub time: Stringify<i64>,
    pub randomness: Randomness,
    pub exp: i64,
    pub is_session: bool,
    pub workos_session_id: SessionId,
}

// 定义所有常量
crate::define_typed_constants! {
    usize => {
        FIELD_COUNT = 8,
    }
    &'static str => {
        // 结构体名称
        STRUCT_NAME = "TokenPayload",

        // 字段名称
        FIELD_SUB = "sub",
        FIELD_TIME = "time",
        FIELD_RANDOMNESS = "randomness",
        FIELD_EXP = "exp",
        FIELD_ISS = "iss",
        FIELD_SCOPE = "scope",
        FIELD_AUD = "aud",
        FIELD_TYPE = "type",
        FIELD_WORKOS_SESSION_ID = "workosSessionId",

        // 错误消息
        MSG_EXPECTING_FIELD = "字段名称",
        MSG_EXPECTING_STRUCT = "结构体 TokenPayload",
        MSG_TYPE_INVALID = "type 字段值必须为 ",
        MSG_ISS_INVALID = "iss 字段值必须为 ",
        MSG_SCOPE_INVALID = "scope 字段值必须为 ",
        MSG_AUD_INVALID = "aud 字段值必须为 ",
    }
    &'static [&'static str] => {
        FIELD_NAMES = &[
            FIELD_SUB,
            FIELD_TIME,
            FIELD_RANDOMNESS,
            FIELD_EXP,
            FIELD_ISS,
            FIELD_SCOPE,
            FIELD_AUD,
            FIELD_TYPE,
            FIELD_WORKOS_SESSION_ID,
        ],
    }
}

impl ::serde::Serialize for TokenPayload {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: ::serde::Serializer {
        use serde::ser::SerializeStruct as _;

        let exist_workos_session_id = !self.workos_session_id.is_empty();
        let count = FIELD_COUNT + exist_workos_session_id as usize;
        let mut state = serializer.serialize_struct(STRUCT_NAME, count)?;
        state.serialize_field(FIELD_SUB, &self.sub)?;
        state.serialize_field(FIELD_TIME, &self.time)?;
        state.serialize_field(FIELD_RANDOMNESS, &self.randomness)?;
        state.serialize_field(FIELD_EXP, &self.exp)?;
        state.serialize_field(FIELD_ISS, ISSUER)?;
        state.serialize_field(FIELD_SCOPE, SCOPE)?;
        state.serialize_field(FIELD_AUD, AUDIENCE)?;
        state.serialize_field(FIELD_TYPE, if self.is_session { TYPE_SESSION } else { TYPE_WEB })?;
        if exist_workos_session_id {
            state.serialize_field(FIELD_WORKOS_SESSION_ID, &self.workos_session_id)?;
        }
        state.end()
    }
}

impl<'de> ::serde::Deserialize<'de> for TokenPayload {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: ::serde::Deserializer<'de> {
        use serde::de::{self, MapAccess, Visitor};

        #[derive(Clone, Copy)]
        enum Field {
            Sub,
            Time,
            Randomness,
            Exp,
            Iss,
            Scope,
            Aud,
            Type,
            WorkosSessionId,
        }

        static FIELDS: ::phf::Map<&'static str, Field> = phf::Map {
            key: 16287231350648472473u64,
            disps: &[(1u32, 3u32), (3u32, 0u32)],
            entries: &[
                (FIELD_SUB, Field::Sub),
                (FIELD_RANDOMNESS, Field::Randomness),
                (FIELD_ISS, Field::Iss),
                (FIELD_TYPE, Field::Type),
                (FIELD_WORKOS_SESSION_ID, Field::WorkosSessionId),
                (FIELD_TIME, Field::Time),
                (FIELD_SCOPE, Field::Scope),
                (FIELD_AUD, Field::Aud),
                (FIELD_EXP, Field::Exp),
            ],
        };

        impl<'de> ::serde::Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
            where D: ::serde::Deserializer<'de> {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(
                        &self,
                        formatter: &mut ::core::fmt::Formatter,
                    ) -> ::core::fmt::Result {
                        formatter.write_str(MSG_EXPECTING_FIELD)
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where E: de::Error {
                        match FIELDS.get(value) {
                            Some(field) => Ok(*field),
                            None => Err(de::Error::unknown_field(value, FIELD_NAMES)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct TokenPayloadVisitor;

        impl<'de> Visitor<'de> for TokenPayloadVisitor {
            type Value = TokenPayload;

            fn expecting(&self, formatter: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                formatter.write_str(MSG_EXPECTING_STRUCT)
            }

            fn visit_map<V>(self, mut map: V) -> Result<TokenPayload, V::Error>
            where V: MapAccess<'de> {
                let mut sub = None;
                let mut time = None;
                let mut randomness = None;
                let mut exp = None;
                let mut is_session = None;
                let mut workos_session_id = None;
                let mut iss_seen = false;
                let mut scope_seen = false;
                let mut aud_seen = false;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Sub => {
                            if sub.is_some() {
                                return Err(de::Error::duplicate_field(FIELD_SUB));
                            }
                            sub = Some(map.next_value()?);
                        }
                        Field::Time => {
                            if time.is_some() {
                                return Err(de::Error::duplicate_field(FIELD_TIME));
                            }
                            time = Some(map.next_value()?);
                        }
                        Field::Randomness => {
                            if randomness.is_some() {
                                return Err(de::Error::duplicate_field(FIELD_RANDOMNESS));
                            }
                            randomness = Some(map.next_value()?);
                        }
                        Field::Exp => {
                            if exp.is_some() {
                                return Err(de::Error::duplicate_field(FIELD_EXP));
                            }
                            exp = Some(map.next_value()?);
                        }
                        Field::Type => {
                            if is_session.is_some() {
                                return Err(de::Error::duplicate_field(FIELD_TYPE));
                            }
                            let value: String = map.next_value()?;
                            is_session = Some(match value.as_str() {
                                TYPE_SESSION => true,
                                TYPE_WEB => false,
                                _ => {
                                    return Err(de::Error::custom(format_args!(
                                        "{MSG_TYPE_INVALID}{TYPE_SESSION} 或 {TYPE_WEB}"
                                    )));
                                }
                            });
                        }
                        Field::WorkosSessionId => {
                            if workos_session_id.is_some() {
                                return Err(de::Error::duplicate_field(FIELD_WORKOS_SESSION_ID));
                            }
                            workos_session_id = Some(map.next_value()?);
                        }
                        Field::Iss => {
                            if iss_seen {
                                return Err(de::Error::duplicate_field(FIELD_ISS));
                            }
                            let value: String = map.next_value()?;
                            if value != ISSUER {
                                return Err(de::Error::custom(format_args!(
                                    "{MSG_ISS_INVALID}{ISSUER}"
                                )));
                            }
                            iss_seen = true;
                        }
                        Field::Scope => {
                            if scope_seen {
                                return Err(de::Error::duplicate_field(FIELD_SCOPE));
                            }
                            let value: String = map.next_value()?;
                            if value != SCOPE {
                                return Err(de::Error::custom(format_args!(
                                    "{MSG_SCOPE_INVALID}{SCOPE}"
                                )));
                            }
                            scope_seen = true;
                        }
                        Field::Aud => {
                            if aud_seen {
                                return Err(de::Error::duplicate_field(FIELD_AUD));
                            }
                            let value: String = map.next_value()?;
                            if value != AUDIENCE {
                                return Err(de::Error::custom(format_args!(
                                    "{MSG_AUD_INVALID}{AUDIENCE}"
                                )));
                            }
                            aud_seen = true;
                        }
                    }
                }

                // 检查必填字段
                let sub = sub.ok_or_else(|| de::Error::missing_field(FIELD_SUB))?;
                let time = time.ok_or_else(|| de::Error::missing_field(FIELD_TIME))?;
                let randomness =
                    randomness.ok_or_else(|| de::Error::missing_field(FIELD_RANDOMNESS))?;
                let exp = exp.ok_or_else(|| de::Error::missing_field(FIELD_EXP))?;
                let is_session = is_session.ok_or_else(|| de::Error::missing_field(FIELD_TYPE))?;
                let workos_session_id = workos_session_id.unwrap_or_else(SessionId::empty);

                // 检查必须存在的常量字段
                if !iss_seen {
                    return Err(de::Error::missing_field(FIELD_ISS));
                }
                if !scope_seen {
                    return Err(de::Error::missing_field(FIELD_SCOPE));
                }
                if !aud_seen {
                    return Err(de::Error::missing_field(FIELD_AUD));
                }

                Ok(TokenPayload { sub, time, randomness, exp, is_session, workos_session_id })
            }
        }

        deserializer.deserialize_struct(STRUCT_NAME, FIELD_NAMES, TokenPayloadVisitor)
    }
}
