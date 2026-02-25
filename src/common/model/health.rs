use super::ApiStatus;
use crate::{
    app::{model::DateTime, route::InfallibleSerialize},
    common::model::raw_json::{RawJson, serialize_as_option_raw_value, serialize_as_raw_value},
};
use serde::Serialize;

#[derive(Serialize)]
pub struct HealthCheckResponse {
    pub status: ApiStatus,
    #[serde(serialize_with = "serialize_as_raw_value")]
    pub service: &'static str,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_as_option_raw_value"
    )]
    pub frontend: Option<&'static str>,
    pub runtime: RuntimeStats,
    pub system: Option<SystemStats>,
    pub capabilities: Capabilities,
}

unsafe impl InfallibleSerialize for HealthCheckResponse {}

#[derive(Serialize)]
pub struct ServiceInfo {
    pub name: &'static str,
    pub version: &'static str,
    pub is_debug: bool,
    pub build: BuildInfo,
}

#[derive(Serialize)]
pub struct BuildInfo {
    // pub commit: Option<&'static str>,
    #[cfg(feature = "__preview")]
    pub version: u32,
    pub timestamp: &'static str,
    pub is_debug: bool,
    pub is_prerelease: bool,
}

#[derive(Serialize)]
pub struct RuntimeStats {
    pub started_at: DateTime,
    pub uptime_seconds: i64,
    pub requests: RequestStats,
}

#[derive(Serialize)]
pub struct RequestStats {
    pub total: u64,
    pub active: u64,
    pub errors: u64,
}

#[derive(Serialize)]
pub struct SystemStats {
    pub memory: MemoryInfo,
    pub cpu: CpuInfo,
}

#[derive(Serialize)]
pub struct MemoryInfo {
    pub used_bytes: u64,
    pub used_percentage: f32,
    pub available_bytes: u64,
}

#[derive(Serialize)]
pub struct CpuInfo {
    pub usage_percentage: f32,
    pub load_average: [f64; 3], // 1min, 5min, 15min
}

#[derive(Serialize)]
pub struct Capabilities {
    pub models: RawJson,
    pub endpoints: &'static [&'static str],
    pub features: &'static [&'static str],
}

static mut SERVICE_INFO: &'static str = "";

pub fn init_service_info() {
    unsafe {
        SERVICE_INFO = Box::leak(
            serde_json::to_string(&ServiceInfo {
                name: crate::app::constant::PKG_NAME,
                version: crate::app::constant::PKG_VERSION,
                is_debug: *crate::app::lazy::log::DEBUG,
                build: BuildInfo {
                    #[cfg(feature = "__preview")]
                    version: crate::app::constant::BUILD_VERSION,
                    timestamp: crate::app::constant::BUILD_TIMESTAMP,
                    is_debug: crate::app::constant::IS_DEBUG,
                    is_prerelease: crate::app::constant::IS_PRERELEASE,
                },
            })
            .unwrap_unchecked()
            .into_boxed_str(),
        )
    };
}

pub const fn service_info() -> &'static str { unsafe { SERVICE_INFO } }
