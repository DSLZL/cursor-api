use super::model::ExchangeMap;
use crate::{
    app::constant::header::{ContentType, get_content_type_by_extension},
    common::utils::parse_from_env,
};
use alloc::borrow::Cow;
use bytes::Bytes;
use http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use manually_init::ManuallyInit;
use std::{
    fs,
    io::{self, Cursor, Read as _},
    path::{Path, PathBuf},
    pin::Pin,
};
use zip::ZipArchive;

type HashMap<K, V> = hashbrown::HashMap<K, V, ahash::RandomState>;
type HashSet<K> = hashbrown::HashSet<K, ahash::RandomState>;

// ============================================================================
// 公共类型定义
// ============================================================================

#[derive(serde::Serialize, serde::Deserialize)]
pub struct RouteRegistry {
    pub name: Option<String>,
    pub version: Option<String>,
    pub description: Option<String>,
    pub author: Option<String>,
    pub license: Option<String>,
    #[serde(skip_serializing, deserialize_with = "deserialize_routes")]
    pub routes: HashMap<String, RouteDef>,
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
pub enum RouteDefinition {
    String(String),
    Object(RouteDef),
}

#[derive(serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RouteDef {
    File { content_type: Option<ContentType>, path: String },
    Empty { status: u16, headers: Vec<(String, String)> },
    Exchange { from: String },
}

/// 反序列化 routes，支持简写格式
///
/// # 支持的格式
///
/// 简写格式（字符串直接映射到文件路径）：
/// ```json
/// {
///   "/index.html": "index.html",
///   "/app.js": "dist/app.js"
/// }
/// ```
///
/// 完整格式（对象）：
/// ```json
/// {
///   "/api": {
///     "type": "file",
///     "path": "index.html",
///     "content_type": "text/html"
///   },
///   "/redirect": {
///     "type": "empty",
///     "status": 301,
///     "headers": [["Location", "https://example.com"]]
///   }
/// }
/// ```
///
/// 混合格式：
/// ```json
/// {
///   "/": "index.html",
///   "/api": { "type": "file", "path": "api.json" }
/// }
/// ```
fn deserialize_routes<'de, D>(deserializer: D) -> Result<HashMap<String, RouteDef>, D::Error>
where D: serde::Deserializer<'de> {
    let raw_routes =
        <HashMap<String, RouteDefinition> as serde::Deserialize>::deserialize(deserializer)?;

    let mut routes =
        HashMap::with_capacity_and_hasher(raw_routes.len(), raw_routes.hasher().clone());

    for (path, definition) in raw_routes {
        let route_def = match definition {
            RouteDefinition::String(file_path) => {
                // 简写格式：字符串 → File 类型（无 Content-Type 指定）
                RouteDef::File { content_type: None, path: file_path }
            }
            RouteDefinition::Object(def) => {
                // 完整格式：直接使用对象
                def
            }
        };

        routes.insert(path, route_def);
    }

    Ok(routes)
}

#[derive(Clone)]
pub enum RouteService {
    File { content_type: HeaderValue, content: Bytes },
    Empty { status: StatusCode, headers: HeaderMap },
}

impl axum::response::IntoResponse for RouteService {
    fn into_response(self) -> axum::response::Response {
        match self {
            Self::File { content_type, content } => {
                let mut res = axum::body::Body::from(content).into_response();
                res.headers_mut().insert(http::header::CONTENT_TYPE, content_type);
                res
            }
            Self::Empty { status, headers } => {
                let mut res = ().into_response();
                *res.status_mut() = status;
                *res.headers_mut() = headers;
                res
            }
        }
    }
}

// ============================================================================
// 错误类型
// ============================================================================

#[derive(Debug)]
pub enum FrontendError {
    PathNotFound(Cow<'static, Path>),
    NotZipOrDirectory(Cow<'static, Path>),
    RegistryNotFound { searched_at: String },
    FileMissing { path: String, referenced_by: Vec<String> },
    InvalidJson { context: String, source: serde_json::Error },
    InvalidHeader { route: String, name: String, value: String, reason: String },
    InvalidStatus { route: String, status: u16 },
    NoExtension { route: String, path: String },
    IoError(io::Error),
    ZipError(zip::result::ZipError),
}

impl core::fmt::Display for FrontendError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::PathNotFound(path) => write!(f, "路径不存在: {}", path.display()),
            Self::NotZipOrDirectory(path) => {
                write!(f, "路径必须是目录或 .zip 文件: {}", path.display())
            }
            Self::RegistryNotFound { searched_at } => {
                write!(f, "找不到 route_registry.json，查找位置: {searched_at}")
            }
            Self::FileMissing { path, referenced_by } => {
                write!(f, "文件不存在: {}\n  被以下路由引用: {}", path, referenced_by.join(", "))
            }
            Self::InvalidJson { context, source } => {
                write!(f, "JSON 解析失败 ({context}): {source}")
            }
            Self::InvalidHeader { route, name, value, reason } => {
                write!(f, "路由 '{route}' 的 header 无效: {name} = {value} ({reason})")
            }
            Self::InvalidStatus { route, status } => {
                write!(f, "路由 '{route}' 的状态码无效: {status}")
            }
            Self::NoExtension { route, path } => {
                write!(f, "路由 '{route}' 的文件 '{path}' 缺少扩展名，无法推断 Content-Type")
            }
            Self::IoError(e) => write!(f, "IO 错误: {e}"),
            Self::ZipError(e) => write!(f, "ZIP 错误: {e}"),
        }
    }
}

impl ::core::error::Error for FrontendError {
    fn source(&self) -> Option<&(dyn ::core::error::Error + 'static)> {
        match self {
            Self::InvalidJson { source, .. } => Some(source),
            Self::IoError(e) => Some(e),
            Self::ZipError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for FrontendError {
    fn from(e: io::Error) -> Self { Self::IoError(e) }
}

impl From<zip::result::ZipError> for FrontendError {
    fn from(e: zip::result::ZipError) -> Self { Self::ZipError(e) }
}

static ROUTE_SERVICES: ManuallyInit<Vec<RouteService>> = ManuallyInit::new_with(Vec::new());

#[derive(Clone, Copy)]
pub struct RouteServiceFn {
    idx: usize,
}

impl FnOnce<()> for RouteServiceFn {
    type Output = Pin<Box<dyn Future<Output = RouteService> + Send>>;
    extern "rust-call" fn call_once(self, _: ()) -> Self::Output {
        Box::pin(async move { unsafe { ROUTE_SERVICES.get_unchecked(self.idx) }.clone() })
    }
}

// ============================================================================
// 资源提供者抽象
// ============================================================================

trait ResourceProvider {
    fn read_file(&mut self, relative_path: &str) -> Result<Vec<u8>, FrontendError>;
}

// 目录模式
struct DirectoryProvider {
    base_path: PathBuf,
}

impl ResourceProvider for DirectoryProvider {
    fn read_file(&mut self, relative_path: &str) -> Result<Vec<u8>, FrontendError> {
        let full_path = self.base_path.join(relative_path);
        fs::read(&full_path).map_err(|e| {
            if e.kind() == io::ErrorKind::NotFound {
                FrontendError::FileMissing {
                    path: relative_path.to_string(),
                    referenced_by: vec![],
                }
            } else {
                FrontendError::IoError(e)
            }
        })
    }
}

// ZIP 模式
struct ZipProvider {
    archive: ZipArchive<Cursor<Vec<u8>>>,
}

impl ResourceProvider for ZipProvider {
    fn read_file(&mut self, relative_path: &str) -> Result<Vec<u8>, FrontendError> {
        let mut file = self.archive.by_name(relative_path).map_err(|e| match e {
            zip::result::ZipError::FileNotFound => FrontendError::FileMissing {
                path: relative_path.to_string(),
                referenced_by: vec![],
            },
            other => FrontendError::ZipError(other),
        })?;

        let mut buffer = Vec::with_capacity(file.size() as usize);
        file.read_to_end(&mut buffer)?;
        Ok(buffer)
    }
}

// ============================================================================
// 前端加载器
// ============================================================================

struct FrontendLoader {
    provider: Box<dyn ResourceProvider>,
    registry: RouteRegistry,
}

impl FrontendLoader {
    fn from_path(path: Cow<'static, Path>) -> Result<Self, FrontendError> {
        if !path.exists() {
            return Err(FrontendError::PathNotFound(path));
        }

        if path.is_dir() {
            Self::from_directory(path)
        } else if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("zip") {
            Self::from_zip_file(path)
        } else {
            Err(FrontendError::NotZipOrDirectory(path))
        }
    }

    fn from_directory(dir_path: Cow<'static, Path>) -> Result<Self, FrontendError> {
        let registry_path = dir_path.join("route_registry.json");

        let json = fs::read_to_string(&registry_path).map_err(|e| {
            if e.kind() == io::ErrorKind::NotFound {
                FrontendError::RegistryNotFound { searched_at: registry_path.display().to_string() }
            } else {
                FrontendError::IoError(e)
            }
        })?;

        let registry: RouteRegistry =
            serde_json::from_str(&json).map_err(|e| FrontendError::InvalidJson {
                context: format!("route_registry.json at {}", registry_path.display()),
                source: e,
            })?;

        let provider = Box::new(DirectoryProvider { base_path: dir_path.into_owned() });

        Ok(Self { provider, registry })
    }

    fn from_zip_file(zip_path: Cow<'static, Path>) -> Result<Self, FrontendError> {
        let bytes = fs::read(&zip_path)?;
        let cursor = Cursor::new(bytes);
        let mut archive = ZipArchive::new(cursor)?;

        // 读取 route_registry.json
        let registry = {
            let mut registry_file =
                archive.by_name("route_registry.json").map_err(|e| match e {
                    zip::result::ZipError::FileNotFound => FrontendError::RegistryNotFound {
                        searched_at: format!("{}/route_registry.json", zip_path.display()),
                    },
                    other => FrontendError::ZipError(other),
                })?;

            let mut json = String::new();
            registry_file.read_to_string(&mut json)?;

            serde_json::from_str(&json).map_err(|e| FrontendError::InvalidJson {
                context: format!("route_registry.json in {}", zip_path.display()),
                source: e,
            })?
        };

        let provider = Box::new(ZipProvider { archive });

        Ok(Self { provider, registry })
    }

    fn build_routes(
        mut self,
    ) -> Result<(HashMap<&'static str, RouteServiceFn>, ExchangeMap), FrontendError> {
        // 收集唯一路径和反向映射
        let random_state = ahash::RandomState::new();
        let mut unique_paths = HashSet::with_hasher(random_state.clone());
        let mut path_to_routes: HashMap<String, Vec<String>> =
            HashMap::with_hasher(random_state.clone());

        for (route_path, route_def) in &self.registry.routes {
            if let RouteDef::File { path, .. } = route_def {
                unique_paths.insert(path.as_str());
                path_to_routes.entry(path.clone()).or_default().push(route_path.clone());
            }
        }

        // 批量读取文件（去重缓存）
        let mut file_cache: HashMap<String, Bytes> =
            HashMap::with_capacity_and_hasher(unique_paths.len(), random_state.clone());

        for path in unique_paths {
            let data = self.provider.read_file(path).map_err(|e| match e {
                FrontendError::FileMissing { path, .. } => FrontendError::FileMissing {
                    referenced_by: __unwrap!(path_to_routes.get(&path).cloned()),
                    path,
                },
                other => other,
            })?;

            file_cache.insert(path.to_string(), Bytes::from(data));
        }

        // 构建 RouteService
        let exchange_len = self
            .registry
            .routes
            .values()
            .filter(|v| matches!(*v, RouteDef::Exchange { .. }))
            .count();
        let mut exchange_map =
            HashMap::with_capacity_and_hasher(exchange_len, random_state.clone());
        let mut routes = HashMap::with_capacity_and_hasher(
            self.registry.routes.len() - exchange_len,
            random_state,
        );

        for (route_path, route_def) in self.registry.routes {
            let opt = match route_def {
                RouteDef::File { content_type, path } => {
                    let content = __unwrap!(file_cache.get(&path)).clone();

                    let ct = Self::resolve_content_type(content_type, &path, &route_path)?;

                    let service = RouteService::File { content_type: ct, content };
                    Some((route_path, service))
                }
                RouteDef::Empty { status, headers } => {
                    let Ok(status) = StatusCode::from_u16(status) else {
                        return Err(FrontendError::InvalidStatus { route: route_path, status });
                    };

                    let headers = Self::build_header_map(&route_path, headers)?;

                    let service = RouteService::Empty { status, headers };
                    Some((route_path, service))
                }
                RouteDef::Exchange { from } => {
                    exchange_map.insert(from, route_path);
                    None
                }
            };
            if let Some((route_path, service)) = opt {
                routes.insert(
                    match unsafe { &mut *ROUTE_REGISTRY.get_ptr() }
                        .entry(Box::leak(route_path.into_boxed_str()))
                    {
                        hashbrown::hash_set::Entry::Vacant(entry) => *entry.insert().get(),
                        _ => unreachable!(), // 键不会重复，因为route_path是从一个HashMap取出的键
                    },
                    {
                        let idx = ROUTE_SERVICES.len();
                        unsafe { &mut *ROUTE_SERVICES.get_ptr() }.push(service);
                        RouteServiceFn { idx }
                    },
                );
            }
        }

        Ok((routes, exchange_map.try_into().unwrap_or_default()))
    }

    fn resolve_content_type(
        content_type: Option<ContentType>,
        file_path: &str,
        route_path: &str,
    ) -> Result<HeaderValue, FrontendError> {
        match content_type {
            Some(ct) => Ok(ct.into()),
            None => {
                let extension = Path::new(file_path)
                    .extension()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| FrontendError::NoExtension {
                        route: route_path.to_string(),
                        path: file_path.to_string(),
                    })?;

                Ok(get_content_type_by_extension(extension))
            }
        }
    }

    fn build_header_map(
        route_path: &str,
        headers: Vec<(String, String)>,
    ) -> Result<HeaderMap, FrontendError> {
        let mut map = HeaderMap::with_capacity(headers.len());

        for (name, value) in headers {
            let header_name = match HeaderName::from_bytes(name.as_bytes()) {
                Ok(n) => n,
                Err(e) => {
                    return Err(FrontendError::InvalidHeader {
                        route: route_path.to_string(),
                        name,
                        value,
                        reason: format!("invalid header name: {e}"),
                    });
                }
            };

            let header_value = match HeaderValue::from_str(&value) {
                Ok(v) => v,
                Err(e) => {
                    return Err(FrontendError::InvalidHeader {
                        route: route_path.to_string(),
                        name,
                        value,
                        reason: format!("invalid header value: {e}"),
                    });
                }
            };

            map.append(header_name, header_value);
        }

        Ok(map)
    }
}

// ============================================================================
// 公共 API
// ============================================================================

static ROUTE_REGISTRY: ManuallyInit<HashSet<&'static str>> = ManuallyInit::new();

/// 初始化前端资源系统
///
/// # 环境变量
/// - `FRONTEND_PATH`: 前端资源路径（目录或 .zip 文件，默认: "frontend.zip"）
///
/// # 返回
/// - `Ok(Option<ExchangeMap>)`: 初始化成功
/// - `Err(FrontendError)`: 初始化失败
pub fn init_frontend() -> Result<(HashMap<&'static str, RouteServiceFn>, ExchangeMap), FrontendError>
{
    ROUTE_REGISTRY.init(HashSet::with_hasher(ahash::RandomState::new()));

    let path_str = parse_from_env("FRONTEND_PATH", "frontend.zip");
    let path = match path_str {
        Cow::Borrowed(p) => Cow::Borrowed(Path::new(p)),
        Cow::Owned(p) => Cow::Owned(PathBuf::from(p)),
    };

    let loader = FrontendLoader::from_path(path)?;

    print_metadata(&loader.registry);

    let result = loader.build_routes()?;

    println!("前端路由加载完成，共 {} 个路由", result.0.len());

    Ok(result)
}

pub fn paths() -> impl Iterator<Item = &'static str> {
    let registry = &*ROUTE_REGISTRY;

    registry.iter().copied()
}

static mut METADATA: Option<&'static str> = None;

pub const fn metadata() -> Option<&'static str> { unsafe { METADATA } }

// /// 处理前端请求
// pub async fn handle_frontend(parts: http::request::Parts) -> RouteService {
//     let registry = &*ROUTE_REGISTRY;

//     // registry.get(parts.uri.path()).cloned().expect("route not found in registry")
//     __unwrap!(registry.get(parts.uri.path())).clone()
// }

fn print_metadata(registry: &RouteRegistry) {
    unsafe {
        METADATA =
            Some(Box::leak(serde_json::to_string(registry).unwrap_unchecked().into_boxed_str()))
    };

    // println!(
    //     "========================================\n\
    //         前端资源包信息\n\
    //         ========================================"
    // );

    // if let Some(name) = &registry.name {
    //     println!("名称:       {name}");
    // }

    // if let Some(version) = &registry.version {
    //     println!("版本:       {version}");
    // }

    // if let Some(description) = &registry.description {
    //     println!("描述:       {description}");
    // }

    // if let Some(author) = &registry.author {
    //     println!("作者:       {author}");
    // }

    // if let Some(license) = &registry.license {
    //     println!("许可证:     {license}");
    // }

    // println!("========================================");
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use zip::write::{FileOptions, ZipWriter};

    impl ZipProvider {
        fn new(bytes: Vec<u8>) -> Result<Self, FrontendError> {
            let cursor = Cursor::new(bytes);
            let archive = ZipArchive::new(cursor)?;
            Ok(Self { archive })
        }

        // fn from_file(path: &Path) -> Result<Self, FrontendError> {
        //     let bytes = fs::read(path)?;
        //     Self::new(bytes)
        // }
    }

    fn create_test_zip() -> Vec<u8> {
        let mut buffer = Vec::new();
        let mut zip = ZipWriter::new(Cursor::new(&mut buffer));

        let options =
            FileOptions::<()>::default().compression_method(zip::CompressionMethod::Deflated);

        zip.start_file("route_registry.json", options).unwrap();
        zip.write_all(br#"{"routes":{}}"#).unwrap();

        zip.start_file("test.txt", options).unwrap();
        zip.write_all(b"hello from zip").unwrap();

        zip.finish().unwrap();
        buffer
    }

    #[test]
    fn test_directory_provider() {
        let temp_dir = std::env::temp_dir().join("frontend_test_dir");
        fs::create_dir_all(&temp_dir).unwrap();

        fs::write(temp_dir.join("test.txt"), b"hello world").unwrap();

        let mut provider = DirectoryProvider { base_path: temp_dir.clone() };
        let content = provider.read_file("test.txt").unwrap();
        assert_eq!(content, b"hello world");

        fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[test]
    fn test_zip_provider() {
        let zip_data = create_test_zip();
        let mut provider = ZipProvider::new(zip_data).unwrap();

        let content = provider.read_file("test.txt").unwrap();
        assert_eq!(content, b"hello from zip");
    }

    #[test]
    fn test_build_routes_deduplication() {
        let temp_dir = std::env::temp_dir().join("frontend_test_dedup");
        fs::create_dir_all(&temp_dir).unwrap();

        fs::write(temp_dir.join("shared.js"), b"shared content").unwrap();

        let registry_content = r#"{
            "routes": {
                "/v1/shared": {"type": "file", "path": "shared.js"},
                "/v2/shared": {"type": "file", "path": "shared.js"}
            }
        }"#;
        fs::write(temp_dir.join("route_registry.json"), registry_content).unwrap();

        let loader = FrontendLoader::from_path(temp_dir.clone().into()).unwrap();
        let routes = loader.build_routes().unwrap().0;

        assert_eq!(routes.len(), 2);

        fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[test]
    fn test_error_context() {
        let temp_dir = std::env::temp_dir().join("frontend_test_error");
        fs::create_dir_all(&temp_dir).unwrap();

        let registry_content = r#"{
            "routes": {
                "/route1": {"type": "file", "path": "missing.js"},
                "/route2": {"type": "file", "path": "missing.js"}
            }
        }"#;
        fs::write(temp_dir.join("route_registry.json"), registry_content).unwrap();

        let loader = FrontendLoader::from_path(temp_dir.clone().into()).unwrap();
        let result = loader.build_routes();

        assert!(result.is_err());
        if let Err(FrontendError::FileMissing { path, referenced_by }) = result {
            assert_eq!(path, "missing.js");
            assert_eq!(referenced_by.len(), 2);
        } else {
            panic!("Expected FileMissing error");
        }

        fs::remove_dir_all(&temp_dir).unwrap();
    }
}
