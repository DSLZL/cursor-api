use crate::app::constant::EMPTY_STRING;
use chrono::{DateTime, TimeDelta, Utc};
use core::{
    net::{Ipv4Addr, SocketAddrV4},
    sync::atomic::{AtomicI64, Ordering},
    time::Duration,
};
use manually_init::ManuallyInit;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::UdpSocket;

const PORT: u16 = 123;
const TIMEOUT_SECS: u64 = 5;
const PACKET_SIZE: usize = 48;
const VERSION: u8 = 4;
const MODE_CLIENT: u8 = 3;
const MODE_SERVER: u8 = 4;
/// 1900年1月1日到1970年1月1日的秒数差
const EPOCH_DELTA: i64 = 0x83AA7E80;

static SERVERS: ManuallyInit<Servers> = ManuallyInit::new();
/// 系统时间与准确时间的偏移量（纳秒）
/// 满足：系统时间 + DELTA = 准确时间
pub static DELTA: AtomicI64 = AtomicI64::new(0);

// ========== 错误类型定义 ==========

#[derive(Debug)]
pub enum NtpError {
    /// NTP 协议层错误
    Protocol(&'static str),
    /// 网络 I/O 错误
    Io(std::io::Error),
    /// 请求超时
    Timeout,
    /// 时间解析错误
    TimeParse,
}

impl std::fmt::Display for NtpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NtpError::Protocol(msg) => write!(f, "NTP协议错误: {msg}"),
            NtpError::Io(e) => write!(f, "IO错误: {e}"),
            NtpError::Timeout => write!(f, "NTP请求超时"),
            NtpError::TimeParse => write!(f, "时间解析错误"),
        }
    }
}

impl ::core::error::Error for NtpError {
    fn source(&self) -> Option<&(dyn ::core::error::Error + 'static)> {
        match self {
            NtpError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for NtpError {
    fn from(e: std::io::Error) -> Self { NtpError::Io(e) }
}

impl From<tokio::time::error::Elapsed> for NtpError {
    fn from(_: tokio::time::error::Elapsed) -> Self { NtpError::Timeout }
}

// ========== 服务器列表 ==========

pub struct Servers {
    inner: Box<[String]>,
}

impl Servers {
    /// 从环境变量 NTP_SERVERS 初始化服务器列表
    /// 格式：逗号分隔的服务器地址，如 "pool.ntp.org,time.cloudflare.com"
    pub fn init() {
        let env = crate::common::utils::parse_from_env("NTP_SERVERS", EMPTY_STRING);
        let servers: Vec<String> =
            env.split(',').map(str::trim).filter(|s| !s.is_empty()).map(String::from).collect();

        SERVERS.init(Self { inner: servers.into_boxed_slice() });
    }
}

impl IntoIterator for &'static Servers {
    type Item = &'static str;
    type IntoIter =
        core::iter::Map<core::slice::Iter<'static, String>, fn(&'static String) -> &'static str>;

    fn into_iter(self) -> Self::IntoIter { self.inner.iter().map(String::as_str) }
}

// ========== 时间转换函数 ==========

/// 将 NTP 64位时间戳转换为 Unix DateTime
/// NTP 时间戳格式：高32位为秒，低32位为秒的小数部分
fn ntp_to_unix_timestamp(ntp_ts: u64) -> DateTime<Utc> {
    let ntp_secs = (ntp_ts >> 32) as i64;
    let ntp_frac = ntp_ts & 0xFFFFFFFF;
    let unix_secs = ntp_secs - EPOCH_DELTA;
    let nanos = ((ntp_frac * 1_000_000_000) >> 32) as u32;

    unsafe { DateTime::from_timestamp(unix_secs, nanos).unwrap_unchecked() }
}

/// 将 SystemTime 转换为 NTP 64位时间戳
fn system_time_to_ntp_timestamp(t: SystemTime) -> Result<u64, NtpError> {
    let duration = t.duration_since(UNIX_EPOCH).map_err(|_| NtpError::TimeParse)?;
    let secs = duration.as_secs() + EPOCH_DELTA as u64;
    let nanos = duration.subsec_nanos() as u64;
    let frac = (nanos << 32) / 1_000_000_000;

    Ok((secs << 32) | frac)
}

// ========== NTP 时间戳结构 ==========

/// NTP 协议的四个关键时间戳
struct NtpTimestamps {
    t1: DateTime<Utc>, // 客户端发送时刻
    t2: DateTime<Utc>, // 服务器接收时刻
    t3: DateTime<Utc>, // 服务器发送时刻
    t4: DateTime<Utc>, // 客户端接收时刻
}

impl NtpTimestamps {
    /// 计算时钟偏移
    /// 公式：θ = [(T2-T1) + (T4-T3)] / 2
    /// 该公式假设网络延迟对称（去程与回程相等）
    fn clock_offset(&self) -> TimeDelta {
        let term1 = self.t2.signed_duration_since(self.t1);
        let term2 = self.t4.signed_duration_since(self.t3);
        (term1 + term2) / 2
    }

    /// 计算往返延迟（RTT）
    /// 公式：RTT = (T4-T1) - (T3-T2)
    #[allow(dead_code)]
    fn round_trip_delay(&self) -> TimeDelta {
        let total_time = self.t4.signed_duration_since(self.t1);
        let server_time = self.t3.signed_duration_since(self.t2);
        total_time - server_time
    }
}

/// 验证 NTP 响应包的有效性
/// 检查协议版本、模式、stratum 层级等字段
#[inline]
fn validate_ntp_response(packet: [u8; 48]) -> Result<(), NtpError> {
    let mode = packet[0] & 0x7;
    let version = (packet[0] & 0x38) >> 3;
    let stratum = packet[1];

    let stratum_desc = match stratum {
        0 => "未指定",
        1 => "主参考源",
        2..=15 => "二级服务器",
        16 => "未同步",
        _ => "无效",
    };

    crate::debug!("NTP响应: 版本={version}, 模式={mode}, 层级={stratum}({stratum_desc})");

    if mode != MODE_SERVER {
        return Err(NtpError::Protocol("响应模式不正确"));
    }

    match stratum {
        0 => Err(NtpError::Protocol("服务器返回Kiss-o'-Death包")),
        16 => Err(NtpError::Protocol("服务器未同步")),
        17..=255 => Err(NtpError::Protocol("服务器stratum值无效")),
        _ => Ok(()),
    }
}

// ========== 异步 UDP 操作 ==========

/// 创建并绑定 UDP 套接字到随机端口
#[inline]
async fn create_udp_socket() -> Result<UdpSocket, NtpError> {
    let socket = UdpSocket::bind(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0)).await?;
    Ok(socket)
}

/// 连接到可用的 NTP 服务器
/// 串行尝试服务器列表中的每一个，直到成功连接
async fn connect_to_ntp_server(socket: &UdpSocket) -> Result<&'static str, NtpError> {
    for server in SERVERS.get() {
        if socket.connect((server, PORT)).await.is_ok() {
            return Ok(server);
        }
    }
    Err(NtpError::Protocol("无法连接到任何NTP服务器"))
}

/// 发送 NTP 请求并接收响应
/// 返回：响应数据包和四个关键时间戳
async fn send_and_receive_ntp_packet(
    socket: &UdpSocket,
) -> Result<([u8; PACKET_SIZE], NtpTimestamps), NtpError> {
    let mut packet = [0u8; PACKET_SIZE];
    packet[0] = (VERSION << 3) | MODE_CLIENT;

    // 记录 T1：客户端发送时刻
    let t1_system = SystemTime::now();
    let t1_ntp = system_time_to_ntp_timestamp(t1_system)?;

    // 将 T1 写入数据包的 Transmit Timestamp 字段（字节 40-47）
    packet[40..48].copy_from_slice(&t1_ntp.to_be_bytes());

    // 发送数据包（带超时保护）
    tokio::time::timeout(Duration::from_secs(TIMEOUT_SECS), socket.send(&packet)).await??;

    // 接收响应（带超时保护）
    tokio::time::timeout(Duration::from_secs(TIMEOUT_SECS), socket.recv(&mut packet)).await??;

    // 记录 T4：客户端接收时刻（尽可能接近接收瞬间）
    let t4_system = SystemTime::now();
    let t4_ntp = system_time_to_ntp_timestamp(t4_system)?;

    // 从响应包中提取 T2 和 T3
    let t2_ntp = u64::from_be_bytes(packet[32..40].try_into().unwrap());
    let t3_ntp = u64::from_be_bytes(packet[40..48].try_into().unwrap());

    Ok((
        packet,
        NtpTimestamps {
            t1: ntp_to_unix_timestamp(t1_ntp),
            t2: ntp_to_unix_timestamp(t2_ntp),
            t3: ntp_to_unix_timestamp(t3_ntp),
            t4: ntp_to_unix_timestamp(t4_ntp),
        },
    ))
}

// ========== 核心同步函数 ==========

/// 执行一次 NTP 测量
/// 返回：(时钟偏移纳秒, 往返延迟纳秒)
async fn measure_once() -> Result<(i64, i64), NtpError> {
    let socket = create_udp_socket().await?;
    let server = connect_to_ntp_server(&socket).await?;

    let (packet, timestamps) = send_and_receive_ntp_packet(&socket).await?;
    validate_ntp_response(packet)?;
    let offset = timestamps.clock_offset();
    let rtt = timestamps.round_trip_delay();
    let offset_nanos =
        offset.num_nanoseconds().ok_or(NtpError::Protocol("时间偏移超出 i64 范围"))?;

    let rtt_nanos = rtt.num_nanoseconds().ok_or(NtpError::Protocol("RTT 超出 i64 范围"))?;
    crate::debug!(
        "NTP采样: 服务器={}, 偏移={}ms, RTT={}ms",
        server,
        offset_nanos / 1_000_000,
        rtt_nanos / 1_000_000
    );
    Ok((offset_nanos, rtt_nanos))
}

/// 执行一次完整的 NTP 同步流程（多次采样 + 加权平均）
///
/// 流程：
/// 1. 根据配置多次调用 `measure_once()`（默认 8 次，间隔 50ms）
/// 2. 收集成功的样本
/// 3. 按 RTT 排序，过滤掉最大的几个
/// 4. 清洗数据：跳过 RTT <= 0 的异常样本
/// 5. 对剩余样本按 1/RTT 加权平均
///
/// 返回：加权平均后的时钟偏移量（纳秒）
pub async fn sync_once() -> Result<i64, NtpError> {
    let (sample_count, interval_ms) = parse_sample_config();
    let mut samples = Vec::with_capacity(sample_count);

    // 1. 多次采样
    for i in 0..sample_count {
        match measure_once().await {
            Ok((delta, rtt)) => {
                samples.push((delta, rtt));
            }
            Err(e) => {
                crate::debug!("NTP采样失败 ({}/{}): {}", i + 1, sample_count, e);
            }
        }

        // 采样间隔（最后一次不需要等待）
        if i + 1 < sample_count {
            tokio::time::sleep(Duration::from_millis(interval_ms)).await;
        }
    }

    // 2. 检查成功样本数
    let success_count = samples.len();
    if success_count < 3 {
        return Err(NtpError::Protocol("成功采样数不足 3 个，无法计算可靠结果"));
    }

    crate::debug!("NTP采样完成: 成功 {success_count}/{sample_count} 次");

    // 3. 按 RTT 排序（从小到大）
    samples.sort_by_key(|(_, rtt)| *rtt);

    // 4. 动态过滤：基于比例策略
    // 保留 RTT 最小的 70% 样本，去除长尾噪声（网络延迟服从长尾分布）
    // 至少保留 4 个样本以保证统计可靠性，但不超过实际样本数
    const KEEP_RATIO: f64 = 0.70;
    const MIN_KEEP: usize = 4;

    let keep_count =
        ((success_count as f64 * KEEP_RATIO).ceil() as usize).max(MIN_KEEP).min(success_count);

    let valid_samples = &samples[..keep_count];
    let filter_count = success_count - keep_count;

    crate::debug!(
        "NTP过滤: 采样={}, 保留={}({:.0}%), 过滤={}",
        success_count,
        keep_count,
        (keep_count as f64 / success_count as f64) * 100.0,
        filter_count
    );

    // 5. 数据清洗：跳过 RTT <= 0 的异常样本（已排序，连续在前）
    let first_valid_idx = valid_samples
        .iter()
        .position(|(_, rtt)| *rtt > 0)
        .ok_or(NtpError::Protocol("所有样本的 RTT 异常(<=0)"))?;

    if first_valid_idx > 0 {
        crate::debug!("跳过 {} 个 RTT<=0 的异常样本", first_valid_idx);
    }

    let clean_samples = &valid_samples[first_valid_idx..];

    // 6. 加权平均：权重 = 1 / RTT
    let mut weighted_sum = 0.0f64;
    let mut weight_sum = 0.0f64;

    for (delta, rtt) in clean_samples {
        let weight = 1.0 / (*rtt as f64);
        weighted_sum += *delta as f64 * weight;
        weight_sum += weight;
    }

    let final_delta = (weighted_sum / weight_sum) as i64;

    // 7. 业务合理性检查：拒绝超过 ±1 天的偏移
    const MAX_REASONABLE_DELTA: i64 = 86400 * 1_000_000_000;

    if final_delta.abs() > MAX_REASONABLE_DELTA {
        crate::debug!("NTP偏移超出合理范围: δ={}s", final_delta / 1_000_000_000);
        return Err(NtpError::Protocol("时间偏移超出合理范围(±1天)"));
    }

    // 8. 统计信息
    let min_rtt = clean_samples.first().map(|(_, rtt)| rtt / 1_000_000).unwrap_or(0);
    let max_rtt = clean_samples.last().map(|(_, rtt)| rtt / 1_000_000).unwrap_or(0);

    crate::debug!(
        "NTP同步完成: δ = {}ms, RTT范围 = [{}, {}]ms",
        final_delta / 1_000_000,
        min_rtt,
        max_rtt
    );

    Ok(final_delta)
}

// ========== 环境变量解析 ==========

/// 从环境变量读取同步间隔
/// 默认值：3600 秒（1 小时）
fn parse_sync_interval() -> u64 {
    crate::common::utils::parse_from_env("NTP_SYNC_INTERVAL_SECS", 3600u64)
}

/// 解析采样配置
/// 返回：(采样次数, 采样间隔毫秒)
fn parse_sample_config() -> (usize, u64) {
    let count = crate::common::utils::parse_from_env("NTP_SAMPLE_COUNT", 8usize);
    let interval_ms = crate::common::utils::parse_from_env("NTP_SAMPLE_INTERVAL_MS", 50u64);
    (count, interval_ms)
}

// ========== 公共接口 ==========

/// 启动时执行一次 NTP 同步
///
/// 行为：
/// - 无服务器配置：静默返回，DELTA 保持为 0
/// - 同步失败：打印错误到标准输出，DELTA 保持为 0
/// - 同步成功：更新 DELTA，记录日志，**自动启动周期性同步任务**
pub async fn init_sync(stdout_ready: alloc::sync::Arc<tokio::sync::Notify>) {
    let servers = SERVERS.get();

    if servers.inner.is_empty() {
        stdout_ready.notified().await;
        __println!("\r\x1B[2K无NTP服务器配置, 若系统时间不准确可能导致UB");
        return;
    }

    let instant = std::time::Instant::now();

    match sync_once().await {
        Ok(delta_nanos) => {
            DELTA.store(delta_nanos, Ordering::Relaxed);
            let d = crate::common::utils::format_time_ms(instant.elapsed().as_secs_f64());
            stdout_ready.notified().await;
            println!("\r\x1B[2KNTP初始化完成: δ = {}ms, 耗时: {}s", delta_nanos / 1_000_000, d);

            // 初始化成功后自动启动周期性同步任务
            spawn_periodic_sync();
        }
        Err(e) => {
            stdout_ready.notified().await;
            println!("\r\x1B[2KNTP同步失败: {e}");
        }
    }
}

/// 启动周期性后台同步任务（由 `init_sync` 自动调用）
///
/// 启动条件（需同时满足）：
/// 1. 配置了 NTP 服务器（已由 init_sync 检查）
/// 2. 同步间隔大于 0（NTP_SYNC_INTERVAL_SECS > 0）
///
/// 任务行为：
/// - 在后台持续运行
/// - 按配置的间隔执行同步
/// - 首次 tick 被跳过（启动时已由 init_sync 完成同步）
/// - 同步失败时记录到日志文件，不影响后续同步
fn spawn_periodic_sync() {
    let interval_secs = parse_sync_interval();

    if interval_secs == 0 {
        return;
    }

    crate::debug!("启动NTP周期性同步任务: 间隔={interval_secs}秒");

    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));

        // 跳过首次 tick（init_sync 已完成初始同步）
        interval.tick().await;

        loop {
            interval.tick().await;

            match sync_once().await {
                Ok(delta_nanos) => {
                    DELTA.store(delta_nanos, Ordering::Relaxed);
                    crate::debug!("NTP周期性同步成功: δ = {}ms", delta_nanos / 1_000_000);
                }
                Err(e) => {
                    crate::debug!("NTP周期性同步失败: {e}");
                }
            }
        }
    });
}
