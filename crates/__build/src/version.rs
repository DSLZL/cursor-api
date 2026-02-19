/// 版本发布阶段
#[derive(Debug, Clone, Copy)]
pub enum ReleaseStage {
  /// 正式发布版本
  Release,
  /// 预览版本，格式如 `-pre.6` 或 `-pre.6+build.8`
  Preview {
    /// 预览版本号
    version: u16,
    /// 构建号（可选）
    build: Option<u16>,
  },
}

impl core::fmt::Display for ReleaseStage {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    match self {
      ReleaseStage::Release => Ok(()),
      ReleaseStage::Preview { version, build: None } => {
        write!(f, "-pre.{version}")
      }
      ReleaseStage::Preview { version, build: Some(build) } => {
        write!(f, "-pre.{version}+build.{build}")
      }
    }
  }
}

/// 遵循格式：v0.4.0-pre.6+build.8
#[derive(Debug, Clone, Copy)]
pub struct Version {
  pub major: u16,
  pub minor: u16,
  pub patch: u16,
  pub stage: ReleaseStage,
}

impl core::fmt::Display for Version {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
    self.stage.fmt(f)
  }
}

/// 版本字符串解析错误
#[allow(clippy::enum_variant_names)]
#[derive(Debug)]
pub enum ParseError {
  /// 整体格式错误（如缺少必需部分）
  InvalidFormat,
  /// 数字解析失败
  InvalidNumber,
  /// pre 部分格式错误
  InvalidPreRelease,
  /// build 部分格式错误
  InvalidBuild,
  // /// 正式版不能带 build 标识
  // BuildWithoutPreview,
}

impl core::fmt::Display for ParseError {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    match self {
      ParseError::InvalidFormat => write!(f, "invalid version format"),
      ParseError::InvalidNumber => write!(f, "invalid number in version"),
      ParseError::InvalidPreRelease => write!(f, "invalid pre-release format"),
      ParseError::InvalidBuild => write!(f, "invalid build format"),
      // ParseError::BuildWithoutPreview => {
      //     write!(f, "build metadata cannot exist without pre-release version")
      // }
    }
  }
}

impl ::core::error::Error for ParseError {}

impl core::str::FromStr for Version {
  type Err = ParseError;
  fn from_str(s: &str) -> core::result::Result<Self, Self::Err> {
    // 按 '-' 分割基础版本号和扩展部分
    let (base, extension) = match s.split_once('-') {
      Some((base, ext)) => (base, Some(ext)),
      None => (s, None),
    };

    // 解析基础版本号 major.minor.patch
    let mut parts: [u16; 3] = [0, 0, 0];
    let mut parsed_count = 0;
    for (i, s) in base.split('.').enumerate() {
      if i >= parts.len() {
        return Err(ParseError::InvalidFormat);
      }
      parts[i] = s.parse().map_err(|_| ParseError::InvalidNumber)?;
      parsed_count += 1;
    }
    if parsed_count != 3 {
      return Err(ParseError::InvalidFormat);
    }

    let major = parts[0];
    let minor = parts[1];
    let patch = parts[2];

    // 解析扩展部分（如果存在）
    let stage =
      if let Some(ext) = extension { parse_extension(ext)? } else { ReleaseStage::Release };

    Ok(Version { major, minor, patch, stage })
  }
}

/// 解析扩展部分：pre.X 或 pre.X+build.Y
fn parse_extension(s: &str) -> core::result::Result<ReleaseStage, ParseError> {
  // 检查是否以 "pre." 开头
  // 移除 "pre." 前缀
  let Some(after_pre) = s.strip_prefix("pre.") else {
    return Err(ParseError::InvalidPreRelease);
  };

  // 按 '+' 分割 version 和 build 部分
  let (version_str, build_str) = match after_pre.split_once('+') {
    Some((ver, build_part)) => (ver, Some(build_part)),
    None => (after_pre, None),
  };

  // 解析 pre 版本号
  let version = version_str.parse().map_err(|_| ParseError::InvalidPreRelease)?;

  // 解析 build 号（如果存在）
  let build = if let Some(build_part) = build_str {
    // 检查格式是否为 "build.X"
    let Some(build_num_str) = build_part.strip_prefix("build.") else {
      return Err(ParseError::InvalidBuild);
    };

    let build_num = build_num_str.parse().map_err(|_| ParseError::InvalidBuild)?;

    Some(build_num)
  } else {
    None
  };

  Ok(ReleaseStage::Preview { version, build })
}
