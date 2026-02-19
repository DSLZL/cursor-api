#![feature(const_trait_impl)]
#![feature(const_convert)]
#![feature(const_cmp)]
#![feature(const_default)]
#![feature(const_result_unwrap_unchecked)]
#![feature(core_intrinsics)]
#![allow(internal_features)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(non_camel_case_types)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

extern crate alloc;

mod arc_str;
mod str;

pub use arc_str::ArcStr;
pub use str::Str;

pub type InternedStr = ArcStr;
pub type StaticStr = &'static str;
pub type string = Str;

#[inline]
pub fn init() { arc_str::__init() }
