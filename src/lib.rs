#![allow(internal_features)]
#![feature(const_trait_impl)]
#![feature(allow_internal_unsafe)]
#![feature(allow_internal_unstable)]

pub const trait UnwrapUnchecked<T: Sized>: Sized {
    fn unwrap_unchecked(self) -> T;
}

impl<T> const UnwrapUnchecked<T> for Option<T> {
    #[inline(always)]
    fn unwrap_unchecked(self) -> T { unsafe { self.unwrap_unchecked() } }
}

impl<T, E> UnwrapUnchecked<T> for Result<T, E> {
    #[inline(always)]
    fn unwrap_unchecked(self) -> T { unsafe { self.unwrap_unchecked() } }
}

#[cfg(debug_assertions)]
#[doc(hidden)]
#[macro_export]
macro_rules! __unwrap {
    ($expr:expr) => {
        $expr.unwrap()
    };
}

#[cfg(not(debug_assertions))]
#[doc(hidden)]
#[macro_export]
macro_rules! __unwrap {
    ($expr:expr) => {
        $crate::UnwrapUnchecked::unwrap_unchecked($expr)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __unwrap_panic {
    ($result:expr) => {
        match $result {
            ::core::result::Result::Ok(t) => t,
            ::core::result::Result::Err(e) => $crate::__panic(&e),
        }
    };
}

#[doc(hidden)]
#[inline(never)]
#[cold]
pub fn __panic(error: &dyn ::core::fmt::Display) -> ! { ::core::panic!("{error}") }

#[doc(hidden)]
#[macro_export]
macro_rules! __unreachable {
    () => {{
        #[cfg(debug_assertions)]
        {
            unreachable!()
        }
        #[cfg(not(debug_assertions))]
        unsafe {
            ::core::hint::unreachable_unchecked()
        }
    }};
}

#[doc(hidden)]
#[allow_internal_unstable(cold_path)]
#[macro_export]
macro_rules! __cold_path {
    () => {
        ::core::hint::cold_path()
    };
}

#[cfg(unix)]
pub const LF: &[u8] = b"\n";

#[cfg(windows)]
pub const LF: &[u8] = b"\r\n";

#[doc(hidden)]
#[macro_export]
macro_rules! __print {
    ($expr:expr) => {
        $crate::_print($expr.as_bytes())
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __println {
    () => {
        $crate::_print($crate::LF)
    };
    ($expr:expr) => {
        $crate::_print(concat!($expr, "\n").as_bytes())
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __eprint {
    ($expr:expr) => {
        $crate::_eprint($expr.as_bytes())
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __eprintln {
    () => {
        $crate::_eprint($crate::LF)
    };
    ($expr:expr) => {
        $crate::_eprint(concat!($expr, "\n").as_bytes())
    };
}

fn print_to<T>(bytes: &'_ [u8], global_s: fn() -> T, label: &str)
where T: std::io::Write {
    if let Err(e) = global_s().write_all(bytes) {
        panic!("failed printing to {label}: {e}");
    }
}

#[doc(hidden)]
#[inline]
pub fn _print(bytes: &'_ [u8]) { print_to(bytes, std::io::stdout, "stdout"); }

#[doc(hidden)]
#[inline]
pub fn _eprint(bytes: &'_ [u8]) { print_to(bytes, std::io::stderr, "stderr"); }

/// See: https://github.com/rust-lang/rust/blob/main/library/core/src/num/niche_types.rs
/// [`core::num::niche_types`]
#[allow_internal_unsafe]
#[allow_internal_unstable(rustc_attrs, structural_match)]
#[macro_export]
macro_rules! define_valid_range_type {
    ($(
        $(#[$m:meta])*
        $vis:vis struct $name:ident($int:ident as $uint:ident in $low:literal..=$high:literal);
    )+) => {$(
        #[derive(Clone, Copy, Eq)]
        #[repr(transparent)]
        #[rustc_layout_scalar_valid_range_start($low)]
        #[rustc_layout_scalar_valid_range_end($high)]
        $(#[$m])*
        $vis struct $name($int);

        const _: () = {
            // With the `valid_range` attributes, it's always specified as unsigned
            ::core::assert!(<$uint>::MIN == 0);
            let ulow: $uint = $low;
            let uhigh: $uint = $high;
            ::core::assert!(ulow <= uhigh);

            ::core::assert!(::core::mem::size_of::<$int>() == ::core::mem::size_of::<$uint>());
        };

        impl $name {
            pub const MIN: $name = unsafe { $name($low as $int) };
            pub const MAX: $name = unsafe { $name($high as $int) };

            #[inline]
            pub const fn new(val: $int) -> Option<Self> {
                if (val as $uint) >= ($low as $uint) && (val as $uint) <= ($high as $uint) {
                    // SAFETY: just checked the inclusive range
                    Some(unsafe { $name(val) })
                } else {
                    None
                }
            }

            /// Constructs an instance of this type from the underlying integer
            /// primitive without checking whether its zero.
            ///
            /// # Safety
            /// Immediate language UB if `val` is not within the valid range for this
            /// type, as it violates the validity invariant.
            #[inline]
            pub const unsafe fn new_unchecked(val: $int) -> Self {
                // SAFETY: Caller promised that `val` is within the valid range.
                unsafe { $name(val) }
            }

            #[inline]
            pub const fn as_inner(self) -> $int {
                // SAFETY: This is a transparent wrapper, so unwrapping it is sound
                // (Not using `.0` due to MCP#807.)
                unsafe { ::core::mem::transmute(self) }
            }
        }

        // This is required to allow matching a constant.  We don't get it from a derive
        // because the derived `PartialEq` would do a field projection, which is banned
        // by <https://github.com/rust-lang/compiler-team/issues/807>.
        impl ::core::marker::StructuralPartialEq for $name {}

        impl ::core::cmp::PartialEq for $name {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.as_inner() == other.as_inner()
            }
        }

        impl ::core::cmp::Ord for $name {
            #[inline]
            fn cmp(&self, other: &Self) -> ::core::cmp::Ordering {
                ::core::cmp::Ord::cmp(&self.as_inner(), &other.as_inner())
            }
        }

        impl ::core::cmp::PartialOrd for $name {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> ::core::option::Option<::core::cmp::Ordering> {
                ::core::option::Option::Some(::core::cmp::Ord::cmp(self, other))
            }
        }

        impl ::core::hash::Hash for $name {
            // Required method
            fn hash<H: ::core::hash::Hasher>(&self, state: &mut H) {
                ::core::hash::Hash::hash(&self.as_inner(), state);
            }
        }

        impl ::core::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                <$int as ::core::fmt::Debug>::fmt(&self.as_inner(), f)
            }
        }
    )+};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __impl_deref {
    ($ident:ident) => {
        impl<T> std::ops::Deref for $ident<T> {
            type Target = T;

            #[inline]
            fn deref(&self) -> &Self::Target { &self.0 }
        }

        impl<T> std::ops::DerefMut for $ident<T> {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
        }
    };

    ($ident:ident: $ty:ty) => {
        impl std::ops::Deref for $ident {
            type Target = $ty;

            #[inline]
            fn deref(&self) -> &Self::Target { &self.0 }
        }

        impl std::ops::DerefMut for $ident {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
        }
    };
}
