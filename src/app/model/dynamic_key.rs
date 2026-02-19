use super::token::RawToken;
use crate::{
    app::constant::{TYPE_SESSION, TYPE_WEB},
    common::utils::hex_to_byte,
};
use alloc::sync::Arc;
use arc_swap::ArcSwap;
use hmac::{Hmac, KeyInit as _, Mac as _};
use manually_init::ManuallyInit;
use sha2::{
    Digest as _, Sha256,
    digest::{FixedOutput, array::Array},
};

pub struct Secret(pub Option<[u8; 64]>);

impl Secret {
    pub fn parse_str(s: &str) -> Self {
        let Some(s) = s.split_whitespace().next() else {
            return Secret(None);
        };

        let mut result = [0; 64];

        if let Some(s) = s.strip_prefix("hex:")
            && s.len() >= 64
        {
            let bytes = s.as_bytes();
            let (hex_pairs, hex_rests) = bytes.as_chunks();

            let mut ok = true;

            for (&[hi, lo], dst) in hex_pairs.iter().zip(&mut result) {
                let Some(byte) = hex_to_byte(hi, lo) else {
                    ok = false;
                    break;
                };
                *dst = byte;
            }

            if ok
                && hex_pairs.len() < result.len()
                && let Some(&hi) = hex_rests.first()
            {
                if let Some(byte) = hex_to_byte(hi, b'0') {
                    result[hex_pairs.len()] = byte;
                } else {
                    ok = false;
                };
            }

            if ok {
                return Secret(Some(result));
            }
        }

        let out = {
            let ptr: *mut [u8; 32] = result.as_mut_ptr().cast();
            unsafe { &mut *ptr }
        };
        FixedOutput::finalize_into(Sha256::new().chain_update(s), out.into());

        Secret(Some(result))
    }
}

static INSTANCE: ManuallyInit<ArcSwap<Hmac<Sha256>>> = ManuallyInit::new();

pub fn init(secret: [u8; 64]) { INSTANCE.init(ArcSwap::from_pointee(Hmac::new(&Array(secret)))) }

pub fn update(secret: [u8; 64]) { INSTANCE.store(Arc::new(Hmac::new(&Array(secret)))) }

pub fn get_hash(raw: &RawToken) -> [u8; 32] {
    let mut hmac = (**INSTANCE.get().load()).clone();
    hmac.update(b"subject");
    hmac.update(raw.subject.provider.as_str().as_bytes());
    hmac.update(raw.subject.id.as_bytes());
    hmac.update(b"signature");
    hmac.update(&raw.signature);
    hmac.update(b"duration");
    hmac.update(&raw.duration.start.to_ne_bytes());
    hmac.update(&raw.duration.end.to_ne_bytes());
    hmac.update(b"randomness");
    hmac.update(&raw.randomness.to_bytes());
    hmac.update(b"type");
    hmac.update(if raw.is_session { TYPE_SESSION } else { TYPE_WEB }.as_bytes());
    hmac.update(b"workos\x1fsession\x1fid");
    hmac.update(raw.workos_session_id.as_bytes());
    hmac.finalize_fixed().0
}
