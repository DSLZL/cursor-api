use core::fmt;

pub const ULID_LEN: usize = 26;
const ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

const LOOKUP: [u8; 256] = {
    let mut table = [255; 256];
    let mut i = 0;
    while i < ALPHABET.len() {
        table[ALPHABET[i] as usize] = i as u8;
        i += 1;
    }
    table
};

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub enum DecodeError {
    InvalidChar,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "invalid character")
    }
}

impl ::core::error::Error for DecodeError {}

#[inline]
pub fn from_bytes(encoded: &[u8; ULID_LEN]) -> Result<u128, DecodeError> {
    // SAFETY: We explicitly trust the indices within 0..26 and LOOKUP within 0..256.
    unsafe {
        // Fast Path: Check 1st char.
        // v0 > 7 covers both Overflow (8..31) and Invalid (255)
        let v0 = *LOOKUP.get_unchecked(*encoded.get_unchecked(0) as usize);
        if v0 > 7 {
            return Err(DecodeError::InvalidChar);
        }

        let mut value = v0 as u128;
        let mut check = 0;
        let mut i = 1;

        while i < ULID_LEN {
            // Using get_unchecked removes the bounds check panic infrastructure entirely.
            let byte = *encoded.get_unchecked(i);
            let val = *LOOKUP.get_unchecked(byte as usize);

            check |= val;
            value = (value << 5) | (val & 0x1f) as u128;
            i += 1;
        }

        if check > 31 {
            return Err(DecodeError::InvalidChar);
        }

        Ok(value)
    }
}

#[inline]
pub fn to_str(mut value: u128, buffer: &mut [u8; ULID_LEN]) -> &mut str {
    unsafe {
        let mut i = ULID_LEN;
        while i > 0 {
            i -= 1;
            // Write directly without bounds checks.
            *buffer.get_unchecked_mut(i) = *ALPHABET.get_unchecked((value & 0x1f) as usize);
            value >>= 5;
        }

        core::str::from_utf8_unchecked_mut(buffer)
    }
}
