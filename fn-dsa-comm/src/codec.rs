#![allow(clippy::identity_op)]
#![allow(clippy::needless_range_loop)]

use core::fmt;

/// Error type for codec operations.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CodecError {
    /// The requested bit width is unsupported.
    InvalidNBits { nbits: u32 },

    /// The destination buffer is too short.
    DestinationTooShort { needed: usize, actual: usize },

    /// The source buffer is too short.
    SourceTooShort { needed: usize, actual: usize },

    /// The source buffer length does not match the expected length.
    SourceLengthMismatch { expected: usize, actual: usize },

    /// The coefficient count is invalid for the codec.
    InvalidCoefficientCount { actual: usize },

    /// A source coefficient is outside of the supported range.
    CoefficientOutOfRange,

    /// The encoded data is invalid or non-canonical.
    InvalidEncoding,
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidNBits { nbits } => {
                write!(f, "invalid coefficient bit width: {nbits}")
            }
            Self::DestinationTooShort { needed, actual } => write!(
                f,
                "destination buffer too short: need {needed} bytes, got {actual}",
            ),
            Self::SourceTooShort { needed, actual } => write!(
                f,
                "source buffer too short: need {needed} bytes, got {actual}",
            ),
            Self::SourceLengthMismatch { expected, actual } => write!(
                f,
                "source buffer length mismatch: expected {expected} bytes, got {actual}",
            ),
            Self::InvalidCoefficientCount { actual } => write!(
                f,
                "invalid coefficient count for codec: {actual}",
            ),
            Self::CoefficientOutOfRange => {
                write!(f, "source coefficient is outside of the supported range")
            }
            Self::InvalidEncoding => {
                write!(f, "invalid or non-canonical encoded data")
            }
        }
    }
}

const fn check_trim_nbits(nbits: u32) -> Result<(), CodecError> {
    if nbits < 2 || nbits > 8 {
        Err(CodecError::InvalidNBits { nbits })
    } else {
        Ok(())
    }
}

const fn trim_i8_size(n: usize, nbits: u32) -> Result<usize, CodecError> {
    if let Err(err) = check_trim_nbits(nbits) {
        Err(err)
    } else {
        Ok(((n * (nbits as usize)) + 7) >> 3)
    }
}

const fn modq_size(n: usize) -> Result<usize, CodecError> {
    if (n & 3) != 0 {
        Err(CodecError::InvalidCoefficientCount { actual: n })
    } else {
        Ok(7 * (n >> 2))
    }
}

/// Encode small integers into bytes, with a fixed size per value.
///
/// Encode the provided sequence of signed integers `f`, with `nbits` bits per
/// value, into the destination buffer `d`. The actual number of written bytes
/// is returned. If the total encoded size is not an integral number of bytes,
/// then extra padding bits of value 0 are used.
///
/// The source coefficients must all lie in the `[-(2^(nbits-1)-1),
/// +(2^(nbits-1)-1)]` range.
pub fn trim_i8_encode(
    f: &[i8],
    nbits: u32,
    d: &mut [u8],
) -> Result<usize, CodecError> {
    let needed = trim_i8_size(f.len(), nbits)?;
    if d.len() < needed {
        return Err(CodecError::DestinationTooShort {
            needed,
            actual: d.len(),
        });
    }
    let mut k = 0;
    let mut acc = 0;
    let mut acc_len = 0;
    let mask = (1u32 << nbits) - 1;
    let maxv = (1i32 << (nbits - 1)) - 1;
    let minv = -maxv;
    for i in 0..f.len() {
        let x = f[i] as i32;
        if x < minv || x > maxv {
            return Err(CodecError::CoefficientOutOfRange);
        }
        acc = (acc << nbits) | (((f[i] as u8) as u32) & mask);
        acc_len += nbits;
        while acc_len >= 8 {
            acc_len -= 8;
            d[k] = (acc >> acc_len) as u8;
            k += 1;
        }
    }
    if acc_len > 0 {
        d[k] = (acc << (8 - acc_len)) as u8;
        k += 1;
    }
    Ok(k)
}

/// Decode small integers from bytes, with a fixed size per value.
///
/// Decode the provided bytes `d` into the signed integers `f`, using
/// `nbits` bits per value. Exactly as many bytes as necessary are read
/// from `d` in order to fill the slice `f` entirely. The actual number
/// of bytes read from `d` is returned. An error is returned if any of the
/// following happens:
/// 
///  - Source buffer is not large enough.
///  - An invalid encoding (`-2^(nbits-1)`) is encountered.
///  - Some bits are unused in the last byte and are not all zero.
/// 
/// The number of bits per coefficient (nbits) MUST lie between 2 and 8
/// (inclusive).
pub fn trim_i8_decode(
    d: &[u8],
    f: &mut [i8],
    nbits: u32,
) -> Result<usize, CodecError> {
    let n = f.len();
    let needed = trim_i8_size(n, nbits)?;
    if d.len() < needed {
        return Err(CodecError::SourceTooShort {
            needed,
            actual: d.len(),
        });
    }
    let mut j = 0;
    let mut acc = 0;
    let mut acc_len = 0;
    let mask1 = (1 << nbits) - 1;
    let mask2 = 1 << (nbits - 1);
    for i in 0..needed {
        acc = (acc << 8) | (d[i] as u32);
        acc_len += 8;
        while acc_len >= nbits {
            acc_len -= nbits;
            let w = (acc >> acc_len) & mask1;
            let w = w | (w & mask2).wrapping_neg();
            if w == mask2.wrapping_neg() {
                return Err(CodecError::InvalidEncoding);
            }
            f[j] = w as i8;
            j += 1;
            if j >= n {
                break;
            }
        }
    }
    if (acc & ((1u32 << acc_len) - 1)) != 0 {
        // Some of the extra bits are non-zero.
        return Err(CodecError::InvalidEncoding);
    }
    Ok(needed)
}

/// Encode integers modulo 12289 into bytes, with 14 bits per value.
///
/// Encode the provided sequence of integers modulo q = 12289 into the
/// destination buffer `d`. Exactly 14 bits are used for each value.
/// The values MUST be in the `[0,q-1]` range. The number of source values
/// MUST be a multiple of 4.
pub fn modq_encode(h: &[u16], d: &mut [u8]) -> Result<usize, CodecError> {
    let needed = modq_size(h.len())?;
    if d.len() < needed {
        return Err(CodecError::DestinationTooShort {
            needed,
            actual: d.len(),
        });
    }
    let mut j = 0;
    for i in 0..(h.len() >> 2) {
        let x0 = h[4 * i + 0] as u64;
        let x1 = h[4 * i + 1] as u64;
        let x2 = h[4 * i + 2] as u64;
        let x3 = h[4 * i + 3] as u64;
        if x0 >= 12289 || x1 >= 12289 || x2 >= 12289 || x3 >= 12289 {
            return Err(CodecError::CoefficientOutOfRange);
        }
        let x = (x0 << 42) | (x1 << 28) | (x2 << 14) | x3;
        d[j..(j + 7)].copy_from_slice(&x.to_be_bytes()[1..8]);
        j += 7;
    }
    Ok(j)
}

/// Decode integers modulo 12289 from bytes, with 14 bits per value.
///
/// Decode some bytes into integers modulo q = 12289. Exactly as many
/// bytes as necessary are read from the source `d` to fill all values in
/// the destination slice `h`. The number of elements in `h` MUST be a
/// multiple of 4. The total number of read bytes is returned. If the
/// source is too short, of if any of the decoded values is invalid (i.e.
/// not in the `[0,q-1]` range), then this function returns an error.
pub fn modq_decode(d: &[u8], h: &mut [u16]) -> Result<usize, CodecError> {
    let n = h.len();
    let needed = modq_size(n)?;
    if d.len() != needed {
        return Err(CodecError::SourceLengthMismatch {
            expected: needed,
            actual: d.len(),
        });
    }
    if n == 0 {
        return Ok(0);
    }
    let mut ov = 0xFFFF;
    let x = ((d[0] as u64) << 48)
        | ((d[1] as u64) << 40)
        | ((d[2] as u64) << 32)
        | ((d[3] as u64) << 24)
        | ((d[4] as u64) << 16)
        | ((d[5] as u64) << 8)
        | (d[6] as u64);
    let h0 = ((x >> 42) as u32) & 0x3FFF;
    let h1 = ((x >> 28) as u32) & 0x3FFF;
    let h2 = ((x >> 14) as u32) & 0x3FFF;
    let h3 = (x as u32) & 0x3FFF;
    ov &= h0.wrapping_sub(12289);
    ov &= h1.wrapping_sub(12289);
    ov &= h2.wrapping_sub(12289);
    ov &= h3.wrapping_sub(12289);
    h[0] = h0 as u16;
    h[1] = h1 as u16;
    h[2] = h2 as u16;
    h[3] = h3 as u16;
    for i in 1..(n >> 2) {
        let x = u64::from_be_bytes(
            *<&[u8; 8]>::try_from(&d[(7 * i - 1)..(7 * i + 7)]).unwrap());
        let h0 = ((x >> 42) as u32) & 0x3FFF;
        let h1 = ((x >> 28) as u32) & 0x3FFF;
        let h2 = ((x >> 14) as u32) & 0x3FFF;
        let h3 = (x as u32) & 0x3FFF;
        ov &= h0.wrapping_sub(12289);
        ov &= h1.wrapping_sub(12289);
        ov &= h2.wrapping_sub(12289);
        ov &= h3.wrapping_sub(12289);
        h[4 * i + 0] = h0 as u16;
        h[4 * i + 1] = h1 as u16;
        h[4 * i + 2] = h2 as u16;
        h[4 * i + 3] = h3 as u16;
    }
    if (ov & 0x8000) == 0 {
        return Err(CodecError::InvalidEncoding);
    }
    Ok(needed)
}

/// Encode small integers into bytes using a compressed (Golomb-Rice) format.
///
/// Encode the provided source values `s` with compressed encoding. If
/// any of the source values is larger than 2047 (in absolute value),
/// then this function returns an error. If the destination buffer `d` is
/// not large enough, then this function returns an error. Otherwise, all
/// output buffer bytes are set (padding bits/bytes of value zero are
/// appended if necessary) and the number of payload bytes is returned.
pub fn comp_encode(s: &[i16], d: &mut [u8]) -> Result<usize, CodecError> {
    let mut acc = 0;
    let mut acc_len = 0;
    let mut j = 0;
    for i in 0..s.len() {
        // Invariant: acc_len <= 7 at the beginning of each iteration.

        let x = s[i] as i32;
        if !(-2047..=2047).contains(&x) {
            return Err(CodecError::CoefficientOutOfRange);
        }

        // Get sign and absolute value.
        let sw = (x >> 16) as u32;
        let w = ((x as u32) ^ sw).wrapping_sub(sw);

        // Encode sign bit then low 7 bits of the absolute value.
        acc <<= 8;
        acc |= sw & 0x80;
        acc |= w & 0x7F;
        acc_len += 8;

        // Encode the high bits. Since |x| <= 2047, the value in the high
        // bits is at most 15.
        let wh = w >> 7;
        acc <<= wh + 1;
        acc |= 1;
        acc_len += wh + 1;

        // We appended at most 8 + 15 + 1 = 24 bits, so the total number of
        // bits still fits in the 32-bit accumulator. We output complete
        // bytes.
        while acc_len >= 8 {
            acc_len -= 8;
            if j >= d.len() {
                return Err(CodecError::DestinationTooShort {
                    needed: j + 1,
                    actual: d.len(),
                });
            }
            d[j] = (acc >> acc_len) as u8;
            j += 1;
        }
    }

    // Flush remaining bits (if any).
    if acc_len > 0 {
        if j >= d.len() {
            return Err(CodecError::DestinationTooShort {
                needed: j + 1,
                actual: d.len(),
            });
        }
        d[j] = (acc << (8 - acc_len)) as u8;
        j += 1;
    }

    // Pad with zeros.
    for k in j..d.len() {
        d[k] = 0;
    }
    Ok(j)
}

/// Encode small integers from bytes using a compressed (Golomb-Rice) format.
///
/// Decode the provided source buffer `d` into signed integers `v`, using
/// the compressed encoding convention. This function returns an error in
/// any of the following cases:
///
///  - Source does not contain enough encoded integers to fill `v` entirely.
///  - An invalid encoding for a value is encountered.
///  - Any of the remaining unused bits in `d` (after all integers have been
///    decoded) is non-zero.
///
/// Valid encodings cover exactly the integers in the `[-2047,+2047]` range.
/// For a given sequence of integers, there is only one valid encoding as
/// a sequence of bytes (of a given length).
pub fn comp_decode(d: &[u8], v: &mut [i16]) -> Result<usize, CodecError> {
    let mut i = 0;
    let mut acc = 0;
    let mut acc_len = 0;
    for j in 0..v.len() {
        // Invariant: acc_len <= 7 at the beginning of each iteration.

        // Get next 8 bits and split them into sign bit (s) and low bits
        // of the absolute value (m).
        if i >= d.len() {
            return Err(CodecError::SourceTooShort {
                needed: i + 1,
                actual: d.len(),
            });
        }
        acc = (acc << 8) | (d[i] as u32);
        i += 1;
        let s = (acc >> (acc_len + 7)) & 1;
        let mut m = (acc >> acc_len) & 0x7F;

        // Get next bits until a 1 is reached.
        loop {
            if acc_len == 0 {
                if i >= d.len() {
                    return Err(CodecError::SourceTooShort {
                        needed: i + 1,
                        actual: d.len(),
                    });
                }
                acc = (acc << 8) | (d[i] as u32);
                i += 1;
                acc_len = 8;
            }
            acc_len -= 1;
            if ((acc >> acc_len) & 1) != 0 {
                break;
            }
            m += 0x80;
            if m > 2047 {
                return Err(CodecError::InvalidEncoding);
            }
        }

        // Reject "-0" (invalid encoding).
        if (s & (m.wrapping_sub(1) >> 31)) != 0 {
            return Err(CodecError::InvalidEncoding);
        }

        // Apply the sign to get the value.
        let sw = s.wrapping_neg();
        let w = (m ^ sw).wrapping_sub(sw);
        v[j] = w as i16;
    }

    // Check that unused bits are all zero.
    if acc_len > 0 && (acc & ((1 << acc_len) - 1)) != 0 {
        return Err(CodecError::InvalidEncoding);
    }
    for k in i..d.len() {
        if d[k] != 0 {
            return Err(CodecError::InvalidEncoding);
        }
    }
    Ok(i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_i8_roundtrip() {
        let src = [-31i8, -1, 0, 1, 31];
        let mut enc = [0u8; 4];
        let used = trim_i8_encode(&src, 6, &mut enc).unwrap();
        assert_eq!(used, 4);
        let mut dec = [0i8; 5];
        let read = trim_i8_decode(&enc, &mut dec, 6).unwrap();
        assert_eq!(read, 4);
        assert_eq!(dec, src);
    }

    #[test]
    fn trim_i8_rejects_invalid_source_values() {
        let src = [-32i8];
        let mut enc = [0u8; 1];
        assert_eq!(
            trim_i8_encode(&src, 6, &mut enc),
            Err(CodecError::CoefficientOutOfRange),
        );
    }

    #[test]
    fn modq_roundtrip() {
        let src = [0u16, 1, 12288, 7];
        let mut enc = [0u8; 7];
        let used = modq_encode(&src, &mut enc).unwrap();
        assert_eq!(used, 7);
        let mut dec = [0u16; 4];
        let read = modq_decode(&enc, &mut dec).unwrap();
        assert_eq!(read, 7);
        assert_eq!(dec, src);
    }

    #[test]
    fn comp_roundtrip() {
        let src = [-2047i16, -1, 0, 1, 2047];
        let mut enc = [0u8; 16];
        let used = comp_encode(&src, &mut enc).unwrap();
        let mut dec = [0i16; 5];
        let read = comp_decode(&enc, &mut dec).unwrap();
        assert_eq!(dec, src);
        assert!(read <= used);
    }

    #[test]
    fn comp_decode_rejects_non_zero_trailing_data() {
        let src = [0i16];
        let mut enc = [0u8; 3];
        let used = comp_encode(&src, &mut enc).unwrap();
        enc[used] = 1;
        assert_eq!(comp_decode(&enc, &mut [0i16; 1]), Err(CodecError::InvalidEncoding));
    }
}
