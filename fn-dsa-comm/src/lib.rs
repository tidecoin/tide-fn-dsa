#![no_std]

//! This crate contains utility functions which are used by FN-DSA for
//! key pair generation, signing, and verifying. It is not meant to
//! be used directly.

use core::fmt;

/// Encoding/decoding primitives.
pub mod codec;

/// Computations with polynomials modulo X^n+1 and modulo q = 12289.
pub mod mq;

/// SHAKE implementation.
pub mod shake;

/// Specialized versions of `mq` which use AVX2 opcodes (on x86 CPUs).
#[cfg(all(not(feature = "no_avx2"),
    any(target_arch = "x86_64", target_arch = "x86")))]
pub mod mq_avx2;

// Re-export RNG traits to get a smooth dependency management.
pub use rand_core::{CryptoRng, RngCore, Error as RngError};

/// Symbolic constant for FN-DSA with degree 512 (`logn = 9`).
pub const FN_DSA_LOGN_512: u32 = 9;

/// Symbolic constant for FN-DSA with degree 1024 (`logn = 10`).
pub const FN_DSA_LOGN_1024: u32 = 10;

/// Error type for invalid degree parameters.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LogNError {
    /// Supported degree parameters are in the `2..=10` range.
    UnsupportedLogN { logn: u32 },
}

impl fmt::Display for LogNError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UnsupportedLogN { logn } => {
                write!(f, "unsupported degree parameter logn={logn}")
            }
        }
    }
}

/// Validate a degree parameter.
pub const fn check_logn(logn: u32) -> Result<(), LogNError> {
    if logn < 2 || logn > 10 {
        Err(LogNError::UnsupportedLogN { logn })
    } else {
        Ok(())
    }
}

const fn sign_key_size_inner(logn: u32) -> usize {
    let n = 1usize << logn;
    let nbits_fg = match logn {
        2..=5 => 8,
        6..=7 => 7,
        8..=9 => 6,
        _ => 5,
    };
    1 + (nbits_fg << (logn - 2)) + n
}

/// Signing key size for degree 512 (`logn = 9`).
pub const SIGN_KEY_SIZE_512: usize = sign_key_size_inner(FN_DSA_LOGN_512);

/// Signing key size for degree 1024 (`logn = 10`).
pub const SIGN_KEY_SIZE_1024: usize = sign_key_size_inner(FN_DSA_LOGN_1024);

/// Get the size (in bytes) of a signing key for the provided degree
/// (degree is `n = 2^logn`, with `2 <= logn <= 10`).
pub const fn sign_key_size(logn: u32) -> Result<usize, LogNError> {
    if let Err(err) = check_logn(logn) {
        Err(err)
    } else {
        Ok(sign_key_size_inner(logn))
    }
}

const fn vrfy_key_size_inner(logn: u32) -> usize {
    1 + (7 << (logn - 2))
}

/// Verifying key size for degree 512 (`logn = 9`).
pub const VRFY_KEY_SIZE_512: usize = vrfy_key_size_inner(FN_DSA_LOGN_512);

/// Verifying key size for degree 1024 (`logn = 10`).
pub const VRFY_KEY_SIZE_1024: usize = vrfy_key_size_inner(FN_DSA_LOGN_1024);

/// Get the size (in bytes) of a verifying key for the provided degree
/// (degree is `n = 2^logn`, with `2 <= logn <= 10`).
pub const fn vrfy_key_size(logn: u32) -> Result<usize, LogNError> {
    if let Err(err) = check_logn(logn) {
        Err(err)
    } else {
        Ok(vrfy_key_size_inner(logn))
    }
}

const fn signature_size_inner(logn: u32) -> usize {
    44 + 3 * (256 >> (10 - logn)) + 2 * (128 >> (10 - logn))
        + 3 * (64 >> (10 - logn)) + 2 * (16 >> (10 - logn))
        - 2 * (2 >> (10 - logn)) - 8 * (1 >> (10 - logn))
}

/// Signature size for degree 512 (`logn = 9`).
pub const SIGNATURE_SIZE_512: usize = signature_size_inner(FN_DSA_LOGN_512);

/// Signature size for degree 1024 (`logn = 10`).
pub const SIGNATURE_SIZE_1024: usize = signature_size_inner(FN_DSA_LOGN_1024);

/// Get the size (in bytes) of a signature for the provided degree
/// (degree is `n = 2^logn`, with `2 <= logn <= 10`).
pub const fn signature_size(logn: u32) -> Result<usize, LogNError> {
    // logn   n      size
    //   2      4      47
    //   3      8      52
    //   4     16      63
    //   5     32      82
    //   6     64     122
    //   7    128     200
    //   8    256     356
    //   9    512     666
    //  10   1024    1280
    if let Err(err) = check_logn(logn) {
        Err(err)
    } else {
        Ok(signature_size_inner(logn))
    }
}

/// Falcon-compatible raw-message profile.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FalconProfile {
    /// Standard Falcon/PQClean raw-message mode.
    PqClean,

    /// Tidecoin-specific Falcon-512 legacy compatibility mode.
    TidecoinLegacyFalcon512,
}

impl fmt::Display for FalconProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::PqClean => write!(f, "FalconPqClean"),
            Self::TidecoinLegacyFalcon512 => {
                write!(f, "TidecoinLegacyFalcon512")
            }
        }
    }
}

/// Falcon nonce length in bytes.
pub const FALCON_NONCE_LEN: usize = 40;

/// Maximum compressed-body length accepted by Tidecoin legacy Falcon-512.
pub const TIDECOIN_LEGACY_FALCON512_SIG_BODY_MAX: usize = 647;

/// Maximum total signature length accepted by Tidecoin legacy Falcon-512.
pub const TIDECOIN_LEGACY_FALCON512_SIG_MAX: usize =
    1 + FALCON_NONCE_LEN + TIDECOIN_LEGACY_FALCON512_SIG_BODY_MAX;

/// The message for which a signature is to be generated or verified is
/// pre-hashed by the caller and provided as a hash value along with
/// an identifier of the used hash function. The identifier is normally
/// an encoded ASN.1 OID. A special identifier is used for "raw" messages
/// (i.e. not pre-hashed at all); it uses a single byte of value 0x00.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct HashIdentifier<'a>(&'a [u8]);

/// Error type for hash identifiers.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HashIdentifierError {
    /// Hash identifiers must not be empty.
    Empty,

    /// One-byte identifiers are reserved for internal special values.
    InvalidSingleByteValue { actual: u8 },
}

impl fmt::Display for HashIdentifierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Empty => write!(f, "hash identifier must not be empty"),
            Self::InvalidSingleByteValue { actual } => write!(
                f,
                "invalid one-byte hash identifier value: 0x{actual:02X}"
            ),
        }
    }
}

impl<'a> HashIdentifier<'a> {
    /// Create a validated hash identifier.
    pub fn new(bytes: &'a [u8]) -> Result<Self, HashIdentifierError> {
        if bytes.is_empty() {
            return Err(HashIdentifierError::Empty);
        }
        if bytes.len() == 1 && bytes[0] != 0x00 {
            return Err(HashIdentifierError::InvalidSingleByteValue {
                actual: bytes[0],
            });
        }
        Ok(Self(bytes))
    }

    /// Get the raw identifier bytes.
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.0
    }

    /// Test whether this is the raw-message identifier.
    pub const fn is_raw(&self) -> bool {
        self.0.len() == 1 && self.0[0] == 0x00
    }
}

/// Hash function identifier: none.
///
/// This is the identifier used internally to specify that signature
/// generation and verification are performed over a raw message, without
/// pre-hashing.
pub const HASH_ID_RAW: HashIdentifier = HashIdentifier(&[0x00]);

/// Hash function identifier: SHA-256
pub const HASH_ID_SHA256: HashIdentifier = HashIdentifier(
    &[0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01]);

/// Hash function identifier: SHA-384
pub const HASH_ID_SHA384: HashIdentifier = HashIdentifier(
    &[0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x02]);

/// Hash function identifier: SHA-512
pub const HASH_ID_SHA512: HashIdentifier = HashIdentifier(
    &[0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x03]);

/// Hash function identifier: SHA-512-256
pub const HASH_ID_SHA512_256: HashIdentifier = HashIdentifier(
    &[0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x06]);

/// Hash function identifier: SHA3-256
pub const HASH_ID_SHA3_256: HashIdentifier = HashIdentifier(
    &[0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x08]);

/// Hash function identifier: SHA3-384
pub const HASH_ID_SHA3_384: HashIdentifier = HashIdentifier(
    &[0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x09]);

/// Hash function identifier: SHA3-512
pub const HASH_ID_SHA3_512: HashIdentifier = HashIdentifier(
    &[0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x0A]);

/// Hash function identifier: SHAKE128
pub const HASH_ID_SHAKE128: HashIdentifier = HashIdentifier(
    &[0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x0B]);

/// Hash function identifier: SHAKE256
pub const HASH_ID_SHAKE256: HashIdentifier = HashIdentifier(
    &[0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x0C]);

/// When a message is signed or verified, it is accompanied with a domain
/// separation context, which is an arbitrary sequence of bytes of length
/// at most 255. Such a context is wrapped in a `DomainContext` structure.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DomainContext<'a>(&'a [u8]);

/// Error type for domain-separation contexts.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DomainContextError {
    /// Domain-separation contexts are limited to 255 bytes.
    Oversized { actual: usize },
}

impl fmt::Display for DomainContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Oversized { actual } => write!(
                f,
                "invalid domain context length: expected at most 255 bytes, got {actual}"
            ),
        }
    }
}

impl<'a> DomainContext<'a> {
    /// Create a validated domain-separation context.
    pub fn new(bytes: &'a [u8]) -> Result<Self, DomainContextError> {
        if bytes.len() > 255 {
            return Err(DomainContextError::Oversized {
                actual: bytes.len(),
            });
        }
        Ok(Self(bytes))
    }

    /// Create an empty domain-separation context.
    pub const fn empty() -> Self {
        Self(b"")
    }

    /// Get the raw bytes for this context.
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.0
    }

    /// Test whether the context is empty.
    pub const fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get the context length in bytes.
    pub const fn len(&self) -> usize {
        self.0.len()
    }
}

/// Empty domain separation context.
pub const DOMAIN_NONE: DomainContext = DomainContext::empty();

/// Error type for message hashing into a point.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HashToPointError {
    /// The nonce must contain exactly 40 bytes.
    InvalidNonceLength { actual: usize },

    /// The hashed verifying key must contain exactly 64 bytes.
    InvalidHashedVerifyingKeyLength { actual: usize },
}

impl fmt::Display for HashToPointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidNonceLength { actual } => {
                write!(f, "invalid nonce length: expected 40 bytes, got {actual}")
            }
            Self::InvalidHashedVerifyingKeyLength { actual } => write!(
                f,
                "invalid hashed verifying key length: expected 64 bytes, got {actual}"
            ),
        }
    }
}

/// Hash a message into a polynomial modulo q = 12289.
///
/// Parameters are:
///
///  - `nonce`:            40-byte random nonce
///  - `hashed_vrfy_key`:  SHAKE256 hash of public (verifying) key (64 bytes)
///  - `ctx`:              domain separation context
///  - `id`:               identifier for pre-hash function
///  - `hv`:               message (pre-hashed)
///  - `c`:                output polynomial
///
/// If `id` is `HASH_ID_RAW`, then no-prehashing is applied and the message
/// itself should be provided as `hv`. Otherwise, the caller is responsible
/// for applying the pre-hashing, and `hv` shall be the hashed message.
pub fn hash_to_point(nonce: &[u8], hashed_vrfy_key: &[u8],
    ctx: &DomainContext, id: &HashIdentifier, hv: &[u8], c: &mut [u16])
    -> Result<(), HashToPointError>
{
    if nonce.len() != FALCON_NONCE_LEN {
        return Err(HashToPointError::InvalidNonceLength {
            actual: nonce.len(),
        });
    }
    if hashed_vrfy_key.len() != 64 {
        return Err(HashToPointError::InvalidHashedVerifyingKeyLength {
            actual: hashed_vrfy_key.len(),
        });
    }
    hash_to_point_inner(nonce, hashed_vrfy_key, ctx, id, hv, c);
    Ok(())
}

/// Hash a raw Falcon/PQClean message into a polynomial modulo q = 12289.
///
/// Parameters are:
///
///  - `nonce`:    40-byte random nonce
///  - `message`:  raw message bytes
///  - `c`:        output polynomial
pub fn hash_to_point_falcon(
    nonce: &[u8],
    message: &[u8],
    c: &mut [u16],
) -> Result<(), HashToPointError> {
    if nonce.len() != FALCON_NONCE_LEN {
        return Err(HashToPointError::InvalidNonceLength {
            actual: nonce.len(),
        });
    }
    hash_to_point_falcon_inner(nonce, message, c);
    Ok(())
}

fn hash_to_point_inner(nonce: &[u8], hashed_vrfy_key: &[u8],
    ctx: &DomainContext, id: &HashIdentifier, hv: &[u8], c: &mut [u16])
{
    debug_assert!(ctx.len() <= 255);
    debug_assert!(!id.as_bytes().is_empty());
    debug_assert!(id.as_bytes().len() != 1 || id.is_raw());

    // Input order:
    //   With pre-hashing:
    //     nonce || hashed_vrfy_key || 0x01 || len(ctx) || ctx || id || hv
    //   Without pre-hashing:
    //     nonce || hashed_vrfy_key || 0x00 || len(ctx) || ctx || message
    // 'len(ctx)' is the length of the context over one byte (0 to 255).
    let raw_message = id.is_raw();
    let mut sh = shake::SHAKE256::new();
    sh.inject(nonce).unwrap();
    sh.inject(hashed_vrfy_key).unwrap();
    sh.inject(&[if raw_message { 0u8 } else { 1u8 }]).unwrap();
    sh.inject(&[ctx.len() as u8]).unwrap();
    sh.inject(ctx.as_bytes()).unwrap();
    if !raw_message {
        sh.inject(id.as_bytes()).unwrap();
    }
    sh.inject(hv).unwrap();
    sh.flip().unwrap();
    let mut i = 0;
    while i < c.len() {
        let mut v = [0u8; 2];
        sh.extract(&mut v).unwrap();
        let mut w = ((v[0] as u16) << 8) | (v[1] as u16);
        if w < 61445 {
            while w >= 12289 {
                w -= 12289;
            }
            c[i] = w;
            i += 1;
        }
    }
}

fn hash_to_point_falcon_inner(nonce: &[u8], message: &[u8], c: &mut [u16]) {
    let mut sh = shake::SHAKE256::new();
    sh.inject(nonce).unwrap();
    sh.inject(message).unwrap();
    sh.flip().unwrap();
    let mut i = 0;
    while i < c.len() {
        let mut v = [0u8; 2];
        sh.extract(&mut v).unwrap();
        let mut w = ((v[0] as u16) << 8) | (v[1] as u16);
        if w < 61445 {
            while w >= 12289 {
                w -= 12289;
            }
            c[i] = w;
            i += 1;
        }
    }
}

/// Trait for a deterministic pseudorandom generator.
///
/// The trait `PRNG` characterizes a stateful object that produces
/// pseudorandom bytes (and larger values) in a cryptographically secure
/// way; the object is created with a source seed, and the output is
/// indistinguishable from uniform randomness up to exhaustive enumeration
/// of the possible values of the seed.
///
/// `PRNG` instances must also implement `Copy` and `Clone` so that they
/// may be embedded in clonable structures. This implies that copying a
/// `PRNG` instance is supposed to clone its internal state, and the copy
/// will output the same values as the original.
pub trait PRNG: Copy + Clone {
    /// Create a new instance over the provided seed.
    fn new(seed: &[u8]) -> Self;
    /// Get the next byte from the PRNG.
    fn next_u8(&mut self) -> u8;
    /// Get the 16-bit value from the PRNG.
    fn next_u16(&mut self) -> u16;
    /// Get the 64-bit value from the PRNG.
    fn next_u64(&mut self) -> u64;
}

#[cfg(all(not(feature = "no_avx2"),
    any(target_arch = "x86_64", target_arch = "x86")))]
cpufeatures::new!(cpuid_avx2, "avx2");

/// Do a rutime check for AVX2 support (x86 and x86_64 only).
///
/// This is a specialized subcase of the is_x86_feature_detected macro,
/// except that this function is compatible with `no_std` builds.
#[cfg(all(not(feature = "no_avx2"),
    any(target_arch = "x86_64", target_arch = "x86")))]
pub fn has_avx2() -> bool {
    cpuid_avx2::get()
}
