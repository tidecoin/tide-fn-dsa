#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use fn_dsa_comm::codec;
use fn_dsa_comm::mq;
use fn_dsa_comm::shake::SHAKE256;
use fn_dsa_comm::{sign_key_size, vrfy_key_size};
use zeroize::{Zeroize, Zeroizing};

use crate::fxp::FXR;
use crate::ntru;
use crate::FALCON_KEYGEN_SEED_SIZE;

const MAX_FALCON_N: usize = 1024;
const TMP_I8_LEN: usize = 4 * MAX_FALCON_N;
const TMP_U16_LEN: usize = 2 * MAX_FALCON_N;
const TMP_U32_LEN: usize = 6 * MAX_FALCON_N;
const TMP_FXR_LEN: usize = 3 * MAX_FALCON_N;

// CDT table for discrete Gaussian sampling over Z with standard
// deviation ~1.17*sqrt(q/2N) for N=1024, q=12289.
// This matches the PQClean / Falcon reference implementation.
static GAUSS_1024_12289: [u64; 27] = [
    1283868770400643928,
    6416574995475331444,
    4078260278032692663,
    2353523259288686585,
    1227179971273316331,
    575931623374121527,
    242543240509105209,
    91437049221049666,
    30799446349977173,
    9255276791179340,
    2478152334826140,
    590642893610164,
    125206034929641,
    23590435911403,
    3948334035941,
    586753615614,
    77391054539,
    9056793210,
    940121950,
    86539696,
    7062824,
    510971,
    32764,
    1862,
    94,
    4,
    0,
];

fn get_rng_u64(shake: &mut SHAKE256) -> u64 {
    let mut tmp = [0u8; 8];
    shake.extract(&mut tmp).unwrap();
    let value = u64::from_le_bytes(tmp);
    tmp.zeroize();
    value
}

fn mkgauss(shake: &mut SHAKE256, logn: u32) -> i32 {
    let g: u32 = 1u32 << (10 - logn);
    let mut val: i32 = 0;
    for _ in 0..g {
        let mut r = get_rng_u64(shake);
        let neg: u32 = (r >> 63) as u32;
        r &= !(1u64 << 63);
        let mut f: u32 = ((r.wrapping_sub(GAUSS_1024_12289[0])) >> 63) as u32;

        let mut v: u32 = 0;
        r = get_rng_u64(shake);
        r &= !(1u64 << 63);
        for k in 1..GAUSS_1024_12289.len() {
            let t: u32 = (((r.wrapping_sub(GAUSS_1024_12289[k])) >> 63) ^ 1) as u32;
            v |= (k as u32) & (!(t & (f ^ 1))).wrapping_add(1);
            f |= t;
        }

        v = (v ^ (!neg).wrapping_add(1)).wrapping_add(neg);
        val += v as i32;
    }
    val
}

fn poly_small_mkgauss(shake: &mut SHAKE256, logn: u32, f: &mut [i8]) {
    let n = 1usize << logn;
    let mut mod2: u32 = 0;
    let mut u = 0;
    while u < n {
        let s = mkgauss(shake, logn);
        if !(-127..=127).contains(&s) {
            continue;
        }
        if u == n - 1 {
            if (mod2 ^ (s & 1) as u32) == 0 {
                continue;
            }
        } else {
            mod2 ^= (s & 1) as u32;
        }
        f[u] = s as i8;
        u += 1;
    }
}

fn nbits_fg(logn: u32) -> u32 {
    match logn {
        2..=5 => 8,
        6..=7 => 7,
        8..=9 => 6,
        _ => 5,
    }
}

fn coeffs_within_bound(f: &[i8], g: &[i8], bound: i32) -> bool {
    debug_assert!(f.len() == g.len());
    let mut reject = 0u32;
    for i in 0..f.len() {
        let xf = f[i] as i32;
        let xg = g[i] as i32;
        reject |= ((xf >= bound || xf <= -bound) as u32)
            | ((xg >= bound || xg <= -bound) as u32);
    }
    reject == 0
}

fn squared_fg_norm(f: &[i8], g: &[i8]) -> u32 {
    debug_assert!(f.len() == g.len());
    let mut sn = 0u32;
    for i in 0..f.len() {
        let xf = f[i] as i32;
        let xg = g[i] as i32;
        sn = sn.wrapping_add((xf * xf + xg * xg) as u32);
    }
    sn
}

fn encode_pqclean_keypair(
    logn: u32,
    nbits_fg: u32,
    f: &[i8],
    g: &[i8],
    cap_f: &[i8],
    h: &[u16],
    sign_key: &mut [u8],
    vrfy_key: &mut [u8],
) {
    sign_key[0] = 0x50 + (logn as u8);
    let j = 1 + codec::trim_i8_encode(f, nbits_fg, &mut sign_key[1..]).unwrap();
    let j = j + codec::trim_i8_encode(g, nbits_fg, &mut sign_key[j..]).unwrap();
    let j = j + codec::trim_i8_encode(cap_f, 8, &mut sign_key[j..]).unwrap();
    debug_assert!(j == sign_key.len());

    vrfy_key[0] = logn as u8;
    let j = 1 + codec::modq_encode(h, &mut vrfy_key[1..]).unwrap();
    debug_assert!(j == vrfy_key.len());
}

pub(crate) fn keygen_pqclean(
    logn: u32,
    seed: &[u8; FALCON_KEYGEN_SEED_SIZE],
    sign_key: &mut [u8],
    vrfy_key: &mut [u8],
) -> bool {
    if logn < 2 || logn > 10 {
        return false;
    }
    if sign_key.len() != sign_key_size(logn).unwrap() {
        return false;
    }
    if vrfy_key.len() != vrfy_key_size(logn).unwrap() {
        return false;
    }

    let n = 1usize << logn;
    let nbits_fg = nbits_fg(logn);
    let bound = 1i32 << (nbits_fg - 1);

    let mut shake = SHAKE256::new();
    shake.inject(seed).unwrap();
    shake.flip().unwrap();

    let mut tmp_i8 = Zeroizing::new([0i8; TMP_I8_LEN]);
    let mut tmp_u16 = Zeroizing::new([0u16; TMP_U16_LEN]);
    let mut tmp_u32 = Zeroizing::new([0u32; TMP_U32_LEN]);
    let mut tmp_fxr = Zeroizing::new([FXR::ZERO; TMP_FXR_LEN]);

    let (f, rest) = tmp_i8[..4 * n].split_at_mut(n);
    let (g, rest) = rest.split_at_mut(n);
    let (cap_f, rest) = rest.split_at_mut(n);
    let (cap_g, _) = rest.split_at_mut(n);
    let (h, t16) = tmp_u16[..2 * n].split_at_mut(n);

    loop {
        poly_small_mkgauss(&mut shake, logn, f);
        poly_small_mkgauss(&mut shake, logn, g);

        if !coeffs_within_bound(f, g, bound) {
            continue;
        }

        if squared_fg_norm(f, g) >= 16823 {
            continue;
        }

        if !mq::mqpoly_small_is_invertible(logn, f, t16) {
            continue;
        }
        if !ntru::check_ortho_norm(logn, f, g, &mut tmp_fxr[..(3 * n)]) {
            continue;
        }
        if !ntru::solve_NTRU(
            logn,
            f,
            g,
            cap_f,
            cap_g,
            &mut tmp_u32[..(6 * n)],
            &mut tmp_fxr[..(3 * n)],
        ) {
            continue;
        }

        mq::mqpoly_div_small(logn, f, g, h, t16);
        encode_pqclean_keypair(logn, nbits_fg, f, g, cap_f, h, sign_key, vrfy_key);
        shake.zeroize();

        return true;
    }
}
