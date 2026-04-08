#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

extern crate alloc;

use alloc::vec;

use super::mp31::*;
use super::poly::*;
use super::pqclean_float::{
    flr_to_f64, iFFT, poly_add, poly_add_muladj_fft, poly_adj_fft, poly_div_autoadj_fft,
    poly_invnorm2_fft, poly_mul_autoadj_fft, poly_mul_fft, poly_mul_selfadj_fft, poly_sub, FFT,
    FLR,
};
use super::zint31::*;

const Q: u32 = 12289;

const MAX_BL_SMALL: [usize; 11] = [1, 1, 2, 2, 4, 7, 14, 27, 53, 106, 209];
const MAX_BL_LARGE: [usize; 10] = [2, 2, 5, 7, 12, 21, 40, 78, 157, 308];
const BITLENGTH: [(i32, i32); 11] = [
    (4, 0),
    (11, 1),
    (24, 1),
    (50, 1),
    (102, 1),
    (202, 2),
    (401, 4),
    (794, 5),
    (1577, 8),
    (3138, 13),
    (6308, 25),
];

#[inline(always)]
fn zint_one_to_plain(x: &[u32]) -> i32 {
    let mut w = x[0];
    w |= (w & 0x40000000) << 1;
    w as i32
}

fn poly_big_to_flr_word_major(d: &mut [FLR], f: &[u32], flen: usize, fstride: usize, logn: u32) {
    let n = 1usize << logn;
    if flen == 0 {
        for x in d[..n].iter_mut() {
            *x = FLR::ZERO;
        }
        return;
    }
    for u in 0..n {
        let mut neg = (f[u + (flen - 1) * fstride] >> 30).wrapping_neg();
        let xm = neg >> 1;
        let mut cc = neg & 1;
        let mut x = FLR::ZERO;
        let mut fsc = FLR::ONE;
        for v in 0..flen {
            let mut w = (f[u + v * fstride] ^ xm).wrapping_add(cc);
            cc = w >> 31;
            w &= 0x7FFFFFFF;
            w = w.wrapping_sub((w << 1) & neg);
            x += FLR::from_i32(w as i32) * fsc;
            fsc = fsc * FLR::scaled(1, 31);
        }
        d[u] = x;
        neg = 0;
        cc = 0;
        let _ = (neg, cc);
    }
}

#[cfg(test)]
pub(crate) fn debug_poly_big_to_flr(
    f: &[u32],
    flen: usize,
    fstride: usize,
    logn: u32,
) -> vec::Vec<FLR> {
    let mut d = vec![FLR::ZERO; 1usize << logn];
    poly_big_to_flr_coeff_major(&mut d, f, flen, fstride, logn);
    d
}

fn poly_big_to_flr_coeff_major(d: &mut [FLR], f: &[u32], flen: usize, fstride: usize, logn: u32) {
    let n = 1usize << logn;
    if flen == 0 {
        for x in d[..n].iter_mut() {
            *x = FLR::ZERO;
        }
        return;
    }
    for u in 0..n {
        let base = u * fstride;
        let neg = (f[base + flen - 1] >> 30).wrapping_neg();
        let xm = neg >> 1;
        let mut cc = neg & 1;
        let mut x = FLR::ZERO;
        let mut fsc = FLR::ONE;
        for v in 0..flen {
            let mut w = (f[base + v] ^ xm).wrapping_add(cc);
            cc = w >> 31;
            w &= 0x7FFFFFFF;
            w = w.wrapping_sub((w << 1) & neg);
            x += FLR::from_i32(w as i32) * fsc;
            fsc = fsc * FLR::scaled(1, 31);
        }
        d[u] = x;
    }
}

#[inline(always)]
fn poly_big_to_small(d: &mut [i8], s: &[u32], lim: i32, logn: u32) -> bool {
    let n = 1usize << logn;
    for u in 0..n {
        let z = zint_one_to_plain(&s[u..]);
        if z < -lim || z > lim {
            return false;
        }
        d[u] = z as i8;
    }
    true
}

fn mp_ntt_column(
    logn: u32,
    data: &mut [u32],
    stride: usize,
    gm: &[u32],
    p: u32,
    p0i: u32,
    tmp: &mut [u32],
) {
    let n = 1usize << logn;
    for i in 0..n {
        tmp[i] = data[i * stride];
    }
    mp_NTT(logn, &mut tmp[..n], gm, p, p0i);
    for i in 0..n {
        data[i * stride] = tmp[i];
    }
}

fn mp_intt_column(
    logn: u32,
    data: &mut [u32],
    stride: usize,
    igm: &[u32],
    p: u32,
    p0i: u32,
    tmp: &mut [u32],
) {
    let n = 1usize << logn;
    for i in 0..n {
        tmp[i] = data[i * stride];
    }
    mp_iNTT(logn, &mut tmp[..n], igm, p, p0i);
    for i in 0..n {
        data[i * stride] = tmp[i];
    }
}

fn make_fg_step_exact(data: &mut [u32], logn: u32, depth: u32, in_ntt: bool, out_ntt: bool) {
    let n = 1usize << logn;
    let hn = n >> 1;
    let slen = MAX_BL_SMALL[depth as usize];
    let tlen = MAX_BL_SMALL[(depth + 1) as usize];
    let mut gm = [0u32; 1024];
    let mut igm = [0u32; 1024];
    let mut t1 = [0u32; 1024];

    let fd_off = 0;
    let gd_off = fd_off + hn * tlen;
    let fs_off = gd_off + hn * tlen;
    let gs_off = fs_off + n * slen;
    data.copy_within(0..(2 * n * slen), fs_off);

    for u in 0..slen {
        let p = PRIMES[u].p;
        let p0i = PRIMES[u].p0i;
        let R2 = PRIMES[u].R2;
        mp_mkgmigm(
            logn,
            PRIMES[u].g,
            PRIMES[u].ig,
            p,
            p0i,
            &mut gm[..n],
            &mut igm[..n],
        );

        for v in 0..n {
            t1[v] = data[fs_off + v * slen + u];
        }
        if !in_ntt {
            mp_NTT(logn, &mut t1[..n], &gm[..n], p, p0i);
        }
        for v in 0..hn {
            let w0 = t1[(v << 1) + 0];
            let w1 = t1[(v << 1) + 1];
            data[fd_off + v * tlen + u] = mp_mmul(mp_mmul(w0, w1, p, p0i), R2, p, p0i);
        }
        if in_ntt {
            let fs = &mut data[(fs_off + u)..];
            mp_intt_column(logn, fs, slen, &igm[..n], p, p0i, &mut t1[..n]);
        }

        for v in 0..n {
            t1[v] = data[gs_off + v * slen + u];
        }
        if !in_ntt {
            mp_NTT(logn, &mut t1[..n], &gm[..n], p, p0i);
        }
        for v in 0..hn {
            let w0 = t1[(v << 1) + 0];
            let w1 = t1[(v << 1) + 1];
            data[gd_off + v * tlen + u] = mp_mmul(mp_mmul(w0, w1, p, p0i), R2, p, p0i);
        }
        if in_ntt {
            let gs = &mut data[(gs_off + u)..];
            mp_intt_column(logn, gs, slen, &igm[..n], p, p0i, &mut t1[..n]);
        }

        if !out_ntt {
            let fd = &mut data[(fd_off + u)..];
            mp_intt_column(logn - 1, fd, tlen, &igm[..n], p, p0i, &mut t1[..n]);
            let gd = &mut data[(gd_off + u)..];
            mp_intt_column(logn - 1, gd, tlen, &igm[..n], p, p0i, &mut t1[..n]);
        }
    }

    zint_rebuild_CRT(&mut data[fs_off..gs_off], slen, 1, n, true, &mut gm[..slen]);
    zint_rebuild_CRT(
        &mut data[gs_off..(gs_off + n * slen)],
        slen,
        1,
        n,
        true,
        &mut gm[..slen],
    );

    for u in slen..tlen {
        let p = PRIMES[u].p;
        let p0i = PRIMES[u].p0i;
        let R2 = PRIMES[u].R2;
        let Rx = mp_Rx31(slen as u32, p, p0i, R2);
        mp_mkgmigm(
            logn,
            PRIMES[u].g,
            PRIMES[u].ig,
            p,
            p0i,
            &mut gm[..n],
            &mut igm[..n],
        );

        for v in 0..n {
            let x = &data[(fs_off + v * slen)..];
            t1[v] = zint_mod_small_signed(x, slen, 1, p, p0i, R2, Rx);
        }
        mp_NTT(logn, &mut t1[..n], &gm[..n], p, p0i);
        for v in 0..hn {
            let w0 = t1[(v << 1) + 0];
            let w1 = t1[(v << 1) + 1];
            data[fd_off + v * tlen + u] = mp_mmul(mp_mmul(w0, w1, p, p0i), R2, p, p0i);
        }

        for v in 0..n {
            let x = &data[(gs_off + v * slen)..];
            t1[v] = zint_mod_small_signed(x, slen, 1, p, p0i, R2, Rx);
        }
        mp_NTT(logn, &mut t1[..n], &gm[..n], p, p0i);
        for v in 0..hn {
            let w0 = t1[(v << 1) + 0];
            let w1 = t1[(v << 1) + 1];
            data[gd_off + v * tlen + u] = mp_mmul(mp_mmul(w0, w1, p, p0i), R2, p, p0i);
        }

        if !out_ntt {
            let fd = &mut data[(fd_off + u)..];
            mp_intt_column(logn - 1, fd, tlen, &igm[..n], p, p0i, &mut t1[..n]);
            let gd = &mut data[(gd_off + u)..];
            mp_intt_column(logn - 1, gd, tlen, &igm[..n], p, p0i, &mut t1[..n]);
        }
    }
}

fn make_fg_exact(data: &mut [u32], f: &[i8], g: &[i8], logn: u32, depth: u32, out_ntt: bool) {
    let n = 1usize << logn;
    let p0 = PRIMES[0].p;
    let mut gm = [0u32; 1024];
    let mut igm = [0u32; 1024];
    let (ft, gt) = data.split_at_mut(n);
    for u in 0..n {
        ft[u] = mp_set(f[u] as i32, p0);
        gt[u] = mp_set(g[u] as i32, p0);
    }

    if depth == 0 {
        if out_ntt {
            mp_mkgmigm(
                logn,
                PRIMES[0].g,
                PRIMES[0].ig,
                p0,
                PRIMES[0].p0i,
                &mut gm[..n],
                &mut igm[..n],
            );
            mp_NTT(logn, ft, &gm[..n], p0, PRIMES[0].p0i);
            mp_NTT(logn, gt, &gm[..n], p0, PRIMES[0].p0i);
        }
        return;
    }

    if depth == 1 {
        make_fg_step_exact(data, logn, 0, false, out_ntt);
        return;
    }

    make_fg_step_exact(data, logn, 0, false, true);
    for d in 1..depth {
        let is_last = d + 1 == depth;
        make_fg_step_exact(
            data,
            logn - d,
            d,
            true,
            if is_last { out_ntt } else { true },
        );
    }
}

fn make_fg_depth0(logn: u32, f: &[i8], g: &[i8], tmp: &mut [u32]) {
    let n = 1usize << logn;
    let p = P0.p;
    let p0i = P0.p0i;
    let (ft, tmp) = tmp.split_at_mut(n);
    let (gt, tmp) = tmp.split_at_mut(n);
    let (gm, _) = tmp.split_at_mut(n);
    poly_mp_set_small(logn, f, p, ft);
    poly_mp_set_small(logn, g, p, gt);
    mp_mkgm(logn, P0.g, p, p0i, gm);
    mp_NTT(logn, ft, gm, p, p0i);
    mp_NTT(logn, gt, gm, p, p0i);
}

fn make_fg_step(logn_top: u32, depth: u32, work: &mut [u32]) {
    let logn = logn_top - depth;
    let n = 1usize << logn;
    let hn = n >> 1;
    let slen = MAX_BL_SMALL[depth as usize];
    let tlen = MAX_BL_SMALL[(depth + 1) as usize];

    let data = work;
    data.copy_within(0..(2 * n * slen), 2 * hn * tlen);
    let (fd, data) = data.split_at_mut(hn * tlen);
    let (gd, data) = data.split_at_mut(hn * tlen);
    let (fgs, data) = data.split_at_mut(2 * n * slen);

    {
        let (fs, gs) = fgs.split_at_mut(n * slen);
        let (igm, _) = data.split_at_mut(n);
        for i in 0..slen {
            let p = PRIMES[i].p;
            let p0i = PRIMES[i].p0i;
            let R2 = PRIMES[i].R2;
            let ks = i * n;
            let kd = i * hn;
            for j in 0..hn {
                fd[kd + j] = mp_mmul(
                    mp_mmul(fs[ks + 2 * j], fs[ks + 2 * j + 1], p, p0i),
                    R2,
                    p,
                    p0i,
                );
                gd[kd + j] = mp_mmul(
                    mp_mmul(gs[ks + 2 * j], gs[ks + 2 * j + 1], p, p0i),
                    R2,
                    p,
                    p0i,
                );
            }
            mp_mkigm(logn, PRIMES[i].ig, p, p0i, igm);
            mp_iNTT(logn, &mut fs[ks..(ks + n)], igm, p, p0i);
            mp_iNTT(logn, &mut gs[ks..(ks + n)], igm, p, p0i);
        }
    }

    if tlen > slen {
        zint_rebuild_CRT(fgs, slen, n, 2, true, data);

        let (fs, gs) = fgs.split_at_mut(n * slen);
        let (gm, data) = data.split_at_mut(n);
        let (t2, _) = data.split_at_mut(n);
        for i in slen..tlen {
            let p = PRIMES[i].p;
            let p0i = PRIMES[i].p0i;
            let R2 = PRIMES[i].R2;
            let Rx = mp_Rx31(slen as u32, p, p0i, R2);
            mp_mkgm(logn, PRIMES[i].g, p, p0i, gm);
            let kd = i * hn;

            for j in 0..n {
                t2[j] = zint_mod_small_signed(&fs[j..], slen, n, p, p0i, R2, Rx);
            }
            mp_NTT(logn, t2, gm, p, p0i);
            for j in 0..hn {
                fd[kd + j] = mp_mmul(mp_mmul(t2[2 * j], t2[2 * j + 1], p, p0i), R2, p, p0i);
            }

            for j in 0..n {
                t2[j] = zint_mod_small_signed(&gs[j..], slen, n, p, p0i, R2, Rx);
            }
            mp_NTT(logn, t2, gm, p, p0i);
            for j in 0..hn {
                gd[kd + j] = mp_mmul(mp_mmul(t2[2 * j], t2[2 * j + 1], p, p0i), R2, p, p0i);
            }
        }
    }
}

fn make_fg_intermediate(logn_top: u32, f: &[i8], g: &[i8], depth: u32, work: &mut [u32]) {
    // Generate depth-(f,g) with the exact PQClean recursion, then convert
    // from coefficient-major layout (PQClean) to the word-major layout used
    // by the existing Rust intermediate solver state.
    let logn = logn_top - depth;
    let n = 1usize << logn;
    let slen = MAX_BL_SMALL[depth as usize];
    let need = 7usize << logn_top;
    assert!(work.len() >= need);

    make_fg_exact(work, f, g, logn_top, depth, true);

    let fg_len = 2 * n * slen;
    let mut transposed = vec![0u32; fg_len];
    let (src_f, src_g) = work[..fg_len].split_at(n * slen);
    let (dst_f, dst_g) = transposed.split_at_mut(n * slen);
    for coeff in 0..n {
        for word in 0..slen {
            dst_f[word * n + coeff] = src_f[coeff * slen + word];
            dst_g[word * n + coeff] = src_g[coeff * slen + word];
        }
    }
    work[..fg_len].copy_from_slice(&transposed);
}

fn make_fg_deepest(logn: u32, f: &[i8], g: &[i8], work: &mut [u32]) -> bool {
    make_fg_intermediate(logn, f, g, logn, work);
    let n = 1usize << logn;
    let mut b = 0u32;
    for i in 0..n {
        b |= work[i].wrapping_sub(1);
    }
    (b >> 31) == 0
}

fn zint_rebuild_CRT_stride(
    xx: &mut [u32],
    xlen: usize,
    xstride: usize,
    num: usize,
    normalize_signed: bool,
    tmp: &mut [u32],
) {
    tmp[0] = PRIMES[0].p;
    for u in 1..xlen {
        let p = PRIMES[u].p;
        let p0i = PRIMES[u].p0i;
        let R2 = PRIMES[u].R2;
        let s = PRIMES[u].s;
        for v in 0..num {
            let x = &mut xx[v * xstride..];
            let xp = x[u];
            let xq = zint_mod_small_unsigned(x, u, 1, p, p0i, R2);
            let xr = mp_mmul(s, mp_sub(xp, xq, p), p, p0i);
            zint_add_mul_small(x, 1, &tmp[..u], xr);
        }
        tmp[u] = zint_mul_small(&mut tmp[..u], p);
    }
    if normalize_signed {
        for u in 0..num {
            zint_norm_zero(&mut xx[u * xstride..], 1, &tmp[..xlen]);
        }
    }
}

fn solve_NTRU_deepest(logn: u32, f: &[i8], g: &[i8], tmp: &mut [u32]) -> bool {
    let slen = MAX_BL_SMALL[logn as usize];
    make_fg_exact(tmp, f, g, logn, logn, false);
    let (fg, tmp) = tmp.split_at_mut(2 * slen);
    let (cap_f, tmp) = tmp.split_at_mut(slen);
    let (cap_g, t1) = tmp.split_at_mut(slen);
    zint_rebuild_CRT(fg, slen, 1, 2, false, t1);
    let (fp, gp) = fg.split_at_mut(slen);
    if zint_bezout(cap_g, cap_f, fp, gp, t1) != 0xFFFFFFFF {
        return false;
    }
    if zint_mul_small(cap_f, Q) != 0 || zint_mul_small(cap_g, Q) != 0 {
        return false;
    }
    fg[..slen].copy_from_slice(cap_f);
    fg[slen..].copy_from_slice(cap_g);
    true
}

fn pow2_scale(dc: i32) -> FLR {
    let mut d = dc;
    let mut pt = if d < 0 {
        d = -d;
        FLR::ONE.double()
    } else {
        FLR::ONE.half()
    };
    let mut pdc = FLR::ONE;
    while d != 0 {
        if (d & 1) != 0 {
            pdc *= pt;
        }
        d >>= 1;
        pt = pt.square();
    }
    pdc
}

fn solve_NTRU_intermediate(
    logn_top: u32,
    f: &[i8],
    g: &[i8],
    depth: u32,
    tmp_u32: &mut [u32],
    tmp_flr: &mut [FLR],
) -> bool {
    let logn = logn_top - depth;
    let n = 1usize << logn;
    let hn = n >> 1;
    let slen = MAX_BL_SMALL[depth as usize];
    let tlen = MAX_BL_SMALL[(depth + 1) as usize];
    let llen = MAX_BL_LARGE[depth as usize];

    make_fg_intermediate(logn_top, f, g, depth, &mut tmp_u32[(2 * tlen * hn)..]);
    tmp_u32.copy_within(0..(2 * tlen * hn), 2 * (llen + slen) * n);
    tmp_u32.copy_within(
        (2 * tlen * hn)..(2 * tlen * hn + 2 * slen * n),
        2 * llen * n,
    );

    {
        let (Ft, work) = tmp_u32.split_at_mut(llen * n);
        let (Gt, work) = work.split_at_mut(llen * n);
        let (_, work) = work.split_at_mut(2 * slen * n);
        let (Fd, work) = work.split_at_mut(tlen * hn);
        let (Gd, _) = work.split_at_mut(tlen * hn);
        for i in 0..llen {
            let p = PRIMES[i].p;
            let p0i = PRIMES[i].p0i;
            let R2 = PRIMES[i].R2;
            let Rx = mp_Rx31(tlen as u32, p, p0i, R2);
            let kt = i * n + hn;
            for j in 0..hn {
                Ft[kt + j] = zint_mod_small_signed(&Fd[j..], tlen, hn, p, p0i, R2, Rx);
                Gt[kt + j] = zint_mod_small_signed(&Gd[j..], tlen, hn, p, p0i, R2, Rx);
            }
        }
    }

    {
        let (FGt, work) = tmp_u32.split_at_mut(2 * llen * n);
        let (fgt, work) = work.split_at_mut(2 * slen * n);
        for i in 0..llen {
            let p = PRIMES[i].p;
            let p0i = PRIMES[i].p0i;
            let R2 = PRIMES[i].R2;
            let (Ft, Gt) = FGt.split_at_mut(llen * n);
            let (ft, gt) = fgt.split_at_mut(slen * n);
            let (gm, work) = work.split_at_mut(n);
            let (igm, work) = work.split_at_mut(n);
            let (fx, work) = work.split_at_mut(n);
            let (gx, _) = work.split_at_mut(n);

            mp_mkgmigm(logn, PRIMES[i].g, PRIMES[i].ig, p, p0i, gm, igm);
            if i < slen {
                fx.copy_from_slice(&ft[(i * n)..((i + 1) * n)]);
                gx.copy_from_slice(&gt[(i * n)..((i + 1) * n)]);
                mp_iNTT(logn, &mut ft[(i * n)..((i + 1) * n)], igm, p, p0i);
                mp_iNTT(logn, &mut gt[(i * n)..((i + 1) * n)], igm, p, p0i);
            } else {
                let Rx = mp_Rx31(slen as u32, p, p0i, R2);
                for j in 0..n {
                    fx[j] = zint_mod_small_signed(&ft[j..], slen, n, p, p0i, R2, Rx);
                    gx[j] = zint_mod_small_signed(&gt[j..], slen, n, p, p0i, R2, Rx);
                }
                mp_NTT(logn, fx, gm, p, p0i);
                mp_NTT(logn, gx, gm, p, p0i);
            }

            let kt = i * n + hn;
            mp_NTT(logn - 1, &mut Ft[kt..(kt + hn)], gm, p, p0i);
            mp_NTT(logn - 1, &mut Gt[kt..(kt + hn)], gm, p, p0i);

            let kt = i * n;
            for j in 0..hn {
                let fa = fx[2 * j];
                let fb = fx[2 * j + 1];
                let ga = gx[2 * j];
                let gb = gx[2 * j + 1];
                let mFp = mp_mmul(Ft[i * n + hn + j], R2, p, p0i);
                let mGp = mp_mmul(Gt[i * n + hn + j], R2, p, p0i);
                Ft[kt + 2 * j] = mp_mmul(gb, mFp, p, p0i);
                Ft[kt + 2 * j + 1] = mp_mmul(ga, mFp, p, p0i);
                Gt[kt + 2 * j] = mp_mmul(fb, mGp, p, p0i);
                Gt[kt + 2 * j + 1] = mp_mmul(fa, mGp, p, p0i);
            }
            mp_iNTT(logn, &mut Ft[kt..(kt + n)], igm, p, p0i);
            mp_iNTT(logn, &mut Gt[kt..(kt + n)], igm, p, p0i);

            if i + 1 == slen {
                zint_rebuild_CRT(fgt, slen, n, 2, true, work);
            }
        }
        zint_rebuild_CRT(FGt, llen, n, 2, true, work);
    }

    let (avg, std) = BITLENGTH[depth as usize];
    let minbl_fg = avg - 6 * std;
    let maxbl_fg = avg + 6 * std;

    let mut rlen = if slen > 10 { 10 } else { slen };
    {
        let (_, work) = tmp_u32.split_at_mut(2 * llen * n);
        let (ft, gt) = work.split_at_mut(slen * n);
        let (rt3, work) = tmp_flr.split_at_mut(n);
        let (rt4, work) = work.split_at_mut(n);
        let (rt5, work) = work.split_at_mut(hn);
        let (rt1, work) = work.split_at_mut(n);
        let (rt2, _) = work.split_at_mut(n);

        poly_big_to_flr_word_major(rt3, &ft[(slen - rlen) * n..], rlen, n, logn);
        poly_big_to_flr_word_major(rt4, &gt[(slen - rlen) * n..], rlen, n, logn);
        let scale_fg = 31 * ((slen - rlen) as i32);

        FFT(logn, rt3);
        FFT(logn, rt4);
        poly_invnorm2_fft(logn, rt5, rt3, rt4);
        poly_adj_fft(logn, rt3);
        poly_adj_fft(logn, rt4);

        let mut FGlen = llen;
        let mut maxbl_FG = 31 * (llen as i32);
        let mut scale_k = maxbl_FG - minbl_fg;
        loop {
            rlen = if FGlen > 10 { 10 } else { FGlen };
            let scale_FG = 31 * ((FGlen - rlen) as i32);
            poly_big_to_flr_word_major(rt1, &tmp_u32[(FGlen - rlen) * n..], rlen, n, logn);
            poly_big_to_flr_word_major(
                rt2,
                &tmp_u32[llen * n + (FGlen - rlen) * n..],
                rlen,
                n,
                logn,
            );

            FFT(logn, rt1);
            FFT(logn, rt2);
            poly_mul_fft(logn, rt1, rt3);
            poly_mul_fft(logn, rt2, rt4);
            poly_add(logn, rt2, rt1);
            poly_mul_selfadj_fft(logn, rt2, rt5);
            iFFT(logn, rt2);

            let pdc = pow2_scale(scale_k - scale_FG + scale_fg);
            let k_base = 2 * (llen + slen) * n;
            for u in 0..n {
                let xv = rt2[u] * pdc;
                let xf = flr_to_f64(xv);
                if xf <= -2147483647.0 || xf >= 2147483647.0 {
                    return false;
                }
                tmp_u32[k_base + u] = xv.rint() as i32 as u32;
            }

            let (Ft, work) = tmp_u32.split_at_mut(llen * n);
            let (Gt, work) = work.split_at_mut(llen * n);
            let (ft, work) = work.split_at_mut(slen * n);
            let (gt, work) = work.split_at_mut(slen * n);
            let (k, scratch) = work.split_at_mut(n);
            let scale_k_u = scale_k as u32;
            if depth == 1 {
                poly_sub_kfg_scaled_depth1(
                    logn_top,
                    Ft,
                    Gt,
                    FGlen,
                    &mut k[..n],
                    scale_k_u,
                    f,
                    g,
                    scratch,
                );
            } else {
                poly_sub_scaled(logn, Ft, FGlen, ft, slen, &k[..n], scale_k_u);
                poly_sub_scaled(logn, Gt, FGlen, gt, slen, &k[..n], scale_k_u);
            }

            let new_maxbl_FG = scale_k + maxbl_fg + 10;
            if new_maxbl_FG < maxbl_FG {
                maxbl_FG = new_maxbl_FG;
                if (FGlen as i32) * 31 >= maxbl_FG + 31 {
                    FGlen -= 1;
                }
            }

            if scale_k <= 0 {
                break;
            }
            scale_k -= 25;
            if scale_k < 0 {
                scale_k = 0;
            }
        }

        let final_FGlen = FGlen;
        let (Ft_u32, work_u32) = tmp_u32.split_at_mut(llen * n);
        let (Gt_u32, _) = work_u32.split_at_mut(llen * n);
        if final_FGlen < slen {
            for u in 0..n {
                let swf = (Ft_u32[u + (final_FGlen - 1) * n] >> 30).wrapping_neg() >> 1;
                let swg = (Gt_u32[u + (final_FGlen - 1) * n] >> 30).wrapping_neg() >> 1;
                for v in final_FGlen..slen {
                    Ft_u32[u + v * n] = swf;
                    Gt_u32[u + v * n] = swg;
                }
            }
        }
    }

    // Keep only the low slen words of F and G.
    // Layout here is word-major (word blocks of n coefficients), so F is
    // already in place and we only need to pack G after F.
    tmp_u32.copy_within((llen * n)..(llen * n + slen * n), slen * n);
    true
}

fn solve_NTRU_binary_depth1(logn_top: u32, f: &[i8], g: &[i8], tmp_u32: &mut [u32]) -> bool {
    let depth = 1u32;
    let logn = logn_top - depth;
    let n = 1usize << logn;
    let n_top = 1usize << logn_top;
    let hn = n >> 1;
    let slen = MAX_BL_SMALL[depth as usize];
    let dlen = MAX_BL_SMALL[(depth + 1) as usize];
    let llen = MAX_BL_LARGE[depth as usize];

    let mut Fd = vec![0u32; dlen * hn];
    let mut Gd = vec![0u32; dlen * hn];
    for v in 0..hn {
        for u in 0..dlen {
            Fd[v * dlen + u] = tmp_u32[u * hn + v];
            Gd[v * dlen + u] = tmp_u32[dlen * hn + u * hn + v];
        }
    }

    let mut FGt = vec![0u32; 2 * llen * n];
    let (Ft, Gt) = FGt.split_at_mut(llen * n);
    let mut gm = vec![0u32; n_top];
    let mut igm = vec![0u32; n_top];
    let mut fx = vec![0u32; n_top];
    let mut gx = vec![0u32; n_top];
    let mut Fp = vec![0u32; hn];
    let mut Gp = vec![0u32; hn];
    let mut tmp_crt = vec![0u32; llen.max(slen)];

    for u in 0..llen {
        let p = PRIMES[u].p;
        let p0i = PRIMES[u].p0i;
        let R2 = PRIMES[u].R2;
        let Rx = mp_Rx31(dlen as u32, p, p0i, R2);

        for v in 0..hn {
            Ft[v * llen + u] = zint_mod_small_signed(&Fd[v * dlen..], dlen, 1, p, p0i, R2, Rx);
            Gt[v * llen + u] = zint_mod_small_signed(&Gd[v * dlen..], dlen, 1, p, p0i, R2, Rx);
        }

        mp_mkgmigm(
            logn_top,
            PRIMES[u].g,
            PRIMES[u].ig,
            p,
            p0i,
            &mut gm,
            &mut igm,
        );
        for v in 0..n_top {
            fx[v] = mp_set(f[v] as i32, p);
            gx[v] = mp_set(g[v] as i32, p);
        }
        mp_NTT(logn_top, &mut fx, &gm, p, p0i);
        mp_NTT(logn_top, &mut gx, &gm, p, p0i);
        for e in ((logn + 1)..=logn_top).rev() {
            let h = 1usize << (e - 1);
            for v in 0..h {
                let fa = fx[(v << 1) + 0];
                let fb = fx[(v << 1) + 1];
                let ga = gx[(v << 1) + 0];
                let gb = gx[(v << 1) + 1];
                fx[v] = mp_mmul(mp_mmul(fa, fb, p, p0i), R2, p, p0i);
                gx[v] = mp_mmul(mp_mmul(ga, gb, p, p0i), R2, p, p0i);
            }
        }

        for v in 0..hn {
            Fp[v] = Ft[v * llen + u];
            Gp[v] = Gt[v * llen + u];
        }
        mp_NTT(logn - 1, &mut Fp, &gm, p, p0i);
        mp_NTT(logn - 1, &mut Gp, &gm, p, p0i);

        for v in 0..hn {
            let ftA = fx[(v << 1) + 0];
            let ftB = fx[(v << 1) + 1];
            let gtA = gx[(v << 1) + 0];
            let gtB = gx[(v << 1) + 1];
            let mFp = mp_mmul(Fp[v], R2, p, p0i);
            let mGp = mp_mmul(Gp[v], R2, p, p0i);
            Ft[((v << 1) + 0) * llen + u] = mp_mmul(gtB, mFp, p, p0i);
            Ft[((v << 1) + 1) * llen + u] = mp_mmul(gtA, mFp, p, p0i);
            Gt[((v << 1) + 0) * llen + u] = mp_mmul(ftB, mGp, p, p0i);
            Gt[((v << 1) + 1) * llen + u] = mp_mmul(ftA, mGp, p, p0i);
        }
        mp_intt_column(logn, &mut Ft[u..], llen, &igm[..n], p, p0i, &mut fx[..n]);
        mp_intt_column(logn, &mut Gt[u..], llen, &igm[..n], p, p0i, &mut gx[..n]);
    }

    {
        let mut combined_fg = vec![0u32; llen * (n << 1)];
        for u in 0..llen {
            for v in 0..n {
                combined_fg[v * llen + u] = Ft[v * llen + u];
                combined_fg[(n + v) * llen + u] = Gt[v * llen + u];
            }
        }
        zint_rebuild_CRT_stride(
            &mut combined_fg,
            llen,
            llen,
            n << 1,
            true,
            &mut tmp_crt[..llen],
        );
        for u in 0..llen {
            for v in 0..n {
                Ft[v * llen + u] = combined_fg[v * llen + u];
                Gt[v * llen + u] = combined_fg[(n + v) * llen + u];
            }
        }
    }

    let mut fg_plain = vec![0u32; 7 * n_top];
    make_fg_exact(&mut fg_plain, f, g, logn_top, 1, false);
    let (ft, rest) = fg_plain.split_at_mut(n);
    let (gt, _) = rest.split_at_mut(n);
    zint_rebuild_CRT_stride(ft, slen, slen, n, true, &mut tmp_crt[..slen]);
    zint_rebuild_CRT_stride(gt, slen, slen, n, true, &mut tmp_crt[..slen]);

    let mut cap_f = vec![FLR::ZERO; n];
    let mut cap_g = vec![FLR::ZERO; n];
    poly_big_to_flr_coeff_major(&mut cap_f, Ft, llen, llen, logn);
    poly_big_to_flr_coeff_major(&mut cap_g, Gt, llen, llen, logn);

    let mut flr_f = vec![FLR::ZERO; n];
    let mut flr_g = vec![FLR::ZERO; n];
    poly_big_to_flr_coeff_major(&mut flr_f, &ft, slen, slen, logn);
    poly_big_to_flr_coeff_major(&mut flr_g, &gt, slen, slen, logn);

    FFT(logn, &mut cap_f);
    FFT(logn, &mut cap_g);
    FFT(logn, &mut flr_f);
    FFT(logn, &mut flr_g);

    let mut num = vec![FLR::ZERO; n];
    let mut den = vec![FLR::ZERO; n];
    poly_add_muladj_fft(logn, &mut num, &cap_f, &cap_g, &flr_f, &flr_g);
    poly_invnorm2_fft(logn, &mut den[..(n >> 1)], &flr_f, &flr_g);
    poly_mul_autoadj_fft(logn, &mut num, &den[..(n >> 1)]);
    iFFT(logn, &mut num);
    for u in 0..n {
        let z = flr_to_f64(num[u]);
        if !(z > -(i64::MAX as f64) && z < (i64::MAX as f64)) {
            return false;
        }
        num[u] = FLR::from_i64(num[u].rint());
    }
    FFT(logn, &mut num);

    let mut kf = flr_f.clone();
    let mut kg = flr_g.clone();
    poly_mul_fft(logn, &mut kf, &num);
    poly_mul_fft(logn, &mut kg, &num);
    poly_sub(logn, &mut cap_f, &kf);
    poly_sub(logn, &mut cap_g, &kg);
    iFFT(logn, &mut cap_f);
    iFFT(logn, &mut cap_g);

    for u in 0..n {
        tmp_u32[u] = cap_f[u].rint() as i32 as u32;
        tmp_u32[n + u] = cap_g[u].rint() as i32 as u32;
    }
    true
}

fn solve_NTRU_binary_depth0(logn: u32, f: &[i8], g: &[i8], tmp_u32: &mut [u32]) -> bool {
    let n = 1usize << logn;
    let hn = n >> 1;
    let p = PRIMES[0].p;
    let p0i = PRIMES[0].p0i;
    let R2 = PRIMES[0].R2;

    let (Fp, work) = tmp_u32.split_at_mut(hn);
    let (Gp, work) = work.split_at_mut(hn);
    let (ft, work) = work.split_at_mut(n);
    let (gt, work) = work.split_at_mut(n);
    let (gm, igm) = work.split_at_mut(n);

    mp_mkgmigm(logn, PRIMES[0].g, PRIMES[0].ig, p, p0i, gm, igm);
    for u in 0..hn {
        Fp[u] = mp_set(zint_one_to_plain(&Fp[u..]), p);
        Gp[u] = mp_set(zint_one_to_plain(&Gp[u..]), p);
    }
    mp_NTT(logn - 1, Fp, gm, p, p0i);
    mp_NTT(logn - 1, Gp, gm, p, p0i);
    for u in 0..n {
        ft[u] = mp_set(f[u] as i32, p);
        gt[u] = mp_set(g[u] as i32, p);
    }
    mp_NTT(logn, ft, gm, p, p0i);
    mp_NTT(logn, gt, gm, p, p0i);
    for u in (0..n).step_by(2) {
        let ftA = ft[u + 0];
        let ftB = ft[u + 1];
        let gtA = gt[u + 0];
        let gtB = gt[u + 1];
        let mFp = mp_mmul(Fp[u >> 1], R2, p, p0i);
        let mGp = mp_mmul(Gp[u >> 1], R2, p, p0i);
        ft[u + 0] = mp_mmul(gtB, mFp, p, p0i);
        ft[u + 1] = mp_mmul(gtA, mFp, p, p0i);
        gt[u + 0] = mp_mmul(ftB, mGp, p, p0i);
        gt[u + 1] = mp_mmul(ftA, mGp, p, p0i);
    }
    mp_iNTT(logn, ft, igm, p, p0i);
    mp_iNTT(logn, gt, igm, p, p0i);

    tmp_u32.copy_within(2 * hn..(2 * hn + 2 * n), 0);
    let (Fp, work) = tmp_u32.split_at_mut(n);
    let (Gp, work) = work.split_at_mut(n);
    let (t1, work) = work.split_at_mut(n);
    let (t2, work) = work.split_at_mut(n);
    let (t3, work) = work.split_at_mut(n);
    let (t4, t5) = work.split_at_mut(n);

    mp_mkgmigm(logn, PRIMES[0].g, PRIMES[0].ig, p, p0i, t1, t4);
    mp_NTT(logn, Fp, t1, p, p0i);
    mp_NTT(logn, Gp, t1, p, p0i);

    t4[0] = mp_set(f[0] as i32, p);
    t5[0] = t4[0];
    for u in 1..n {
        t4[u] = mp_set(f[u] as i32, p);
        t5[n - u] = mp_set(-(f[u] as i32), p);
    }
    mp_NTT(logn, t4, t1, p, p0i);
    mp_NTT(logn, t5, t1, p, p0i);
    for u in 0..n {
        let w = mp_mmul(t5[u], R2, p, p0i);
        t2[u] = mp_mmul(w, Fp[u], p, p0i);
        t3[u] = mp_mmul(w, t4[u], p, p0i);
    }

    t4[0] = mp_set(g[0] as i32, p);
    t5[0] = t4[0];
    for u in 1..n {
        t4[u] = mp_set(g[u] as i32, p);
        t5[n - u] = mp_set(-(g[u] as i32), p);
    }
    mp_NTT(logn, t4, t1, p, p0i);
    mp_NTT(logn, t5, t1, p, p0i);
    for u in 0..n {
        let w = mp_mmul(t5[u], R2, p, p0i);
        t2[u] = mp_add(t2[u], mp_mmul(w, Gp[u], p, p0i), p);
        t3[u] = mp_add(t3[u], mp_mmul(w, t4[u], p, p0i), p);
    }

    mp_mkgmigm(logn, PRIMES[0].g, PRIMES[0].ig, p, p0i, t1, t4);
    mp_iNTT(logn, t2, t4, p, p0i);
    mp_iNTT(logn, t3, t4, p, p0i);
    for u in 0..n {
        t1[u] = mp_norm(t2[u], p) as u32;
        t2[u] = mp_norm(t3[u], p) as u32;
    }

    let mut rt2 = vec![FLR::ZERO; hn];
    let mut rt3 = vec![FLR::ZERO; n];
    for u in 0..n {
        rt3[u] = FLR::from_i32(t2[u] as i32);
    }
    FFT(logn, &mut rt3);
    rt2.copy_from_slice(&rt3[..hn]);
    for u in 0..n {
        rt3[u] = FLR::from_i32(t1[u] as i32);
    }
    FFT(logn, &mut rt3);
    poly_div_autoadj_fft(logn, &mut rt3, &rt2);
    iFFT(logn, &mut rt3);
    for u in 0..n {
        t1[u] = mp_set(rt3[u].rint() as i32, p);
    }

    mp_mkgmigm(logn, PRIMES[0].g, PRIMES[0].ig, p, p0i, t4, t5);
    mp_NTT(logn, t1, t4, p, p0i);
    for u in 0..n {
        t2[u] = mp_set(f[u] as i32, p);
        t3[u] = mp_set(g[u] as i32, p);
    }
    mp_NTT(logn, t2, t4, p, p0i);
    mp_NTT(logn, t3, t4, p, p0i);
    for u in 0..n {
        let kw = mp_mmul(t1[u], R2, p, p0i);
        Fp[u] = mp_sub(Fp[u], mp_mmul(kw, t2[u], p, p0i), p);
        Gp[u] = mp_sub(Gp[u], mp_mmul(kw, t3[u], p, p0i), p);
    }
    mp_iNTT(logn, Fp, t5, p, p0i);
    mp_iNTT(logn, Gp, t5, p, p0i);
    poly_mp_norm(logn, Fp, p);
    poly_mp_norm(logn, Gp, p);
    true
}

pub(crate) fn solve_NTRU(
    logn: u32,
    f: &[i8],
    g: &[i8],
    cap_f: &mut [i8],
    cap_g: &mut [i8],
    tmp_u32: &mut [u32],
    tmp_flr: &mut [FLR],
) -> bool {
    debug_solve_NTRU_stage(logn, f, g, cap_f, cap_g, tmp_u32, tmp_flr) == 0
}

pub(crate) fn debug_solve_NTRU_stage(
    logn: u32,
    f: &[i8],
    g: &[i8],
    cap_f: &mut [i8],
    cap_g: &mut [i8],
    tmp_u32: &mut [u32],
    tmp_flr: &mut [FLR],
) -> u32 {
    assert!((1..=10).contains(&logn));
    let n = 1usize << logn;
    if !solve_NTRU_deepest(logn, f, g, tmp_u32) {
        return 1;
    }
    for depth in ((if logn >= 3 { 2 } else { 1 })..logn).rev() {
        if !solve_NTRU_intermediate(logn, f, g, depth, tmp_u32, tmp_flr) {
            return 10 + depth;
        }
    }
    if logn >= 3 {
        if !solve_NTRU_binary_depth1(logn, f, g, tmp_u32) {
            return 101;
        }
        if !solve_NTRU_binary_depth0(logn, f, g, tmp_u32) {
            return 100;
        }
    } else if !solve_NTRU_binary_depth0(logn, f, g, tmp_u32) {
        return 2;
    }
    for i in 0..n {
        let zf = zint_one_to_plain(&tmp_u32[i..]);
        let zg = zint_one_to_plain(&tmp_u32[i + n..]);
        if !(-127..=127).contains(&zf) || !(-127..=127).contains(&zg) {
            return 3;
        }
        cap_f[i] = zf as i8;
        cap_g[i] = zg as i8;
    }
    0
}

#[cfg(test)]
pub(crate) fn debug_depth1_components(
    logn: u32,
    f: &[i8],
    g: &[i8],
    tmp_u32: &mut [u32],
    tmp_flr: &mut [FLR],
) -> Option<(vec::Vec<i32>, vec::Vec<i32>)> {
    assert!(logn == 10);
    if !solve_NTRU_deepest(logn, f, g, tmp_u32) {
        return None;
    }
    for depth in (2..logn).rev() {
        if !solve_NTRU_intermediate(logn, f, g, depth, tmp_u32, tmp_flr) {
            return None;
        }
    }
    if !solve_NTRU_binary_depth1(logn, f, g, tmp_u32) {
        return None;
    }
    let n = 1usize << (logn - 1);
    let mut cap_f = vec![0i32; n];
    let mut cap_g = vec![0i32; n];
    for i in 0..n {
        cap_f[i] = tmp_u32[i] as i32;
        cap_g[i] = tmp_u32[n + i] as i32;
    }
    Some((cap_f, cap_g))
}

#[cfg(test)]
pub(crate) fn debug_depth1_prebabai(
    logn_top: u32,
    f: &[i8],
    g: &[i8],
    tmp_u32: &mut [u32],
) -> Option<(vec::Vec<u32>, vec::Vec<u32>, vec::Vec<u32>, vec::Vec<u32>)> {
    assert!(logn_top == 10);
    let depth = 1u32;
    let logn = logn_top - depth;
    let n = 1usize << logn;
    let n_top = 1usize << logn_top;
    let hn = n >> 1;
    let slen = MAX_BL_SMALL[depth as usize];
    let dlen = MAX_BL_SMALL[(depth + 1) as usize];
    let llen = MAX_BL_LARGE[depth as usize];
    if !solve_NTRU_deepest(logn_top, f, g, tmp_u32) {
        return None;
    }
    let mut tmp_flr = [FLR::ZERO; 5 * 1024];
    for depth in (2..logn_top).rev() {
        if !solve_NTRU_intermediate(logn_top, f, g, depth, tmp_u32, &mut tmp_flr) {
            return None;
        }
    }

    let mut Fd = vec![0u32; dlen * hn];
    let mut Gd = vec![0u32; dlen * hn];
    for v in 0..hn {
        for u in 0..dlen {
            Fd[v * dlen + u] = tmp_u32[u * hn + v];
            Gd[v * dlen + u] = tmp_u32[dlen * hn + u * hn + v];
        }
    }

    let mut FGt = vec![0u32; 2 * llen * n];
    let (Ft, Gt) = FGt.split_at_mut(llen * n);
    let mut gm = vec![0u32; n_top];
    let mut igm = vec![0u32; n_top];
    let mut fx = vec![0u32; n_top];
    let mut gx = vec![0u32; n_top];
    let mut Fp = vec![0u32; hn];
    let mut Gp = vec![0u32; hn];
    let mut tmp_crt = vec![0u32; llen.max(slen)];

    for u in 0..llen {
        let p = PRIMES[u].p;
        let p0i = PRIMES[u].p0i;
        let R2 = PRIMES[u].R2;
        let Rx = mp_Rx31(dlen as u32, p, p0i, R2);

        for v in 0..hn {
            Ft[v * llen + u] = zint_mod_small_signed(&Fd[v * dlen..], dlen, 1, p, p0i, R2, Rx);
            Gt[v * llen + u] = zint_mod_small_signed(&Gd[v * dlen..], dlen, 1, p, p0i, R2, Rx);
        }

        mp_mkgmigm(
            logn_top,
            PRIMES[u].g,
            PRIMES[u].ig,
            p,
            p0i,
            &mut gm,
            &mut igm,
        );
        for v in 0..n_top {
            fx[v] = mp_set(f[v] as i32, p);
            gx[v] = mp_set(g[v] as i32, p);
        }
        mp_NTT(logn_top, &mut fx, &gm, p, p0i);
        mp_NTT(logn_top, &mut gx, &gm, p, p0i);
        for e in ((logn + 1)..=logn_top).rev() {
            let h = 1usize << (e - 1);
            for v in 0..h {
                let fa = fx[(v << 1) + 0];
                let fb = fx[(v << 1) + 1];
                let ga = gx[(v << 1) + 0];
                let gb = gx[(v << 1) + 1];
                fx[v] = mp_mmul(mp_mmul(fa, fb, p, p0i), R2, p, p0i);
                gx[v] = mp_mmul(mp_mmul(ga, gb, p, p0i), R2, p, p0i);
            }
        }

        for v in 0..hn {
            Fp[v] = Ft[v * llen + u];
            Gp[v] = Gt[v * llen + u];
        }
        mp_NTT(logn - 1, &mut Fp, &gm, p, p0i);
        mp_NTT(logn - 1, &mut Gp, &gm, p, p0i);

        for v in 0..hn {
            let ftA = fx[(v << 1) + 0];
            let ftB = fx[(v << 1) + 1];
            let gtA = gx[(v << 1) + 0];
            let gtB = gx[(v << 1) + 1];
            let mFp = mp_mmul(Fp[v], R2, p, p0i);
            let mGp = mp_mmul(Gp[v], R2, p, p0i);
            Ft[((v << 1) + 0) * llen + u] = mp_mmul(gtB, mFp, p, p0i);
            Ft[((v << 1) + 1) * llen + u] = mp_mmul(gtA, mFp, p, p0i);
            Gt[((v << 1) + 0) * llen + u] = mp_mmul(ftB, mGp, p, p0i);
            Gt[((v << 1) + 1) * llen + u] = mp_mmul(ftA, mGp, p, p0i);
        }
        mp_intt_column(logn, &mut Ft[u..], llen, &igm[..n], p, p0i, &mut fx[..n]);
        mp_intt_column(logn, &mut Gt[u..], llen, &igm[..n], p, p0i, &mut gx[..n]);
    }

    zint_rebuild_CRT_stride(Ft, llen, llen, n, true, &mut tmp_crt[..llen]);
    zint_rebuild_CRT_stride(Gt, llen, llen, n, true, &mut tmp_crt[..llen]);
    let mut fg_plain = vec![0u32; 7 * n_top];
    make_fg_exact(&mut fg_plain, f, g, logn_top, 1, false);
    let (ft, rest) = fg_plain.split_at_mut(n);
    let (gt, _) = rest.split_at_mut(n);
    zint_rebuild_CRT_stride(ft, slen, slen, n, true, &mut tmp_crt[..slen]);
    zint_rebuild_CRT_stride(gt, slen, slen, n, true, &mut tmp_crt[..slen]);
    Some((Ft.to_vec(), Gt.to_vec(), ft.to_vec(), gt.to_vec()))
}

#[cfg(test)]
pub(crate) fn debug_depth1_input(logn_top: u32, f: &[i8], g: &[i8], tmp_u32: &mut [u32]) -> bool {
    assert!(logn_top == 10);
    if !solve_NTRU_deepest(logn_top, f, g, tmp_u32) {
        return false;
    }
    let mut tmp_flr = [FLR::ZERO; 5 * 1024];
    for depth in (2..logn_top).rev() {
        if !solve_NTRU_intermediate(logn_top, f, g, depth, tmp_u32, &mut tmp_flr) {
            return false;
        }
    }
    let n = 1usize << (logn_top - 2);
    let slen = MAX_BL_SMALL[2];
    let mut transposed_f = vec![0u32; slen * n];
    let mut transposed_g = vec![0u32; slen * n];
    for coeff in 0..n {
        for word in 0..slen {
            transposed_f[coeff * slen + word] = tmp_u32[word * n + coeff];
            transposed_g[coeff * slen + word] = tmp_u32[slen * n + word * n + coeff];
        }
    }
    tmp_u32[..(slen * n)].copy_from_slice(&transposed_f);
    tmp_u32[(slen * n)..(2 * slen * n)].copy_from_slice(&transposed_g);
    true
}

#[cfg(test)]
pub(crate) fn debug_intermediate_output(
    logn_top: u32,
    f: &[i8],
    g: &[i8],
    target_depth: u32,
    tmp_u32: &mut [u32],
) -> bool {
    assert_eq!(logn_top, 10);
    assert!((2..logn_top).contains(&target_depth));
    if !solve_NTRU_deepest(logn_top, f, g, tmp_u32) {
        return false;
    }
    let mut tmp_flr = [FLR::ZERO; 5 * 1024];
    for depth in (target_depth..logn_top).rev() {
        if !solve_NTRU_intermediate(logn_top, f, g, depth, tmp_u32, &mut tmp_flr) {
            return false;
        }
        if depth == target_depth {
            let n = 1usize << (logn_top - target_depth);
            let slen = MAX_BL_SMALL[target_depth as usize];
            let mut transposed_f = vec![0u32; slen * n];
            let mut transposed_g = vec![0u32; slen * n];
            for coeff in 0..n {
                for word in 0..slen {
                    transposed_f[coeff * slen + word] = tmp_u32[word * n + coeff];
                    transposed_g[coeff * slen + word] = tmp_u32[slen * n + word * n + coeff];
                }
            }
            tmp_u32[..(slen * n)].copy_from_slice(&transposed_f);
            tmp_u32[(slen * n)..(2 * slen * n)].copy_from_slice(&transposed_g);
            return true;
        }
    }
    false
}

#[cfg(test)]
pub(crate) fn debug_deepest_resultants(
    logn_top: u32,
    f: &[i8],
    g: &[i8],
    fp: &mut [u32],
    gp: &mut [u32],
) -> bool {
    let slen = MAX_BL_SMALL[logn_top as usize];
    assert_eq!(fp.len(), slen);
    assert_eq!(gp.len(), slen);
    let n = 1usize << logn_top;
    let mut work = vec![0u32; 7 * n];
    make_fg_exact(&mut work, f, g, logn_top, logn_top, false);
    let mut fg = vec![0u32; 5 * slen];
    fg[..(2 * slen)].copy_from_slice(&work[..(2 * slen)]);
    let (fg, tail) = fg.split_at_mut(2 * slen);
    let (_, tmp) = tail.split_at_mut(2 * slen);
    zint_rebuild_CRT(fg, slen, 1, 2, false, tmp);
    fp.copy_from_slice(&fg[..slen]);
    gp.copy_from_slice(&fg[slen..(2 * slen)]);
    true
}

#[cfg(test)]
pub(crate) fn debug_deepest_bezout(
    logn_top: u32,
    f: &[i8],
    g: &[i8],
    Fp: &mut [u32],
    Gp: &mut [u32],
) -> bool {
    let slen = MAX_BL_SMALL[logn_top as usize];
    assert_eq!(Fp.len(), slen);
    assert_eq!(Gp.len(), slen);
    let mut fp = vec![0u32; slen];
    let mut gp = vec![0u32; slen];
    let mut out_g = vec![0u32; slen];
    let mut out_f = vec![0u32; slen];
    if !debug_deepest_resultants(logn_top, f, g, &mut fp, &mut gp) {
        return false;
    }
    let mut tmp = vec![0u32; 4 * slen];
    if zint_bezout(&mut out_g, &mut out_f, &mut fp, &mut gp, &mut tmp) != 0xFFFFFFFF {
        return false;
    }
    Fp.copy_from_slice(&out_f);
    Gp.copy_from_slice(&out_g);
    true
}

#[cfg(test)]
pub(crate) fn debug_top_output(
    logn: u32,
    f: &[i8],
    g: &[i8],
    F: &mut [u32],
    G: &mut [u32],
    tmp_u32: &mut [u32],
    tmp_flr: &mut [FLR],
) -> bool {
    let n = 1usize << logn;
    assert_eq!(F.len(), n);
    assert_eq!(G.len(), n);
    if !solve_NTRU_deepest(logn, f, g, tmp_u32) {
        return false;
    }
    for depth in (2..logn).rev() {
        if !solve_NTRU_intermediate(logn, f, g, depth, tmp_u32, tmp_flr) {
            return false;
        }
    }
    if !solve_NTRU_binary_depth1(logn, f, g, tmp_u32) {
        return false;
    }
    if !solve_NTRU_binary_depth0(logn, f, g, tmp_u32) {
        return false;
    }
    F.copy_from_slice(&tmp_u32[..n]);
    G.copy_from_slice(&tmp_u32[n..(2 * n)]);
    true
}

#[cfg(test)]
pub(crate) fn debug_depth1_fg(logn_top: u32, f: &[i8], g: &[i8]) -> (vec::Vec<u32>, vec::Vec<u32>) {
    assert_eq!(logn_top, 10);
    let depth = 1u32;
    let logn = logn_top - depth;
    let n = 1usize << logn;
    let logn_top_n = 1usize << logn_top;
    let mut fg_data = vec![0u32; 7 * logn_top_n];
    make_fg_exact(&mut fg_data, f, g, logn_top, depth, true);
    let (ft, gt) = fg_data.split_at_mut(n);
    let p = PRIMES[0].p;
    let p0i = PRIMES[0].p0i;
    let mut gm = vec![0u32; n];
    let mut igm = vec![0u32; n];
    let mut tmp = vec![0u32; n];
    mp_mkgmigm(logn, PRIMES[0].g, PRIMES[0].ig, p, p0i, &mut gm, &mut igm);
    mp_intt_column(logn, ft, 1, &igm, p, p0i, &mut tmp);
    mp_intt_column(logn, gt, 1, &igm, p, p0i, &mut tmp);
    (ft.to_vec(), gt.to_vec())
}
