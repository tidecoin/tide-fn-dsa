#![allow(dead_code)]

use alloc::vec;

use super::*;

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
