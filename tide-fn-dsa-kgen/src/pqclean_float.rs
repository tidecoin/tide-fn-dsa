#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

#[path = "../../tide-fn-dsa-sign/src/flr.rs"]
mod flr;
#[path = "../../tide-fn-dsa-sign/src/poly.rs"]
mod poly;

pub(crate) use flr::FLR;
pub(crate) use poly::{iFFT, poly_add, poly_mul_fft, poly_sub, FFT};

#[inline(always)]
fn flc_mul(x_re: FLR, x_im: FLR, y_re: FLR, y_im: FLR) -> (FLR, FLR) {
    (x_re * y_re - x_im * y_im, x_re * y_im + x_im * y_re)
}

#[inline(always)]
pub(crate) fn poly_adj_fft(logn: u32, a: &mut [FLR]) {
    let n = 1usize << logn;
    for i in (n >> 1)..n {
        a[i] = -a[i];
    }
}

#[inline(always)]
pub(crate) fn poly_invnorm2_fft(logn: u32, d: &mut [FLR], f: &[FLR], g: &[FLR]) {
    let hn = 1usize << (logn - 1);
    for i in 0..hn {
        let nf = f[i].square() + f[i + hn].square();
        let ng = g[i].square() + g[i + hn].square();
        d[i] = FLR::ONE / (nf + ng);
    }
}

#[inline(always)]
pub(crate) fn poly_add_muladj_fft(
    logn: u32,
    d: &mut [FLR],
    cap_f: &[FLR],
    cap_g: &[FLR],
    f: &[FLR],
    g: &[FLR],
) {
    let hn = 1usize << (logn - 1);
    for i in 0..hn {
        let (a_re, a_im) = flc_mul(cap_f[i], cap_f[i + hn], f[i], -f[i + hn]);
        let (b_re, b_im) = flc_mul(cap_g[i], cap_g[i + hn], g[i], -g[i + hn]);
        d[i] = a_re + b_re;
        d[i + hn] = a_im + b_im;
    }
}

#[inline(always)]
pub(crate) fn poly_mul_selfadj_fft(logn: u32, a: &mut [FLR], b: &[FLR]) {
    let hn = 1usize << (logn - 1);
    for i in 0..hn {
        a[i] *= b[i];
        a[i + hn] *= b[i];
    }
}

#[inline(always)]
pub(crate) fn poly_div_selfadj_fft(logn: u32, a: &mut [FLR], b: &[FLR]) {
    let hn = 1usize << (logn - 1);
    for i in 0..hn {
        let x = FLR::ONE / b[i];
        a[i] *= x;
        a[i + hn] *= x;
    }
}

#[inline(always)]
pub(crate) fn poly_mul_autoadj_fft(logn: u32, a: &mut [FLR], b: &[FLR]) {
    poly_mul_selfadj_fft(logn, a, b);
}

#[inline(always)]
pub(crate) fn poly_div_autoadj_fft(logn: u32, a: &mut [FLR], b: &[FLR]) {
    poly_div_selfadj_fft(logn, a, b);
}

#[inline(always)]
pub(crate) fn flr_to_f64(x: FLR) -> f64 {
    f64::from_le_bytes(x.encode())
}

#[inline(always)]
pub(crate) fn poly_set_small(logn: u32, d: &mut [FLR], f: &[i8]) {
    let n = 1usize << logn;
    for i in 0..n {
        d[i] = FLR::from_i32(f[i] as i32);
    }
}

#[inline(always)]
pub(crate) fn poly_mulconst(logn: u32, a: &mut [FLR], x: FLR) {
    let n = 1usize << logn;
    for i in 0..n {
        a[i] *= x;
    }
}
