extern crate alloc;

use crate::PqcleanCandidateDebug;
use alloc::vec;
use alloc::vec::Vec;
use tide_fn_dsa_comm::{sign_key_size, vrfy_key_size};

pub(crate) const PQCLEAN_MAX_BL_SMALL: [usize; 11] = [1, 1, 2, 2, 4, 7, 14, 27, 53, 106, 209];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct PqcleanRefCandidateDebugRaw {
    within_bound: u32,
    within_norm: u32,
    invertible: u32,
    ortho_ok: u32,
    solve_ok: u32,
    solve_stage: u32,
}

unsafe extern "C" {
    fn pqclean_ref_keygen_seeded_512(
        seed: *const u8,
        seed_len: usize,
        pk: *mut u8,
        sk: *mut u8,
    ) -> i32;

    fn pqclean_ref_keygen_seeded_1024(
        seed: *const u8,
        seed_len: usize,
        pk: *mut u8,
        sk: *mut u8,
    ) -> i32;

    fn pqclean_ref_trace_attempts_512(
        seed: *const u8,
        seed_len: usize,
        max_attempts: usize,
        out: *mut PqcleanRefCandidateDebugRaw,
    ) -> usize;

    fn pqclean_ref_trace_attempts_1024(
        seed: *const u8,
        seed_len: usize,
        max_attempts: usize,
        out: *mut PqcleanRefCandidateDebugRaw,
    ) -> usize;

    fn pqclean_ref_keygen_components_512(
        seed: *const u8,
        seed_len: usize,
        f: *mut i8,
        g: *mut i8,
        F: *mut i8,
    ) -> i32;

    fn pqclean_ref_keygen_components_1024(
        seed: *const u8,
        seed_len: usize,
        f: *mut i8,
        g: *mut i8,
        F: *mut i8,
    ) -> i32;

    fn pqclean_ref_depth1_components_1024(
        f: *const i8,
        g: *const i8,
        F: *mut i32,
        G: *mut i32,
    ) -> i32;

    fn pqclean_ref_depth1_prebabai_1024(
        f: *const i8,
        g: *const i8,
        Ft: *mut u32,
        Gt: *mut u32,
        ft: *mut u32,
        gt: *mut u32,
    ) -> i32;

    fn pqclean_ref_depth1_input_1024(f: *const i8, g: *const i8, Fd: *mut u32, Gd: *mut u32)
        -> i32;

    fn pqclean_ref_intermediate_output_1024(
        f: *const i8,
        g: *const i8,
        target_depth: u32,
        F: *mut u32,
        G: *mut u32,
    ) -> i32;

    fn pqclean_ref_deepest_resultants_1024(
        f: *const i8,
        g: *const i8,
        fp: *mut u32,
        gp: *mut u32,
    ) -> i32;

    fn pqclean_ref_deepest_bezout_1024(
        f: *const i8,
        g: *const i8,
        Fp: *mut u32,
        Gp: *mut u32,
    ) -> i32;

    fn pqclean_ref_top_output_1024(f: *const i8, g: *const i8, F: *mut u32, G: *mut u32) -> i32;

    fn pqclean_ref_depth1_fg_1024(f: *const i8, g: *const i8, ft: *mut u32, gt: *mut u32) -> i32;

    fn pqclean_ref_poly_big_to_fp_1024(
        f: *const u32,
        flen: usize,
        fstride: usize,
        logn: u32,
        d: *mut u64,
    ) -> i32;

    fn pqclean_ref_fft_1024(logn: u32, f: *mut u64);
    fn pqclean_ref_ifft_1024(logn: u32, f: *mut u64);
    fn pqclean_ref_poly_add_muladj_fft_1024(
        logn: u32,
        d: *mut u64,
        cap_f: *const u64,
        cap_g: *const u64,
        f: *const u64,
        g: *const u64,
    );
    fn pqclean_ref_poly_invnorm2_fft_1024(logn: u32, d: *mut u64, f: *const u64, g: *const u64);
    fn pqclean_ref_poly_mul_autoadj_fft_1024(logn: u32, a: *mut u64, b: *const u64);
    fn pqclean_ref_poly_mul_fft_1024(logn: u32, a: *mut u64, b: *const u64);
}

fn convert_debug(raw: PqcleanRefCandidateDebugRaw) -> PqcleanCandidateDebug {
    PqcleanCandidateDebug {
        within_bound: raw.within_bound != 0,
        within_norm: raw.within_norm != 0,
        invertible: raw.invertible != 0,
        ortho_ok: raw.ortho_ok != 0,
        solve_ok: raw.solve_ok != 0,
        solve_stage: raw.solve_stage,
    }
}

pub(crate) fn deterministic_keygen_from_seed(
    logn: u32,
    seed: &[u8; crate::FALCON_KEYGEN_SEED_SIZE],
    sign_key: &mut [u8],
    vrfy_key: &mut [u8],
) -> bool {
    let expected_sk_len = sign_key_size(logn).expect("valid pqclean logn");
    let expected_vk_len = vrfy_key_size(logn).expect("valid pqclean logn");
    assert_eq!(sign_key.len(), expected_sk_len);
    assert_eq!(vrfy_key.len(), expected_vk_len);

    let rc = unsafe {
        match logn {
            9 => pqclean_ref_keygen_seeded_512(
                seed.as_ptr(),
                seed.len(),
                vrfy_key.as_mut_ptr(),
                sign_key.as_mut_ptr(),
            ),
            10 => pqclean_ref_keygen_seeded_1024(
                seed.as_ptr(),
                seed.len(),
                vrfy_key.as_mut_ptr(),
                sign_key.as_mut_ptr(),
            ),
            _ => panic!("unsupported logn for pqclean reference: {logn}"),
        }
    };
    rc == 0
}

pub(crate) fn candidate_trace_from_seed(
    logn: u32,
    seed: &[u8; crate::FALCON_KEYGEN_SEED_SIZE],
    max_attempts: usize,
) -> Vec<PqcleanCandidateDebug> {
    let mut raw = vec![PqcleanRefCandidateDebugRaw::default(); max_attempts];
    let count = unsafe {
        match logn {
            9 => pqclean_ref_trace_attempts_512(
                seed.as_ptr(),
                seed.len(),
                max_attempts,
                raw.as_mut_ptr(),
            ),
            10 => pqclean_ref_trace_attempts_1024(
                seed.as_ptr(),
                seed.len(),
                max_attempts,
                raw.as_mut_ptr(),
            ),
            _ => panic!("unsupported logn for pqclean reference: {logn}"),
        }
    };
    raw.truncate(count);
    raw.into_iter().map(convert_debug).collect()
}

pub(crate) fn keygen_components_from_seed(
    logn: u32,
    seed: &[u8; crate::FALCON_KEYGEN_SEED_SIZE],
    f: &mut [i8],
    g: &mut [i8],
    F: &mut [i8],
) -> bool {
    let rc = unsafe {
        match logn {
            9 => pqclean_ref_keygen_components_512(
                seed.as_ptr(),
                seed.len(),
                f.as_mut_ptr(),
                g.as_mut_ptr(),
                F.as_mut_ptr(),
            ),
            10 => pqclean_ref_keygen_components_1024(
                seed.as_ptr(),
                seed.len(),
                f.as_mut_ptr(),
                g.as_mut_ptr(),
                F.as_mut_ptr(),
            ),
            _ => panic!("unsupported logn for pqclean reference: {logn}"),
        }
    };
    rc == 0
}

pub(crate) fn depth1_components_from_fg_1024(
    f: &[i8],
    g: &[i8],
    F: &mut [i32],
    G: &mut [i32],
) -> bool {
    assert_eq!(f.len(), 1024);
    assert_eq!(g.len(), 1024);
    assert_eq!(F.len(), 512);
    assert_eq!(G.len(), 512);
    let rc = unsafe {
        pqclean_ref_depth1_components_1024(f.as_ptr(), g.as_ptr(), F.as_mut_ptr(), G.as_mut_ptr())
    };
    rc == 0
}

pub(crate) fn depth1_prebabai_from_fg_1024(
    f: &[i8],
    g: &[i8],
    Ft: &mut [u32],
    Gt: &mut [u32],
    ft: &mut [u32],
    gt: &mut [u32],
) -> bool {
    assert_eq!(f.len(), 1024);
    assert_eq!(g.len(), 1024);
    assert_eq!(Ft.len(), 2 * 512);
    assert_eq!(Gt.len(), 2 * 512);
    assert_eq!(ft.len(), 1 * 512);
    assert_eq!(gt.len(), 1 * 512);
    let mut out_ft = [0u32; 1024];
    let mut out_gt = [0u32; 1024];
    let mut out_ft_small = [0u32; 512];
    let mut out_gt_small = [0u32; 512];

    let rc = unsafe {
        pqclean_ref_depth1_prebabai_1024(
            f.as_ptr(),
            g.as_ptr(),
            out_ft.as_mut_ptr(),
            out_gt.as_mut_ptr(),
            out_ft_small.as_mut_ptr(),
            out_gt_small.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return false;
    }
    Ft.copy_from_slice(&out_ft);
    Gt.copy_from_slice(&out_gt);
    ft.copy_from_slice(&out_ft_small);
    gt.copy_from_slice(&out_gt_small);
    true
}

pub(crate) fn depth1_input_from_fg_1024(
    f: &[i8],
    g: &[i8],
    Fd: &mut [u32],
    Gd: &mut [u32],
) -> bool {
    assert_eq!(f.len(), 1024);
    assert_eq!(g.len(), 1024);
    let rc = unsafe {
        pqclean_ref_depth1_input_1024(f.as_ptr(), g.as_ptr(), Fd.as_mut_ptr(), Gd.as_mut_ptr())
    };
    rc == 0
}

pub(crate) fn intermediate_output_from_fg_1024(
    f: &[i8],
    g: &[i8],
    target_depth: u32,
    F: &mut [u32],
    G: &mut [u32],
) -> bool {
    assert_eq!(f.len(), 1024);
    assert_eq!(g.len(), 1024);
    let n = 1usize << (10 - target_depth);
    let slen = PQCLEAN_MAX_BL_SMALL[target_depth as usize];
    assert_eq!(F.len(), slen * n);
    assert_eq!(G.len(), slen * n);
    let rc = unsafe {
        pqclean_ref_intermediate_output_1024(
            f.as_ptr(),
            g.as_ptr(),
            target_depth,
            F.as_mut_ptr(),
            G.as_mut_ptr(),
        )
    };
    rc == 0
}

pub(crate) fn deepest_resultants_from_fg_1024(
    f: &[i8],
    g: &[i8],
    fp: &mut [u32],
    gp: &mut [u32],
) -> bool {
    assert_eq!(f.len(), 1024);
    assert_eq!(g.len(), 1024);
    assert_eq!(fp.len(), PQCLEAN_MAX_BL_SMALL[10]);
    assert_eq!(gp.len(), PQCLEAN_MAX_BL_SMALL[10]);
    let rc = unsafe {
        pqclean_ref_deepest_resultants_1024(
            f.as_ptr(),
            g.as_ptr(),
            fp.as_mut_ptr(),
            gp.as_mut_ptr(),
        )
    };
    rc == 0
}

pub(crate) fn deepest_bezout_from_fg_1024(
    f: &[i8],
    g: &[i8],
    Fp: &mut [u32],
    Gp: &mut [u32],
) -> bool {
    assert_eq!(f.len(), 1024);
    assert_eq!(g.len(), 1024);
    assert_eq!(Fp.len(), PQCLEAN_MAX_BL_SMALL[10]);
    assert_eq!(Gp.len(), PQCLEAN_MAX_BL_SMALL[10]);
    let rc = unsafe {
        pqclean_ref_deepest_bezout_1024(f.as_ptr(), g.as_ptr(), Fp.as_mut_ptr(), Gp.as_mut_ptr())
    };
    rc == 0
}

pub(crate) fn top_output_from_fg_1024(f: &[i8], g: &[i8], F: &mut [u32], G: &mut [u32]) -> bool {
    assert_eq!(f.len(), 1024);
    assert_eq!(g.len(), 1024);
    assert_eq!(F.len(), 1024);
    assert_eq!(G.len(), 1024);
    let rc = unsafe {
        pqclean_ref_top_output_1024(f.as_ptr(), g.as_ptr(), F.as_mut_ptr(), G.as_mut_ptr())
    };
    rc == 0
}

pub(crate) fn depth1_fg_from_fg_1024(f: &[i8], g: &[i8], ft: &mut [u32], gt: &mut [u32]) -> bool {
    assert_eq!(f.len(), 1024);
    assert_eq!(g.len(), 1024);
    assert_eq!(ft.len(), 512);
    assert_eq!(gt.len(), 512);
    let rc = unsafe {
        pqclean_ref_depth1_fg_1024(f.as_ptr(), g.as_ptr(), ft.as_mut_ptr(), gt.as_mut_ptr())
    };
    rc == 0
}

pub(crate) fn poly_big_to_fp_1024(
    f: &[u32],
    flen: usize,
    fstride: usize,
    logn: u32,
    d: &mut [u64],
) -> bool {
    assert!(d.len() >= (1usize << logn));
    let rc = unsafe {
        pqclean_ref_poly_big_to_fp_1024(f.as_ptr(), flen, fstride, logn, d.as_mut_ptr())
    };
    rc == 0
}

pub(crate) fn fft_1024(logn: u32, f: &mut [u64]) {
    unsafe { pqclean_ref_fft_1024(logn, f.as_mut_ptr()) }
}

pub(crate) fn ifft_1024(logn: u32, f: &mut [u64]) {
    unsafe { pqclean_ref_ifft_1024(logn, f.as_mut_ptr()) }
}

pub(crate) fn poly_add_muladj_fft_1024(
    logn: u32,
    d: &mut [u64],
    cap_f: &[u64],
    cap_g: &[u64],
    f: &[u64],
    g: &[u64],
) {
    unsafe {
        pqclean_ref_poly_add_muladj_fft_1024(
            logn,
            d.as_mut_ptr(),
            cap_f.as_ptr(),
            cap_g.as_ptr(),
            f.as_ptr(),
            g.as_ptr(),
        )
    }
}

pub(crate) fn poly_invnorm2_fft_1024(logn: u32, d: &mut [u64], f: &[u64], g: &[u64]) {
    unsafe { pqclean_ref_poly_invnorm2_fft_1024(logn, d.as_mut_ptr(), f.as_ptr(), g.as_ptr()) }
}

pub(crate) fn poly_mul_autoadj_fft_1024(logn: u32, a: &mut [u64], b: &[u64]) {
    unsafe { pqclean_ref_poly_mul_autoadj_fft_1024(logn, a.as_mut_ptr(), b.as_ptr()) }
}

pub(crate) fn poly_mul_fft_1024(logn: u32, a: &mut [u64], b: &[u64]) {
    unsafe { pqclean_ref_poly_mul_fft_1024(logn, a.as_mut_ptr(), b.as_ptr()) }
}
