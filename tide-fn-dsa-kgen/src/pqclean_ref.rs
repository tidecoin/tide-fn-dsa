extern crate alloc;

use tide_fn_dsa_comm::{sign_key_size, vrfy_key_size};

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

pub(crate) fn keygen_components_from_seed_1024(
    seed: &[u8; crate::FALCON_KEYGEN_SEED_SIZE],
    f: &mut [i8],
    g: &mut [i8],
    F: &mut [i8],
) -> bool {
    let rc = unsafe {
        pqclean_ref_keygen_components_1024(
            seed.as_ptr(),
            seed.len(),
            f.as_mut_ptr(),
            g.as_mut_ptr(),
            F.as_mut_ptr(),
        )
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
