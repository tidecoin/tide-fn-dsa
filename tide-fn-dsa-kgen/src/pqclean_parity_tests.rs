extern crate alloc;

use alloc::vec;

use crate::pqclean_ntru::debug::{
    debug_depth1_components, debug_depth1_input, debug_depth1_prebabai,
};
use crate::pqclean_ref::{
    depth1_components_from_fg_1024, depth1_input_from_fg_1024, depth1_prebabai_from_fg_1024,
    deterministic_keygen_from_seed, keygen_components_from_seed_1024,
};
use crate::{
    sign_key_size, vrfy_key_size, FALCON_KEYGEN_SEED_SIZE, FN_DSA_LOGN_1024, FN_DSA_LOGN_512,
    KeyPairGeneratorStandard,
};

fn zero_seed() -> [u8; FALCON_KEYGEN_SEED_SIZE] {
    [0u8; FALCON_KEYGEN_SEED_SIZE]
}

fn pqclean_fg_1024_from_seed(seed: &[u8; FALCON_KEYGEN_SEED_SIZE]) -> ([i8; 1024], [i8; 1024]) {
    let mut f = [0i8; 1024];
    let mut g = [0i8; 1024];
    let mut cap_f = [0i8; 1024];
    assert!(keygen_components_from_seed_1024(seed, &mut f, &mut g, &mut cap_f));
    (f, g)
}

#[test]
fn pqclean_parity_seeded_keygen_matches_local_oracle_512() {
    let seed = zero_seed();
    let mut kg = KeyPairGeneratorStandard::default();
    let mut rust_sk = [0u8; crate::SIGN_KEY_SIZE_512];
    let mut rust_vk = [0u8; crate::VRFY_KEY_SIZE_512];
    let mut ref_sk = [0u8; crate::SIGN_KEY_SIZE_512];
    let mut ref_vk = [0u8; crate::VRFY_KEY_SIZE_512];

    kg.keygen_from_seed_pqclean(FN_DSA_LOGN_512, &seed, &mut rust_sk, &mut rust_vk)
        .unwrap();
    assert!(deterministic_keygen_from_seed(
        FN_DSA_LOGN_512,
        &seed,
        &mut ref_sk,
        &mut ref_vk,
    ));

    assert_eq!(rust_sk, ref_sk);
    assert_eq!(rust_vk, ref_vk);
}

#[test]
fn pqclean_parity_seeded_keygen_matches_local_oracle_1024() {
    let seed = zero_seed();
    let mut kg = KeyPairGeneratorStandard::default();
    let mut rust_sk = vec![0u8; sign_key_size(FN_DSA_LOGN_1024).unwrap()];
    let mut rust_vk = vec![0u8; vrfy_key_size(FN_DSA_LOGN_1024).unwrap()];
    let mut ref_sk = vec![0u8; sign_key_size(FN_DSA_LOGN_1024).unwrap()];
    let mut ref_vk = vec![0u8; vrfy_key_size(FN_DSA_LOGN_1024).unwrap()];

    kg.keygen_from_seed_pqclean(FN_DSA_LOGN_1024, &seed, &mut rust_sk, &mut rust_vk)
        .unwrap();
    assert!(deterministic_keygen_from_seed(
        FN_DSA_LOGN_1024,
        &seed,
        &mut ref_sk,
        &mut ref_vk,
    ));

    assert_eq!(rust_sk, ref_sk);
    assert_eq!(rust_vk, ref_vk);
}

#[test]
fn pqclean_parity_depth1_input_matches_local_oracle_1024() {
    let seed = zero_seed();
    let (f, g) = pqclean_fg_1024_from_seed(&seed);
    let mut tmp_u32 = [0u32; 8 * 1024];
    let mut rust_fd = [0u32; 512];
    let mut rust_gd = [0u32; 512];
    let mut ref_fd = [0u32; 512];
    let mut ref_gd = [0u32; 512];

    assert!(debug_depth1_input(FN_DSA_LOGN_1024, &f, &g, &mut tmp_u32));
    rust_fd.copy_from_slice(&tmp_u32[..512]);
    rust_gd.copy_from_slice(&tmp_u32[512..1024]);
    assert!(depth1_input_from_fg_1024(&f, &g, &mut ref_fd, &mut ref_gd));

    assert_eq!(rust_fd, ref_fd);
    assert_eq!(rust_gd, ref_gd);
}

#[test]
fn pqclean_parity_depth1_prebabai_matches_local_oracle_1024() {
    let seed = zero_seed();
    let (f, g) = pqclean_fg_1024_from_seed(&seed);
    let mut tmp_u32 = [0u32; 8 * 1024];
    let (rust_ft, rust_gt, rust_f_small, rust_g_small) =
        debug_depth1_prebabai(FN_DSA_LOGN_1024, &f, &g, &mut tmp_u32).unwrap();
    let mut ref_ft = [0u32; 1024];
    let mut ref_gt = [0u32; 1024];
    let mut ref_f_small = [0u32; 512];
    let mut ref_g_small = [0u32; 512];

    assert!(depth1_prebabai_from_fg_1024(
        &f,
        &g,
        &mut ref_ft,
        &mut ref_gt,
        &mut ref_f_small,
        &mut ref_g_small,
    ));

    assert_eq!(rust_ft, ref_ft);
    assert_eq!(rust_gt, ref_gt);
    assert_eq!(rust_f_small, ref_f_small);
    assert_eq!(rust_g_small, ref_g_small);
}

#[test]
fn pqclean_parity_depth1_components_matches_local_oracle_1024() {
    let seed = zero_seed();
    let (f, g) = pqclean_fg_1024_from_seed(&seed);
    let mut tmp_u32 = [0u32; 8 * 1024];
    let mut tmp_flr = [crate::flr::FLR::ZERO; 5 * 1024];
    let (rust_cap_f, rust_cap_g) =
        debug_depth1_components(FN_DSA_LOGN_1024, &f, &g, &mut tmp_u32, &mut tmp_flr).unwrap();
    let mut ref_cap_f = [0i32; 512];
    let mut ref_cap_g = [0i32; 512];

    assert!(depth1_components_from_fg_1024(
        &f,
        &g,
        &mut ref_cap_f,
        &mut ref_cap_g,
    ));

    assert_eq!(rust_cap_f, ref_cap_f);
    assert_eq!(rust_cap_g, ref_cap_g);
}
