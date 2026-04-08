#ifndef TIDE_FN_DSA_PQCLEAN_REF_H__
#define TIDE_FN_DSA_PQCLEAN_REF_H__

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t within_bound;
    uint32_t within_norm;
    uint32_t invertible;
    uint32_t ortho_ok;
    uint32_t solve_ok;
    uint32_t solve_stage;
} pqclean_ref_candidate_debug;

int pqclean_ref_keygen_seeded_512(
    const uint8_t *seed,
    size_t seed_len,
    uint8_t *pk,
    uint8_t *sk
);

int pqclean_ref_keygen_seeded_1024(
    const uint8_t *seed,
    size_t seed_len,
    uint8_t *pk,
    uint8_t *sk
);

size_t pqclean_ref_trace_attempts_512(
    const uint8_t *seed,
    size_t seed_len,
    size_t max_attempts,
    pqclean_ref_candidate_debug *out
);

size_t pqclean_ref_trace_attempts_1024(
    const uint8_t *seed,
    size_t seed_len,
    size_t max_attempts,
    pqclean_ref_candidate_debug *out
);

int pqclean_ref_keygen_components_512(
    const uint8_t *seed,
    size_t seed_len,
    int8_t *f,
    int8_t *g,
    int8_t *F
);

int pqclean_ref_keygen_components_1024(
    const uint8_t *seed,
    size_t seed_len,
    int8_t *f,
    int8_t *g,
    int8_t *F
);

int pqclean_ref_depth1_components_1024(
    const int8_t *f,
    const int8_t *g,
    int32_t *F,
    int32_t *G
);

int pqclean_ref_depth1_prebabai_1024(
    const int8_t *f,
    const int8_t *g,
    uint32_t *Ft,
    uint32_t *Gt,
    uint32_t *ft,
    uint32_t *gt
);

int pqclean_ref_depth1_input_1024(
    const int8_t *f,
    const int8_t *g,
    uint32_t *Fd,
    uint32_t *Gd
);

int pqclean_ref_intermediate_output_1024(
    const int8_t *f,
    const int8_t *g,
    uint32_t target_depth,
    uint32_t *F,
    uint32_t *G
);

int pqclean_ref_deepest_resultants_1024(
    const int8_t *f,
    const int8_t *g,
    uint32_t *fp,
    uint32_t *gp
);

int pqclean_ref_deepest_bezout_1024(
    const int8_t *f,
    const int8_t *g,
    uint32_t *Fp,
    uint32_t *Gp
);

int pqclean_ref_top_output_1024(
    const int8_t *f,
    const int8_t *g,
    uint32_t *F,
    uint32_t *G
);

int pqclean_ref_depth1_fg_1024(
    const int8_t *f,
    const int8_t *g,
    uint32_t *ft,
    uint32_t *gt
);

int pqclean_ref_depth1_unreduced_fg_1024(
    const int8_t *f,
    const int8_t *g,
    uint32_t *Ft,
    uint32_t *Gt
);

int pqclean_ref_poly_big_to_fp_1024(
    const uint32_t *f,
    size_t flen,
    size_t fstride,
    uint32_t logn,
    uint64_t *d
);

void pqclean_ref_fft_1024(
    uint32_t logn,
    uint64_t *f
);

void pqclean_ref_ifft_1024(
    uint32_t logn,
    uint64_t *f
);

void pqclean_ref_poly_add_muladj_fft_1024(
    uint32_t logn,
    uint64_t *d,
    const uint64_t *F,
    const uint64_t *G,
    const uint64_t *f,
    const uint64_t *g
);

void pqclean_ref_poly_invnorm2_fft_1024(
    uint32_t logn,
    uint64_t *d,
    const uint64_t *f,
    const uint64_t *g
);

void pqclean_ref_poly_mul_autoadj_fft_1024(
    uint32_t logn,
    uint64_t *a,
    const uint64_t *b
);

void pqclean_ref_poly_mul_fft_1024(
    uint32_t logn,
    uint64_t *a,
    const uint64_t *b
);

#endif
