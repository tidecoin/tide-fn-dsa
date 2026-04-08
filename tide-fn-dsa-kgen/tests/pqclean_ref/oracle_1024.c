#include <string.h>

#include "pqclean_ref.h"
#include "falcon-1024/api.h"
#include "falcon-1024/inner.h"

/*
 * Debug depth-1 pre-Babai trace path needs more scratch than
 * FALCON_KEYGEN_TEMP_10 because it reuses the same tmp region with
 * overlapping staged buffers before calling make_fg().
 */
#define TIDE_PQCLEAN_DEBUG_TMP_1024_PREBABAI 32768

void PQCLEAN_FALCON1024_CLEAN_debug_sample_fg(
    inner_shake256_context *rng,
    int8_t *f,
    int8_t *g,
    unsigned logn
);

void PQCLEAN_FALCON1024_CLEAN_debug_eval_candidate(
    unsigned logn,
    const int8_t *f,
    const int8_t *g,
    pqclean_ref_candidate_debug *dbg,
    uint8_t *tmp
);

int PQCLEAN_FALCON1024_CLEAN_debug_solve_depth1(
    const int8_t *f,
    const int8_t *g,
    int32_t *F,
    int32_t *G,
    uint8_t *tmp
);

int PQCLEAN_FALCON1024_CLEAN_debug_solve_depth1_prebabai(
    const int8_t *f,
    const int8_t *g,
    uint32_t *Ft,
    uint32_t *Gt,
    uint32_t *ft,
    uint32_t *gt,
    uint8_t *tmp
);

int PQCLEAN_FALCON1024_CLEAN_debug_depth1_input(
    const int8_t *f,
    const int8_t *g,
    uint32_t *Fd,
    uint32_t *Gd,
    uint8_t *tmp
);

int PQCLEAN_FALCON1024_CLEAN_debug_intermediate_output(
    const int8_t *f,
    const int8_t *g,
    uint32_t target_depth,
    uint32_t *F,
    uint32_t *G,
    uint8_t *tmp
);

int PQCLEAN_FALCON1024_CLEAN_debug_deepest_resultants(
    const int8_t *f,
    const int8_t *g,
    uint32_t *fp,
    uint32_t *gp,
    uint8_t *tmp
);

int PQCLEAN_FALCON1024_CLEAN_debug_deepest_bezout(
    const int8_t *f,
    const int8_t *g,
    uint32_t *Fp,
    uint32_t *Gp,
    uint8_t *tmp
);

int PQCLEAN_FALCON1024_CLEAN_debug_top_output(
    const int8_t *f,
    const int8_t *g,
    uint32_t *F,
    uint32_t *G,
    uint8_t *tmp
);

int PQCLEAN_FALCON1024_CLEAN_debug_depth1_fg(
    const int8_t *f,
    const int8_t *g,
    uint32_t *ft,
    uint32_t *gt,
    uint8_t *tmp
);

int PQCLEAN_FALCON1024_CLEAN_debug_depth1_unreduced_fg(
    const int8_t *f,
    const int8_t *g,
    uint32_t *Ft,
    uint32_t *Gt,
    uint8_t *tmp
);

void PQCLEAN_FALCON1024_CLEAN_debug_poly_big_to_fp(
    fpr *d,
    const uint32_t *f,
    size_t flen,
    size_t fstride,
    unsigned logn
);

int
pqclean_ref_keygen_seeded_1024(
    const uint8_t *seed,
    size_t seed_len,
    uint8_t *pk,
    uint8_t *sk
) {
    union {
        uint8_t b[FALCON_KEYGEN_TEMP_10];
        uint64_t dummy_u64;
        fpr dummy_fpr;
    } tmp;
    inner_shake256_context rng;
    int8_t f[1024], g[1024], F[1024];
    uint16_t h[1024];
    size_t u, v;

    if (seed_len != 48) {
        return -1;
    }

    inner_shake256_init(&rng);
    inner_shake256_inject(&rng, seed, seed_len);
    inner_shake256_flip(&rng);
    PQCLEAN_FALCON1024_CLEAN_keygen(&rng, f, g, F, NULL, h, 10, tmp.b);
    inner_shake256_ctx_release(&rng);

    sk[0] = 0x50 + 10;
    u = 1;
    v = PQCLEAN_FALCON1024_CLEAN_trim_i8_encode(
        sk + u,
        PQCLEAN_FALCON1024_CLEAN_CRYPTO_SECRETKEYBYTES - u,
        f,
        10,
        PQCLEAN_FALCON1024_CLEAN_max_fg_bits[10]
    );
    if (v == 0) {
        return -1;
    }
    u += v;
    v = PQCLEAN_FALCON1024_CLEAN_trim_i8_encode(
        sk + u,
        PQCLEAN_FALCON1024_CLEAN_CRYPTO_SECRETKEYBYTES - u,
        g,
        10,
        PQCLEAN_FALCON1024_CLEAN_max_fg_bits[10]
    );
    if (v == 0) {
        return -1;
    }
    u += v;
    v = PQCLEAN_FALCON1024_CLEAN_trim_i8_encode(
        sk + u,
        PQCLEAN_FALCON1024_CLEAN_CRYPTO_SECRETKEYBYTES - u,
        F,
        10,
        PQCLEAN_FALCON1024_CLEAN_max_FG_bits[10]
    );
    if (v == 0) {
        return -1;
    }
    u += v;
    if (u != PQCLEAN_FALCON1024_CLEAN_CRYPTO_SECRETKEYBYTES) {
        return -1;
    }

    pk[0] = 10;
    v = PQCLEAN_FALCON1024_CLEAN_modq_encode(
        pk + 1,
        PQCLEAN_FALCON1024_CLEAN_CRYPTO_PUBLICKEYBYTES - 1,
        h,
        10
    );
    if (v != PQCLEAN_FALCON1024_CLEAN_CRYPTO_PUBLICKEYBYTES - 1) {
        return -1;
    }

    return 0;
}

size_t
pqclean_ref_trace_attempts_1024(
    const uint8_t *seed,
    size_t seed_len,
    size_t max_attempts,
    pqclean_ref_candidate_debug *out
) {
    union {
        uint8_t b[FALCON_KEYGEN_TEMP_10];
        uint64_t dummy_u64;
        fpr dummy_fpr;
    } tmp;
    inner_shake256_context rng;
    int8_t f[1024], g[1024];
    size_t u;

    if (seed_len != 48) {
        return 0;
    }

    inner_shake256_init(&rng);
    inner_shake256_inject(&rng, seed, seed_len);
    inner_shake256_flip(&rng);
    for (u = 0; u < max_attempts; u ++) {
        PQCLEAN_FALCON1024_CLEAN_debug_sample_fg(&rng, f, g, 10);
        PQCLEAN_FALCON1024_CLEAN_debug_eval_candidate(10, f, g, &out[u], tmp.b);
        if (out[u].solve_ok) {
            inner_shake256_ctx_release(&rng);
            return u + 1;
        }
    }
    inner_shake256_ctx_release(&rng);
    return max_attempts;
}

int
pqclean_ref_keygen_components_1024(
    const uint8_t *seed,
    size_t seed_len,
    int8_t *f,
    int8_t *g,
    int8_t *F
) {
    union {
        uint8_t b[FALCON_KEYGEN_TEMP_10];
        uint64_t dummy_u64;
        fpr dummy_fpr;
    } tmp;
    inner_shake256_context rng;
    uint16_t h[1024];

    if (seed_len != 48) {
        return -1;
    }

    inner_shake256_init(&rng);
    inner_shake256_inject(&rng, seed, seed_len);
    inner_shake256_flip(&rng);
    PQCLEAN_FALCON1024_CLEAN_keygen(&rng, f, g, F, NULL, h, 10, tmp.b);
    inner_shake256_ctx_release(&rng);
    return 0;
}

int
pqclean_ref_depth1_components_1024(
    const int8_t *f,
    const int8_t *g,
    int32_t *F,
    int32_t *G
) {
    union {
        uint8_t b[FALCON_KEYGEN_TEMP_10];
        uint64_t dummy_u64;
        fpr dummy_fpr;
    } tmp;

    return PQCLEAN_FALCON1024_CLEAN_debug_solve_depth1(f, g, F, G, tmp.b);
}

int
pqclean_ref_depth1_prebabai_1024(
    const int8_t *f,
    const int8_t *g,
    uint32_t *Ft,
    uint32_t *Gt,
    uint32_t *ft,
    uint32_t *gt
) {
    union {
        uint8_t b[TIDE_PQCLEAN_DEBUG_TMP_1024_PREBABAI];
        uint64_t dummy_u64;
        fpr dummy_fpr;
    } tmp;

    return PQCLEAN_FALCON1024_CLEAN_debug_solve_depth1_prebabai(
        f, g, Ft, Gt, ft, gt, tmp.b);
}

int
pqclean_ref_depth1_input_1024(
    const int8_t *f,
    const int8_t *g,
    uint32_t *Fd,
    uint32_t *Gd
) {
    union {
        uint8_t b[FALCON_KEYGEN_TEMP_10];
        uint64_t dummy_u64;
        fpr dummy_fpr;
    } tmp;

    return PQCLEAN_FALCON1024_CLEAN_debug_depth1_input(f, g, Fd, Gd, tmp.b);
}

int
pqclean_ref_intermediate_output_1024(
    const int8_t *f,
    const int8_t *g,
    uint32_t target_depth,
    uint32_t *F,
    uint32_t *G
) {
    union {
        uint8_t b[FALCON_KEYGEN_TEMP_10];
        uint64_t dummy_u64;
        fpr dummy_fpr;
    } tmp;

    return PQCLEAN_FALCON1024_CLEAN_debug_intermediate_output(
        f, g, target_depth, F, G, tmp.b);
}

int
pqclean_ref_deepest_resultants_1024(
    const int8_t *f,
    const int8_t *g,
    uint32_t *fp,
    uint32_t *gp
) {
    union {
        uint8_t b[FALCON_KEYGEN_TEMP_10];
        uint64_t dummy_u64;
        fpr dummy_fpr;
    } tmp;

    return PQCLEAN_FALCON1024_CLEAN_debug_deepest_resultants(
        f, g, fp, gp, tmp.b);
}

int
pqclean_ref_deepest_bezout_1024(
    const int8_t *f,
    const int8_t *g,
    uint32_t *Fp,
    uint32_t *Gp
) {
    union {
        uint8_t b[FALCON_KEYGEN_TEMP_10];
        uint64_t dummy_u64;
        fpr dummy_fpr;
    } tmp;

    return PQCLEAN_FALCON1024_CLEAN_debug_deepest_bezout(
        f, g, Fp, Gp, tmp.b);
}

int
pqclean_ref_top_output_1024(
    const int8_t *f,
    const int8_t *g,
    uint32_t *F,
    uint32_t *G
) {
    union {
        uint8_t b[FALCON_KEYGEN_TEMP_10];
        uint64_t dummy_u64;
        fpr dummy_fpr;
    } tmp;

    return PQCLEAN_FALCON1024_CLEAN_debug_top_output(
        f, g, F, G, tmp.b);
}

int
pqclean_ref_depth1_fg_1024(
    const int8_t *f,
    const int8_t *g,
    uint32_t *ft,
    uint32_t *gt
) {
    union {
        uint8_t b[FALCON_KEYGEN_TEMP_10];
        uint64_t dummy_u64;
        fpr dummy_fpr;
    } tmp;

    return PQCLEAN_FALCON1024_CLEAN_debug_depth1_fg(
        f, g, ft, gt, tmp.b);
}

int
pqclean_ref_depth1_unreduced_fg_1024(
    const int8_t *f,
    const int8_t *g,
    uint32_t *Ft,
    uint32_t *Gt
) {
    union {
        uint8_t b[FALCON_KEYGEN_TEMP_10];
        uint64_t dummy_u64;
        fpr dummy_fpr;
    } tmp;

    return PQCLEAN_FALCON1024_CLEAN_debug_depth1_unreduced_fg(
        f, g, Ft, Gt, tmp.b);
}

int
pqclean_ref_poly_big_to_fp_1024(
    const uint32_t *f,
    size_t flen,
    size_t fstride,
    uint32_t logn,
    uint64_t *d
) {
    PQCLEAN_FALCON1024_CLEAN_debug_poly_big_to_fp((fpr *)d, f, flen, fstride, logn);
    return 0;
}

void
pqclean_ref_fft_1024(
    uint32_t logn,
    uint64_t *f
) {
    PQCLEAN_FALCON1024_CLEAN_FFT((fpr *)f, logn);
}

void
pqclean_ref_ifft_1024(
    uint32_t logn,
    uint64_t *f
) {
    PQCLEAN_FALCON1024_CLEAN_iFFT((fpr *)f, logn);
}

void
pqclean_ref_poly_add_muladj_fft_1024(
    uint32_t logn,
    uint64_t *d,
    const uint64_t *F,
    const uint64_t *G,
    const uint64_t *f,
    const uint64_t *g
) {
    PQCLEAN_FALCON1024_CLEAN_poly_add_muladj_fft(
        (fpr *)d, (const fpr *)F, (const fpr *)G, (const fpr *)f, (const fpr *)g, logn);
}

void
pqclean_ref_poly_invnorm2_fft_1024(
    uint32_t logn,
    uint64_t *d,
    const uint64_t *f,
    const uint64_t *g
) {
    PQCLEAN_FALCON1024_CLEAN_poly_invnorm2_fft(
        (fpr *)d, (const fpr *)f, (const fpr *)g, logn);
}

void
pqclean_ref_poly_mul_autoadj_fft_1024(
    uint32_t logn,
    uint64_t *a,
    const uint64_t *b
) {
    PQCLEAN_FALCON1024_CLEAN_poly_mul_autoadj_fft(
        (fpr *)a, (const fpr *)b, logn);
}

void
pqclean_ref_poly_mul_fft_1024(
    uint32_t logn,
    uint64_t *a,
    const uint64_t *b
) {
    PQCLEAN_FALCON1024_CLEAN_poly_mul_fft((fpr *)a, (const fpr *)b, logn);
}
