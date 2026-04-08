#include "pqclean_ref.h"
#include "falcon-1024/api.h"
#include "falcon-1024/inner.h"

/*
 * Debug depth-1 pre-Babai trace path needs more scratch than
 * FALCON_KEYGEN_TEMP_10 because it reuses the same tmp region with
 * overlapping staged buffers before calling make_fg().
 */
#define TIDE_PQCLEAN_DEBUG_TMP_1024_PREBABAI 32768

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
