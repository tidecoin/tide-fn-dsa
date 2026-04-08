#include "pqclean_ref.h"
#include "falcon-512/api.h"
#include "falcon-512/inner.h"

int
pqclean_ref_keygen_seeded_512(
    const uint8_t *seed,
    size_t seed_len,
    uint8_t *pk,
    uint8_t *sk
) {
    union {
        uint8_t b[FALCON_KEYGEN_TEMP_9];
        uint64_t dummy_u64;
        fpr dummy_fpr;
    } tmp;
    inner_shake256_context rng;
    int8_t f[512], g[512], F[512];
    uint16_t h[512];
    size_t u, v;

    if (seed_len != 48) {
        return -1;
    }

    inner_shake256_init(&rng);
    inner_shake256_inject(&rng, seed, seed_len);
    inner_shake256_flip(&rng);
    PQCLEAN_FALCON512_CLEAN_keygen(&rng, f, g, F, NULL, h, 9, tmp.b);
    inner_shake256_ctx_release(&rng);

    sk[0] = 0x50 + 9;
    u = 1;
    v = PQCLEAN_FALCON512_CLEAN_trim_i8_encode(
        sk + u,
        PQCLEAN_FALCON512_CLEAN_CRYPTO_SECRETKEYBYTES - u,
        f,
        9,
        PQCLEAN_FALCON512_CLEAN_max_fg_bits[9]
    );
    if (v == 0) {
        return -1;
    }
    u += v;
    v = PQCLEAN_FALCON512_CLEAN_trim_i8_encode(
        sk + u,
        PQCLEAN_FALCON512_CLEAN_CRYPTO_SECRETKEYBYTES - u,
        g,
        9,
        PQCLEAN_FALCON512_CLEAN_max_fg_bits[9]
    );
    if (v == 0) {
        return -1;
    }
    u += v;
    v = PQCLEAN_FALCON512_CLEAN_trim_i8_encode(
        sk + u,
        PQCLEAN_FALCON512_CLEAN_CRYPTO_SECRETKEYBYTES - u,
        F,
        9,
        PQCLEAN_FALCON512_CLEAN_max_FG_bits[9]
    );
    if (v == 0) {
        return -1;
    }
    u += v;
    if (u != PQCLEAN_FALCON512_CLEAN_CRYPTO_SECRETKEYBYTES) {
        return -1;
    }

    pk[0] = 9;
    v = PQCLEAN_FALCON512_CLEAN_modq_encode(
        pk + 1,
        PQCLEAN_FALCON512_CLEAN_CRYPTO_PUBLICKEYBYTES - 1,
        h,
        9
    );
    if (v != PQCLEAN_FALCON512_CLEAN_CRYPTO_PUBLICKEYBYTES - 1) {
        return -1;
    }

    return 0;
}
