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

#endif
