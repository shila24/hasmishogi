#ifndef MCTS_C_API_H
#define MCTS_C_API_H
#include <stdint.h>
#include <stddef.h>
#if defined(_WIN32)
# if defined(MCTS_BUILD)
#  define MZ_API __declspec(dllexport)
# else
#  define MZ_API __declspec(dllimport)
# endif
#else
# define MZ_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
#define MZ_ABI_VERSION 3
/* Opaque owned search; calls on one handle must be serialized. Separate handles
 * are independent. No exception crosses this boundary. Status is >=0 or -1;
 * create returns NULL on error. After any failed operation destroy the search.
 * All lengths/capacities count elements, except SFEN/error strings (bytes).
 * Buffers must be correctly aligned and valid for their declared capacities.
 * Inputs and outputs must not overlap. No pointers are retained after a call.
 * Destruction joins all threads. Error text belongs to the calling thread.
 */
MZ_API int32_t mz_abi_version(void);
/* Returns the required length excluding NUL; NULL/0 queries it. */
MZ_API int32_t mz_last_error(char* out, size_t capacity);
/* kind: 0 Hasami, 1 shogi. threads >= 1, c_puct >= 0, truncation_value in [-1,1]. */
MZ_API void* mz_create(int32_t kind, int32_t threads, double c_puct, double truncation_value);
/* Set before adding trees. Default is zero contempt and no selected reasons.
 * value in [-1,1]; reason bits: 1 repetition, 2 max_moves (shogi only).
 * The utility is symmetric for both players; decisive outcomes are unchanged.
 */
MZ_API int32_t mz_draw_policy(void* handle, double value, uint32_t reasons);
MZ_API void mz_destroy(void* handle);
/* Import the COMPLETE replay, never a current board with invented history.
 * Hasami boards: initial black lo/hi, white lo/hi. Settings: initial turn,
 * pending, edge_capture, capture_target, lead_target, repetition_limit, no_moves_draw.
 * Shogi settings: max_moves, declaration (0 none, 1 csa27). Codes are engine moves.
 * The optional resignation occurs after the complete move list.
 * Add returns the new tree index. Adds must precede prepare/observe/results.
 */
MZ_API int32_t mz_add_hasami(void* handle, const uint64_t* boards, size_t boards_capacity,
    const int32_t* settings, size_t settings_capacity, const int32_t* moves, size_t move_count);
MZ_API int32_t mz_add_shogi(void* handle, const char* sfen, size_t sfen_length,
    const int32_t* settings, size_t settings_capacity, const int32_t* moves, size_t move_count,
    int32_t resigned);
/* Prepare roots (descend=0) or one simulation/tree (descend=1). Tree ids must be
 * unique and valid. Outputs have capacity for count rows, row-major float32.
 * Only leaves needing evaluation are emitted, compacted in input order; returns
 * their count. Each output id identifies its tree's single outstanding leaf.
 * Finish every evaluation, then backup every prepared simulation before reuse.
 */
MZ_API int32_t mz_prepare(void* handle, const int32_t* ids, size_t count, int32_t descend,
    float* planes, size_t planes_capacity, float* masks, size_t masks_capacity,
    int32_t* leaf_ids, size_t ids_capacity);
MZ_API int32_t mz_expand(void* handle, const int32_t* ids, size_t count,
    const double* probabilities, size_t probabilities_capacity,
    const double* values, size_t values_capacity);
/* Noise is drawn by the caller in legal-action order, not by native threads. */
MZ_API int32_t mz_noise(void* handle, int32_t tree, const double* noise, size_t count, double epsilon);
/* Enable Gumbel after root expansion, before descent. Noise is one finite sample
 * per legal action (ascending order); pass zeros for noiseless evaluation.
 * simulations is the TOTAL decision budget, m >= 1, constants finite >= 0.
 */
MZ_API int32_t mz_gumbel(void* handle, int32_t tree, const double* noise, size_t count,
    int64_t simulations, int64_t m, double c_visit, double c_scale);
/* Gumbel policy targets, dense float64, and winning action ids (terminal: -1).
 * PUCT rows are zero / -1: the bridge normalizes the original float32 visits.
 */
MZ_API int32_t mz_policy(void* handle, double* policy, size_t capacity,
    int32_t* selected, size_t selected_capacity);
MZ_API int32_t mz_backup(void* handle, const int32_t* ids, size_t count);
/* All trees, including terminal roots, in insertion order. */
MZ_API int32_t mz_observe(void* handle, float* planes, size_t planes_capacity,
    float* masks, size_t masks_capacity);
MZ_API int32_t mz_results(void* handle, float* visits, size_t visits_capacity,
    float* q_values, size_t q_capacity, double* root_values, size_t values_capacity,
    int64_t* simulations, size_t simulations_capacity);
#ifdef __cplusplus
}
#endif
#endif
