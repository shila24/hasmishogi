/* SPDX-License-Identifier: MIT */
#ifndef SHOGI_SEARCH_C_API_H
#define SHOGI_SEARCH_C_API_H
#include <stdint.h>
#include <stddef.h>
#if defined(_WIN32)
# if defined(SHOGI_SEARCH_BUILD)
#  define SS_API __declspec(dllexport)
# else
#  define SS_API __declspec(dllimport)
# endif
#else
# define SS_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
#define SS_ABI_VERSION 1
#define SS_ALL_FEATURES 2047
/* All calls except stop require exclusive access to their handle. Independent handles
 * share no writable state. stop is atomic; destroy must wait for go to return.
 * Status -1 / pointer NULL means error; last_error is thread-local. Exceptions never escape.
 * Buffers: NULL + capacity 0 queries element count; all other capacities must suffice.
 * set_position replays the complete history and publishes only after successful validation.
 * go leaves the root unchanged; a budget interruption returns the last complete iteration
 * (or the first legal move, depth 0). A stopped handle stays stopped until clear_stop.
 */
typedef struct ss_handle ss_handle;
typedef struct ss_options { uint32_t features; int32_t hash_mb, qdepth; } ss_options;
typedef struct ss_limits {
    uint64_t nodes;
    int32_t depth, time_ms, soft_time_ms, multipv;
} ss_limits;
typedef struct ss_result {
    uint64_t nodes;
    double elapsed_ms;
    int32_t move, score, depth, completed, stopped, root_count, mate_plies;
} ss_result;
typedef struct ss_root_score { int32_t move, score, depth; } ss_root_score;
typedef struct ss_state { uint64_t key; int32_t turn, ply, winner, reason; } ss_state;
SS_API int32_t ss_abi_version(void);
SS_API size_t ss_sizeof_options(void);
SS_API size_t ss_sizeof_limits(void);
SS_API size_t ss_sizeof_result(void);
SS_API size_t ss_sizeof_root_score(void);
SS_API size_t ss_sizeof_state(void);
SS_API const char* ss_last_error(void);
SS_API ss_handle* ss_create(const ss_options* options);
SS_API void ss_destroy(ss_handle* handle);
SS_API int32_t ss_set_position(ss_handle*, const char* initial_sfen, int32_t max_moves,
                               int32_t declaration, const int32_t* moves, int32_t count, int32_t resigned);
SS_API int32_t ss_get_state(const ss_handle*, ss_state* out);
SS_API int32_t ss_legal_moves(ss_handle*, int32_t* out, int32_t capacity);
SS_API int64_t ss_perft(ss_handle*, int32_t depth);
SS_API int32_t ss_evaluate(ss_handle*, int32_t* out);
/* Additive NNUE ABI. Loading is transactional; NULL/empty unloads. No learning dependencies.
 * evaluator: "bootstrap-v1" or architecture id. Borrowed string valid until load/destroy.
 * features perspective is absolute player 0/1; output is ascending int32 indices.
 * float evaluation is a fresh float32 reference; integer search scores round and clamp
 * to +/-900000 cp, outside the reserved mate interval. */
SS_API int32_t ss_load_nnue(ss_handle*, const char* path);
SS_API const char* ss_evaluator(const ss_handle*);
SS_API int32_t ss_evaluate_float(ss_handle*, float* out);
SS_API int32_t ss_nnue_features(ss_handle*, int32_t perspective, int32_t* out, int32_t capacity);
SS_API int32_t ss_save_nnue(const ss_handle*, const char* path);
SS_API int32_t ss_data_roundtrip(const char* input, const char* output);
/* Decode canonical hlnn1 records into [count,2,39] indices in mover order.
 * Additive ABI, no handle. Capacity is in int32 elements; -1 pads absent slots.
 * On failure output may be partial and must be discarded. */
SS_API int32_t ss_nnue_encode_records(const uint8_t* records, size_t bytes,
                                      int32_t* out, size_t capacity);
SS_API int32_t ss_see(ss_handle*, int32_t move, int32_t* out);
SS_API int32_t ss_go(ss_handle*, const ss_limits*, const int32_t* root_moves, int32_t count, ss_result* out);
/* Additive ABI v1 entry point. The callback runs on the go thread after each
 * completed iteration. Result and PV pointers are valid only during the call;
 * mate_plies is zero until the final proof. Return nonzero to request stop.
 * Do not throw across the callback or reenter the handle except for ss_stop.
 * NULL callback is equivalent to ss_go. user_data is passed through unchanged.
 */
typedef int32_t (*ss_progress)(const ss_result*, const int32_t* pv, int32_t count, void* user_data);
SS_API int32_t ss_go_with_progress(ss_handle*, const ss_limits*, const int32_t* root_moves,
                                  int32_t count, ss_result* out, ss_progress, void* user_data);
SS_API int32_t ss_root_scores(const ss_handle*, ss_root_score* out, int32_t capacity);
/* index -1 selects best PV, nonnegative selects a root_scores row. */
SS_API int32_t ss_pv(const ss_handle*, int32_t index, int32_t* out, int32_t capacity);
SS_API int32_t ss_stop(ss_handle*);
SS_API int32_t ss_clear_stop(ss_handle*);
#ifdef __cplusplus
}
#endif
#endif
