#ifndef HASAMI_C_API_H
#define HASAMI_C_API_H
#include <stdint.h>
#if defined(_WIN32)
#  if defined(HASAMI_BUILD)
#    define HS_API __declspec(dllexport)
#  else
#    define HS_API __declspec(dllimport)
#  endif
#else
#  define HS_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
#define HS_ABI_VERSION 1
/* All functions use the cdecl convention. Handles are opaque and not thread-safe. */
typedef struct {
    uint64_t black_lo, black_hi, white_lo, white_hi;
    int32_t turn, pending;
} hs_position;
typedef struct {
    int32_t edge_capture, capture_target, lead_target, repetition_limit, no_moves_draw;
} hs_rules;
typedef struct {
    hs_position position;
    int32_t winner, reason, ply;
    uint64_t captured_lo, captured_hi;
} hs_state;
typedef struct {
    int32_t action, score, depth, completed;
    uint64_t nodes;
    double elapsed_ms;
} hs_search_result;
HS_API int32_t hs_abi_version(void);
HS_API int32_t hs_sizeof_position(void);
HS_API int32_t hs_sizeof_rules(void);
HS_API int32_t hs_sizeof_state(void);
HS_API int32_t hs_sizeof_search_result(void);
HS_API const char* hs_last_error(void);
/* A null rules/position pointer selects the default rules/initial position. */
HS_API void* hs_create(const hs_rules* rules, const hs_position* position);
HS_API void hs_destroy(void* handle);
HS_API void* hs_clone(const void* handle);
HS_API int32_t hs_get_state(const void* handle, hs_state* out);
/* Query with out=NULL and capacity=0 to obtain the required length. */
HS_API int32_t hs_legal_actions(const void* handle, int32_t* out, int32_t capacity);
HS_API int32_t hs_history(const void* handle, hs_position* out, int32_t capacity);
HS_API int32_t hs_play(void* handle, int32_t action);
/* Return 1 when undone, 0 when no move exists, and -1 on error. */
HS_API int32_t hs_undo(void* handle);
HS_API int32_t hs_search(const void* handle, int32_t depth, int32_t time_ms,
                         uint64_t node_limit, hs_search_result* out);
HS_API int32_t hs_perft(const void* handle, int32_t depth, uint64_t* out);
#ifdef __cplusplus
}
#endif
#endif
