#ifndef SHOGI_C_API_H
#define SHOGI_C_API_H
/*
 * C ABI of the native shogi engine (shogi-std-v1). Contract owned by the orchestrator;
 * see shogi/native.py for the Python binding. The Python package shogi/ is the reference implementation:
 * every function here must return exactly what the corresponding Python code returns
 * (same values, same order, same accept/reject decisions, same error message text).
 *
 * Conventions
 * - cdecl, opaque handles, not thread-safe per handle. Distinct handles may be used from
 *   distinct threads concurrently.
 * - No C++ exception crosses this boundary. Status-returning functions return >= 0 on
 *   success and -1 on error; sg_last_error() then returns a thread-local, NUL-terminated
 *   message that stays valid until the next call on the same thread. A failed call leaves
 *   every handle unchanged (strong guarantee).
 * - Squares 0..80 (row * 9 + col, row 0 = rank a, col 0 = file 9). Board cells: 0 empty,
 *   1..14 sente piece types, 17..30 gote piece types (type + 16). Hands: 7 counts per player
 *   in the order P L N S G B R, sente first. Move codes and policy labels as in the spec.
 * - Booleans are int32 0/1.
 * - Variable-length outputs: call with out=NULL and capacity=0 to obtain the required
 *   length. Element outputs return the element count. String outputs return the byte length
 *   excluding the terminating NUL and write string + NUL only when capacity > length; a too
 *   small, non-zero capacity is an error.
 */
#include <stdint.h>
#include <stddef.h>
#if defined(_WIN32)
#  if defined(SHOGI_BUILD)
#    define SG_API __declspec(dllexport)
#  else
#    define SG_API __declspec(dllimport)
#  endif
#else
#  define SG_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

#define SG_ABI_VERSION 1
#define SG_SQUARES 81
#define SG_HAND_TYPES 7
#define SG_MOVE_SPACE 13690
#define SG_DECLARE 13689
#define SG_ACTION_SIZE 2188
#define SG_DECLARE_INDEX 2187
#define SG_NUM_PLANES 49
#define SG_PLANES_SIZE (SG_NUM_PLANES * SG_SQUARES) /* 3969 float32 */
#define SG_MAX_LEGAL 600 /* upper bound on legal moves of any position, incl. non-promotions */

/* Outcome reasons, in the order of the spec's reason list. */
enum { SG_END_ONGOING = 0, SG_END_CHECKMATE = 1, SG_END_NO_LEGAL_MOVES = 2, SG_END_DECLARATION = 3,
       SG_END_PERPETUAL_CHECK = 4, SG_END_REPETITION = 5, SG_END_MAX_MOVES = 6, SG_END_RESIGN = 7 };
/* Declaration rule. */
enum { SG_DECLARATION_NONE = 0, SG_DECLARATION_CSA27 = 1 };

typedef struct sg_rules {
    int32_t max_moves;   /* >= 0; 0 disables */
    int32_t declaration; /* SG_DECLARATION_* */
} sg_rules;

typedef struct sg_state {
    int32_t turn;        /* player to move: 0 sente, 1 gote */
    int32_t ply;         /* moves played (len(moves)) */
    int32_t move_number; /* SFEN move number of the current position */
    int32_t winner;      /* -2 ongoing, -1 draw, 0/1 winner */
    int32_t reason;      /* SG_END_* */
    int32_t in_check;    /* player to move is in check */
    int32_t can_declare; /* DECLARE is legal now */
} sg_state;

typedef struct sg_game sg_game;

SG_API int32_t sg_abi_version(void);
SG_API size_t sg_sizeof_rules(void);
SG_API size_t sg_sizeof_state(void);
SG_API const char *sg_last_error(void);
SG_API int32_t sg_default_rules(sg_rules *out);

/* Lifecycle. sfen == NULL means the start position. Validation as in spec section 5. */
SG_API sg_game *sg_create(const sg_rules *rules, const char *sfen);
SG_API sg_game *sg_clone(const sg_game *game);
SG_API void sg_destroy(sg_game *game); /* NULL is a no-op */

/* State. */
SG_API int32_t sg_get_state(const sg_game *game, sg_state *out);
SG_API int32_t sg_board(const sg_game *game, int32_t *out, int32_t capacity);  /* 81 cells */
SG_API int32_t sg_hands(const sg_game *game, int32_t *out, int32_t capacity);  /* 14 counts */
SG_API int32_t sg_moves(const sg_game *game, int32_t *out, int32_t capacity);  /* codes played */
SG_API int32_t sg_sfen(const sg_game *game, char *out, int32_t capacity);
/* History key i (0 = initial position, ply = current): SFEN without the move number. */
SG_API int32_t sg_history_key(const sg_game *game, int32_t index, char *out, int32_t capacity);

/* Moves. Legal codes ascending; labels in the same order. */
SG_API int32_t sg_legal_moves(const sg_game *game, int32_t *out, int32_t capacity);
SG_API int32_t sg_legal_labels(const sg_game *game, int32_t *out, int32_t capacity);
SG_API int32_t sg_is_legal(const sg_game *game, int32_t code);              /* 0/1, never -1 for a valid handle */
SG_API int32_t sg_move_label(const sg_game *game, int32_t code);            /* -1 when the code is not legal */
SG_API int32_t sg_label_to_move(const sg_game *game, int32_t label);        /* -1 when no legal move has the label */
/* Play a move; on success writes the captured piece type (0 none) and whether it gives check. */
SG_API int32_t sg_play(sg_game *game, int32_t code, int32_t *captured, int32_t *check);
SG_API int32_t sg_undo(sg_game *game);                                      /* 1 undone, 0 nothing to undo */
SG_API int32_t sg_resign(sg_game *game);

/* Learning fast paths (shogi-enc-v1): player must be the side to move. */
SG_API int32_t sg_encode(const sg_game *game, int32_t player, float *out, int32_t capacity);      /* SG_PLANES_SIZE */
SG_API int32_t sg_legal_mask(const sg_game *game, int32_t player, float *out, int32_t capacity); /* SG_ACTION_SIZE */

/* Verification. */
SG_API int64_t sg_perft(const sg_game *game, int32_t depth); /* -1 on error */

#ifdef __cplusplus
}
#endif
#endif
