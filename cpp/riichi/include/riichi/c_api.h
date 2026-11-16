#ifndef RIICHI_C_API_H
#define RIICHI_C_API_H
/*
 * C ABI of the native riichi engine (riichi-4p-v1). Contract owned by the orchestrator;
 * see riichi/native.py for the Python binding. The Python package riichi/ is the reference
 * implementation: every function here must return exactly what the corresponding Python
 * code returns (same values, same order, same accept/reject decisions).
 *
 * Conventions
 * - cdecl, opaque handles, not thread-safe per handle. Distinct handles may be used from
 *   distinct threads concurrently.
 * - No C++ exception crosses this boundary. Status-returning functions return >= 0 on
 *   success and -1 on error (-2 where documented); rq_last_error() then returns a
 *   thread-local, NUL-terminated message that stays valid until the next call on the
 *   same thread. A failed call leaves every handle unchanged (strong guarantee).
 * - Tile ids are physical ids 0..135 (type = id / 4; red fives are 16, 52, 88). Tile
 *   types are 0..33. Seats are absolute 0..3 unless a JSON document says otherwise.
 * - Booleans are int32 0/1. Inputs other than 0/1 for a boolean field are rejected.
 * - Variable-length outputs: call with out=NULL and capacity=0 to obtain the required
 *   length. Element outputs return the element count. String outputs return the byte
 *   length excluding the terminating NUL and write string + NUL only when
 *   capacity > length; a too small, non-zero capacity is an error.
 * - JSON outputs are compact UTF-8 (ASCII only) objects that json.loads parses into
 *   exactly the dict the Python engine returns, except that Python tuples appear as
 *   JSON arrays (the Python bridge restores the documented tuples).
 */
#include <stdint.h>
#if defined(_WIN32)
#  if defined(RIICHI_BUILD)
#    define RQ_API __declspec(dllexport)
#  else
#    define RQ_API __declspec(dllimport)
#  endif
#else
#  define RQ_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

#define RQ_ABI_VERSION 1
#define RQ_TILE_IDS 136
#define RQ_TILE_TYPES 34
#define RQ_ACTION_SIZE 111
#define RQ_NUM_PLANES 120
#define RQ_ROWS 4
#define RQ_PLANES_SIZE (RQ_NUM_PLANES * RQ_ROWS * RQ_TILE_TYPES) /* 16320 float32 */
#define RQ_LIVE_WALL 122
#define RQ_DEAD_WALL 14
#define RQ_MAX_HAND 14
#define RQ_NO_ACTION (-1)
#define RQ_NO_TILE (-1)

/* Phases, in the order of riichi.game.RiichiGame.phase strings. */
enum { RQ_PHASE_DRAW = 0, RQ_PHASE_DISCARD = 1, RQ_PHASE_CLAIM = 2, RQ_PHASE_KAN_CLAIM = 3,
       RQ_PHASE_TERMINAL = 4 };
/* Match outcome reasons (riichi.game.Outcome.reason). */
enum { RQ_END_ONGOING = 0, RQ_END_TOBI = 1, RQ_END_AGARI_YAME = 2, RQ_END_ROUNDS_COMPLETE = 3 };
/* Meld kinds (riichi.agari.Meld.kind). */
enum { RQ_MELD_CHI = 0, RQ_MELD_PON = 1, RQ_MELD_KAN = 2 };
/* RiichiRules.rounds and RiichiRules.renhou. */
enum { RQ_ROUNDS_HANCHAN = 0, RQ_ROUNDS_TONPUU = 1 };
enum { RQ_RENHOU_NONE = 0, RQ_RENHOU_MANGAN = 1, RQ_RENHOU_YAKUMAN = 2 };

/*
 * riichi.rules.RiichiRules without ruleset_id (always riichi-4p-v1), fields in declaration
 * order. Validation must accept and reject exactly what RiichiRules.__post_init__ does,
 * plus RiichiGame's own restriction (pao=0 and atozuke=1) in rq_create.
 */
typedef struct {
    int32_t rounds;            /* RQ_ROUNDS_* */
    int32_t start_points, return_points;
    int32_t uma[4];
    int32_t red_fives, kuitan, atozuke, ippatsu, ura_dora, kan_dora;
    int32_t open_kan_dora_after_discard, closed_kan_dora_immediate;
    int32_t double_ron, atama_hane, triple_ron_abort, kyuushu_kyuuhai, suufon_renda;
    int32_t suucha_riichi, suukaikan;
    int32_t tenpai_payment, riichi_stick, honba;
    int32_t tobi, agari_yame, kazoe_yakuman, double_yakuman, multiple_yakuman, pao;
    int32_t renhou;            /* RQ_RENHOU_* */
    int32_t kiriage_mangan, double_wind_pair_fu, nagashi_mangan;
    int32_t discard_furiten, temporary_furiten, riichi_furiten, track_tsumogiri;
} rq_rules;

/* riichi.agari.Meld: tile TYPES; tiles[3] is ignored unless kind == RQ_MELD_KAN. */
typedef struct {
    int32_t kind;
    int32_t open;
    int32_t tiles[4];
} rq_meld;

/* riichi.yaku.WinContext; indicators are tile types. */
typedef struct {
    int32_t tsumo, seat_wind, round_wind;
    int32_t riichi, double_riichi, ippatsu, rinshan, chankan, haitei, houtei;
    int32_t tenhou, chiihou, renhou;
    int32_t n_dora, dora_indicators[5];
    int32_t n_ura, ura_indicators[5];
    int32_t red_dora;
    int32_t yaku_established;  /* -1 = None, 0 = False, 1 = True */
} rq_win_context;

/* Per-game scalar state. */
typedef struct {
    int32_t terminal, winner, reason;      /* riichi.game.Outcome; winner -2 while ongoing */
    int32_t phase, turn, dealer, round_index, honba, riichi_sticks;
    int32_t hand_number, hand_start, num_moves, remaining_wall;
    int32_t scores[4];
    double final_points[4];                /* placement_points(); zeros until terminal */
    uint64_t seed;
} rq_state;

/* Replacement hidden state for rq_redeal (riichi.game.RiichiGame.redeal). */
typedef struct {
    int32_t hand_sizes[4];
    int32_t hands[4][RQ_MAX_HAND];
    int32_t live_count;
    int32_t live_wall[RQ_LIVE_WALL];
    int32_t dead_wall[RQ_DEAD_WALL];
    int32_t drawn_tiles[4];                /* RQ_NO_TILE for None */
    int32_t temporary_furiten[4], riichi_furiten[4];
    uint64_t seed;
} rq_hidden;

/* ---- ABI identity --------------------------------------------------------------- */
RQ_API int32_t rq_abi_version(void);
RQ_API int32_t rq_sizeof_rules(void);
RQ_API int32_t rq_sizeof_meld(void);
RQ_API int32_t rq_sizeof_win_context(void);
RQ_API int32_t rq_sizeof_state(void);
RQ_API int32_t rq_sizeof_hidden(void);
RQ_API const char* rq_last_error(void);

/* ---- Core ----------------------------------------------------------------------- */
/* Fill *out with RiichiRules() defaults (nagashi_mangan = 1). */
RQ_API int32_t rq_default_rules(rq_rules* out);
/* random.Random(seed): `count` successive riichi.tiles.build_wall(rng=...) shuffles of one
 * generator, 136 ids each (live = [0, 122), dead = [122, 136)). */
RQ_API int32_t rq_walls(uint64_t seed, int32_t count, int32_t* out, int32_t capacity);
/* shanten.shanten / standard_shanten / chiitoitsu_shanten / kokushi_shanten for 34
 * concealed type counts plus declared melds; out = {overall, standard, chiitoitsu, kokushi}. */
RQ_API int32_t rq_shanten(const int32_t counts[RQ_TILE_TYPES], const rq_meld* melds,
                          int32_t n_melds, int32_t out[4]);
/* shanten.waits: out[t] = 1 when type t is a wait; returns the number of waits. */
RQ_API int32_t rq_waits(const int32_t counts[RQ_TILE_TYPES], const rq_meld* melds,
                        int32_t n_melds, int32_t out[RQ_TILE_TYPES]);
/* agari.is_agari: 1 or 0. */
RQ_API int32_t rq_is_agari(const int32_t counts[RQ_TILE_TYPES], const rq_meld* melds,
                           int32_t n_melds);
/* scoring.score_hand. counts include the winning tile; tile_ids may be NULL (None).
 * Writes {"han","fu","yakuman","base_points","payment":{"ron","tsumo_dealer",
 * "tsumo_non_dealer","riichi_bonus","total"},"yaku":[[name,han,yakuman],...]} with yaku in
 * Python's order. Returns -2 for NoYakuError, -1 for any other ValueError. */
RQ_API int32_t rq_score_hand_json(const int32_t counts[RQ_TILE_TYPES], int32_t winning_tile,
                                  const rq_meld* melds, int32_t n_melds,
                                  const rq_win_context* context, const rq_rules* rules,
                                  int32_t honba, int32_t riichi_sticks,
                                  const int32_t* tile_ids, int32_t n_tile_ids,
                                  char* out, int32_t capacity);
/* scoring.placement_points with the default tie order. */
RQ_API int32_t rq_placement_points(const int32_t scores[4], const rq_rules* rules,
                                   double out[4]);

/* ---- Game ----------------------------------------------------------------------- */
/* RiichiGame(seed=seed, rules=rules); rules == NULL selects RiichiGame's default,
 * RiichiRules(nagashi_mangan=False). Returns NULL on error. */
RQ_API void* rq_create(uint64_t seed, const rq_rules* rules);
RQ_API void rq_destroy(void* handle);
/* Deep copy (copy.deepcopy semantics). NULL on error. */
RQ_API void* rq_clone(const void* handle);
RQ_API int32_t rq_get_state(const void* handle, rq_state* out);
RQ_API int32_t rq_get_rules(const void* handle, rq_rules* out);
/* RiichiGame.actors() in Python's order, which is NOT sorted: the seat to move, or in claim and
 * kan-claim phases the three other seats in turn order after the discarder/declarer (e.g. {2, 3, 0}).
 * Returns the count (0..3). */
RQ_API int32_t rq_actors(const void* handle, int32_t out[4]);
/* RiichiGame.legal_actions(player): same ids, same order; non-actors yield 0 elements. */
RQ_API int32_t rq_legal_actions(const void* handle, int32_t player, int32_t* out,
                                int32_t capacity);
/* RiichiGame.play_joint({p: actions[p] for p with actions[p] != RQ_NO_ACTION}). */
RQ_API int32_t rq_play_joint(void* handle, const int32_t actions[4]);
/* riichi-enc-v1 planes, row-major (plane, row, tile type); bitwise equal to
 * numpy.array(riichi.encoding.encode(game, player), dtype=float32). size == RQ_PLANES_SIZE. */
RQ_API int32_t rq_encode(const void* handle, int32_t player, float* out, int32_t size);
/* 1.0 at legal action ids, else 0.0; size == RQ_ACTION_SIZE. */
RQ_API int32_t rq_legal_mask(const void* handle, int32_t player, float* out, int32_t size);
/* RiichiGame.moves as 4 int32 per joint step (RQ_NO_ACTION for seats without an action). */
RQ_API int32_t rq_moves(const void* handle, int32_t* out, int32_t capacity);
/* RiichiGame.observation(player) as JSON. */
RQ_API int32_t rq_observation_json(const void* handle, int32_t player, char* out,
                                   int32_t capacity);
/* RiichiGame.last_hand as JSON ("{}" before the first hand ends). */
RQ_API int32_t rq_last_hand_json(const void* handle, char* out, int32_t capacity);
/* RiichiGame.redeal(...): replace hidden state for determinised search. */
RQ_API int32_t rq_redeal(void* handle, const rq_hidden* hidden);

#ifdef __cplusplus
}
#endif
#endif
