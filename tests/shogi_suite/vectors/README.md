# Shogi test vectors

Data only (no code). Generated on 2026-11-13 with two independent open-source shogi move generators
(cshogi 1.1.1 and python-shogi 1.1.1) that were used once, outside this repository, as oracles; neither is a
dependency of this project.

- `perft_vectors.jsonl`: the start position (perft 1-4) and 240 positions reached by seeded random playouts
  (196 with pieces in hand). Each row has the SFEN, perft for depths 1-3 (1-4 for the start position), and the
  sorted USI strings of every legal move. Both oracles produced identical legal-move sets for every row; perft values
  come from cshogi.
- `rule_cases.json`: hand-built positions for pawn-drop mate, two pawns on a file, dead-piece drops and moves,
  forced and optional promotion, pins and king safety (`move_cases`: is the USI move legal, and the number of legal
  moves), and the CSA 27-point entering-king declaration (`declaration_cases`), checked against both oracles
  (declaration against cshogi's `is_nyugyoku`).
- Every position is valid under `shogi/engine.py` section 5: in particular the side not to move is never in
  check (audited with cshogi on 2026-11-13 after three hand-built pawn-drop cases were found to violate it and were
  rebuilt with a knight protecting the drop square).
