"""Slow array/set oracle, intentionally independent of bitboard operations."""

def adjacent(square):
    r, c = divmod(square, 9)
    return [rr * 9 + cc for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
            if 0 <= rr < 9 and 0 <= cc < 9]


def array_board(position):
    return [0 if (position.black >> s) & 1 else 1 if (position.white >> s) & 1 else -1
            for s in range(81)]


def legal(board, turn):
    actions = []
    for source, value in enumerate(board):
        if value != turn:
            continue
        row, col = divmod(source, 9)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = row + dr, col + dc
            while 0 <= r < 9 and 0 <= c < 9 and board[r * 9 + c] == -1:
                actions.append(source * 81 + r * 9 + c)
                r, c = r + dr, c + dc
    return sorted(actions)


def apply(board, turn, action, edge=True):
    board = list(board)
    source, target = divmod(action, 81)
    if action not in legal(board, turn):
        raise ValueError("Illegal oracle move")
    board[source], board[target] = -1, turn
    taken = set()
    row, col = divmod(target, 9)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r, c, run = row + dr, col + dc, []
        while 0 <= r < 9 and 0 <= c < 9 and board[r * 9 + c] == 1 - turn:
            run.append(r * 9 + c)
            r, c = r + dr, c + dc
        if 0 <= r < 9 and 0 <= c < 9 and board[r * 9 + c] == turn:
            taken.update(run)
    if edge:
        unseen = {s for s, piece in enumerate(board) if piece == 1 - turn}
        while unseen:
            start = min(unseen)
            stack = [start]
            group = {start}
            while stack:
                s = stack.pop()
                for n in adjacent(s):
                    if board[n] == 1 - turn and n not in group:
                        group.add(n)
                        stack.append(n)
            unseen.difference_update(group)
            boundary = {n for s in group for n in adjacent(s)} - group
            at_edge = any(s // 9 in (0, 8) or s % 9 in (0, 8) for s in group)
            if at_edge and target in boundary and all(board[n] == turn for n in boundary):
                taken.update(group)
    for s in taken:
        board[s] = -1
    return board, taken
