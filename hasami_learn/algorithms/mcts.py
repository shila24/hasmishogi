"""Compatibility alias: the PUCT search moved to ``hasami_learn.search.mcts``.

The alias makes ``hasami_learn.algorithms.mcts`` the very same module object, so existing imports, private helpers and
patches of its attributes keep working.
"""

import sys

from ..search import mcts as _search_mcts

sys.modules[__name__] = _search_mcts
