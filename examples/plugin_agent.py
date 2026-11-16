"""A trusted Python agent plugin demonstrating the factory contract."""
from hasami.agents import Decision, check_cancel


class FirstLegalAgent:
    def choose(self, game, limits, cancel=None):
        check_cancel(cancel)
        actions = game.legal_actions()
        if not actions:
            raise ValueError("No action is available")
        return Decision(actions[0])


def create_agent():
    return FirstLegalAgent()
