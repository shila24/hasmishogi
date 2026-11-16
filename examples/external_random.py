"""A complete external agent example; no project imports are required."""
import json
import random
import sys


def main():
    request = json.loads(sys.stdin.readline())
    if request.get("protocol") != "hasami-agent-v1":
        raise ValueError("Unsupported protocol")
    observation = request["observation"]
    rng = random.Random(observation["ply"])
    action = rng.choice(observation["legal_actions"])
    # Send diagnostics to stderr, never to the protocol stream.
    print(json.dumps({"protocol": "hasami-agent-v1", "action": action}), flush=True)


if __name__ == "__main__":
    main()
