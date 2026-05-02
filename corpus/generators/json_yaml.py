import json
import random
from corpus.generators.base import Generator


def _random_json_value(rng: random.Random, depth: int):
    if depth >= 4:
        return rng.choice([rng.randint(0, 1000), True, False, None, _random_string(rng)])
    kind = rng.choice(["dict", "list", "str", "int", "bool"])
    if kind == "dict":
        n = rng.randint(1, 6)
        return {_random_string(rng): _random_json_value(rng, depth + 1) for _ in range(n)}
    if kind == "list":
        n = rng.randint(1, 5)
        return [_random_json_value(rng, depth + 1) for _ in range(n)]
    if kind == "str":
        return _random_string(rng)
    if kind == "int":
        return rng.randint(-1000, 10000)
    return rng.choice([True, False])


def _random_string(rng: random.Random) -> str:
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz_") for _ in range(rng.randint(3, 10)))


class JsonYamlGenerator(Generator):
    archetype = "json"

    def generate(self, rng: random.Random) -> str:
        # JSON only for v1 — YAML adds parsing complexity we don't need
        obj = _random_json_value(rng, 0)
        if not isinstance(obj, (dict, list)):
            obj = {"value": obj}
        return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
