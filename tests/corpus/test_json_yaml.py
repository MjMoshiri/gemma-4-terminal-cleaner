import json
import random
from corpus.generators.json_yaml import JsonYamlGenerator


def test_json_archetype():
    assert JsonYamlGenerator().archetype == "json"


def test_json_deterministic():
    g = JsonYamlGenerator()
    assert g.generate(random.Random(0)) == g.generate(random.Random(0))


def test_json_no_ansi():
    assert "\x1b[" not in JsonYamlGenerator().generate(random.Random(0))


def test_json_parses():
    out = JsonYamlGenerator().generate(random.Random(5))
    parsed = json.loads(out)
    assert parsed is not None


def test_json_is_object_or_array():
    out = JsonYamlGenerator().generate(random.Random(5))
    parsed = json.loads(out)
    assert isinstance(parsed, (dict, list))
