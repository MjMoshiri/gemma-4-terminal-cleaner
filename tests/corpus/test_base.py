import random
import pytest
from corpus.generators.base import Generator


def test_generator_is_abstract():
    with pytest.raises(TypeError):
        Generator()  # cannot instantiate ABC


def test_generator_subclass_must_implement_generate():
    class Incomplete(Generator):
        archetype = "incomplete"
    with pytest.raises(TypeError):
        Incomplete()


def test_generator_subclass_works():
    class Concrete(Generator):
        archetype = "concrete"
        def generate(self, rng: random.Random) -> str:
            return "hello"
    g = Concrete()
    assert g.generate(random.Random(0)) == "hello"
    assert g.archetype == "concrete"
