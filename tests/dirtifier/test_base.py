import random
import pytest
from dirtifier.transforms.base import Transform


def test_transform_is_abstract():
    with pytest.raises(TypeError):
        Transform()


def test_transform_subclass_works():
    class Concrete(Transform):
        name = "concrete"
        def apply(self, clean, rng):
            return clean + "!"
    t = Concrete()
    assert t.apply("hi", random.Random(0)) == "hi!"
    assert t.name == "concrete"
