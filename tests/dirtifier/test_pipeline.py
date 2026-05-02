import random
from dirtifier.pipeline import apply_recipe, pick_recipe
from dirtifier.recipes import RECIPES


def test_apply_recipe_deterministic():
    clean = "hello world\nfoo bar baz\n"
    a = apply_recipe(clean, RECIPES["cli_colored_table"], random.Random(0))
    b = apply_recipe(clean, RECIPES["cli_colored_table"], random.Random(0))
    assert a == b


def test_apply_passthrough_returns_unchanged():
    clean = "hello world\nfoo bar\n"
    out = apply_recipe(clean, RECIPES["passthrough"], random.Random(0))
    assert out == clean


def test_apply_install_with_progress_adds_cr():
    clean = "installed package foo-1.2.3\n"
    out = apply_recipe(clean, RECIPES["install_with_progress"], random.Random(1))
    # Should contain at least one \r (progress / spinner)
    assert "\r" in out


def test_pick_recipe_deterministic():
    rng_a = random.Random(0)
    rng_b = random.Random(0)
    assert pick_recipe(rng_a).name == pick_recipe(rng_b).name
