from dirtifier.recipes import RECIPES, Recipe


def test_all_five_recipes_defined():
    expected = {"cli_colored_table", "install_with_progress", "tui_redraw", "noisy_logs", "passthrough"}
    assert expected <= set(RECIPES.keys())


def test_passthrough_is_empty():
    r = RECIPES["passthrough"]
    assert isinstance(r, Recipe)
    assert r.steps == []
