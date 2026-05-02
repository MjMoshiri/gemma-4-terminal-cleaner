from dirtifier.generate import _pick_split_recipes


def test_split_excludes_held_out_recipes_from_train():
    train_recipes, val_recipes = _pick_split_recipes(
        all_recipes=["a", "b", "c", "d"],
        held_out_for_val=["c"],
    )
    assert "c" not in train_recipes
    assert "c" in val_recipes
    assert set(train_recipes) == {"a", "b", "d"}
    assert set(val_recipes) == {"a", "b", "c", "d"}
