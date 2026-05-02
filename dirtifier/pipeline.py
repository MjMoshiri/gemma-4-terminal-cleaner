import random
from dirtifier.recipes import RECIPES, Recipe


def apply_recipe(clean: str, recipe: Recipe, rng: random.Random) -> str:
    """Apply each step of the recipe in order, with each step's probability."""
    out = clean
    for transform, prob in recipe.steps:
        if rng.random() < prob:
            out = transform.apply(out, rng)
    return out


def pick_recipe(rng: random.Random, recipes: dict[str, Recipe] = RECIPES) -> Recipe:
    """Weighted random selection of a recipe."""
    names = list(recipes.keys())
    weights = [recipes[n].weight for n in names]
    return recipes[rng.choices(names, weights=weights, k=1)[0]]
