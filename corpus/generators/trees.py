import random
from corpus.generators.base import Generator


def _random_subtree(rng: random.Random, depth: int, max_depth: int) -> list[tuple[str, list]]:
    """Returns nested (name, children) tuples. Empty children = leaf file."""
    if depth >= max_depth:
        return []
    n = rng.randint(1, 6)
    nodes = []
    for _ in range(n):
        is_dir = depth < max_depth - 1 and rng.random() < 0.4
        name = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz_-") for _ in range(rng.randint(3, 12)))
        if not is_dir:
            name += rng.choice([".py", ".js", ".rs", ".md", ".json", ".txt"])
            children = []
        else:
            children = _random_subtree(rng, depth + 1, max_depth)
        nodes.append((name, children))
    return nodes


def _render_tree(nodes: list[tuple[str, list]], prefix: str = "") -> list[str]:
    lines = []
    for i, (name, children) in enumerate(nodes):
        is_last = i == len(nodes) - 1
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + name)
        if children:
            extension = "    " if is_last else "│   "
            lines.extend(_render_tree(children, prefix + extension))
    return lines


class TreeGenerator(Generator):
    archetype = "tree"

    def generate(self, rng: random.Random) -> str:
        root_name = rng.choice(["src", "project", "app", "lib", "."])
        depth = rng.randint(2, 4)
        children = _random_subtree(rng, 0, depth)
        lines = [root_name] + _render_tree(children)
        n_files = sum(1 for line in lines if "." in line.split("── ")[-1])
        n_dirs = sum(1 for line in lines if "── " in line and "." not in line.split("── ")[-1])
        lines.append("")
        lines.append(f"{n_dirs} directories, {n_files} files")
        return "\n".join(lines) + "\n"
