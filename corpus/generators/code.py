import random
from corpus.generators.base import Generator


_PY_TEMPLATES = [
    'def {fn}({args}):\n    """{doc}"""\n    return {ret}\n\n',
    'class {cls}:\n    def __init__(self, {args}):\n        self.{attr} = {attr}\n\n',
    'if __name__ == "__main__":\n    main()\n',
]
_JS_TEMPLATES = [
    'function {fn}({args}) {{\n  return {ret};\n}}\n\n',
    'const {var} = ({args}) => {{\n  return {ret};\n}};\n\n',
    'export class {cls} {{\n  constructor({args}) {{\n    this.{attr} = {attr};\n  }}\n}}\n\n',
]


def _name(rng: random.Random) -> str:
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz_") for _ in range(rng.randint(3, 10)))


class CodeGenerator(Generator):
    archetype = "code"

    def generate(self, rng: random.Random) -> str:
        lang = rng.choice(["python", "javascript"])
        templates = _PY_TEMPLATES if lang == "python" else _JS_TEMPLATES
        n_blocks = rng.randint(2, 8)
        out = []
        for _ in range(n_blocks):
            t = rng.choice(templates)
            out.append(t.format(
                fn=_name(rng), args=", ".join(_name(rng) for _ in range(rng.randint(0, 4))),
                ret=_name(rng), doc=" ".join(_name(rng) for _ in range(rng.randint(2, 6))),
                cls=_name(rng).capitalize(), attr=_name(rng), var=_name(rng),
            ))
        return "".join(out)
