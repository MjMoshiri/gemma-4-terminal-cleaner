import random
from dirtifier.transforms.base import Transform


class WhitespacePadding(Transform):
    """Add extra spaces between whitespace runs (mimicking column padding)."""
    name = "whitespace_padding"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        out_lines = []
        for line in clean.split("\n"):
            # Replace each space run with a longer run with low prob
            chars = []
            i = 0
            while i < len(line):
                if line[i] == " ":
                    j = i
                    while j < len(line) and line[j] == " ":
                        j += 1
                    n = j - i
                    if rng.random() < 0.3:
                        n += rng.randint(1, 4)
                    chars.append(" " * n)
                    i = j
                else:
                    chars.append(line[i])
                    i += 1
            out_lines.append("".join(chars))
        return "\n".join(out_lines)


class TrailingWhitespace(Transform):
    """Add trailing spaces per line + extra blank lines."""
    name = "trailing_whitespace"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        out_lines = []
        for line in clean.split("\n"):
            if rng.random() < 0.3:
                line = line + " " * rng.randint(1, 8)
            out_lines.append(line)
            if rng.random() < 0.05:
                out_lines.append("")
        return "\n".join(out_lines)


class WindowsLineEndings(Transform):
    """Replace some \\n with \\r\\n."""
    name = "windows_line_endings"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        out = []
        for ch in clean:
            if ch == "\n" and rng.random() < 0.4:
                out.append("\r\n")
            else:
                out.append(ch)
        return "".join(out)
