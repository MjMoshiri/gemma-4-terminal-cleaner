from dataclasses import dataclass, field
from dirtifier.transforms.base import Transform
from dirtifier.transforms.ansi import AnsiColor, AnsiBold, AnsiUnderline, AnsiOsc, Hyperlink
from dirtifier.transforms.progress import ProgressBar, Spinner, CursorMovement
from dirtifier.transforms.box import BoxDrawing
from dirtifier.transforms.repeat import RepeatedLines
from dirtifier.transforms.whitespace import WhitespacePadding, TrailingWhitespace, WindowsLineEndings
from dirtifier.transforms.locale import LocaleVariants
from dirtifier.transforms.timestamps import Timestamps
from dirtifier.transforms.streams import MixedStreams
from dirtifier.transforms.edge import BellChars, NulBytes


@dataclass
class Recipe:
    name: str
    # Each step: (transform, probability of applying)
    steps: list[tuple[Transform, float]] = field(default_factory=list)
    # For weighted sampling at the pipeline level
    weight: float = 1.0


RECIPES: dict[str, Recipe] = {
    "cli_colored_table": Recipe(
        name="cli_colored_table",
        steps=[
            (AnsiColor(), 0.9),
            (AnsiBold(), 0.3),
            (WhitespacePadding(), 0.5),
            (TrailingWhitespace(), 0.3),
        ],
        weight=3.0,  # most common archetype dirt
    ),
    "install_with_progress": Recipe(
        name="install_with_progress",
        steps=[
            (ProgressBar(), 0.7),
            (Spinner(), 0.5),
            (AnsiColor(), 0.6),
            (AnsiBold(), 0.4),
            (RepeatedLines(), 0.4),
        ],
        weight=2.0,
    ),
    "tui_redraw": Recipe(
        name="tui_redraw",
        steps=[
            (CursorMovement(), 0.9),
            (AnsiColor(), 0.7),
            (BoxDrawing(), 0.3),
        ],
        weight=1.0,
    ),
    "noisy_logs": Recipe(
        name="noisy_logs",
        steps=[
            (Timestamps(), 0.6),
            (RepeatedLines(), 0.5),
            (AnsiColor(), 0.4),
            (MixedStreams(), 0.4),
        ],
        weight=2.0,
    ),
    "passthrough": Recipe(
        name="passthrough",
        steps=[],
        weight=1.5,  # important: model must learn to leave clean text alone
    ),
}
