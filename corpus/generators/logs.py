import random
from corpus.generators.base import Generator


_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR"]
_LEVEL_WEIGHTS = [10, 60, 20, 10]
_COMPONENTS = ["server", "db", "cache", "auth", "api", "worker", "scheduler"]
_MESSAGES = [
    "request handled in {ms}ms",
    "connected to {host}",
    "retrying after {n} attempts",
    "config loaded from {path}",
    "cache miss for key={key}",
    "rate limit exceeded for ip={ip}",
    "shutting down gracefully",
    "starting up version {ver}",
    "connection closed by peer",
    "deprecated API used: {fn}",
]


class LogGenerator(Generator):
    archetype = "log"

    def generate(self, rng: random.Random) -> str:
        n = rng.randint(10, 80)
        lines = []
        hour = rng.randint(0, 23)
        minute = rng.randint(0, 59)
        for _ in range(n):
            second = rng.randint(0, 59)
            ms = rng.randint(0, 999)
            ts = f"2026-04-{rng.randint(1, 28):02d}T{hour:02d}:{minute:02d}:{second:02d}.{ms:03d}Z"
            level = rng.choices(_LEVELS, weights=_LEVEL_WEIGHTS, k=1)[0]
            comp = rng.choice(_COMPONENTS)
            tmpl = rng.choice(_MESSAGES)
            msg = tmpl.format(
                ms=rng.randint(1, 9999), host=f"db-{rng.randint(0,99)}.internal",
                n=rng.randint(1, 10), path=f"/etc/{rng.choice(['app','svc','daemon'])}.yaml",
                key=f"user:{rng.randint(1000, 99999)}", ip=f"10.0.{rng.randint(0,255)}.{rng.randint(0,255)}",
                ver=f"{rng.randint(1,3)}.{rng.randint(0,9)}.{rng.randint(0,9)}",
                fn=rng.choice(["legacy_login", "old_api", "v1_handler"]),
            )
            lines.append(f"{ts} [{level:<5}] {comp}: {msg}")
        return "\n".join(lines) + "\n"
