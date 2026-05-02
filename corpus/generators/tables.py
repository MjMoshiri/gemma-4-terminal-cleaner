import random
from corpus.generators.base import Generator


_USERS = ["alice", "bob", "carol", "dave", "eve", "root", "deploy", "www-data"]
_GROUPS = ["users", "staff", "wheel", "nogroup", "deploy", "www-data"]
_FILE_SUFFIXES = [".py", ".js", ".ts", ".rs", ".go", ".md", ".json", ".yaml",
                  ".toml", ".lock", ".txt", ".log", ".sh", ".html", ".css"]
_DIR_NAMES = ["src", "tests", "docs", "build", "dist", "node_modules",
              ".git", "vendor", "target", "venv", "logs", "tmp", "data"]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _random_perm(rng: random.Random) -> str:
    is_dir = rng.random() < 0.2
    type_char = "d" if is_dir else "-"
    perms = ""
    for _ in range(3):
        perms += rng.choice(["r", "-"])
        perms += rng.choice(["w", "-"])
        perms += rng.choice(["x", "-"])
    return type_char + perms


def _random_filename(rng: random.Random) -> str:
    if rng.random() < 0.25:
        return rng.choice(_DIR_NAMES)
    base_chars = "abcdefghijklmnopqrstuvwxyz_-0123456789"
    base_len = rng.randint(3, 12)
    base = "".join(rng.choice(base_chars) for _ in range(base_len))
    return base + rng.choice(_FILE_SUFFIXES)


def _random_size(rng: random.Random) -> int:
    # log-uniform-ish: many small files, occasional large
    exponent = rng.randint(0, 8)
    return rng.randint(1, 9) * (10 ** exponent)


def _random_date_columns(rng: random.Random) -> str:
    month = rng.choice(_MONTHS)
    day = rng.randint(1, 28)
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    return f"{month} {day:>2} {hour:02d}:{minute:02d}"


class LsListingGenerator(Generator):
    archetype = "table"

    def generate(self, rng: random.Random) -> str:
        n = rng.randint(3, 40)
        rows = []
        total = 0
        for _ in range(n):
            perms = _random_perm(rng)
            links = rng.randint(1, 9)
            owner = rng.choice(_USERS)
            group = rng.choice(_GROUPS)
            size = _random_size(rng)
            total += size // 1024 + 1
            date = _random_date_columns(rng)
            name = _random_filename(rng)
            rows.append(f"{perms} {links:>2} {owner:<8} {group:<8} {size:>8} {date} {name}")
        return f"total {total}\n" + "\n".join(rows) + "\n"


class PsListingGenerator(Generator):
    archetype = "table"

    def generate(self, rng: random.Random) -> str:
        cmds = ["python", "node", "bash", "ssh", "vim", "make", "cargo build",
                "go test ./...", "pytest", "docker", "kubectl", "ruby",
                "java -jar app.jar", "/usr/bin/containerd", "systemd"]
        header = "  PID TTY          TIME CMD"
        rows = [header]
        n = rng.randint(5, 30)
        for _ in range(n):
            pid = rng.randint(1, 99999)
            tty = rng.choice(["pts/0", "pts/1", "?", "tty1"])
            mins = rng.randint(0, 59)
            secs = rng.randint(0, 59)
            cmd = rng.choice(cmds)
            rows.append(f"{pid:>5} {tty:<8} 00:{mins:02d}:{secs:02d} {cmd}")
        return "\n".join(rows) + "\n"


class DfListingGenerator(Generator):
    archetype = "table"

    def generate(self, rng: random.Random) -> str:
        header = "Filesystem      1K-blocks      Used  Available Use% Mounted on"
        rows = [header]
        mounts = ["/", "/home", "/var", "/tmp", "/dev/shm", "/boot", "/data"]
        for mount in rng.sample(mounts, k=rng.randint(3, len(mounts))):
            blocks = rng.randint(100_000, 100_000_000)
            used = rng.randint(0, blocks)
            avail = blocks - used
            pct = int(used / blocks * 100)
            fs = f"/dev/{rng.choice(['sda1','sda2','nvme0n1p1','vda1'])}"
            rows.append(f"{fs:<14} {blocks:>10} {used:>9} {avail:>10} {pct:>3}% {mount}")
        return "\n".join(rows) + "\n"
