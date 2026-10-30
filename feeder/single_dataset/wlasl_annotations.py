"""Parse WLASL clip names without treating spaces in glosses as columns."""
import os


def read_wlasl_annotations(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                name, label = line.rsplit(None, 1)
                int(label)
            except ValueError as exc:
                raise ValueError(f"Invalid WLASL annotation at {path}:{number}") from exc
            rows.append([name, label])
    return rows


def clip_stem(filename):
    return os.path.splitext(str(filename))[0]
