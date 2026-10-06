import os


def write_atomic(path, text):
    """A crash mid-write must not leave resume or a restart a torn show or recipe."""
    staged = path.with_name(f".{path.name}.tmp")
    staged.write_text(text, encoding="utf-8")
    os.replace(staged, path)
