def setup_imports():
    from pathlib import Path
    import sys

    cur = Path().resolve()
    while not (cur / "src").is_dir():
        if cur == cur.parent:
            raise RuntimeError("No 'src' dir found")
        cur = cur.parent

    src_path = cur / "src"
    sys.path.insert(0, str(src_path))
    print(f"[setup_imports] Added '{src_path}' to sys.path")
