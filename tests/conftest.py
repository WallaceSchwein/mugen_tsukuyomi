from glob import glob

def path_to_module(path: str) -> str:
    """Transforms a path to a module location to import"""
    return path.replace("/", ".").replace("\\", ".").replace(".py", "")

pytest_plugins = [path_to_module(p) for p in glob("tests/fixtures/*py")if "__" not in p]
