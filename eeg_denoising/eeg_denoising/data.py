import config


_cache = {}

def __getattr__(name: str):
    if name == "MASTER":
        if "MASTER" not in _cache:
            with open("master.json") as f:
                _cache["MASTER"] = objectpath.Tree(json.load(f))
        return _cache["MASTER"]
    raise AttributeError(f"module 'data' has no attribute {name!r}")


