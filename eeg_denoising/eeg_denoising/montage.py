import config

_MONTAGE_FILES = {
    "TUAR_TCP_AR": config.PATH_TO_TUAR_01_TCP_AR_CUSTOM_MONTAGE,
}

_montage_cache = {}

def __getattr__(name: str):
    if name not in _MONTAGE_FILES:
        raise AttributeError(f"module 'montages' has no attribute {name!r}")

    if name not in _montage_cache:
        fpath = os.path.join(_MONTAGE_DIR, _MONTAGE_FILES[name])
        _montage_cache[name] = read_nedc_montage(fpath)

    return _montage_cache[name]


def read_nedc_montage(fpath: str) -> list[dict]:
    montage = []

    with open(fpath, "r") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()

            if not line or line.startswith("#") or line.startswith("["):
                continue

            try:
                _, rest = line.split("=", 1)
                idx_name, electrodes = rest.split(":", 1)
                idx_str, name = idx_name.split(",", 1)
                anode, cathode = electrodes.split("--", 1)
            except ValueError:
                raise ValueError(f"Cannot parse montage line {lineno}: {raw!r}")

            montage.append({
                "index":   int(idx_str.strip()),
                "name":    name.strip(),
                "anode":   anode.strip(),
                "cathode": cathode.strip(),
            })

    return montage


def apply_nedc_montage(raw: mne.io.BaseRaw, fpath: str) -> mne.io.BaseRaw:
    channels = read_nedc_montage(fpath)

    anodes   = [ch["anode"]   for ch in channels]
    cathodes = [ch["cathode"] for ch in channels]
    names    = [ch["name"]    for ch in channels]

    return mne.set_bipolar_reference(raw, anode=anodes, cathode=cathodes, ch_name=names)
