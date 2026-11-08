import os

_HERE = os.path.dirname(os.path.abspath(__file__))
MERGES_PATH = os.path.join(_HERE, "bpe_merges.txt")


def load_merges(path):
    merges = {}
    if not os.path.exists(path):
        return merges
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b = map(int, line.split())
            merges[(a, b)] = 256 + len(merges)
    return merges


def build_ntos(merges):
    ntos = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in merges.items():
        ntos[new_id] = ntos[a] + ntos[b]
    return ntos


merges = load_merges(MERGES_PATH)
ntos = build_ntos(merges)
LENTH = 256 + len(merges)


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def apply_merge(ids, pair, new_id):
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def tokenizerEncoding(text):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = apply_merge(ids, pair, merges[pair])
    return ids


def tokenizerDecoding(ids):
    return b"".join(ntos[i] for i in ids).decode("utf-8", errors="replace")

