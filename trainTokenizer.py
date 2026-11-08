import os
import regex as re
from collections import Counter

import tokenizer

VOCAB_SIZE = 3000
_HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(_HERE, "shakespeare_complete_pg100.txt")
MERGES_PATH = os.path.join(_HERE, "bpe_merges.txt")
PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def pre_tokenizer(text):
    return Counter(m.group() for m in PAT.finditer(text))


def words_count(words):
    vocabs = Counter()
    for word, num in words.items():
        vocabs[tuple(word.encode("utf-8"))] += num
    return vocabs


def full_stats(vocabs):
    stats = Counter()
    for vocab, num in vocabs.items():
        for i in range(len(vocab) - 1):
            stats[(vocab[i], vocab[i + 1])] += num
    return stats


def merge_and_update(vocabs, new_id, pair, stats):
    new_vocabs = Counter()
    for vocab, num in vocabs.items():
        i, out, changed = 0, [], False
        while i < len(vocab):
            if i < len(vocab) - 1 and (vocab[i], vocab[i + 1]) == pair:
                out.append(new_id)
                i += 2
                changed = True
            else:
                out.append(vocab[i])
                i += 1
        if changed:
            for k in range(len(vocab) - 1):
                stats[(vocab[k], vocab[k + 1])] -= num
            out = tuple(out)
            for k in range(len(out) - 1):
                stats[(out[k], out[k + 1])] += num
            new_vocabs[out] += num
        else:
            new_vocabs[vocab] += num
    return new_vocabs


def main():
    text = open(TRAIN_PATH, encoding="utf-8").read()
    text = text[: (len(text) // 10) * 9]        

    vocabs = words_count(pre_tokenizer(text))
    stats = full_stats(vocabs)

    merges = {}
    new_id = 256
    for _ in range(VOCAB_SIZE - 256):
        pair = max(stats, key=stats.get)
        if stats[pair] <= 0:
            break
        vocabs = merge_and_update(vocabs, new_id, pair, stats)
        merges[pair] = new_id
        new_id += 1

    with open(tokenizer.MERGES_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("# bpe merges v1\n")
        for (a, b) in merges:
            f.write(f"{a} {b}\n")
    print("saved:", tokenizer.MERGES_PATH, "merges:", len(merges))


if __name__ == "__main__":
    main()
    print("已保存:", MERGES_PATH)