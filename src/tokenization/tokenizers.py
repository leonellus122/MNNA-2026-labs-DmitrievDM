import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple



PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"

SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]


def _sample_texts(
    texts: List[str],
    max_samples: Optional[int],
    seed: int
) -> List[str]:
    """
    Если max_samples задано и текстов больше, возвращает случайную часть.
    Иначе возвращает все тексты.
    """
    if max_samples is None or len(texts) <= max_samples:
        return texts

    rng = random.Random(seed)
    return rng.sample(texts, max_samples)


# ============================================================
# Character tokenizer
# ============================================================

def fit_char_tokenizer(
    texts: List[str],
    max_samples: Optional[int] = None,
    seed: int = 42
) -> Dict[str, int]:
    """
    Строит символьный словарь.

    Возвращает словарь:
        {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "а": 4, ...}
    """
    if len(texts) == 0:
        raise ValueError("Список текстов пуст")

    train_texts = _sample_texts(texts, max_samples, seed)

    token2id = {token: i for i, token in enumerate(SPECIAL_TOKENS)}

    for text in train_texts:
        for char in text:
            if char not in token2id:
                token2id[char] = len(token2id)

    return token2id


def encode_char(
    text: str,
    token2id: Dict[str, int],
    add_special: bool = True
) -> List[int]:
    """
    Кодирует текст в последовательность id символов.

    Если add_special=True:
        добавляет BOS в начало и EOS в конец.
    """
    unk_id = token2id[UNK]

    if add_special:
        ids = [token2id[BOS]]
    else:
        ids = []

    for char in text:
        ids.append(token2id.get(char, unk_id))

    if add_special:
        ids.append(token2id[EOS])

    return ids


# ============================================================
# Word tokenizer
# ============================================================

def fit_word_tokenizer(
    texts: List[str],
    max_samples: Optional[int] = 100_000,
    max_vocab_size: Optional[int] = 30_000,
    min_freq: int = 2,
    seed: int = 42
) -> Dict[str, int]:
    """
    Строит словарь слов.

    Если текстов много, можно обучать только на части:
        max_samples=100_000

    Если не хватает памяти:
        уменьшайте max_samples, max_vocab_size или увеличивайте min_freq.
    """
    if len(texts) == 0:
        raise ValueError("Список текстов пуст")

    train_texts = _sample_texts(texts, max_samples, seed)

    counter = Counter()

    for text in train_texts:
        words = text.split()
        counter.update(words)

    token2id = {token: i for i, token in enumerate(SPECIAL_TOKENS)}

    if max_vocab_size is None:
        common_items = counter.most_common()
    else:
        common_items = counter.most_common(max_vocab_size)

    for token, freq in common_items:
        if freq < min_freq:
            break

        if token not in token2id:
            token2id[token] = len(token2id)

    return token2id


def encode_word(
    text: str,
    token2id: Dict[str, int],
    add_special: bool = True
) -> List[int]:
    """
    Кодирует текст в последовательность id слов.

    Если add_special=True:
        добавляет BOS в начало и EOS в конец.
    """
    unk_id = token2id[UNK]

    if add_special:
        ids = [token2id[BOS]]
    else:
        ids = []

    for word in text.split():
        ids.append(token2id.get(word, unk_id))

    if add_special:
        ids.append(token2id[EOS])

    return ids


# ============================================================
# BPE tokenizer (собственная реализация, без сторонних библиотек)
# ============================================================

END_OF_WORD = "</w>"  # маркер конца слова, чтобы BPE не сливал символы через границу слов

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _pre_tokenize(text: str) -> List[str]:
    """Разбивает текст на слова/знаки препинания"""
    return _WORD_RE.findall(text)


class _EncodingResult:
    """Обёртка, повторяющая интерфейс tokenizers.Encoding (доступ через .ids)."""

    def __init__(self, ids: List[int]):
        self.ids = ids


class BPETokenizer:

    def __init__(self, unk_token: str = UNK):
        self.unk_token = unk_token
        self.token_to_id_map: Dict[str, int] = {}
        self.id_to_token_map: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.merge_rank: Dict[Tuple[str, str], int] = {}

    def train(
        self,
        texts: List[str],
        vocab_size: int = 10_000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
    ) -> "BPETokenizer":
        special_tokens = special_tokens or SPECIAL_TOKENS

        word_freq: Counter = Counter()
        for text in texts:
            for word in _pre_tokenize(text):
                word_freq[word] += 1

        words: List[Tuple[str, ...]] = []
        freqs: List[int] = []
        for word, freq in word_freq.items():
            words.append(tuple(word) + (END_OF_WORD,))
            freqs.append(freq)

        base_vocab = set()
        for w in words:
            base_vocab.update(w)

        pair_freq: Counter = Counter()
        pair_to_words: Dict[Tuple[str, str], set] = defaultdict(set)

        for idx, w in enumerate(words):
            for a, b in zip(w, w[1:]):
                pair_freq[(a, b)] += freqs[idx]
                pair_to_words[(a, b)].add(idx)

        merges: List[Tuple[str, str]] = []
        target_merges = max(0, vocab_size - len(base_vocab) - len(special_tokens))

        while len(merges) < target_merges and pair_freq:
            best_pair, best_count = pair_freq.most_common(1)[0]
            if best_count < min_frequency:
                break

            merges.append(best_pair)
            a, b = best_pair
            merged_symbol = a + b
            affected = list(pair_to_words[best_pair])

            for idx in affected:
                word = words[idx]
                if best_pair not in zip(word, word[1:]):
                    continue  # запись устарела из-за предыдущих слияний

                for x, y in zip(word, word[1:]):
                    pair_freq[(x, y)] -= freqs[idx]
                    if pair_freq[(x, y)] <= 0:
                        del pair_freq[(x, y)]
                    pair_to_words[(x, y)].discard(idx)

                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                        new_word.append(merged_symbol)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word = tuple(new_word)
                words[idx] = new_word

                for x, y in zip(new_word, new_word[1:]):
                    pair_freq[(x, y)] += freqs[idx]
                    pair_to_words[(x, y)].add(idx)

            base_vocab.add(merged_symbol)
            pair_to_words.pop(best_pair, None)

        self.merges = merges
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}

        vocab_tokens = list(special_tokens)
        for token in sorted(base_vocab):
            if token not in vocab_tokens:
                vocab_tokens.append(token)

        self.token_to_id_map = {tok: i for i, tok in enumerate(vocab_tokens)}
        self.id_to_token_map = {i: tok for tok, i in self.token_to_id_map.items()}

        return self

    def _bpe_word(self, word: str) -> List[str]:
        symbols = list(word) + [END_OF_WORD]
        if len(symbols) == 1:
            return symbols

        while True:
            pairs = list(zip(symbols, symbols[1:]))
            ranked = [(self.merge_rank[p], i) for i, p in enumerate(pairs) if p in self.merge_rank]
            if not ranked:
                break
            _, merge_idx = min(ranked)
            a, b = pairs[merge_idx]
            symbols = symbols[:merge_idx] + [a + b] + symbols[merge_idx + 2:]

        return symbols

    def _encode_ids(self, text: str) -> List[int]:
        unk_id = self.token_to_id_map[self.unk_token]
        ids: List[int] = []
        for word in _pre_tokenize(text):
            for symbol in self._bpe_word(word):
                ids.append(self.token_to_id_map.get(symbol, unk_id))
        return ids

    def encode(self, text: str) -> _EncodingResult:
        """Совместимо с tokenizers.Tokenizer.encode(text).ids."""
        return _EncodingResult(self._encode_ids(text))

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token_map.get(i, self.unk_token) for i in ids]
        return "".join(tokens).replace(END_OF_WORD, " ").strip()

    def token_to_id(self, token: str) -> Optional[int]:
        return self.token_to_id_map.get(token)

    def id_to_token(self, idx: int) -> Optional[str]:
        return self.id_to_token_map.get(idx)

    def get_vocab_size(self) -> int:
        return len(self.token_to_id_map)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab": self.token_to_id_map,
            "merges": [list(pair) for pair in self.merges],
            "unk_token": self.unk_token,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(unk_token=data.get("unk_token", UNK))
        tok.token_to_id_map = {k: int(v) for k, v in data["vocab"].items()}
        tok.id_to_token_map = {v: k for k, v in tok.token_to_id_map.items()}
        tok.merges = [tuple(pair) for pair in data["merges"]]
        tok.merge_rank = {pair: i for i, pair in enumerate(tok.merges)}
        return tok


def train_bpe_tokenizer(
    texts: List[str],
    max_samples: Optional[int] = 100_000, # Максимальное количество текстов для обучения
    vocab_size: int = 10_000, # Размер словаря
    min_freq: int = 2, # Минимальная частота для включения токена
    seed: int = 42
) -> BPETokenizer:

    if len(texts) == 0:
        raise ValueError("Список текстов пуст")

    train_texts = _sample_texts(texts, max_samples, seed)

    tokenizer = BPETokenizer(unk_token=UNK)
    tokenizer.train(
        train_texts,
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=SPECIAL_TOKENS
    )

    return tokenizer


def encode_bpe(
    text: str,
    tokenizer: BPETokenizer,
    add_special: bool = True
) -> List[int]:
    """
    Кодирует текст с помощью BPE.

    Если add_special=True:
        добавляет BOS в начало и EOS в конец.
    """
    encoded = tokenizer.encode(text)
    ids = encoded.ids

    if add_special:
        bos_id = tokenizer.token_to_id(BOS)
        eos_id = tokenizer.token_to_id(EOS)

        if bos_id is None or eos_id is None:
            raise ValueError("В BPE-токенизаторе отсутствуют токены BOS/EOS")

        ids = [bos_id] + ids + [eos_id]

    return ids


# ============================================================
# Save / Load
# ============================================================

def save_vocab(token2id: Dict[str, int], path) -> Path:
    """
    Сохраняет символьный или словесный словарь в JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(token2id, f, ensure_ascii=False, indent=2)

    return path


def load_vocab(path) -> Dict[str, int]:
    """
    Загружает символьный или словесный словарь из JSON.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_bpe_tokenizer(tokenizer: BPETokenizer, path) -> Path:
    """
    Сохраняет BPE-токенизатор в файл.
    """
    return tokenizer.save(path)


def load_bpe_tokenizer(path) -> BPETokenizer:
    """
    Загружает BPE-токенизатор из файла.
    """
    return BPETokenizer.load(path)