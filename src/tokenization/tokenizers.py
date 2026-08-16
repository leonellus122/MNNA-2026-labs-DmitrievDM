import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from tokenizers import Tokenizer, models, trainers, pre_tokenizers



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
# BPE tokenizer
# ============================================================

def train_bpe_tokenizer(
    texts: List[str],
    max_samples: Optional[int] = 100_000,
    vocab_size: int = 10_000,
    min_freq: int = 2,
    seed: int = 42
) -> Tokenizer:
    """
    Обучает BPE-токенизатор.

    Если текстов много, можно обучать только на части:
        max_samples=100_000

    Размер словаря можно менять:
        vocab_size=10_000, 20_000, 30_000
    """
    if len(texts) == 0:
        raise ValueError("Список текстов пуст")

    train_texts = _sample_texts(texts, max_samples, seed)

    tokenizer = Tokenizer(models.BPE(unk_token=UNK))

    # Разбивает текст по пробелам и знакам препинания.
    # Если пунктуация уже удалена при очистке, это не страшно.
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=SPECIAL_TOKENS
    )

    tokenizer.train_from_iterator(train_texts, trainer=trainer)

    return tokenizer


def encode_bpe(
    text: str,
    tokenizer: Tokenizer,
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


def save_bpe_tokenizer(tokenizer: Tokenizer, path) -> Path:
    """
    Сохраняет BPE-токенизатор в файл.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer.save(str(path))

    return path


def load_bpe_tokenizer(path) -> Tokenizer:
    """
    Загружает BPE-токенизатор из файла.
    """
    return Tokenizer.from_file(str(path))