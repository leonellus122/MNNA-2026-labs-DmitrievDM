import warnings
import logging
import json
from pathlib import Path
import re
from langdetect import detect, LangDetectException
import unicodedata
import tiktoken

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ============================================================
# ЛОГГЕР
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# ЗАДАНИЕ 3.2.1. Очистка текста от HTML разметки
# ============================================================
def remove_html(raw_html: str) -> str:
    if not raw_html:
        return ""

    soup = BeautifulSoup(raw_html, "lxml")

    for tag in soup(["header"]): # Удаляем header-ы но также можно в дальнейшем убрать и теги: "script", "style", "header", "footer", "nav", "noscript", "iframe"
        tag.decompose()

    text = soup.get_text(separator=" ")

    return text

# ============================================================
# ЗАДАНИЕ 3.2.2. Удаление неизвестных языков и посторонних символов
# ============================================================
allowed_languages = {"ru", "en"}

allowed_chars_pattern = re.compile(
    r"[^a-zA-Zа-яА-ЯёЁ0-9\s"
    r".,!?;:\-—–\"'«»()\[\]{}%№@#&*/+=<>_$€₽…]"
)
def detect_language(text: str) -> str | None:
    # langdetect не умеет работать с пустым/слишком коротким текстом, вроде как
    sample = text.strip()
    if len(sample) < 20:
        return None

    

    try:
        return detect(sample)
    except LangDetectException:
        return None

def is_allowed_language(text: str) -> bool:
    lang = detect_language(text)
    return lang in allowed_languages

def strip_unknown_symbols(text: str) -> str:
    return allowed_chars_pattern.sub(" ", text)

# ============================================================
# ЗАДАНИЕ 3.2.3. Фильтрация по ключевым словам (токсичный контент) 
# ============================================================
toxic_keywords = {}


def contains_toxic_keywords(text: str, keywords: set[str] = toxic_keywords) -> bool:
    # if not keywords:
    #     return False

    # lowered = text.lower()
    # return any(keyword.lower() in lowered for keyword in keywords)
    pass

# ============================================================
# ЗАДАНИЕ 3.2.4. Нормализация пробелов и Unicode
# ============================================================
# 4. Нормализация пробелов и Unicode
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text) # Нормализация, приведение текста к единому виду
    text = re.sub(r"\s+", " ", text) # Замена последовательностей пробельных символов на один пробел
    return text.strip() # Удаление пробелов в начале и конце текста

# ============================================================
# ЗАДАНИЕ 3.2.5. Удаление пустых строк
# ============================================================
def is_empty(text: str, min_length: int = 1) -> bool:
    return len(text.strip()) < min_length # Строка пустая если ее длина меньше минимальной длины (по умолчанию 1, то есть пустая строка)

# ============================================================
# ЗАДАНИЕ 3.2.6. Разбиение слишком длинных объектов на части
# ============================================================


_ENC = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))

# Границы длины объекта в токенах (см. задание: 512-1024 токена)
MAX_TOKENS_PER_CHUNK = 1024

# Простое разбиение на предложения по знакам конца предложения
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?…])\s+")

def _hard_split_by_words(text: str, max_tokens: int) -> list[str]:
    """Аварийное разбиение аномально длинного предложения по словам."""
    words = text.split()
    chunks, current = [], []

    for word in words:
        current.append(word)
        if count_tokens(" ".join(current)) >= max_tokens:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks

def split_long_text(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> list[str]:
    """
    Если текст длиннее max_tokens токенов — разбивает его на несколько
    частей по границам предложений (чтобы не разрывать предложение
    посередине). Части набираются "жадно" до достижения лимита.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    sentences = _SENTENCE_SPLIT_PATTERN.split(text)

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_tokens = count_tokens(sentence)
        if sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_tokens = [], 0
            chunks.extend(_hard_split_by_words(sentence, max_tokens))
            continue

        if current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_tokens = [], 0

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ============================================================
# ЗАДАНИЕ 3.2. Объединение всех шагов очистки в один пайплайн
# ============================================================
def clean_text(raw_html: str, use_toxic_filter: bool = False) -> list[str]:
    # 1. HTML и заголовки
    text = remove_html(raw_html)

    # 2. Неизвестные языки и посторонние символы
    if not is_allowed_language(text):
        return []
    text = strip_unknown_symbols(text)

    # 3. (опционально) Токсичные ключевые слова
    if use_toxic_filter and contains_toxic_keywords(text):
        return []

    # 4. Нормализация пробелов и Unicode
    text = normalize_text(text)

    # 5. Удаление пустых строк
    if is_empty(text, min_length=20):
        return []

    # 6. Разбиение длинных объектов
    return split_long_text(text, max_tokens=MAX_TOKENS_PER_CHUNK)


# Обработка целого JSONL-файла
def clean_jsonl_file(
    input_path: str,
    output_path: str,
    text_field: str = "content",
    use_toxic_filter: bool = False,
) -> None:
    """
    Читает JSONL построчно, применяет clean_text() к каждому объекту
    и пишет результат в новый JSONL. Каждая исходная запись может
    превратиться в 0, 1 или несколько итоговых записей.
    """
    stats = {"read": 0, "kept": 0, "dropped": 0, "chunks_written": 0}

    with open(input_path, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            stats["read"] += 1

            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Пропущена битая строка JSON (raw): %r", line[:100])
                continue

            raw_html = item.get(text_field, "")
            chunks = clean_text(raw_html, use_toxic_filter=use_toxic_filter)

            if not chunks:
                stats["dropped"] += 1
                continue

            stats["kept"] += 1

            for i, chunk in enumerate(chunks):
                out_item = {
                    "url": item.get("url"),
                    "record_id": item.get("record_id"),
                    "chunk_index": i,
                    "text": chunk,
                }
                fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                stats["chunks_written"] += 1

    logger.info(
        "Файл %s обработан: прочитано=%d, оставлено объектов=%d, "
        "отброшено=%d, итоговых чанков записано=%d",
        input_path,
        stats["read"],
        stats["kept"],
        stats["dropped"],
        stats["chunks_written"],
    )


def clean_all_jsonl(
    input_dir: str,
    output_dir: str,
    use_toxic_filter: bool = False,
) -> None:
    """Применяет clean_jsonl_file() ко всем .jsonl файлам в папке."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(input_path.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning("В папке %s не найдено .jsonl файлов", input_path)
        return

    for file in jsonl_files:
        out_file = output_path / file.name
        logger.info("Обработка: %s", file)
        clean_jsonl_file(
            input_path=str(file),
            output_path=str(out_file),
            use_toxic_filter=use_toxic_filter,
        )

# if __name__ == "__main__":
#     input="D:/Folders/Master Degree/labs/MNNA-2026-labs-DmitrievDM/MNNA-2026-labs-DmitrievDM/data/converted"
#     output="D:/Folders/Master Degree/labs/MNNA-2026-labs-DmitrievDM/MNNA-2026-labs-DmitrievDM/data/cleaned"

#     clean_all_jsonl(
#         input_dir=input,
#         output_dir=output,
#         use_toxic_filter=False,
#     )

if __name__ == "__main__":

    input="D:/Folders/Master Degree/labs/MNNA-2026-labs-DmitrievDM/MNNA-2026-labs-DmitrievDM/data/converted"
    output="D:/Folders/Master Degree/labs/MNNA-2026-labs-DmitrievDM/MNNA-2026-labs-DmitrievDM/data/cleaned"
    
    try:
        clean_all_jsonl(
            input_dir=input,
            output_dir=output,
            use_toxic_filter=False,
        )
    except KeyboardInterrupt:
        logger.warning("Прервано пользователем — показываю собранную статистику")
    finally:
        print_profile_report()