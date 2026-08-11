import gzip
import json
from pathlib import Path
import logging
from tqdm.notebook import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from warcio.archiveiterator import ArchiveIterator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

HTML_TYPES = {
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "text/xml",

}

# Открывает WARC-файл, поддерживая сжатые файлы .gz
def open_warc(path: str):
    path_obj = Path(path)

    if path_obj.suffix == ".gz":
        return gzip.open(path_obj, "rb")

    return open(path_obj, "rb")

# Извлечение кодировки из заголовка Content-Type
def get_charset(content_type: str | None) -> str:
    if not content_type:
        return "utf-8"

# Для регистронезависимости, 
# например, может быть, что charset=UTF-8 или charset="utf-8", 
# все приводим к utf-8
    content_type_lower = content_type.lower()

    if "charset" not in content_type_lower:
        return "utf-8"


    # Что убрать пробелы вокруг знака равенства, чтобы корректно извлечь кодировку
    normalized = content_type_lower.replace(" =", "=").replace("= ", "=")
    
    charset = normalized.split("charset=", 1)[1]
    charset = charset.split(";", 1)[0]
    charset = charset.strip().strip('"').strip("'")
    
    return charset or "utf-8"



def convert_warc_to_jsonl(
    input_path: str,
    output_path: str,
    html_only: bool = True,
) -> None:
    
    records_total = 0
    records_written = 0
    records_failed = 0

    with open_warc(input_path) as warc_stream, \
            open(output_path, "w", encoding="utf-8") as output:

# Итерация по записям в warc файле
        for record in ArchiveIterator(warc_stream):
            records_total += 1

            if record.rec_type != "response":
                continue

            content_type = record.http_headers.get_header("Content-Type")

            if html_only:
                normalized_content_type = (
                    content_type.split(";", 1)[0].strip().lower()
                    if content_type
                    else ""
                )
# Нопмализуем типы записей, если нет в нашем списке(HTML_TYPES) пропускаем
                if normalized_content_type not in HTML_TYPES:
                    continue

            try:
                payload = record.content_stream().read()

                charset = get_charset(content_type)
# Если кодировка не поддерживается, используем utf-8, все непонятные символы заменяем на �, чтобы не было ошибок при записи
                try:
                    content = payload.decode(charset, errors="replace")
                except LookupError:
                    content = payload.decode("utf-8", errors="replace")

                item = {
                    "url": record.rec_headers.get_header("WARC-Target-URI"),
                    "record_id": record.rec_headers.get_header(
                        "WARC-Record-ID"
                    ),
                    "content_type": content_type,
                    "content_length": len(payload),
                    "content": content,
                }

                output.write(
                    # Коевертируем словарь в JSON-строку и добавляем перенос строки
                    json.dumps(
                        item,
                        ensure_ascii=False,
                    ) + "\n"
                )

                records_written += 1

            except Exception:
                records_failed += 1

    logger.info(
    "Файл %s обработан: всего записей=%d, "
    "записано=%d, ошибок=%d",
    input_path,
    records_total,
    records_written,
    records_failed,
)


# Конвертация всех WARC-файлов в папке
def convert_all_warc(
    input_dir: str,
    output_dir: str,
    recursive: bool = False,
    overwrite: bool = False,
) -> None:

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Входная папка не найдена: {input_path}"
        )

    if not input_path.is_dir():
        raise NotADirectoryError(
            f"Путь не является папкой: {input_path}"
        )

    output_path.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.warc.gz" if recursive else "*.warc.gz"
    warc_files = sorted(input_path.glob(pattern))

    if not warc_files:
        logger.warning(
            "В папке %s не найдены файлы по шаблону %s",
            input_path,
            pattern,
        )
        return

    converted = 0
    skipped = 0
    failed = 0

    logger.info(
        "Найдено WARC-файлов: %d",
        len(warc_files),
    )

    with logging_redirect_tqdm():

        for warc_file in tqdm(
            warc_files,
            desc="Конвертация WARC",
            unit="file",
        ):
            relative_path = warc_file.relative_to(input_path)

            output_file = output_path / relative_path
            output_file = output_file.with_suffix("")
            output_file = output_file.with_suffix(".jsonl")

            output_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if output_file.exists() and not overwrite:
                logger.info(
                    "Пропуск %s: результат уже существует",
                    warc_file,
                )
                skipped += 1
                continue

            logger.info(
                "Обработка файла: %s",
                warc_file,
            )

            try:
                convert_warc_to_jsonl(
                    input_path=str(warc_file),
                    output_path=str(output_file),
                )

                converted += 1

                logger.info(
                    "Готово: %s",
                    output_file,
                )

            except Exception:
                failed += 1

                logger.exception(
                    "Ошибка при обработке файла: %s",
                    warc_file,
                )

    logger.info("Обработка завершена")
    logger.info("Файлов найдено: %d", len(warc_files))
    logger.info("Сконвертировано: %d", converted)
    logger.info("Пропущено: %d", skipped)
    logger.info("Ошибок: %d", failed)