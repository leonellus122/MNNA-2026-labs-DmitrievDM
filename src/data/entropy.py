import json
import logging
from pathlib import Path
import hashlib

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger(__name__)

# ============================================================
# ЗАДАНИЕ 3.3.1 Вычисление энтропии при помощи GPT-2
# ============================================================
def read_jsonl(path: Path, text_field: str = "text"):
    """
    Читает объекты из JSONL.

    Ожидаемый формат строки:
    {"text": "some cleaned text"}

    Если поле называется иначе, можно передать text_field="content".
    """
    index = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping line %d: invalid JSON", line_no)
                continue

            if isinstance(obj, str):
                text = obj
            elif isinstance(obj, dict):
                text = obj.get(text_field, "")
            else:
                logger.warning("Skipping line %d: unsupported JSON type", line_no)
                continue

            if not isinstance(text, str):
                text = str(text)

            text = text.strip()
            if not text:
                continue

            yield {
                "index": index,
                "text": text,
            }
            index += 1


def batched(items, batch_size: int):
    """
    Разбивает объекты на батчи.
    """
    batch = []

    for item in items:
        batch.append(item)

        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


@torch.no_grad()
def compute_entropy_batch(texts, tokenizer, model, device, max_length: int):
    """
    Считает энтропию для батча текстов.

    Возвращает список кортежей:
    (total_nll, num_tokens, entropy_per_token)
    """
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    if input_ids.size(1) < 2:
        return [(0.0, 0, 0.0) for _ in texts]

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits

    # GPT-2 предсказывает следующий токен:
    # logits[:, t, :] -> input_ids[:, t + 1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()

    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.size())

    total_nll = (loss * shift_mask).sum(dim=1)
    num_tokens = shift_mask.sum(dim=1).long()
    entropy_per_token = total_nll / num_tokens.clamp(min=1).float()

    return list(
        zip(
            total_nll.tolist(),
            num_tokens.tolist(),
            entropy_per_token.tolist(),
        )
    )


def compute_dataset_entropy(
    input_path,
    output_path,
    stats_path,
    model_name: str = "gpt2",
    text_field: str = "text",
    batch_size: int = 8,
    max_length: int = 1024,
):
    """
    Вычисляет энтропию каждого объекта датасета и информационную плотность.

    Args:
        input_path: Путь к очищенному JSONL после 3.2.
        output_path: Куда сохранить энтропию каждого объекта.
        stats_path: Куда сохранить итоговую статистику.
        model_name: Модель GPT-2 с Hugging Face.
        text_field: Поле с текстом в JSONL.
        batch_size: Размер батча.
        max_length: Максимальная длина последовательности.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    input_path = Path(input_path)
    output_path = Path(output_path)
    stats_path = Path(stats_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    total_nll = 0.0
    total_tokens = 0

    num_objects = 0
    num_skipped_short = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        objects = read_jsonl(input_path, text_field=text_field)
        batches = batched(objects, batch_size=batch_size)

        for batch in tqdm(batches, desc="Computing GPT-2 entropy", unit="batch"):
            texts = [obj["text"] for obj in batch]

            results = compute_entropy_batch(
                texts=texts,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=max_length,
            )

            for obj, (obj_nll, obj_tokens, obj_entropy) in zip(batch, results):
                if obj_tokens == 0:
                    num_skipped_short += 1
                else:
                    total_nll += obj_nll
                    total_tokens += obj_tokens

                record = {
                    "index": obj["index"],
                    "num_tokens": obj_tokens,
                    "total_nll": obj_nll,
                    "entropy_per_token": obj_entropy,
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                num_objects += 1

    info_density = total_nll / total_tokens if total_tokens > 0 else 0.0

    stats = {
        "model_name": model_name,
        "device": device,
        "batch_size": batch_size,
        "max_length": max_length,
        "num_objects": num_objects,
        "num_skipped_short": num_skipped_short,
        "total_tokens": total_tokens,
        "total_nll": total_nll,
        "info_density_nats_per_token": info_density,
    }

    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Objects processed: %d", num_objects)
    logger.info("Total tokens: %d", total_tokens)
    logger.info("Info density: %.6f", info_density)

    return stats



def compute_dataset_entropy_dir(
    input_dir,
    output_dir,
    stats_path,
    pattern: str = "*.jsonl",
    model_name: str = "gpt2",
    text_field: str = "text",
    batch_size: int = 8,
    max_length: int = 1024,
):
    """
    Вычисляет энтропию для всех JSONL-файлов в директории.

    Args:
        input_dir: Папка с очищенными JSONL-файлами.
        output_dir: Папка, куда сохранить JSONL-файлы с энтропией.
        stats_path: Путь к итоговому файлу статистики.
        pattern: Шаблон файлов, например "*.jsonl".
        model_name: Модель GPT-2 с Hugging Face.
        text_field: Поле с текстом в JSONL.
        batch_size: Размер батча.
        max_length: Максимальная длина последовательности.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    stats_path = Path(stats_path)

    files = sorted(input_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files found in {input_dir} with pattern {pattern}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    agg_total_nll = 0.0
    agg_total_tokens = 0
    agg_num_objects = 0
    agg_num_skipped_short = 0

    per_file_stats = []

    for file_path in files:
        logger.info("Processing file: %s", file_path)

        output_path = output_dir / file_path.name

        file_total_nll = 0.0
        file_total_tokens = 0
        file_num_objects = 0
        file_num_skipped_short = 0

        objects = read_jsonl(file_path, text_field=text_field)
        batches = batched(objects, batch_size=batch_size)

        with output_path.open("w", encoding="utf-8") as out_f:
            for batch in tqdm(batches, desc=file_path.name, unit="batch"):
                texts = [obj["text"] for obj in batch]

                results = compute_entropy_batch(
                    texts=texts,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_length=max_length,
                )

                for obj, (obj_nll, obj_tokens, obj_entropy) in zip(batch, results):
                    if obj_tokens == 0:
                        file_num_skipped_short += 1
                    else:
                        file_total_nll += obj_nll
                        file_total_tokens += obj_tokens

                    record = {
                        "source_file": file_path.name,
                        "index": obj["index"],
                        "num_tokens": obj_tokens,
                        "total_nll": obj_nll,
                        "entropy_per_token": obj_entropy,
                    }

                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    file_num_objects += 1

        file_info_density = (
            file_total_nll / file_total_tokens
            if file_total_tokens > 0
            else 0.0
        )

        file_stat = {
            "file_name": file_path.name,
            "num_objects": file_num_objects,
            "num_skipped_short": file_num_skipped_short,
            "total_tokens": file_total_tokens,
            "total_nll": file_total_nll,
            "info_density_nats_per_token": file_info_density,
        }

        per_file_stats.append(file_stat)

        agg_total_nll += file_total_nll
        agg_total_tokens += file_total_tokens
        agg_num_objects += file_num_objects
        agg_num_skipped_short += file_num_skipped_short

        logger.info(
            "File %s done. Objects: %d, info density: %.6f",
            file_path.name,
            file_num_objects,
            file_info_density,
        )

    global_info_density = (
        agg_total_nll / agg_total_tokens
        if agg_total_tokens > 0
        else 0.0
    )

    global_stats = {
        "model_name": model_name,
        "device": device,
        "batch_size": batch_size,
        "max_length": max_length,
        "num_files": len(files),
        "num_objects": agg_num_objects,
        "num_skipped_short": agg_num_skipped_short,
        "total_tokens": agg_total_tokens,
        "total_nll": agg_total_nll,
        "info_density_nats_per_token": global_info_density,
        "per_file": per_file_stats,
    }

    stats_path.write_text(
        json.dumps(global_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Global info density: %.6f", global_info_density)

    return global_stats


# ============================================================
# ЗАДАНИЕ 3.3.3 Удаление дубликатов и объектов с высокой или низкой энтропией
# ============================================================
def filter_single_file(
    original_path: Path, 
    metrics_path: Path, 
    output_path: Path, 
    text_field: str = "text",
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
):
    """
    Фильтрует один JSONL файл на основе метрик энтропии и дубликатов.
    """
    # 1. Загружаем метрики энтропии и вычисляем пороги (границы)
    logger.info(f"Загрузка метрик из {metrics_path.name}...")
    entropies = []
    metrics_map = {}

    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj.get("index")
            ent = obj.get("entropy_per_token", 0.0)
            metrics_map[idx] = ent
            entropies.append(ent)

    if not entropies:
        logger.error(f"В файле {metrics_path.name} нет метрик!")
        return None

    # Вычисляем границы (нижний и верхний процентиль)
    low_thr = np.percentile(entropies, lower_percentile)
    high_thr = np.percentile(entropies, upper_percentile)
    
    logger.info(f"Установлены границы энтропии: [{low_thr:.4f}, {high_thr:.4f}]")

    # 2. Читаем исходный датасет и фильтруем его
    seen_hashes = set()
    stats = {
        "total_valid_objects": 0,
        "kept": 0,
        "removed_duplicates": 0,
        "removed_low_entropy": 0,
        "removed_high_entropy": 0,
        "skipped_empty_or_invalid": 0
    }

    index = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with original_path.open("r", encoding="utf-8") as f_in, \
         output_path.open("w", encoding="utf-8") as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["skipped_empty_or_invalid"] += 1
                continue

            # Логика извлечения текста должна ТОЧНО совпадать с read_jsonl из прошлой части,
            # чтобы индексы совпали!
            if isinstance(obj, str):
                text = obj
                save_obj = {"text": obj} # Сохраняем в едином формате
            elif isinstance(obj, dict):
                text = obj.get(text_field, "")
                save_obj = obj # Сохраняем исходный объект со всеми метаданными
            else:
                stats["skipped_empty_or_invalid"] += 1
                continue

            if not isinstance(text, str):
                text = str(text)

            text = text.strip()
            if not text:
                stats["skipped_empty_or_invalid"] += 1
                continue

            stats["total_valid_objects"] += 1

            # Проверка на дубликаты
            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            if text_hash in seen_hashes:
                stats["removed_duplicates"] += 1
                index += 1
                continue
            
            # Добавляем в множество уникальных
            seen_hashes.add(text_hash)

            # Проверка на энтропию
            ent = metrics_map.get(index)
            if ent is None:
                logger.warning(f"Индекс {index} не найден в метриках. Пропускаем.")
                index += 1
                continue

            if ent < low_thr:
                stats["removed_low_entropy"] += 1
            elif ent > high_thr:
                stats["removed_high_entropy"] += 1
            else:
                # Текст уникален и энтропия в норме -> записываем в чистый датасет
                f_out.write(json.dumps(save_obj, ensure_ascii=False) + "\n")
                stats["kept"] += 1

            index += 1

    logger.info(f"Файл {original_path.name} обработан.")
    logger.info(f"Статистика: {stats}")
    return stats


def filter_dataset_dir(
    original_dir, 
    metrics_dir, 
    output_dir, 
    pattern: str = "*.jsonl",
    text_field: str = "text",
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
):
    """
    Фильтрует все файлы в папке.
    """
    original_dir = Path(original_dir)
    metrics_dir = Path(metrics_dir)
    output_dir = Path(output_dir)

    files = sorted(original_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Файлы не найдены в {original_dir}")

    global_stats = {
        "total_valid_objects": 0,
        "kept": 0,
        "removed_duplicates": 0,
        "removed_low_entropy": 0,
        "removed_high_entropy": 0,
        "skipped_empty_or_invalid": 0
    }

    for file_path in files:
        # Предполагается, что файлы с метриками имеют те же имена, что и исходные
        metrics_path = metrics_dir / file_path.name
        output_path = output_dir / file_path.name

        if not metrics_path.exists():
            logger.warning(f"Файл метрик {metrics_path} не найден. Пропуск файла {file_path.name}.")
            continue

        file_stats = filter_single_file(
            original_path=file_path,
            metrics_path=metrics_path,
            output_path=output_path,
            text_field=text_field,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile
        )

        if file_stats:
            for key in global_stats:
                global_stats[key] += file_stats[key]

    logger.info("=" * 40)
    logger.info("ГЛОБАЛЬНАЯ СТАТИСТИКА ОЧИСТКИ:")
    for key, value in global_stats.items():
        logger.info(f"{key}: {value}")
    
    # Сохраняем статистику очистки в JSON
    stats_path = output_dir / "filtering_stats.json"
    stats_path.write_text(json.dumps(global_stats, indent=2), encoding="utf-8")
    
    return global_stats