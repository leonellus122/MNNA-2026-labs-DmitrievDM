import yaml
import logging
import requests
from pathlib import Path
from urllib.parse import urljoin
from tqdm.notebook import tqdm # Для красивых виджетов

from src.utils import load_config
# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def download_warc_files(config_path: str, raw_data_dir: str) -> None:
    """
    Скачивает WARC файлы на основе конфига.
    """
    config = load_config(config_path)
    
    cc_config = config.get('common_crawl', {})

    data_server = cc_config.get('data_server', 'https://data.commoncrawl.org/')
    user_agent = cc_config.get('user_agent', 'Mozilla/5.0 (compatible; Lab1/1.0)')
    warc_paths = cc_config.get('warc_paths', [])

    raw_data_dir = Path(raw_data_dir)
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Директория для сохранения: {raw_data_dir.resolve()}")

    headers = {'User-Agent': user_agent}

    # tqdm.notebook для  прогресс-бара в Jupyter
    for rel_path in tqdm(warc_paths, desc="Общий прогресс"):
        full_url = urljoin(data_server, rel_path)
        filename = full_url.split('/')[-1]
        save_path = raw_data_dir / filename

        if save_path.exists():
            logger.info(f"Файл {filename} уже существует. Пропускаем.")
            continue

        logger.info(f"Начинаем скачивание: {filename}")

        try:
            with requests.get(full_url, headers=headers, stream=True, timeout=30) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename, leave=False) as pbar:
                    with open(save_path, 'wb') as file:
                        for chunk in response.iter_content(chunk_size=1048576): # 1 MB
                            if chunk:
                                size = file.write(chunk)
                                pbar.update(size)
                                
                logger.info(f"✅ Успешно скачано: {save_path.name}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка сети при скачивании {full_url}: {e}")
            if save_path.exists():
                save_path.unlink()
                
        except KeyboardInterrupt:
            logger.warning("Скачивание прервано. Удаляем битый файл...")
            if save_path.exists():
                save_path.unlink()
            break