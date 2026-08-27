import json
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
from src.tokenization.tokenizers import load_bpe_tokenizer
from omegaconf import DictConfig


def create_packed_batches(texts, tokenizer, max_length=512):
    """
    Packed batching с локальной маской:
    0 — PAD, 1 — первый объект, 2 — второй объект и т.д.
    """
    packs = []

    current_input_ids = []
    current_mask = []
    segment_id = 1  # локальный ID внутри текущего пака

    pad_id = tokenizer.token_to_id("<pad>")
    eos_id = tokenizer.token_to_id("<eos>")

    for text in texts:
        ids = tokenizer.encode(text).ids

        if eos_id is not None:
            ids.append(eos_id)

        current_input_ids.extend(ids)
        current_mask.extend([segment_id] * len(ids))
        segment_id += 1

        # Набиваем полные паки
        while len(current_input_ids) >= max_length:
            pack_input_ids = current_input_ids[:max_length]
            pack_mask = current_mask[:max_length]

            # Нормализуем маску: 1, 2, 3...
            # (на случай, если в пак попали куски одного и того же документа)
            unique_ids = sorted(set(pack_mask))
            id_map = {old: new for new, old in enumerate(unique_ids, start=1)}
            pack_mask = [id_map[m] for m in pack_mask]

            packs.append({
                "input_ids": pack_input_ids,
                "mask": pack_mask
            })

            # Остаток переносим в новый пак
            remaining_ids = current_input_ids[max_length:]
            remaining_mask = current_mask[max_length:]

            # Сбрасываем segment_id для нового пака
            # Остаток принадлежит тому же документу, что и конец предыдущего пака
            current_input_ids = remaining_ids
            current_mask = remaining_mask
            segment_id = max(id_map.values()) + 1 if remaining_mask else 1

    # Последний неполный пак — паддим до max_length
    if len(current_input_ids) > 0:
        pad_len = max_length - len(current_input_ids)

        # Нормализуем маску перед паддингом
        unique_ids = sorted(set(current_mask))
        id_map = {old: new for new, old in enumerate(unique_ids, start=1)}
        current_mask = [id_map[m] for m in current_mask]

        pack_input_ids = current_input_ids + [pad_id] * pad_len
        pack_mask = current_mask + [0] * pad_len  # 0 для PAD

        packs.append({
            "input_ids": pack_input_ids,
            "mask": pack_mask
        })

    return packs


class PackedWikiTextDataset(Dataset):
    def __init__(self, packs):
        self.packs = packs
        
    def __len__(self):
        return len(self.packs)
        
    def __getitem__(self, idx):
        pack = self.packs[idx]
        return {
            "input_ids": torch.tensor(pack["input_ids"], dtype=torch.long),
            "mask": torch.tensor(pack["mask"], dtype=torch.long),
        }


class WikiTextDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.save_hyperparameters(config)

        self.data_dir = Path(config.paths.data_dir)
        self.tokenizer_path = Path(config.paths.tokenizer_path)
        self.max_length = config.data.max_length
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        
        # Создаем модель
        
    def setup(self, stage=None):
        # Загружаем BPE, обученный на Common Crawl!
        self.tokenizer = load_bpe_tokenizer(self.tokenizer_path)
        
        self.train_packs = self._pack_split("train.jsonl")
        self.val_packs = self._pack_split("validation.jsonl")
        
    def _pack_split(self, filename):
        texts = []
        file_path = self.data_dir / filename
        if not file_path.exists():
            return []
            
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("text"):
                    texts.append(obj["text"])
                    
        return create_packed_batches(texts, self.tokenizer, self.max_length)
        
    def train_dataloader(self):
        dataset = PackedWikiTextDataset(self.train_packs)
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def val_dataloader(self):
        dataset = PackedWikiTextDataset(self.val_packs)
        return DataLoader(
            dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )