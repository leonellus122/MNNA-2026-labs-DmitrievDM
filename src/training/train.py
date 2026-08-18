import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from clearml import Task
from omegaconf import OmegaConf
from dotenv import load_dotenv
from pathlib import Path
import argparse

from src.data.wikitext_datamodule import WikiTextDataModule
from src.training.lightning_module import GPTLightningModule


def main():
    # Загружаем переменные окружения из .env
    load_dotenv()
    
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(description="Train GPT-like language model")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    config = OmegaConf.load(args.config)
    
    # Разрешаем переменные окружения в конфиге
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)
    
    # Инициализируем ClearML Task
    task = Task.init(
        project_name="GPT-Language-Model",
        task_name="Training",
        config=config.to_container()
    )
    
    # Создаем DataModule
    data_module = WikiTextDataModule(
        data_dir=config.paths.data_dir,
        tokenizer_path=config.paths.tokenizer_path,
        max_length=config.data.max_length,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers
    )
    
    # Создаем LightningModule
    model = GPTLightningModule(config)
    
    # Настраиваем callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.paths.checkpoint_dir,
        filename="best-model-{epoch:02d}-{val_perplexity:.2f}",
        monitor="val_perplexity",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Настраиваем логгеры
    tensorboard_logger = TensorBoardLogger(
        save_dir="logs",
        name="gpt_training"
    )
    
    # Настраиваем Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        gradient_clip_val=config.training.gradient_clip_val,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tensorboard_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=True
    )
    
    # Запускаем обучение
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.fit(model, datamodule=data_module, ckpt_path=args.resume)
    else:
        trainer.fit(model, datamodule=data_module)
    
    # Сохраняем лучший чекпоинт
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    
    # Завершаем ClearML Task
    task.close()


if __name__ == "__main__":
    main()