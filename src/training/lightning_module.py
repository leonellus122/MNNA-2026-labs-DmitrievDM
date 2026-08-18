import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from omegaconf import DictConfig
from typing import Any, Dict

from src.models.gpt_model import GPTModel


class GPTLightningModule(pl.LightningModule):
    """
    LightningModule обертка для GPT модели.
    
    Реализует:
    - training_step: обучение с подсчетом loss и perplexity
    - validation_step: валидация с подсчетом loss и perplexity
    - configure_optimizers: AdamW + CosineAnnealingLR с warm-up
    - Подсчет метрик: loss, perplexity
    - Логирование норм градиентов (глобальная и локальная)
    """
    
    def __init__(self, config: DictConfig):
        """
        Args:
            config: конфигурация из YAML файла
        """
        super().__init__()
        
        # Сохраняем конфиг (важно для воспроизводимости)
        self.save_hyperparameters(config)
        
        # Создаем модель
        self.model = GPTModel(
            vocab_size=config.model.vocab_size,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_layers,
            d_ff=config.model.d_ff,
            max_len=config.model.max_len
        )
        
        # Для подсчета метрик
        self.train_loss_sum = 0.0
        self.train_count = 0
        self.val_loss_sum = 0.0
        self.val_count = 0
    
    def forward(self, input_ids: torch.Tensor, sequence_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass через модель.
        
        Args:
            input_ids: (batch_size, seq_len)
            sequence_ids: (batch_size, seq_len)
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        return self.model(input_ids, sequence_ids)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Один шаг обучения.
        
        Args:
            batch: словарь с 'input_ids' и 'mask' (sequence_ids)
            batch_idx: индекс батча
        
        Returns:
            loss: скаляр
        """
        input_ids = batch["input_ids"]
        sequence_ids = batch["mask"]  # В вашем коде mask - это sequence_ids
        
        # Вычисляем loss с учетом маски
        loss = self.model.compute_loss(input_ids, sequence_ids)
        
        # Вычисляем perplexity
        perplexity = torch.exp(loss)
        
        # Логируем метрики
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_perplexity", perplexity, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Для подсчета средней метрики за эпоху
        self.train_loss_sum += loss.item()
        self.train_count += 1
        
        return loss
    
    def on_before_optimizer_step(self, optimizer):
        """
        Вызывается перед шагом оптимизатора.
        Логирует нормы градиентов (глобальная и локальная).
        """
        # Вычисляем глобальную норму градиентов
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Логируем глобальную норму
        self.log("grad_norm/global", total_norm, on_step=True, on_epoch=False, prog_bar=False)
        
        # Логируем локальные нормы по слоям (опционально, можно отключить для экономии)
        # Раскомментируйте, если нужно отслеживать нормы по каждому слою
        # for name, p in self.named_parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.data.norm(2)
        #         self.log(f"grad_norm/{name}", param_norm.item(), on_step=True, on_epoch=False, prog_bar=False)
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Один шаг валидации.
        
        Args:
            batch: словарь с 'input_ids' и 'mask' (sequence_ids)
            batch_idx: индекс батча
        
        Returns:
            loss: скаляр
        """
        input_ids = batch["input_ids"]
        sequence_ids = batch["mask"]
        
        # Вычисляем loss с учетом маски
        loss = self.model.compute_loss(input_ids, sequence_ids)
        
        # Вычисляем perplexity
        perplexity = torch.exp(loss)
        
        # Логируем метрики
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_perplexity", perplexity, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Для подсчета средней метрики за эпоху
        self.val_loss_sum += loss.item()
        self.val_count += 1
        
        return loss
    
    def configure_optimizers(self):
        """
        Настройка оптимизатора и LR scheduler с warm-up.
        
        Returns:
            optimizer: AdamW
            scheduler: CosineAnnealingLR с warm-up
        """
        # Optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.training.learning_rate,
            betas=tuple(self.hparams.training.optimizer.betas),
            eps=self.hparams.training.optimizer.eps,
            weight_decay=self.hparams.training.weight_decay
        )
        
        # LR Scheduler с warm-up
        warmup_steps = self.hparams.training.warmup_steps
        max_steps = self.hparams.training.scheduler.T_max
        
        def lr_lambda(current_step):
            """
            Warm-up: линейное увеличение LR от 0 до target_lr
            После warm-up: cosine decay от target_lr до eta_min
            """
            if current_step < warmup_steps:
                # Linear warm-up
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                return max(
                    self.hparams.training.scheduler.eta_min / self.hparams.training.learning_rate,
                    0.5 * (1.0 + torch.cos(torch.tensor(3.141592653589793 * progress)))
                )
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Обновляем на каждом шаге
                "frequency": 1
            }
        }
    
    def on_train_epoch_end(self):
        """
        Вызывается в конце каждой тренировочной эпохи.
        Логирует средние метрики за эпоху.
        """
        if self.train_count > 0:
            avg_loss = self.train_loss_sum / self.train_count
            avg_perplexity = torch.exp(torch.tensor(avg_loss))
            
            self.log("train_loss_epoch", avg_loss, prog_bar=False)
            self.log("train_perplexity_epoch", avg_perplexity, prog_bar=False)
        
        # Сбрасываем счетчики
        self.train_loss_sum = 0.0
        self.train_count = 0
    
    def on_validation_epoch_end(self):
        """
        Вызывается в конце каждой валидационной эпохи.
        Логирует средние метрики за эпоху.
        """
        if self.val_count > 0:
            avg_loss = self.val_loss_sum / self.val_count
            avg_perplexity = torch.exp(torch.tensor(avg_loss))
            
            self.log("val_loss_epoch", avg_loss, prog_bar=False)
            self.log("val_perplexity_epoch", avg_perplexity, prog_bar=False)
        
        # Сбрасываем счетчики
        self.val_loss_sum = 0.0
        self.val_count = 0
    
    def generate(
        self,
        prompt: str,
        tokenizer,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        """
        Генерация текста из модели.
        
        Args:
            prompt: начальный текст
            tokenizer: токенизатор
            max_length: максимальная длина генерации
            temperature: температура для sampling
            top_k: top-k sampling
            top_p: top-p (nucleus) sampling
        
        Returns:
            сгенерированный текст
        """
        self.model.eval()
        
        # Токенизируем prompt
        input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=self.device)
        sequence_ids = torch.ones_like(input_ids)  # Все токены принадлежат одной последовательности
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.model(generated, sequence_ids)
                
                # Берем логиты для последней позиции
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[:, indices_to_remove] = float('-inf')
                
                # Sampling
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Добавляем токен к сгенерированной последовательности
                generated = torch.cat([generated, next_token], dim=1)
                sequence_ids = torch.cat([sequence_ids, torch.ones_like(next_token)], dim=1)
                
                # Проверяем, не достигли ли EOS
                if next_token.item() == tokenizer.token_to_id("<eos>"):
                    break
        
        # Декодируем текст
        generated_text = tokenizer.decode(generated[0].tolist())
        
        return generated_text