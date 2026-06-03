"""
模型训练与评估模块
集成全面的指标收集、特征提取和性能监控
"""

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
)
import data
import os
import config
import time
import torch
import pickle
import numpy as np
import glob
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
from datasets import config as dconfig

os.environ["HF_ENDPOINT"] = config.hf_endpoint
os.environ["CUDA_LAUNCH_BLOCKING"] = config.cuda_launch_blocking

# 导入注意力机制相关
try:
    from atten import setup_model_with_attention_and_lora

    ATTEN_AVAILABLE = True
except ImportError:
    ATTEN_AVAILABLE = False
    print(
        "Warning: atten.py not found. Attention mechanisms and LoRA will not be available."
    )


def initialize_model(model_url, num_labels, apply_atten_lora=True):
    """初始化模型，可选择应用注意力机制和LoRA"""
    tokenizer = AutoTokenizer.from_pretrained(model_url)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_url, num_labels=num_labels, ignore_mismatched_sizes=True
    )

    # 应用注意力机制和LoRA
    if apply_atten_lora and ATTEN_AVAILABLE:
        model = setup_model_with_attention_and_lora(model)

    return model, tokenizer


def prepare_dataset(datasets, tokenizer, textCol, batch_size=None):
    """准备数据集，使用config中的配置"""
    if batch_size is None:
        batch_size = config.batch_size

    def tokenize_function(examples):
        return tokenizer(
            examples[textCol],
            truncation=True,
            max_length=config.max_length,
            padding=False,  # DataCollator会处理padding
        )

    tokenized_datasets = datasets.map(
        tokenize_function, batched=True, batch_size=batch_size, remove_columns=[textCol]
    )

    # 过滤无效标签
    tokenized_datasets = tokenized_datasets.filter(lambda x: x["label"] in [0, 1])

    tokenized_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
        output_all_columns=False,
    )

    dconfig.USE_MEMORY_MAPPING = False

    data_collator = DataCollatorWithPadding(
        tokenizer, padding=True, return_tensors="pt"
    )
    return tokenized_datasets, data_collator


def compute_metrics(p):
    """计算全面的评估指标"""
    preds = p.predictions.argmax(-1)
    labels = p.label_ids

    # 基础指标
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    precision = precision_score(labels, preds, average="binary", zero_division=0)
    recall = recall_score(labels, preds, average="binary", zero_division=0)

    # 高级指标
    try:
        # AUC需要概率值
        probabilities = p.predictions[:, 1] if p.predictions.ndim > 1 else None
        auc = roc_auc_score(labels, probabilities) if probabilities is not None else 0.0
        mcc = matthews_corrcoef(labels, preds)
    except Exception as e:
        print(f"Warning: Could not compute AUC or MCC: {e}")
        auc = 0.0
        mcc = 0.0

    # 混淆矩阵
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "mcc": mcc,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


class PerformanceMonitor(TrainerCallback):
    """性能监控回调，收集训练过程中的各种指标"""

    def __init__(self):
        super().__init__()
        self.train_start = 0
        self.epoch_start = 0
        self.total_tokens = 0
        self.loss_history = []
        self.train_metrics_history = []
        self.eval_metrics_history = []
        self.learning_rates = []
        self.gradient_norms = []
        self.peak_memory = 0
        self.batch_times = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f"训练开始，共 {state.max_steps} 步")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
        print(f"\n开始 Epoch {state.epoch + 1}/{args.num_train_epochs}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # 累计 token 数
        bs = args.per_device_train_batch_size
        seq_len = config.max_length
        self.total_tokens += bs * seq_len

        # 记录学习率
        if hasattr(state, "learning_rate"):
            self.learning_rates.append(state.learning_rate)

        # 记录batch时间
        if hasattr(self, "_step_start"):
            batch_time = time.time() - self._step_start
            self.batch_times.append(batch_time)

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1024**3
            self.peak_memory = max(self.peak_memory, mem)
            gpu_info = f"| GPU显存: {mem:.2f}GB"
        else:
            gpu_info = ""

        print(f"Epoch {state.epoch:.0f} 完成 | 耗时: {epoch_time:.1f}s {gpu_info}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            eval_entry = {
                "step": state.global_step,
                "epoch": state.epoch,
            }
            # 分离训练集和验证集指标
            train_metrics = {}
            eval_metrics = {}

            for k, v in metrics.items():
                if k.startswith("eval_"):
                    eval_metrics[k.replace("eval_", "")] = v
                elif k.startswith("train_"):
                    train_metrics[k.replace("train_", "")] = v

            if eval_metrics:
                self.eval_metrics_history.append({**eval_entry, **eval_metrics})
            if train_metrics:
                self.train_metrics_history.append({**eval_entry, **train_metrics})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # 记录损失
            if "loss" in logs:
                self.loss_history.append(
                    {
                        "step": state.global_step,
                        "epoch": state.epoch,
                        "loss": logs["loss"],
                    }
                )
            # 记录学习率
            if "learning_rate" in logs:
                self.learning_rates.append(logs["learning_rate"])


def train(model, tokenized_dataset, data_collator, training_args=None):
    """训练模型，收集全面的训练特征"""
    if training_args is None:
        training_args = config.training_args

    monitor = PerformanceMonitor()

    # 准备训练集和验证集
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset.get("validation", tokenized_dataset.get("test"))

    if eval_dataset is None:
        print("Warning: No validation dataset found. Using test dataset.")
        eval_dataset = tokenized_dataset["test"]

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[monitor],
    )

    # 检查是否有检查点可以恢复训练
    checkpoint_path = None
    if os.path.exists(training_args.output_dir):
        import glob

        checkpoint_dirs = glob.glob(
            os.path.join(training_args.output_dir, "checkpoint-*")
        )
        if checkpoint_dirs:
            # 找到最新的检查点（按步数排序）
            checkpoint_dirs.sort(key=lambda x: int(os.path.basename(x).split("-")[1]))
            checkpoint_path = checkpoint_dirs[-1]
            print(f"\n发现检查点: {checkpoint_path}")
            print(f"将从该检查点恢复训练...")

    print("\n开始训练...")
    if checkpoint_path:
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        trainer.train()

    # 在训练集和验证集上分别评估
    print("\n在训练集上评估...")
    train_results = trainer.evaluate(train_dataset, metric_key_prefix="train_")

    print("在验证集上评估...")
    eval_results = trainer.evaluate(eval_dataset, metric_key_prefix="eval_")

    # 收集所有训练特征
    training_features = {
        "model_info": {
            "model_type": type(model).__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        },
        "training_config": {
            "num_train_epochs": training_args.num_train_epochs,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "warmup_ratio": training_args.warmup_ratio,
        },
        "train_results": train_results,
        "eval_results": eval_results,
        "loss_history": monitor.loss_history,
        "train_metrics_history": monitor.train_metrics_history,
        "eval_metrics_history": monitor.eval_metrics_history,
        "learning_rates": monitor.learning_rates,
        "peak_memory": monitor.peak_memory,
        "total_tokens": monitor.total_tokens,
        "train_start_time": monitor.train_start,
        "total_train_time": time.time() - monitor.train_start,
        "batch_times": monitor.batch_times,
    }

    return trainer.model, eval_results, monitor, training_features


def save_training_features(training_features, results_path):
    """保存训练特征到文件"""
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "wb") as f:
        pickle.dump(training_features, f)
    print(f"训练特征已保存到: {results_path}")

    # 打印关键指标对比
    print("\n" + "=" * 60)
    print("训练集 vs 验证集指标对比")
    print("=" * 60)

    train_metrics = {
        k.replace("train_", ""): v
        for k, v in training_features["train_results"].items()
        if k.startswith("train_")
    }
    eval_metrics = {
        k.replace("eval_", ""): v
        for k, v in training_features["eval_results"].items()
        if k.startswith("eval_")
    }

    metrics_to_compare = ["accuracy", "f1", "precision", "recall", "auc", "mcc"]

    print(f"{'指标':<15} {'训练集':<15} {'验证集':<15} {'差异':<15}")
    print("-" * 60)

    for metric in metrics_to_compare:
        train_val = train_metrics.get(metric, "N/A")
        eval_val = eval_metrics.get(metric, "N/A")

        if train_val != "N/A" and eval_val != "N/A":
            diff = train_val - eval_val
            diff_str = f"{diff:+.4f}"
        else:
            diff_str = "N/A"

        print(f"{metric:<15} {str(train_val):<15} {str(eval_val):<15} {diff_str:<15}")

    print("=" * 60)
