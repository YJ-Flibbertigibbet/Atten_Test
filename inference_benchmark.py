"""
Inference Benchmark and Feature Collection Script
Supports all attention mechanisms (MHA, GQA, MGA, MQA)
"""

import os
import sys
import time
import torch
import numpy as np
import pickle
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig
from datasets import load_dataset
import config
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    average_precision_score,
)
from collections import defaultdict
from visualization_new import generate_visualizations, save_data_to_csv

# 设置环境
os.environ["HF_ENDPOINT"] = config.hf_endpoint
os.environ["CUDA_LAUNCH_BLOCKING"] = config.cuda_launch_blocking


def detect_attention_type(checkpoint_path):
    """Detect attention type from path or config"""
    path_str = checkpoint_path.lower()
    if "gqa" in path_str:
        return "GQA"
    elif "mga" in path_str:
        return "MGA"
    elif "mqa" in path_str:
        return "MQA"
    elif "mha" in path_str:
        return "MHA"
    else:
        return "MHA"


def detect_model_type(checkpoint_path):
    """Detect model type (BERT or RoBERTa) from path"""
    path_str = checkpoint_path.lower()
    if "roberta" in path_str:
        return "roberta"
    elif "bert" in path_str:
        return "bert"
    else:
        return "bert"


def load_model_for_inference(checkpoint_path, attention_type=None):
    """Load model for inference based on attention type and model type"""
    print(f"\nLoading model from: {checkpoint_path}")

    # Detect attention type
    if attention_type is None:
        attention_type = detect_attention_type(checkpoint_path)
    print(f"Detected attention type: {attention_type}")

    # Detect model type
    model_type = detect_model_type(checkpoint_path)
    print(f"Detected model type: {model_type}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    print("Tokenizer loaded successfully")

    # Load base model config based on model type
    if model_type == "roberta":
        from transformers import RobertaConfig
        model_config = RobertaConfig.from_pretrained(config.roberta_url)
        base_model_url = config.roberta_url
    else:
        model_config = BertConfig.from_pretrained(config.bert_url)
        base_model_url = config.bert_url

    model_config.num_labels = config.num_labels
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_url, config=model_config, ignore_mismatched_sizes=True
    )

    # Replace attention mechanism if needed
    if attention_type != "MHA":
        print(f"Replacing attention mechanism with {attention_type}...")
        from atten import replace_attention

        model = replace_attention(model, attention_type, config.num_groups)

    # Load trained weights
    checkpoint_file = os.path.join(checkpoint_path, "model.safetensors")
    if not os.path.exists(checkpoint_file):
        checkpoint_file = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(checkpoint_file):
        print(f"从 {checkpoint_file} 加载权重...")
        if checkpoint_file.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(checkpoint_file)
        else:
            state_dict = torch.load(checkpoint_file, map_location="cpu")

        # 加载权重（允许部分匹配）
        model.load_state_dict(state_dict, strict=False)
        print("权重加载完成")
    else:
        print("警告: 未找到权重文件，使用预训练权重")

    return model, tokenizer


def run_inference(model, tokenizer, device, num_samples=None):
    """运行推理测试"""
    max_length = config.max_length
    batch_size = config.inference_batch_size

    # 加载数据集
    print("加载测试数据集...")
    dataset = load_dataset(config.sst2_dataset_name, config.sst2_dataset_config)
    # Use validation set instead of test set (test set has no labels)
    test_dataset = dataset["validation"]

    if num_samples and num_samples < len(test_dataset):
        test_dataset = test_dataset.shuffle(seed=42).select(range(num_samples))

    # 分词处理
    def tokenize_batch(examples):
        return tokenizer(
            examples[config.sst2_text_col],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    test_dataset = test_dataset.map(tokenize_batch, batched=True)
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )

    # 初始化特征收集
    inference_features = {
        "model_info": {
            "model_type": type(model).__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        },
        "dataset_info": {
            "dataset_name": config.sst2_dataset_name,
            "num_samples": len(test_dataset),
            "max_length": max_length,
        },
        "performance_metrics": {},
        "predictions": [],
        "true_labels": [],
        "probabilities": [],
        "inference_times": [],
        "token_counts": [],
        "per_sample_metrics": defaultdict(list),
        "confusion_matrix": None,
        "classification_report": None,
    }

    total_tokens = 0
    total_samples = len(test_dataset)
    inference_times = []
    all_predictions = []
    all_true_labels = []
    all_probabilities = []

    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print(f"\n开始推理测试，共 {total_samples} 个样本...")
    start_time = time.time()

    # 批量推理
    for i in range(0, total_samples, batch_size):
        batch_end = min(i + batch_size, total_samples)
        batch = test_dataset[i:batch_end]

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].numpy()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_start = time.time()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_time = time.time() - batch_start

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        predictions = logits.argmax(dim=-1).cpu().numpy()

        all_predictions.extend(predictions)
        all_true_labels.extend(labels)
        all_probabilities.extend(probabilities)
        inference_times.append(batch_time)

        batch_tokens = (input_ids != tokenizer.pad_token_id).sum().item()
        total_tokens += batch_tokens

        for j in range(len(predictions)):
            inference_features["per_sample_metrics"]["correct"].append(
                predictions[j] == labels[j]
            )
            inference_features["per_sample_metrics"]["confidence"].append(
                np.max(probabilities[j])
            )

    total_time = time.time() - start_time
    peak_memory = (
        torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    )

    # 计算评估指标
    accuracy = accuracy_score(all_true_labels, all_predictions)

    # 检测是否为二分类任务
    unique_labels = np.unique(all_true_labels)
    is_binary = len(unique_labels) <= 2
    avg_method = "binary" if is_binary else "weighted"

    f1 = f1_score(all_true_labels, all_predictions, average=avg_method, zero_division=0)
    precision = precision_score(all_true_labels, all_predictions, average=avg_method, zero_division=0)
    recall = recall_score(all_true_labels, all_predictions, average=avg_method, zero_division=0)
    mcc = matthews_corrcoef(all_true_labels, all_predictions)

    # 计算AUC和AP（仅对二分类有效）
    if is_binary:
        positive_probs = [p[1] if len(p) > 1 else p[0] for p in all_probabilities]
        try:
            auc = roc_auc_score(all_true_labels, positive_probs)
            ap = average_precision_score(all_true_labels, positive_probs)
        except ValueError:
            auc = 0.0
            ap = 0.0
    else:
        # 多分类使用macro平均
        try:
            auc = roc_auc_score(all_true_labels, all_probabilities, multi_class='ovr', average='weighted')
            ap = 0.0  # AP不适用于多分类
        except ValueError:
            auc = 0.0
            ap = 0.0
    cm = confusion_matrix(all_true_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    cls_report = classification_report(
        all_true_labels, all_predictions, output_dict=True
    )

    inference_features["predictions"] = all_predictions
    inference_features["true_labels"] = all_true_labels
    inference_features["probabilities"] = all_probabilities
    inference_features["inference_times"] = inference_times
    inference_features["token_counts"] = total_tokens

    inference_features["performance_metrics"] = {
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "avg_time_per_sample": total_time / total_samples,
        "avg_time_per_batch": np.mean(inference_times),
        "throughput_samples": total_samples / total_time,
        "throughput_tokens": total_tokens / total_time,
        "peak_memory_gb": peak_memory,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
        "auc": auc,
        "average_precision": ap,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }

    inference_features["confusion_matrix"] = cm.tolist()
    inference_features["classification_report"] = cls_report

    inference_features["detailed_stats"] = {
        "prediction_confidence_mean": float(
            np.mean([np.max(p) for p in all_probabilities])
        ),
        "prediction_confidence_std": float(
            np.std([np.max(p) for p in all_probabilities])
        ),
        "inference_time_mean": float(np.mean(inference_times)),
        "inference_time_std": float(np.std(inference_times)),
        "inference_time_per_token": float(total_time / total_tokens),
    }

    # 打印结果
    print("\n" + "=" * 60)
    print("推理性能指标")
    print("=" * 60)
    print(f"总样本数: {total_samples}")
    print(f"总Token数: {total_tokens}")
    print(f"总推理时间: {total_time:.2f}s")
    print(f"平均每样本耗时: {total_time / total_samples * 1000:.2f}ms")
    print(f"吞吐量 (样本/秒): {total_samples / total_time:.2f}")
    if torch.cuda.is_available():
        print(f"峰值GPU显存: {peak_memory:.2f}GB")
    print("-" * 60)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print("=" * 60)

    return inference_features


def save_results(inference_features, output_path):
    """Save inference results"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(inference_features, f)
    print(f"\nInference results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="推理测试与评估")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint路径（如果未指定，将自动测试所有可用的checkpoint）",
    )
    parser.add_argument(
        "--attention",
        type=str,
        default=None,
        choices=["MHA", "MGA", "GQA", "MQA"],
        help="注意力类型（如果未指定，将从路径自动检测）",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="测试样本数（默认使用config中的设置）",
    )
    args = parser.parse_args()

    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("已清理CUDA缓存")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.use_gpu else "cpu"
    )
    print(f"使用设备: {device}")

    # 确定要测试的checkpoints
    if args.checkpoint:
        checkpoints = [(args.checkpoint, args.attention)]
    else:
        # 自动查找所有checkpoints
        checkpoints = []
        res_dir = "/workspace/task1/res"
        results_dir = "/workspace/results/checkpoints"

        # 从res目录查找
        if os.path.exists(res_dir):
            for model_dir in os.listdir(res_dir):
                checkpoint_path = os.path.join(res_dir, model_dir, "checkpoint")
                if os.path.exists(checkpoint_path):
                    checkpoints.append((checkpoint_path, None))

        # 从results/checkpoints目录查找
        if os.path.exists(results_dir):
            for root, dirs, files in os.walk(results_dir):
                if "model.safetensors" in files or "pytorch_model.bin" in files:
                    checkpoints.append((root, None))

        if not checkpoints:
            print("未找到任何checkpoint！")
            return

        print(f"找到 {len(checkpoints)} 个checkpoint:")
        for cp, _ in checkpoints:
            print(f"  - {cp}")

    # 批量测试所有checkpoints
    all_results = {}

    for checkpoint_path, attention_type in checkpoints:
        print("\n" + "=" * 60)
        print(f"测试checkpoint: {checkpoint_path}")
        print("=" * 60)

        try:
            # 加载模型
            model, tokenizer = load_model_for_inference(checkpoint_path, attention_type)
            model = model.to(device)
            model.eval()

            # 运行推理
            detected_attention = attention_type or detect_attention_type(
                checkpoint_path
            )
            inference_features = run_inference(
                model, tokenizer, device, num_samples=args.num_samples
            )
            inference_features["attention_type"] = detected_attention
            inference_features["checkpoint_path"] = checkpoint_path

            # 保存结果
            result_name = f"inference_{detected_attention}_{os.path.basename(os.path.dirname(checkpoint_path))}"
            output_path = os.path.join(
                config.inference_results_dir, f"{result_name}.pkl"
            )
            save_results(inference_features, output_path)

            # 生成可视化
            vis_dir = os.path.join(config.visualization_dir, result_name)
            generate_visualizations(
                inference_features, vis_dir, prefix=f"{detected_attention}_"
            )
      # Save CSV data
            save_data_to_csv(
        inference_features, vis_dir, prefix=f"{detected_attention}_"
            )

            all_results[checkpoint_path] = inference_features

            # 清理显存
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception as e:
            print(f"测试失败: {e}")
            import traceback

            traceback.print_exc()

            # 清理CUDA状态
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    # 汇总所有结果
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("所有checkpoint的测试结果汇总")
        print("=" * 60)
        print(f"{'Checkpoint':<40} {'Accuracy':<10} {'F1':<10} {'AUC':<10}")
        print("-" * 70)
        for cp, features in all_results.items():
            metrics = features["performance_metrics"]
            att_type = features.get("attention_type", "Unknown")
            print(
                f"{att_type:<40} {metrics['accuracy']:<10.4f} {metrics['f1_score']:<10.4f} {metrics['auc']:<10.4f}"
            )

    print("\n所有测试完成!")


if __name__ == "__main__":
    main()
