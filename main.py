"""
主训练脚本
支持多种模型、数据集和注意力机制配置
"""

from modelProcess import (
    train,
    prepare_dataset,
    initialize_model,
    save_training_features,
)
import config
import data
import os
import pickle
import numpy as np
import argparse

os.environ["CUDA_LAUNCH_BLOCKING"] = config.cuda_launch_blocking
os.environ["HF_ENDPOINT"] = config.hf_endpoint


def run(
    model_url=None,
    num_labels=None,
    dataset=None,
    textCol=None,
    batch_size=None,
    training_args=None,
    results_path=None,
    dataset_name="sst2",
):
    """
    运行训练流程

    参数:
        model_url: 模型URL或路径
        num_labels: 分类标签数
        dataset: 数据集
        textCol: 文本列名
        batch_size: 批次大小
        training_args: 训练参数
        results_path: 结果保存路径
        dataset_name: 数据集名称 (sst2, yelp, chnsenticorp)
    """
    # 设置默认值
    if model_url is None:
        model_url = config.bert_url
    if num_labels is None:
        num_labels = config.num_labels
    if batch_size is None:
        batch_size = config.batch_size

    # 根据数据集名称选择数据集
    if dataset is None:
        if dataset_name == "sst2":
            dataset = data.sst2_dataset
            textCol = data.sst2_textCol
        elif dataset_name == "yelp":
            dataset = data.yelp_dataset
            textCol = data.yelp_textCol
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    if training_args is None:
        if "bert" in model_url.lower():
            training_args = config.bertTraining_args
            if results_path is None:
                results_path = config.bert_train_results
        elif "roberta" in model_url.lower():
            training_args = config.robertaTraining_args
            if results_path is None:
                results_path = config.roberta_train_results
        elif "electra" in model_url.lower():
            training_args = config.electraTraining_args
            if results_path is None:
                results_path = config.electra_train_results
        else:
            training_args = config.training_args
            if results_path is None:
                results_path = config.results_dir + "/train_results.pkl"

    print("=" * 60)
    print("开始训练流程")
    print("=" * 60)
    print(f"模型: {model_url}")
    print(f"数据集: {dataset_name}")
    print(f"批次大小: {batch_size}")
    print(f"最大序列长度: {config.max_length}")
    print(f"注意力机制: {config.attention_type}")
    print(f"使用LoRA: {config.use_lora}")
    if config.use_lora:
        print(f"PEFT方法: {config.peft_method}")
    print("=" * 60)

    # 初始化模型（包含注意力机制和LoRA）
    model, tokenizer = initialize_model(model_url, num_labels, apply_atten_lora=True)

    # 准备数据集
    print("\n准备数据集...")
    tokenized_dataset, dataCollator = prepare_dataset(
        dataset, tokenizer, textCol, batch_size
    )

    print(f"训练集大小: {len(tokenized_dataset['train'])}")
    if "validation" in tokenized_dataset:
        print(f"验证集大小: {len(tokenized_dataset['validation'])}")
    if "test" in tokenized_dataset:
        print(f"测试集大小: {len(tokenized_dataset['test'])}")

    # 训练模型
    TrainModel, eval_results, monitor, training_features = train(
        model, tokenized_dataset, dataCollator, training_args
    )

    # 保存训练特征
    save_training_features(training_features, results_path)

    # 保存模型
    # 根据模型URL判断模型名称
    if "bert" in model_url.lower():
        model_name = "bert"
    elif "roberta" in model_url.lower():
        model_name = "roberta"
    elif "electra" in model_url.lower():
        model_name = "electra"
    else:
        model_name = "unknown"
    model_save_path = os.path.join(
        config.checkpoint_dir, dataset_name, model_name, config.attention_type
    )
    os.makedirs(model_save_path, exist_ok=True)
    TrainModel.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"\n模型已保存到: {model_save_path}")

    return TrainModel, eval_results, training_features


def run_all_models():
    """运行所有模型的训练：一个模型对应三种注意力机制"""
    # 只训练 bert 模型，用三种注意力机制
    model_url = config.bert_url
    model_name = "bert"

    # 三种注意力机制
    attention_types = ["MHA", "MGA", "GQA"]

    results = {}
    for attention_type in attention_types:
        print(f"\n{'#' * 60}")
        print(f"训练模型: {model_name}，注意力机制: {attention_type}")
        print(f"{'#' * 60}")

        # 设置注意力机制
        config.attention_type = attention_type

        # 为当前注意力机制生成独立的训练参数（输出目录区分）
        training_args = config.get_training_args(
            model_name, f"{model_name}_{attention_type}"
        )

        # 训练结果保存路径也加上注意力机制区分
        results_path = f"./results/{model_name}/train_results_{attention_type}.pkl"

        try:
            _, eval_results, training_features = run(
                model_url=model_url,
                training_args=training_args,
                results_path=results_path,
            )
            results[f"{model_name}_{attention_type}"] = {
                "eval_results": eval_results,
                "training_features": training_features,
                "attention_type": attention_type,
            }
        except Exception as e:
            print(f"训练 {model_name} 注意力机制 {attention_type} 时出错: {e}")
            results[f"{model_name}_{attention_type}"] = {"error": str(e)}

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练文本分类模型")
    parser.add_argument(
        "--model",
        type=str,
        default="bert",
        choices=["bert", "roberta", "electra", "all"],
        help="选择模型: bert, roberta, electra, 或 all (训练所有模型)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sst2",
        choices=["sst2", "yelp"],
        help="选择数据集: sst2 或 yelp",
    )
    parser.add_argument(
        "--attention",
        type=str,
        default=None,
        choices=["MHA", "MGA", "GQA", "MQA"],
        help="注意力机制类型",
    )
    parser.add_argument("--use_lora", action="store_true", help="是否使用LoRA")
    parser.add_argument(
        "--peft_method",
        type=str,
        default=None,
        choices=["lora", "prefix_tuning", "prompt_tuning"],
        help="PEFT方法",
    )

    args = parser.parse_args()

    # 更新配置
    if args.attention:
        config.attention_type = args.attention
    if args.use_lora:
        config.use_lora = True
    if args.peft_method:
        config.peft_method = args.peft_method

    if args.model == "all":
        results = run_all_models()
    else:
        model_map = {
            "bert": config.bert_url,
            "roberta": config.roberta_url,
            "electra": config.electra_url,
        }
        run(
            model_url=model_map[args.model],
            dataset_name=args.dataset,
        )
