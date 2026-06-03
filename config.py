from transformers import TrainingArguments

# ==================== 模型配置 ====================
# 模型URL
bert_url = "google-bert/bert-large-uncased"
roberta_url = "roberta-base"
electra_url = "google/electra-base-discriminator"
llama1b_url = "facebook/Llama-1B-Human-Preference"

# 分类标签数
num_labels = 2

# ==================== 数据处理配置 ====================
# 基础数据处理参数
batch_size = 8
max_length = 256
num_workers = 4

# 数据集配置
sst2_dataset_name = "glue"
sst2_dataset_config = "sst2"
sst2_text_col = "sentence"
sst2_label_col = "label"

yelp_dataset_name = "yelp_polarity"
yelp_text_col = "text"
yelp_label_col = "label"

chnsenticorp_dataset_name = "AiNiklaus/ChnSentiCorp"
chnsenticorp_text_col = "text"
chnsenticorp_label_col = "label"

# ==================== 注意力机制配置 ====================
# 注意力类型: "MHA", "MGA", "GQA", "MQA"
attention_type = "GQA"
num_groups = 8 # 用于MGA和GQA

# ==================== LoRA/PEFT配置 ====================
use_lora = True
lora_r = 8  # LoRA秩
lora_alpha = 16  # LoRA alpha参数
lora_dropout = 0.1  # LoRA dropout
lora_target_modules = ["query", "key", "value"]  # 目标模块

# PEFT方法配置: "lora", "prefix_tuning", "prompt_tuning", "p_tuning"
peft_method = "lora"
prefix_tuning_num_virtual_tokens = 20
prompt_tuning_num_virtual_tokens = 20

# ==================== 训练参数配置 ====================
# 基础训练参数
output_dir = "./results"
num_train_epochs = 6
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
learning_rate = 3e-5
weight_decay = 0.01
warmup_ratio = 0.1
eval_strategy = "epoch"
save_strategy = "epoch"
load_best_model_at_end = True
metric_for_best_model = "f1"
greater_is_better = True
fp16 = True
gradient_accumulation_steps = 1
logging_steps = 100
save_total_limit = 3


# 各模型训练参数
def get_training_args(model_name, output_subdir):
    """生成指定模型的训练参数"""
    return TrainingArguments(
        output_dir=f"{output_dir}/{model_name}",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        fp16=fp16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        report_to="none",
    )


# 各模型训练参数实例
bertTraining_args = get_training_args("bert", "bert")
robertaTraining_args = get_training_args("roberta", "roberta")
electraTraining_args = get_training_args("electra", "electra")
llama1bTraining_args = get_training_args("llama1b", "llama1b")
training_args = get_training_args("base", "base")

# ==================== 评估指标配置 ====================
# 需要计算的指标
compute_metrics_list = ["accuracy", "f1", "precision", "recall", "auc", "mcc"]

# ==================== 结果保存路径配置 ====================
# 训练结果保存路径
results_dir = "./results"
bert_train_results = f"{results_dir}/bert/train_results.pkl"
roberta_train_results = f"{results_dir}/roberta/train_results.pkl"
electra_train_results = f"{results_dir}/electra/train_results.pkl"
llama1b_train_results = f"{results_dir}/llama1b/train_results.pkl"

# 推理结果保存路径
inference_results_dir = f"{results_dir}/inference"
bert_inference_results = f"{inference_results_dir}/bert_inference.pkl"
roberta_inference_results = f"{inference_results_dir}/roberta_inference.pkl"
electra_inference_results = f"{inference_results_dir}/electra_inference.pkl"
llama1b_inference_results = f"{inference_results_dir}/llama1b_inference.pkl"

# 特征保存路径
features_dir = f"{results_dir}/features"
bert_features = f"{features_dir}/bert_features.pkl"
roberta_features = f"{features_dir}/roberta_features.pkl"
electra_features = f"{features_dir}/electra_features.pkl"
llama1b_features = f"{features_dir}/llama1b_features.pkl"

# 可视化结果保存路径
visualization_dir = f"{results_dir}/visualizations"

# 模型检查点路径
checkpoint_dir = f"{results_dir}/checkpoints"

# ==================== 推理配置 ====================
inference_batch_size = 32
num_inference_samples = 1000  # 用于基准测试的样例数
measure_latency = True
measure_throughput = True
measure_memory = True

# ==================== 可视化配置 =================
plot_style = "seaborn-v0_8"  # Updated for newer matplotlib
figure_size = (10, 6)
dpi = 100
save_format = "png"

# ==================== 环境配置 ====================
hf_endpoint = "https://hf-mirror.com"
cuda_launch_blocking = "1"
use_gpu = True
