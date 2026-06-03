# Task 1: 文本分类模型训练与评估系统

## 项目简介

基于 Hugging Face Transformers 的文本分类实验框架，支持多种预训练模型、注意力机制和 PEFT 微调方法的对比实验。

## 支持的模型

| 模型 | 路径 |
|------|------|
| BERT | `google-bert/bert-large-uncased` |
| RoBERTa | `roberta-base` |
| ELECTRA | `google/electra-base-discriminator` |

## 支持的注意力机制

- **MHA** (Multi-Head Attention) — 标准多头注意力
- **MGA** (Multi-Group Attention) — 多组注意力
- **GQA** (Grouped Query Attention) — 分组查询注意力
- **MQA** (Multi-Query Attention) — 多查询注意力

## 支持的 PEFT 方法

- LoRA
- Prefix Tuning
- Prompt Tuning

## 数据集

- **SST-2** (GLUE) — 英文电影评论二分类
- **Yelp Polarity** — 英文 Yelp 评论二分类

## 项目结构

| 文件 | 说明 |
|------|------|
| `main.py` | 主训练脚本，支持单模型/全模型训练 |
| `config.py` | 统一配置文件（模型、数据、训练、注意力等参数） |
| `data.py` | 数据集加载 |
| `modelProcess.py` | 模型初始化、训练循环、评估指标、性能监控 |
| `atten.py` | 自定义注意力机制实现 + LoRA/PEFT 集成 |
| `inference_benchmark.py` | 推理基准测试与特征收集 |
| `comprehensive_analysis.py` | 综合分析（图表 + 报告） |
| `final_analysis.py` | 最终综合分析（参数-内存关系、收敛性分析） |
| `simple_analysis.py` | 简单分析脚本 |
| `conftest.py` | Pytest 测试配置与共享 fixtures |

## 使用方法

### 训练

```bash
# 训练 BERT 模型（默认数据集 SST-2）
python main.py --model bert

# 指定注意力机制
python main.py --model bert --attention GQA

# 使用 LoRA
python main.py --model bert --use_lora --peft_method lora

# 训练所有模型
python main.py --model all
```

### 推理基准测试

```bash
python inference_benchmark.py
```

### 分析

```bash
python comprehensive_analysis.py
# 或
python final_analysis.py
```

## 配置说明

主要配置在 `config.py` 中，包括：

- 模型路径与标签数
- 批大小、序列长度
- 注意力类型与分组数
- LoRA 秩、alpha、dropout
- 训练轮数、学习率、优化器参数
- 评估指标（accuracy, f1, precision, recall, auc, mcc）
- 结果与可视化保存路径

## 输出

- `results/` — 训练结果 pkl 文件
- `results/inference/` — 推理结果 pkl 文件
- `results/visualizations/` — 可视化图表（PNG + PDF）
- `results/checkpoints/` — 模型检查点
