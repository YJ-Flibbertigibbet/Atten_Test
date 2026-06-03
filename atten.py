import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PrefixTuningConfig,
        PromptTuningConfig,
        P_TuningConfig,
    )
    from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print(
        "Warning: PEFT not installed. LoRA and other PEFT methods will not be available."
    )

import config


class MultiGroupAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_groups):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        # Ensure num_groups divides num_heads evenly
        if num_attention_heads % num_groups != 0:
            # Find the largest divisor of num_heads that is <= num_groups
            best = 1
            for i in range(num_groups, 0, -1):
                if num_attention_heads % i == 0:
                    best = i
                    break
            num_groups = best

        self.num_groups = num_groups

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, self.head_dim * num_groups)
        self.value = nn.Linear(hidden_size, self.head_dim * num_groups)

    def forward(self, hidden_states, *args, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        output_attentions = kwargs.get("output_attentions", False)

        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)

        q = self.query(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

        k = self.key(hidden_states)
        k = k.view(batch_size, seq_len, self.num_groups, self.head_dim)
        k = k.transpose(1, 2)  # (batch, num_groups, seq_len, head_dim)
        k = k.repeat_interleave(self.num_heads // self.num_groups, dim=1)

        v = self.value(hidden_states)
        v = v.view(batch_size, seq_len, self.num_groups, self.head_dim)
        v = v.transpose(1, 2)  # (batch, num_groups, seq_len, head_dim)
        v = v.repeat_interleave(self.num_heads // self.num_groups, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )
        if output_attentions:
            return (attn_output, attn_weights)
        return (attn_output,)


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_kv_groups):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        # Ensure num_kv_groups divides num_heads evenly
        if num_attention_heads % num_kv_groups != 0:
            # Find the largest divisor of num_heads that is <= num_kv_groups
            best = 1
            for i in range(num_kv_groups, 0, -1):
                if num_attention_heads % i == 0:
                    best = i
                    break
            num_kv_groups = best

        self.num_kv_groups = num_kv_groups

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, self.head_dim * num_kv_groups)
        self.value = nn.Linear(hidden_size, self.head_dim * num_kv_groups)

    def forward(self, hidden_states, *args, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        output_attentions = kwargs.get("output_attentions", False)

        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)

        q = self.query(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

        k = self.key(hidden_states)
        k = k.view(batch_size, seq_len, self.num_kv_groups, self.head_dim)
        k = k.transpose(1, 2)  # (batch, num_kv_groups, seq_len, head_dim)
        k = k.repeat_interleave(
            self.num_heads // self.num_kv_groups, dim=1
        )  # (batch, num_heads, seq_len, head_dim)

        v = self.value(hidden_states)
        v = v.view(batch_size, seq_len, self.num_kv_groups, self.head_dim)
        v = v.transpose(1, 2)  # (batch, num_kv_groups, seq_len, head_dim)
        v = v.repeat_interleave(
            self.num_heads // self.num_kv_groups, dim=1
        )  # (batch, num_heads, seq_len, head_dim)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )
        if output_attentions:
            return (attn_output, attn_weights)
        return (attn_output,)


class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, self.head_dim)
        self.value = nn.Linear(hidden_size, self.head_dim)

    def forward(self, hidden_states, *args, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        output_attentions = kwargs.get("output_attentions", False)

        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)

        q = self.query(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

        k = self.key(hidden_states)  # (batch, seq_len, head_dim)
        k = k.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        k = k.repeat(1, self.num_heads, 1, 1)  # (batch, num_heads, seq_len, head_dim)

        v = self.value(hidden_states)  # (batch, seq_len, head_dim)
        v = v.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        v = v.repeat(1, self.num_heads, 1, 1)  # (batch, num_heads, seq_len, head_dim)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )
        if output_attentions:
            return (attn_output, attn_weights)
        return (attn_output,)


def apply_lora_to_model(model, lora_config=None):
    """为模型应用LoRA配置"""
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT is not installed. Please install it with: pip install peft"
        )

    if lora_config is None:
        # 使用config中的默认配置
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )

    model = get_peft_model(model, lora_config)
    print(
        f"Applied LoRA to model. Trainable parameters: {model.print_trainable_parameters()}"
    )
    return model


def apply_peft_to_model(model, peft_method=None):
    """为模型应用PEFT方法"""
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT is not installed. Please install it with: pip install peft"
        )

    if peft_method is None:
        peft_method = config.peft_method

    if peft_method == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
    elif peft_method == "prefix_tuning":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=config.prefix_tuning_num_virtual_tokens,
        )
    elif peft_method == "prompt_tuning":
        peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=config.prompt_tuning_num_virtual_tokens,
        )
    else:
        raise ValueError(f"Unsupported PEFT method: {peft_method}")

    model = get_peft_model(model, peft_config)
    print(f"Applied {peft_method} to model.")
    return model


def replace_attention(model, atten="MHA", num_groups=2):
    """替换模型的注意力机制"""
    # 从模型配置获取num_attention_heads
    num_attention_heads = None
    if hasattr(model, "config"):
        num_attention_heads = getattr(model.config, "num_attention_heads", None)

    if hasattr(model, "bert"):
        layers = model.bert.encoder.layer
        _replace_layer_attention(layers, atten, num_groups, num_attention_heads)
    elif hasattr(model, "roberta"):
        layers = model.roberta.encoder.layer
        _replace_layer_attention(layers, atten, num_groups, num_attention_heads)
    elif hasattr(model, "electra"):
        layers = model.electra.encoder.layer
        _replace_layer_attention(layers, atten, num_groups, num_attention_heads)
    elif hasattr(model, "encoder"):
        layers = model.encoder.layer
        _replace_layer_attention(layers, atten, num_groups, num_attention_heads)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    return model


def _replace_layer_attention(
    layers, attention_type, num_groups, num_attention_heads=None
):
    """实际替换注意力层的函数"""
    hidden_size = layers[0].attention.self.query.out_features

    # 如果没有传入num_attention_heads，尝试从模型配置或计算
    if num_attention_heads is None:
        # 尝试不同的方式获取num_attention_heads
        attn_layer = layers[0].attention.self
        if hasattr(attn_layer, "num_heads"):
            num_attention_heads = attn_layer.num_heads
        elif hasattr(attn_layer, "num_attention_heads"):
            num_attention_heads = attn_layer.num_attention_heads
        elif hasattr(attn_layer, "n_heads"):
            num_attention_heads = attn_layer.n_heads
        else:
            # 从hidden_size和head_dim计算，或默认使用24（bert-large的默认值）
            head_dim = getattr(attn_layer, "head_dim", None)
            if head_dim is not None:
                num_attention_heads = hidden_size // head_dim
            else:
                num_attention_heads = 24  # bert-large-uncased default
                print(
                    f"Warning: Could not determine num_attention_heads, using default: {num_attention_heads}"
                )

    for layer in layers:
        if attention_type == "MHA":
            continue
        elif attention_type == "MGA":
            layer.attention.self = MultiGroupAttention(
                hidden_size, num_attention_heads, num_groups
            )
        elif attention_type == "GQA":
            layer.attention.self = GroupedQueryAttention(
                hidden_size, num_attention_heads, num_groups
            )
        elif attention_type == "MQA":
            layer.attention.self = MultiQueryAttention(hidden_size, num_attention_heads)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")


def setup_model_with_attention_and_lora(
    model, attention_type=None, use_lora=None, peft_method=None
):
    """综合设置模型的注意力机制和LoRA/PEFT"""
    if attention_type is None:
        attention_type = config.attention_type
    if use_lora is None:
        use_lora = config.use_lora

    # 首先替换注意力机制
    if attention_type != "MHA":
        model = replace_attention(model, attention_type, config.num_groups)
        print(f"Replaced attention with {attention_type}")

    # 然后应用LoRA/PEFT
    if use_lora and PEFT_AVAILABLE:
        if peft_method is None:
            peft_method = config.peft_method
        model = apply_peft_to_model(model, peft_method)

    return model
