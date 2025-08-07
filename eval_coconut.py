# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# --- 导入必要的库 ---
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut import Coconut  # 自定义的 Coconut 模型包装类
from dataset import (
    get_dataset,
    get_question_latent_dataset,
    MyCollator,
)
from tqdm import tqdm
import os
import yaml
import json
import gc
import argparse
import functools
from utils import Config, set_seed


def main():
    # --- 1. 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="coconut evaluation (single sample, single GPU)")
    parser.add_argument("config_file")
    args = parser.parse_args()

    # --- 2. 加载并解析配置文件 ---
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    # 强制只运行评估
    if not config_dict.get('only_eval', False):
        print(f"Warning: 'only_eval' is not True. Forcing it to True.")
        config_dict['only_eval'] = True

    configs = Config(config_dict)
    set_seed(configs.seed)

    # --- 3. 加载基础语言模型 ---
    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 添加特殊 token
    new_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]
    tokenizer.add_tokens(new_tokens)
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # --- 4. 处理新增 token 的嵌入 ---
    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token.")
        target_id = tokenizer.eos_token_id

        for token_id in [latent_id, start_id, end_id]:
            embeddings.weight.data[token_id] = embeddings.weight.data[target_id]
            if hasattr(model, 'lm_head'):
                model.lm_head.weight.data[token_id] = model.lm_head.weight.data[target_id]

    # --- 5. 包装为 Coconut 模型 ---
    model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    # --- 6. 加载检查点 ---
    if configs.load_model_path is None:
        raise ValueError("For evaluation, `load_model_path` must be specified.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_weights = torch.load(configs.load_model_path, map_location=device)

    if configs.coconut and not any(k.startswith("base_causallm") for k in saved_weights.keys()):
        print(f"Loading base model weights into Coconut model...")
        print(model.load_state_dict(saved_weights, strict=False))
    elif not configs.coconut and any(k.startswith("base_causallm") for k in saved_weights.keys()):
        raise ValueError("Cannot load coconut model weights into a causallm model")
    else:
        print(f"Loading model checkpoint...")
        print(model.load_state_dict(saved_weights, strict=False))

    # --- 7. 移动模型到 GPU ---
    model = model.to(device)

    # --- 8. 加载验证集数据 ---
    val_data = json.load(open(configs.val_path))
    question_val = [d["question"] for d in val_data]
    answers_val = [d["answer"].replace(",", "").strip() for d in val_data]
    cot_val = ["\n".join(d["steps"]) for d in val_data]

    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=100000000
    )

    if "gsm" in configs.val_path:
        max_new_tokens = 64
    else:
        max_new_tokens = 128

    # --- 9. 计算 stage 并获取单个样本 ---
    #scheduled_stage = min(configs.resume // configs.epochs_per_stage, configs.max_latent_stage)
    scheduled_stage = 2
    print(f"Using scheduled_stage = {scheduled_stage} for evaluation.")

    dataset_gen_val = get_question_latent_dataset(
        scheduled_stage,
        base_dataset_valid,
        configs,
        start_id,
        latent_id,
        end_id,
        no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
    )

    single_sample = dataset_gen_val[10]
    test_idx = single_sample["idx"]

    # 构造 batch (batch_size=1)
    batch = {
        k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else torch.tensor([v]).to(device)
        for k, v in single_sample.items()
        if v is not None and k not in ["idx", "position_ids"]
    }

    # 获取真实值
    question = question_val[test_idx]
    answer = answers_val[test_idx]
    answer_cot = cot_val[test_idx]

    print(f"Question: {question}")
    print(f"True Answer: {answer}")

    # 推理
    with torch.no_grad():
        model.eval()
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
        )
        text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\nGenerated Output:\n{text_output}")

    answer_output = text_output.split("#")[-1].replace(",", "").strip()
    cot_output = ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()

    is_correct = answer_output == answer
    is_cot_correct = cot_output == answer_cot

    print(f"\n→ Extracted Answer: '{answer_output}' → Correct: {is_correct}")

    # --- 清理 ---
    del model
    gc.collect()
    torch.cuda.empty_cache()


# --- 程序入口 ---
if __name__ == "__main__":
    main()