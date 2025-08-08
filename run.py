# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

import swanlab

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block


from coconut import Coconut
from dataset import (
    get_dataset,
    get_question_latent_dataset,
    get_cot_latent_dataset,
    MyCollator,
)

from tqdm import tqdm
from copy import copy
import itertools
import os, sys
import yaml
import json
import gc
import argparse
import functools
from utils import Config, set_seed




def main():

    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")
    args = parser.parse_args()

    # init distributed environment
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if rank == 0:
        print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    torch.distributed.barrier()
    cur_ckpts = os.listdir(save_dir)

    # check if the job is preempted and resumed.

    if len(cur_ckpts) > 0 and not configs.only_eval:
        # if there are previous checkpoints, and only_eval is False
        # it means the previous run was preempted and the program is restarted.
        # need to find the latest checkpoint and resume from that.

        if rank == 0:
            print(
                f"Warning: found previous run and gonna resume from that. the inputted `resume` argument is ignored!"
            )

        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))

        # Get the last item in the sorted list
        latest_checkpoint = checkpoints[-1] if checkpoints else None
        configs.resume = int(latest_checkpoint.split("_")[1])
        load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)

        configs.load_model_path = load_dir
        print(f"Loading from previous run epoch_{configs.resume}!")

    elif configs.resume != 0:
        # by setting `resume`, we can skip a few epoches at the beginning.
        if configs.load_model_path == "None":
            print(
                f"Warning: you want to skip the first {configs.resume} but you are not loading any existing checkpoint!"
            )
            # not an intended use case at this point
        print(
            f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs"
        )

    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    loaded = False

    if configs.load_model_path != "None":
        saved_weights = torch.load(
            configs.load_model_path, map_location=torch.device(rank)
        )

        if configs.coconut and not any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # we are loading a base model into coconut model
            # e.g., for GSM8k, we used a SFTed model to skip the stage 0
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

        elif not configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            raise ValueError("Cannot load coconut model weights into a causallm model")

        elif configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # loading from preempted run
            # will handle later
            pass

        else:
            # resume or evaluate sft model
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        # if we need new tokens, initialize their embeddings and lm heads
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        # initialize the new token embeddings with a known token
        # it helps stablize the training
        for token_id in [latent_id, start_id, end_id]:
            target_embedding = embeddings.weight.data[token_id]
            embeddings.weight.data[token_id] = target_embedding
            # The input embeddings and lm heads are tied in GPT2. So the code below is not necessary
            lm_head = model.lm_head
            lm_head.weight.data[token_id] = lm_head.weight.data[target_id]

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    if configs.coconut:
        model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    if configs.load_model_path != "None" and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

    print(f"Running FSDP on rank = {rank}, world size = {world_size}")
    model = model.to(rank)

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            GPT2Block,       # for GPT2, we don't need to shard layers (it becomes DDP)
            #LlamaDecoderLayer  # only shard llama's layers.
        },
    )

    if configs.bf16:
        model.to(torch.bfloat16)

    # if only eval, use ddp (to avoid bugs in fsdp)
    if configs.only_eval:
        parallel_model = DDP(model, device_ids=[rank])

    else:
        parallel_model = FSDP(
            model, auto_wrap_policy=llama_auto_wrap_policy, device_id=rank
        )

    del model

    if rank == 0:
        print(parallel_model)

    # prepare the ground truth answer and cot for evaluation
    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [
        d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))
    ]
    cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]

    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=32 if configs.debug else 100000000
    )

    if not configs.only_eval:
        base_dataset_train = get_dataset(
            configs.train_path, tokenizer, max_size=5000 if configs.debug else 100000000
        )

    if "gsm" in configs.val_path:
        max_new_tokens = 64
    else:
        max_new_tokens = 128

    total_train_steps = 0

    if not configs.debug and not configs.only_eval and rank == 0:
        swanlab_run = swanlab.init(
        project=configs.project,
        experiment_name=configs.name,
        config=vars(configs),   # SwanLab 支持 dict / argparse.Namespace
    )

        #text_table = swanlab.Table(columns=["step", "text"])


    else:
        swanlab_run = None

    if configs.reset_optimizer:
        optimizer = None

    else:
        optimizer = optim.AdamW(
            parallel_model.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )

    best_acc = 0

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    theta0 = None
    # 用于存储最近 T0 步的梯度
    recent_gradients = []

    def estimate_iiw(parallel_model, current_gradients, theta0, theta_bar, n_samples, T0, rank, world_size):
        """
        估计 Information in Weights (IIW) 的近似值。
        
        Args:
            parallel_model: FSDP封装的模型
            current_gradients: 当前步骤的梯度列表 (来自backward后的grad)
            theta0: 初始模型权重 (先验均值)
            theta_bar: 移动平均后的模型权重 (论文中的 θ̄_S)
            n_samples: 训练集总样本数
            T0: 信息估计迭代次数 (这里指梯度累积步数)
            rank: 当前进程的rank
            world_size: 总进程数

        Returns:
            I_tilde (float): 估计的IIW值 (在rank0上)
        """
        # 只在rank 0上进行计算和聚合
        if rank != 0:
            # 非0号rank只需要将梯度发送给0号rank
            if current_gradients:
                for grad in current_gradients:
                    dist.send(grad, dst=0)
            return None

        # 1. 收集所有GPU上的梯度并求平均
        # 假设 current_gradients 是一个包含所有参数梯度的列表
        # 我们需要从所有world_size个进程中收集
        all_gradients = [current_gradients]  # 0号rank自己的梯度
        for src_rank in range(1, world_size):
            # 为每个其他rank创建一个梯度列表的占位符
            src_grads = []
            for param in parallel_model.parameters():
                if param.requires_grad:
                    # 创建一个与param.grad形状相同的张量来接收
                    recv_grad = torch.zeros_like(param.grad)
                    dist.recv(recv_grad, src=src_rank)
                    src_grads.append(recv_grad)
            all_gradients.append(src_grads)
        
        # 将所有梯度在 batch 维度上平均 (模拟数据并行的平均效果)
        averaged_gradients = []
        for i in range(len(current_gradients)):
            stacked_grads = torch.stack([grads[i] for grads in all_gradients], dim=0) # [world_size, ...]
            mean_grad = stacked_grads.mean(dim=0) # 在world_size维度上平均
            averaged_gradients.append(mean_grad)

        # 2. 将当前平均梯度添加到 recent_gradients 队列
        recent_gradients.append(averaged_gradients)
        # 保持队列长度为 T0
        if len(recent_gradients) > T0:
            recent_gradients.pop(0)

        # 3. 计算 Δθ = θ_bar - θ0
        # 注意: theta_bar 和 theta0 应该是展平后的向量
        # 这里需要您根据实际情况实现展平和恢复
        # 为了简化，我们假设有一个函数来处理
        def flatten_params(params):
            return torch.cat([param.view(-1) for param in params])

        # 获取当前模型权重 (θ_bar)
        with torch.no_grad():
            current_params = [param.data.clone() for param in parallel_model.parameters() if param.requires_grad]
        flat_theta_bar = flatten_params(current_params)
        flat_theta0 = flatten_params(theta0)

        delta_theta = flat_theta_bar - flat_theta0 # [D,]

        # 4. 计算 I_tilde = n * sum_{t=1}^{T1} (Δθ^T * ∇L_t)^2 / T1
        # 公式 (15): I˜(w; S)= n / T1 * sum_{t=1}^{T1} [Δθ>∇θ`_t(θˆ)]^2
        T1 = len(recent_gradients) # 实际累积的步数
        if T1 == 0:
            return 0.0

        sum_sq_term = 0.0
        for grad_list in recent_gradients:
            # 将梯度展平
            flat_grad = flatten_params(grad_list) # [D,]
            # 计算 Δθ^T * ∇L_t
            inner_product = torch.dot(delta_theta, flat_grad)
            sum_sq_term += inner_product.item() ** 2

        I_tilde = (n_samples / T1) * sum_sq_term
        return I_tilde


    for epoch in range(configs.resume, configs.num_epochs):

        #scheduled_stage = (0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage)
        scheduled_stage = 6
        dataset_gen_val = get_question_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            configs,
            start_id,
            latent_id,
            end_id,
            no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
        )

        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            batch_size=1,
            collate_fn=collator,
            sampler=DistributedSampler(dataset_gen_val, shuffle=False),
        )

        if not configs.only_eval:

            dataset_train = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_train,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
                shuffle=True,
            )

            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_train, shuffle=True),
            )

            # the sampler is deterministic even if shuffle is set to True
            # so we have shuffled the dataset when it's constructed (at every epoch).

            dataset_loss_val = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
            )

            valid_loss_dataloader = torch.utils.data.DataLoader(
                dataset_loss_val,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_loss_val, shuffle=False),
            )

            if configs.reset_optimizer:
                del optimizer

                optimizer = optim.AdamW(
                    parallel_model.parameters(),
                    lr=configs.lr,
                    weight_decay=configs.weight_decay,
                )

            parallel_model.module.train()

            total_length = len(train_dataloader) // configs.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )

            # 关键: 在第一次迭代时获取 theta0
            if theta0 is None and epoch == configs.resume:
                # 获取当前模型的所有参数作为 theta0
                theta0 = [param.data.clone().detach() for param in parallel_model.parameters() if param.requires_grad]
                print(f"Rank {rank}: Captured initial model weights (theta0) at start of epoch {epoch}.")
                # 广播给所有进程，确保一致性
                for param in theta0:
                    dist.broadcast(param, src=0)

            for step, batch in enumerate(train_dataloader):

                total_train_steps += 1
                batch = {
                    key: batch[key].to(rank) for key in batch.keys() if key != "idx"
                }

                outputs = parallel_model(**batch)

                loss = outputs.loss / configs.gradient_accumulation_steps
                loss.backward()



                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    # 获取当前所有参数的梯度
                    current_gradients = []
                    for param in parallel_model.parameters():
                        if param.requires_grad and param.grad is not None:
                            current_gradients.append(param.grad.clone()) # 克隆以避免被清零影响

                    # 调用估计函数
                    # T0 被设为梯度累积步数，n_samples 需要您提供（例如，训练集大小）
                    n_samples = 32  # <-- 请替换为您的实际训练集样本总数
                    T0 = configs.gradient_accumulation_steps
                    I_tilde = estimate_iiw(parallel_model, current_gradients, theta0, None, n_samples, T0, rank, world_size)

                    # 只有 rank 0 会返回值
                    if rank == 0 and I_tilde is not None:
                        print(f"Estimated IIW (I_tilde): {I_tilde}")
                        if swanlab_run:
                            swanlab_run.log({"train/IIW": I_tilde})
                    # ====================================================

                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                if swanlab_run and rank == 0:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float()
                        * configs.gradient_accumulation_steps,
                    }
                    swanlab_run.log(log_dict)

                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}"
                )
            pbar.close()
            dist.barrier()

            if (
                not configs.save_only_improve
                and not configs.debug
                and not configs.only_eval
            ):
                states = parallel_model.state_dict()
                if rank == 0:
                    torch.save(
                        states, os.path.join(save_dir, f"checkpoint_{epoch + 1}")
                    )
                    print("saving model.")

                dist.barrier()
                del states
                gc.collect()
                torch.cuda.empty_cache()



if __name__ == "__main__":
    main()
