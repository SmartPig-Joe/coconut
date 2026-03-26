import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import matplotlib
import torch.distributed as dist
import sys


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Computer Modern', 'DejaVu Serif']
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 11
matplotlib.rcParams['figure.titlesize'] = 16


Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id

        self.embedding = self.base_causallm.transformer.get_input_embeddings()
        # --- 新增代码 ---
        # 获取嵌入向量的维度
        embedding_dim = self.embedding.weight.shape[1] 
        # 定义一个LayerNorm层，用于归一化隐藏状态
        self.latent_norm = nn.LayerNorm(embedding_dim)
        # 用于跨测试集收集 IS（最大核采样概率）值
        self.all_is_values = []


    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):

        logits = []
        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]

        max_n_latents = max([len(l) for l in latent_lists])

        # 定义下一次需要计算的输入范围 (start, end)
        next_compute_range = (0, input_ids.shape[1])
        # 将输入的token ID转换为对应的嵌入向量
        inputs_embeds = self.embedding(input_ids)


        # 如果存在潜在标记，则将第一次计算范围设置为从开头到第一个潜在标记之前
        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache is None:
                # 第一次前向传播，没有可用的KV缓存
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,  # 需要隐藏状态来更新潜在标记
                )
                # 由于没有使用KV缓存，输出的隐藏状态从位置0开始
                hidden_states_offset = 0

            else:
                # 后续前向传播，可以重用之前的KV缓存
                # 从kv_cache中提取出到next_compute_range[0]为止的KV对
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,  # 传入KV缓存
                    output_hidden_states=True,
                )
                # 当使用KV缓存时，outputs.hidden_states中[0, k)位置的隐藏状态会被跳过
                # 因此需要一个偏移量来正确索引隐藏状态
                hidden_states_offset = next_compute_range[0]

            # 将当前轮次的logits加入列表
            logits.append(outputs.logits)

            # 更新下一次计算的范围：
            # start: 从上一轮的end位置开始
            # end: 如果是最后一轮，则到序列末尾；否则，只计算下一个位置（即潜在标记位置）
            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            # 获取基础模型最后一层的隐藏状态
            hidden_states = outputs.hidden_states[-1]
            # 缓存KV对，供下一轮使用
            kv_cache = outputs.past_key_values

            # --- 将推理结果“反馈”到输入嵌入中 ---
            # 决定需要被“连续思考”（即隐藏状态）替换的嵌入位置
            filling_indices = [
                (instance_idx, mask_list[pass_idx])  # (批次索引, 位置索引)
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx  # 确保该批次在当前轮次仍有潜在标记需要处理
            ]

            # 为了避免对inputs_embeds进行in-place操作，先将其分解为一个列表
            # tensor_list[batch_idx][pos] 对应 inputs_embeds[batch_idx, pos, :]
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # 将指定位置的嵌入向量替换为上一轮前向传播得到的隐藏状态
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                # 1. 提取出原始的“思考结果”（即前一个token的隐藏状态）
                thought_vector = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

                # 2. 【新增步骤】对这个向量进行归一化
                normalized_thought_vector = self.latent_norm(thought_vector)
                #normalized_thought_vector = thought_vector


                # 3. 将归一化后的向量赋值给潜在标记的位置
                tensor_list[batch_idx][token_idx] = normalized_thought_vector # <-- 使用归一化后的向量

            # 将修改后的列表重新组装成张量
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # --- 最终前向传播 ---
        # 处理剩余的所有token（包括潜在标记已被替换后的序列）
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=(
                [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )

        # 将最终轮次的logits加入列表
        logits.append(outputs.logits)

        # 更新计数器，记录本次forward调用的总前向传播次数
        self.gen_forward_cnt += max_n_latents + 1

        # 将所有轮次的logits在序列维度（dim=-2）上拼接起来
        logits = torch.cat(logits, dim=-2)

        # --- 计算损失 ---
        # 将logits和labels进行偏移，以计算下一个token的预测损失
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        # 返回损失、最终的输入嵌入（可用于分析）和完整的logits
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        """设置模型为训练模式"""
        self.base_causallm.train()

    def eval(self):
        """设置模型为评估模式"""
        self.base_causallm.eval()

    @staticmethod
    def _compute_IS_values(reasoning_logits, top_p=0.95):
        """计算每个推理步骤的 IS 值（top-p 核采样集合中的最大概率）。

        对于每个步骤，先对 logits 做 softmax，再按降序排序，累积概率超过 top_p
        时截断，IS = 核集合中的最大概率（即排序后的第一个概率值）。

        Args:
            reasoning_logits: shape [num_steps, vocab_size] 的 logits 张量。
            top_p: 核采样的概率阈值，默认 0.95。

        Returns:
            list[float]: 每个步骤对应的 IS 值。
        """
        probs = torch.softmax(reasoning_logits.detach().cpu().float(), dim=-1)
        is_values = []
        for i in range(probs.shape[0]):
            p = probs[i]
            sorted_probs, _ = torch.sort(p, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # 找到核集合的截止位置（累积概率首次超过 top_p）
            cutoff = (cumulative >= top_p).nonzero(as_tuple=True)[0]
            if cutoff.numel() > 0:
                nucleus_probs = sorted_probs[: cutoff[0].item() + 1]
            else:
                # 所有 token 的累积概率仍不足 top_p（数值精度问题），使用全部 token
                nucleus_probs = sorted_probs
            # IS = 核集合中的最大概率（即排序后的第一个）
            is_values.append(nucleus_probs[0].item())
        return is_values

    def _plot_IS_histogram(self, save_path="is_histogram.pdf", title="IS Distribution Across Test Set", n_bins=20):
        """将收集到的 IS 值绘制为直方图并保存。

        横轴为 IS 值（0–1），纵轴为落在每个子区间的步骤频数。
        绘图后会清空 all_is_values，以便下次重新收集。

        Args:
            save_path: 输出文件路径（支持 .pdf 或 .png）。
            title: 图标题，为空则不显示。
            n_bins: 横轴划分的子区间数量，默认 20。
        """
        if not self.all_is_values:
            print("No IS values collected. Skipping histogram.")
            return

        values = np.array(self.all_is_values)
        bins = np.linspace(0, 1, n_bins + 1)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values, bins=bins, color='steelblue', edgecolor='white', linewidth=0.5)

        ax.set_xlabel("IS (Max Nucleus Probability)")
        ax.set_ylabel("Frequency (Steps)")
        if title:
            ax.set_title(title)
        ax.set_xlim(0, 1)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)

        plt.tight_layout()

        if not save_path.endswith(('.pdf', '.png')):
            save_path += '.pdf'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"IS histogram saved to {save_path} (n={len(values)} steps total)")

        # 清空，以便下次重新收集
        self.all_is_values = []

    def _plot_logit_evolution(self, reasoning_logits, tokenizer, top_p=0.95, save_path="logit_evolution_renormalized.pdf"):
        # 1. Convert logits to probabilities on CPU
        probs_over_steps = torch.softmax(reasoning_logits.detach().cpu(), dim=-1)
        num_steps = probs_over_steps.shape[0]

        # 2. Process each step
        selected_tokens_per_step = []
        all_unique_token_ids = set()

        for i in range(num_steps):
            probs_this_step = probs_over_steps[i]
            sorted_probs_full, sorted_indices_full = torch.sort(probs_this_step, descending=True)
            
            if len(sorted_probs_full) <= 1:
                selected_tokens_per_step.append({'indices': torch.tensor([]), 'probs': torch.tensor([])})
                continue

            # a. Remove top-1 token
            sorted_probs_remaining = sorted_probs_full[1:]
            sorted_indices_remaining = sorted_indices_full[1:]
            
            sum_of_remaining_probs = torch.sum(sorted_probs_remaining)
            if sum_of_remaining_probs <= 1e-9:
                selected_tokens_per_step.append({'indices': torch.tensor([]), 'probs': torch.tensor([])})
                continue

            # b. Nucleus sampling on the remaining tokens
            cumulative_probs = torch.cumsum(sorted_probs_remaining, dim=-1)
            cutoff_index = (cumulative_probs > top_p * sum_of_remaining_probs).nonzero(as_tuple=True)[0]

            nucleus_size = cutoff_index[0].item() + 1 if cutoff_index.numel() > 0 else len(sorted_probs_remaining)
            nucleus_indices = sorted_indices_remaining[:nucleus_size]
            nucleus_probs_original = sorted_probs_remaining[:nucleus_size]

            # c. Renormalize over the remaining distribution (excluding top-1)
            renormalized_nucleus_probs = nucleus_probs_original / sum_of_remaining_probs

            step_data = {'indices': nucleus_indices, 'probs': renormalized_nucleus_probs}
            selected_tokens_per_step.append(step_data)
            all_unique_token_ids.update(nucleus_indices.tolist())

        # 3. Assign consistent colors to tokens
        unique_token_list = sorted(list(all_unique_token_ids))
        if unique_token_list:
            # Use a perceptually distinct colormap; limit to 20 for clarity
            n_colors = min(len(unique_token_list), 20)
            cmap = plt.cm.get_cmap('tab10' if n_colors <= 10 else 'tab20', n_colors)
            token_to_color = {
                token_id: cmap(i % n_colors) for i, token_id in enumerate(unique_token_list)
            }
        else:
            token_to_color = {}

        # 4. Plot
        fig, ax = plt.subplots(figsize=(8, max(4, 0.6 * num_steps)))  # Adaptive height

        y_positions = np.arange(num_steps)

        for i, step_data in enumerate(selected_tokens_per_step):
            left = 0.0
            if step_data['indices'].numel() > 0:
                for token_id, prob in zip(step_data['indices'], step_data['probs']):
                    prob_val = prob.item()
                    color = token_to_color.get(token_id.item())
                    if color is not None:
                        ax.barh(y_positions[i], prob_val, left=left, color=color, edgecolor='none')
                        left += prob_val

        # 5. Legend with decoded tokens (clean, not repr)
        legend_elements = []
        for token_id in unique_token_list[:30]:  # 可选：限制最多30个token避免过载
            try:
                decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                label = decoded.replace('\n', '\\n').replace('\t', '\\t')
            except Exception:
                label = f"[UNK:{token_id}]"
            color = token_to_color[token_id]
            legend_elements.append(Patch(facecolor=color, edgecolor='none', label=label))

        if legend_elements:
            ax.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                frameon=False,
                title="Nucleus Tokens",
                title_fontsize=12,
                ncol=2,                 # ← 关键：分两列
                columnspacing=0.8,      # 列间距
                handletextpad=0.3,      # 图标与文字间距
                fontsize=10             # 文字大小微调
            )
        # 6. Formatting
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"Step {j+1}" for j in range(num_steps)])
        ax.invert_yaxis()
        ax.set_xlabel(r"Renormalized Probability")
        ax.set_ylabel("Reasoning Step")
        ax.set_xlim(0, 1.0)

        # Remove spines and grid for clean look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)

        plt.tight_layout(rect=[0, 0, 0.82, 1])

        # 7. Save (prefer PDF for ICLR)
        if not save_path.endswith(('.pdf', '.png')):
            save_path += ".pdf"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"ICLR-style logit evolution plot saved to {save_path}")

    def generate(
            self,
            input_ids,
            attention_mask,
            max_new_tokens=16,
            output_embedding=False,
            synced_gpus=False,
            tokenizer=None,
            **kwargs
        ):
        # ======================= DEBUGGING BLOCK START =======================
        rank = -1
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()

        # This print is CRITICAL. It tells us if the function is even being entered on this rank.
        print(f"[Rank {rank}] --- Entered generate function. Input shape: {input_ids.shape} ---", flush=True)
        # ======================= DEBUGGING BLOCK END =========================

        # 重置计数器
        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        # 初始化生成的token列表，包含输入的token
        tokens = input_ids[0].detach().tolist()

        # 调用forward方法处理初始输入，执行潜在推理过程
        # labels在此处是占位符，不被使用
        labels = input_ids.clone()
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),  # 创建一个全1的attention_mask
            labels,
            # 创建position_ids，从0到序列长度
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        # 获取经过潜在推理后的最终输入嵌入
        inputs_embeds = outputs.inputs_embeds

        # --- 收集潜在推理步骤的 IS 值（所有 rank 均执行）---
        latent_positions = (input_ids[0] == self.latent_token_id).nonzero(as_tuple=True)[0]
        if latent_positions.numel() > 0:
            logit_indices_for_reasoning = latent_positions - 1
            reasoning_logits_latent = outputs.logits[0, logit_indices_for_reasoning, :]
            self.all_is_values.extend(self._compute_IS_values(reasoning_logits_latent))

        # --- 【修改点 1】: 移除原有的可视化代码块 ---
        # 原有的代码块在这里被删除了。

        # --- 【新增代码 1】: 初始化一个列表来收集CoT步骤的logits ---
        cot_logits_list = []

        # Generate first token
        cot_logits_list.append(outputs.logits[0, -1, :])
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # --- Generation loop ---
        print(f"[Rank {rank}] Starting generation loop for {max_new_tokens - 1} steps.", flush=True)
        
        for i in range(max_new_tokens - 1):
            # This print tells us if the loop is actually running.
            print(f"[Rank {rank}] Loop iteration {i}", flush=True)

            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1

            cot_logits_list.append(outputs.logits[0, -1, :])

            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                print(f"[Rank {rank}] EOS token generated. Breaking loop.", flush=True)
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        print(f"[Rank {rank}] Loop finished. Collected {len(cot_logits_list)} sets of logits.", flush=True)

        # --- Plotting logic ---
        is_main_process = (rank == 0 or rank == -1) # rank -1 for non-distributed case
        if is_main_process and tokenizer and cot_logits_list:
            print(f"[Rank {rank}] Conditions met for plotting. Stacking logits.", flush=True)
            cot_logits = torch.stack(cot_logits_list, dim=0)
            
            print(f"[Rank {rank}] Logits stacked, shape: {cot_logits.shape}. Calling plot function.", flush=True)
            self._plot_logit_evolution(
                cot_logits, 
                tokenizer, 
                save_path="cot_logit_evolution.png"
            )
            print(f"[Rank {rank}] Plotting finished.", flush=True)
        elif is_main_process:
            # If we are on the main process but didn't plot, why?
            print(f"[Rank {rank}] On main process but skipping plot. "
                  f"Tokenizer exists: {tokenizer is not None}, "
                  f"Logits exist: {bool(cot_logits_list)}", flush=True)

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds
        else:
            return torch.tensor(tokens).view(1, -1)
