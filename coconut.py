import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


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
                # 将该位置的嵌入替换为前一个token（token_idx-1）的隐藏状态
                # 需要减去hidden_states_offset来校正索引
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

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

    def generate(
        self,
        input_ids,
        attention_mask,  # 注意力掩码在此方法中未被使用
        max_new_tokens=16,  # 最多生成的新token数量
        output_embedding=False,  # 是否返回生成的嵌入向量
        synced_gpus=False,  # 是否在FSDP等分布式训练中同步GPU的前向传播次数
        **kwargs
    ):

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
        '''
        # 1. 将整个序列的嵌入向量重塑为 (seq_len, hidden_size) 以便PCA
        all_embeddings = inputs_embeds[0].detach().cpu()  # Shape: [seq_len, hidden_size]
        seq_len = all_embeddings.shape[0]
        print(f"Total number of tokens in the sequence: {seq_len}")

        # 2. 执行PCA降维
        pca = PCA(n_components=2)
        all_embeddings_2d = pca.fit_transform(all_embeddings)  # Shape: [seq_len, 2]
        explained_variance = pca.explained_variance_ratio_.sum()
        print(f"PCA explained variance ratio: {explained_variance:.2f}")

        # 3. 计算每个嵌入向量的 L2 范数
        # .norm(dim=1) 计算每个 token 向量在特征维度上的 L2 范数
        embedding_norms = all_embeddings.norm(dim=1).numpy()  # Shape: [seq_len]
        print(f"Min Norm: {embedding_norms.min():.3f}, Max Norm: {embedding_norms.max():.3f}, Mean Norm: {embedding_norms.mean():.3f}")

        # 4. 找到潜在标记（latent）的位置
        latent_positions = (input_ids[0] == self.latent_token_id).nonzero().squeeze(-1)
        latent_positions = latent_positions.cpu().numpy()

        # 5. 创建掩码，区分普通token和latent token
        is_latent = torch.zeros(seq_len, dtype=torch.bool)
        is_latent[latent_positions] = True

        # 6. 分离普通token和latent token的坐标 (用于PCA)
        normal_points = all_embeddings_2d[~is_latent]
        latent_points = all_embeddings_2d[is_latent]

        # 7. 创建包含两个子图的画布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))  # 一行两列

        # === 子图 1: PCA Visualization ===
        # 绘制普通 token
        ax1.scatter(normal_points[:, 0], normal_points[:, 1],
                    c='lightgray', s=100, alpha=0.9, label='Normal Tokens', edgecolors='blue')
        # 绘制 latent token
        ax1.scatter(latent_points[:, 0], latent_points[:, 1],
                    c='red', s=100, alpha=0.9, label='Latent Tokens', edgecolors='black', linewidth=1)

        # 连接 latent token 的轨迹
        if len(latent_points) > 1:
            ax1.plot(latent_points[:, 0], latent_points[:, 1],
                     color='red', linewidth=2.5, alpha=0.7, zorder=5)

        # 为每个 latent 点标注序号
        for i, (x, y) in enumerate(latent_points):
            ax1.annotate(f'L{i+1}', (x, y),
                        textcoords="offset points", xytext=(0,10),
                        ha='center', fontsize=12, fontweight='bold', color='darkred')

        ax1.set_title(f"PCA Visualization of Embeddings\nExplained Variance: {explained_variance:.2f}")
        ax1.set_xlabel("First Principal Component")
        ax1.set_ylabel("Second Principal Component")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # === 子图 2: Embedding Norm Curve ===
        # 绘制所有 token 的范数曲线
        ax2.plot(range(seq_len), embedding_norms, color='tab:blue', linewidth=2, label='Embedding Norm')

        ax2.set_title("L2 Norm of Each Token Embedding")
        ax2.set_xlabel("Token Position (Sequence Index)")
        ax2.set_ylabel("L2 Norm")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 10. 保存并关闭
        save_path = 'pca_of_latent.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        '''


        # --- 生成第一个新token ---
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # --- 生成后续token ---
        first_layer_attn = None
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(
                inputs_embeds=new_inputs_embeds,
                output_attentions=True,
                output_hidden_states=False,
            )
            self.gen_forward_cnt += 1

            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break

            first_layer_attn = outputs.attentions[0][-1]
            avg_attn = first_layer_attn.mean(dim=0).cpu().numpy()  # [seq_len, seq_len]
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if output_embedding:
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds
        else:
            return torch.tensor(tokens).view(1, -1)