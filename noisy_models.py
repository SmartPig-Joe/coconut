# noisy_models.py

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithPast

class NoisyGPT2LMHeadModel(GPT2LMHeadModel):
    """
    一个继承自 GPT2LMHeadModel 的自定义模型。
    它可以在训练期间，在 transformer 的输出和 lm_head 之间为隐藏状态添加高斯噪声。
    
    噪声标准差在此类中硬编码，无需从外部传入。
    """
    def __init__(self, config):
        super().__init__(config)
        
        # --- 在这里直接设置噪声标准差 ---
        self.noise_std = 2.58
        # --------------------------------
        
        if self.noise_std > 0:
            print(f"--- INFO: NoisyGPT2LMHeadModel is active with noise_std = {self.noise_std} ---")

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # --- 核心修改：在此处注入噪声 ---
        # 只有在训练模式 (model.train()) 且噪声标准差大于0时，才添加噪声
        noise = torch.randn_like(hidden_states) * self.noise_std
        hidden_states = hidden_states + noise
        
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-int(shift_logits.size(-1))), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )