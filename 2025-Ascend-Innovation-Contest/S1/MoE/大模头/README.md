# MindNLP 模型优化详细说明 (DeepseekMoE & Qwen2-MoE)

本文档详细记录了针对 DeepseekMoE 和 Qwen2-MoE 模型的关键性能优化点，并附带了相应的核心代码实现。

## 1. DeepseekMoE 模型优化

### 1.1 MoE 推理加速：Decode 阶段 (消除 Host-Device 同步)

优化痛点: 原始实现需遍历所有专家（包括无 token 分配的专家），且通过nonzero逐个查找专家对应的 token，操作冗余且计算碎片化；每个专家独立处理 token 时缺乏批量聚集，导致index_add频繁调用，内存访问和计算效率低。

改进方案: 通过展平专家索引、权重并关联 token 索引，先对专家索引排序以聚集同专家的 token，再利用unique_consecutive筛选出有 token 分配的专家并计算其处理范围，仅遍历需实际计算的专家；对同一专家的 token 进行批量处理和单次index_add累加，避免冗余的nonzero查找和零散计算，提升整体计算效率

**源码实现** (`Qwen2MoeSparseMoeBlock`):

**Python**

```
    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        num_tokens = batch_size * sequence_length
        hidden_states = hidden_states.view(num_tokens, hidden_dim)  # 展平为 [num_tokens, hidden_dim]

        # 1. 门控计算：获取路由权重和选中的专家
        router_logits = self.gate(hidden_states)  # [num_tokens, num_experts]
        routing_weights = F.softmax(router_logits, dim=1, dtype=mindspore.float32)
        routing_weights, selected_experts = ops.topk(routing_weights, self.top_k, dim=-1)  # 各为 [num_tokens, top_k]

 
        if self.norm_topk_prob:
            routing_weights /= ops.sum(routing_weights, dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype) 

        # 2. 整理专家- token 映射关系（核心优化）
      
        flat_expert_indices = selected_experts.flatten()  
        flat_weights = routing_weights.flatten()
        token_indices = ops.arange(num_tokens).unsqueeze(1).repeat(1, self.top_k).flatten()  # [num_tokens * top_k]

        # 3. 对专家索引排序，聚集同一专家的 token（减少循环次数）
        sorted_order = flat_expert_indices.argsort()  
        sorted_experts = flat_expert_indices[sorted_order] 
        sorted_weights = flat_weights[sorted_order]  
        sorted_token_indices = token_indices[sorted_order]  

        # 4. 确定每个专家的处理范围（仅处理有 token 的专家）
        unique_experts, counts = ops.unique_consecutive(sorted_experts, return_counts=True)
        offsets = ops.cat([ops.zeros(1, dtype=mindspore.int64), counts.cumsum(0)])

        # 5. 初始化输出缓存
        final_hidden_states = ops.zeros_like(hidden_states)

        # 6. 批量处理每个专家的 token（仅遍历有 token 的专家）
        for i in range(len(unique_experts)):
            exp_id = unique_experts[i].item()  
            start = offsets[i]
            end = offsets[i + 1]
            if start >= end:
                continue  

            # 提取当前专家需要处理的 token 索引和权重
            current_tokens = sorted_token_indices[start:end]
            current_weights = sorted_weights[start:end].unsqueeze(1)  

            # 批量处理 token 并加权
            expert_input = hidden_states[current_tokens]  # [num_tokens_for_exp, hidden_dim]
            expert_output = self.experts[exp_id](expert_input) 
            expert_output = expert_output * current_weights  

            # 累加结果到对应 token 位置（一次 index_add 处理该专家所有 token）
            final_hidden_states = final_hidden_states.index_add(
                0,  # 沿 token 维度累加
                current_tokens.astype(mindspore.int32),  # token 索引（需转为 int32）
                expert_output.to(hidden_states.dtype)
            )

        # 7. 处理共享专家并累加
        shared_gate = F.sigmoid(self.shared_expert_gate(hidden_states))  
        shared_output = self.shared_expert(hidden_states) * shared_gate  
        final_hidden_states += shared_output

        # 恢复原始形状并返回
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
```



## 2. Qwen2-MoE 模型优化

### 2.1 MoE 路由逻辑优化 (索引计算)

优化痛点: 原始实现未区分 LLM 推理的 prefill（长序列）与 decode（单 token）阶段，采用统一的 “排序 - 分组 - 累加” 逻辑，导致单 token 生成时引入全局排序、bincount 等不必要的 “过度计算”，decode 阶段的资源浪费与 prefill 阶段的优化不足并存。

改进方案: 通过序列长度判断推理阶段，将 prefill（长序列）与 decode（单 token）阶段的推理逻辑拆分，prefill 阶段用批量分组处理逻辑以提升并行效率，decode 阶段则采用轻量的逐专家遍历计算以消除冗余开销。

**源码实现** (`DeepseekMoE`):

**Python**

```
    def forward(self, hidden_states):
        identity = hidden_states  # 保存原始输入用于共享专家
        orig_shape = hidden_states.shape  # 原始形状: (batch_size, seq_len, hidden_dim)
        # 门控网络输出：专家索引、权重、辅助损失（训练用）
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        
        # 展平输入为 (total_tokens, hidden_dim)，便于按token处理
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)  # 展平专家索引: (total_tokens * num_experts_per_tok,)
        flat_topk_weight = topk_weight.view(-1, 1)  # 展平专家权重: (total_tokens * num_experts_per_tok, 1)
        
        if self.training:
            raise NotImplementedError("Training is not supported yet.")
        else:
            # 根据序列长度判断阶段：prefill（长序列）或 decode（单token）
            seq_len = orig_shape[1]
            if seq_len == 1:
                # decode阶段（单token生成）：用轻量逐专家计算
                y = self.moe_infer_decode(
                    hidden_states, 
                    flat_topk_idx, 
                    flat_topk_weight
                ).view(*orig_shape)
            else:
                # prefill阶段（长序列处理）：用批量分组计算
                y = self.moe_infer_prefill(
                    hidden_states, 
                    flat_topk_idx, 
                    flat_topk_weight
                ).view(*orig_shape)
        
        # 叠加共享专家输出（若有）
        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    def moe_infer_prefill(self, x, flat_expert_indices, flat_expert_weights):
        """prefill阶段推理（长序列）：批量分组处理同一专家的token，提高并行效率"""
        expert_cache = ops.zeros_like(x)  # 缓存所有专家的加权输出
        # 对专家索引排序，将同一专家的token聚集（减少专家调用次数）
        idxs = flat_expert_indices.argsort()
        # 计算每个专家处理的token数量（累积和，用于划分范围）
        tokens_per_expert = flat_expert_indices.bincount().cumsum(0)
        # 计算排序后每个位置对应的原始token索引（每个token被num_experts_per_tok个专家处理）
        token_idxs = idxs // self.num_experts_per_tok
        
        # 遍历每个专家，批量处理分配给它的token
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue  # 无token分配给当前专家，跳过
            
            expert = self.experts[i]
            # 提取当前专家需要处理的原始token索引
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 提取对应的输入token
            expert_tokens = x[exp_token_idx]
            # 专家计算并加权
            expert_out = expert(expert_tokens)
            expert_out = expert_out.mul(flat_expert_weights[idxs[start_idx:end_idx]])
            # 将结果累加到缓存中对应的token位置
            expert_cache = mindspore.mint.scatter_add(
                expert_cache,
                0,  # 沿token维度（第0维）累加
                exp_token_idx.view(-1, 1).tile((1, x.shape[-1])),  # 扩展索引至与输出同形状
                expert_out
            )
        return expert_cache

    @no_grad()
    def moe_infer_decode(self, x, flat_expert_indices, flat_expert_weights):
        """decode阶段推理（单token）：逐专家遍历计算，避免分组排序开销"""
        expert_cache = ops.zeros_like(x)  # 缓存输出
        total_tokens = x.shape[0]  # decode阶段通常为1（batch_size * 1）
        
        # 遍历每个token的所有专家（每个token对应num_experts_per_tok个专家）
        for tok_idx in range(total_tokens):
            # 当前token的专家索引和权重（从flat数组中切片）
            start = tok_idx * self.num_experts_per_tok
            end = start + self.num_experts_per_tok
            expert_ids = flat_expert_indices[start:end]
            weights = flat_expert_weights[start:end]
            
            # 逐个专家计算并累加
            for exp_id, weight in zip(expert_ids, weights):
                expert = self.experts[exp_id.item()]  # 获取专家
                expert_out = expert(x[tok_idx:tok_idx+1])  # 处理当前token（保持维度）
                expert_cache[tok_idx:tok_idx+1] += expert_out * weight  # 加权累加
        return expert_cache
```

## 最终收益
| model_name | memory_reserved | memory_allocated | avg_prefill_latency | avg_decode_latency |
| :--- | :--- | :--- | :--- | :--- |
| Qwen1.5-MoE-A2.7B-Chat | 31.138512896 | 29.234176512 | 2.6949222882588706 | 0.22898790647852943 |
| deepseek-moe-16b-chat | 34.359738368 | 32.813018112 | 3.7060397466023765 | 0.16409588618487966 |


## 评测结果

| 评测指标 | 平均得分 |
|---------|---------|
| 峰值显存得分 | 100 |
| Prefill时延得分 | 83.907   |
| Decode时延得分 | 304.9096     |
| **总分** | **162.9389** |