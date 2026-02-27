from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DyGCAConfig:
    vocab_size: int
    hidden_size: int
    num_heads: int = 8
    dropout: float = 0.1
    max_position_embeddings: int = 2048
    k_focuses: int = 32
    m_selection: int = 8
    distribution: str = "beta"
    diversity_lambda: float = 0.1
    use_importance: bool = True
    use_chunk_attn: bool = True
    gate_bias_init: float = -2.0  # Section 3.4: b_g = -2
    gl_degree: int = 32  # Gauss-Legendre quadrature degree
    use_gate: bool = True  # Contrast test: Gate vs Add


class DyGCAPlugin(nn.Module):
    def __init__(self, base_model: nn.Module, config: DyGCAConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.focus_param_dim = self._focus_param_dim(config.distribution)
        
        # 3.2 Soft Chunk Construction: MLP maps H(L-1) to {alpha, beta, a}
        self.focus_proj = nn.Linear(config.hidden_size, config.k_focuses * self.focus_param_dim)
        self.importance_proj = nn.Linear(config.hidden_size, config.k_focuses) if config.use_importance else None
        
        self.chunk_attn = None
        if config.use_chunk_attn:
            self.chunk_attn = nn.MultiheadAttention(
                config.hidden_size,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
        
        # 3.3 Dynamic Cross-Attention Fusion
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Initialize K, V as identity matrices (Section 3.3)
        nn.init.eye_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.eye_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        
        # 3.4 Gated Integration: g_t = sigma(w_g^T [H(L-1); C_mean] + b_g)
        self.gate_mlp = None
        if config.use_gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(config.hidden_size * 2, 1),
            )
            # Initialize gate bias to -2 (Section 3.4)
            nn.init.constant_(self.gate_mlp[0].bias, config.gate_bias_init)
        
        # Precompute Gauss-Legendre nodes and weights on [0, 1]
        nodes, weights = np.polynomial.legendre.leggauss(config.gl_degree)
        # Map from [-1, 1] to [0, 1]
        nodes = (nodes + 1) / 2
        weights = weights / 2
        self.register_buffer("gl_nodes", torch.from_numpy(nodes).float())
        self.register_buffer("gl_weights", torch.from_numpy(weights).float())
        
        self.out_ln = nn.LayerNorm(config.hidden_size)

    def _focus_param_dim(self, distribution: str) -> int:
        if distribution in {"beta"}:
            return 2
        if distribution in {"gaussian", "laplace"}:
            return 2
        if distribution in {"studentt"}:
            return 3
        raise ValueError(f"Unsupported distribution: {distribution}")

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask

    def _build_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # 3.2: x_j = (j + 0.5) / N
        j = torch.arange(seq_len, device=device).float()
        positions = (j + 0.5) / seq_len
        return positions

    def _distribution_pdf(self, params: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # positions: (1, 1, 1, num_nodes)
        # params: (bsz, seq_len, k_focuses, dim)
        positions = positions.clamp(1e-5, 1.0 - 1e-5)
        
        if self.config.distribution == "beta":
            # 确保 alpha 和 beta 形状为 (bsz, seq_len, k_focuses, 1) 以便广播
            alpha = F.softplus(params[..., 0:1]) + 1e-4
            beta = F.softplus(params[..., 1:2]) + 1e-4
            dist = torch.distributions.Beta(alpha, beta)
            log_prob = dist.log_prob(positions)
        elif self.config.distribution == "gaussian":
            mean = torch.sigmoid(params[..., 0:1])
            scale = F.softplus(params[..., 1:2]) + 1e-4
            dist = torch.distributions.Normal(mean, scale)
            log_prob = dist.log_prob(positions)
        elif self.config.distribution == "laplace":
            mean = torch.sigmoid(params[..., 0:1])
            scale = F.softplus(params[..., 1:2]) + 1e-4
            dist = torch.distributions.Laplace(mean, scale)
            log_prob = dist.log_prob(positions)
        elif self.config.distribution == "studentt":
            df = F.softplus(params[..., 0:1]) + 1.0
            mean = torch.sigmoid(params[..., 1:2])
            scale = F.softplus(params[..., 2:3]) + 1e-4
            dist = torch.distributions.StudentT(df, mean, scale)
            log_prob = dist.log_prob(positions)
        else:
            raise ValueError(f"Unknown distribution: {self.config.distribution}")
        
        return torch.exp(log_prob)

    def _diversity_loss(self, focus_params: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # 3.5: JS Divergence via Gauss-Legendre Quadrature
        bsz, seq_len, k_focuses, _ = focus_params.shape
        if k_focuses <= 1:
            return torch.zeros([], device=focus_params.device)
        
        # Evaluate PDF at GL nodes: (bsz, seq_len, k, num_nodes)
        nodes = self.gl_nodes.view(1, 1, 1, -1)
        pdf = self._distribution_pdf(focus_params, nodes)
        
        # JS(P, Q) integral is approximated by sum(weights * JS_at_nodes)
        p_i = pdf.unsqueeze(3)  # (bsz, seq_len, k, 1, num_nodes)
        p_j = pdf.unsqueeze(2)  # (bsz, seq_len, 1, k, num_nodes)
        
        m = 0.5 * (p_i + p_j)
        
        # Continuous KL approximation: integral(P * log(P/M)) dx approx sum(w * P * log(P/M))
        eps = 1e-9
        kl_i_m = (p_i * (torch.log(p_i + eps) - torch.log(m + eps)))
        kl_j_m = (p_j * (torch.log(p_j + eps) - torch.log(m + eps)))
        
        js_at_nodes = 0.5 * (kl_i_m + kl_j_m)
        
        # Quadrature integration over nodes
        gl_weights = self.gl_weights.view(1, 1, 1, 1, -1)
        js_pair = (js_at_nodes * gl_weights).sum(dim=-1) # (bsz, seq_len, k, k)
        
        upper = torch.triu(js_pair, diagonal=1)
        
        # 应用 attention_mask 排除 Padding Token 的损失
        if attention_mask is not None:
            # attention_mask: (bsz, seq_len) -> (bsz, seq_len, 1, 1)
            mask = attention_mask.view(bsz, seq_len, 1, 1).to(upper.dtype)
            upper = upper * mask
            # 修正分母为有效 Token 数量
            active_tokens = attention_mask.sum()
            denom = active_tokens * (k_focuses * (k_focuses - 1) / 2)
        else:
            denom = bsz * seq_len * (k_focuses * (k_focuses - 1) / 2)
            
        return upper.sum() / max(denom, 1.0)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        # 3.2: Position-aware representations via RoPE
        # Try to use base model's rotary_emb if available
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "rotary_emb"):
            rotary_emb = self.base_model.model.rotary_emb
            bsz, seq_len, h = x.shape
            # Qwen2RotaryEmbedding expects (seq_len, device)
            # It returns (cos, sin) each of shape (1, seq_len, 1, h)
            cos, sin = rotary_emb(x, torch.arange(seq_len, device=x.device).unsqueeze(0))
            
            # Standard RoPE rotation logic
            def rotate_half(x):
                x1 = x[..., : x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1)
            
            # cos/sin are (1, seq_len, 1, head_dim) -> (1, seq_len, head_dim)
            cos = cos.squeeze(2)
            sin = sin.squeeze(2)
            
            # If embedding dim > rotary dim, we repeat RoPE or only apply to part
            # Standard Qwen applies RoPE to the whole head. 
            # Here we treat the embedding as multiple heads of head_dim
            head_dim = cos.shape[-1]
            if h > head_dim:
                # Repeat cos/sin to match hidden_size
                num_repeats = h // head_dim
                cos = cos.repeat(1, 1, num_repeats)
                sin = sin.repeat(1, 1, num_repeats)
            
            x_rope = (x * cos) + (rotate_half(x) * sin)
            return x_rope
        return x

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None) -> dict:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        
        # 3.2: Input embeddings E_j
        base = self.base_model.get_input_embeddings()(input_ids)
        # Apply RoPE to input embeddings (Section 3.2: tilde{E}_j = RoPE(E_j, j))
        base = self._apply_rope(base)
        
        outputs = self.base_model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        
        # Section 3.1: Hidden representation of penultimate layer (L-1)
        # outputs.hidden_states[0] is embedding, [1...L] are layer outputs
        h_penultimate = outputs.hidden_states[-2] 
        # Section 3.3: Query from final layer (L)
        h_final = outputs.hidden_states[-1]
        
        target_dtype = self.focus_proj.weight.dtype
        h_penultimate = h_penultimate.to(target_dtype)
        h_final = h_final.to(target_dtype)
        base = base.to(target_dtype)
        
        # 3.2 Soft Chunk Construction: {alpha, beta, a} = MLP(H_t(L-1))
        focus_params_raw = self.focus_proj(h_penultimate)
        focus_params_raw = focus_params_raw.view(bsz, seq_len, self.config.k_focuses, self.focus_param_dim)
        
        if self.importance_proj is None:
            importance_raw = torch.ones(bsz, seq_len, self.config.k_focuses, device=device, dtype=target_dtype)
        else:
            importance_raw = F.softplus(self.importance_proj(h_penultimate))
        
        # 时间错位处理: 用 t-1 步的信息预测 t 步的焦点 (因果性)
        focus_params = torch.zeros_like(focus_params_raw)
        importance = torch.zeros_like(importance_raw)
        focus_params[:, 1:] = focus_params_raw[:, :-1]
        importance[:, 1:] = importance_raw[:, :-1]
        
        # 3.2: Normalized coordinates x_j = (j + 0.5) / N
        pos = self._build_positions(seq_len, device).view(seq_len, 1, 1, 1)
        pdf = self._distribution_pdf(focus_params, pos) # (seq_len, bsz, s, k)
        pdf = pdf.permute(1, 2, 3, 0) # (bsz, seq_len, k, seq_len)
        
        # 3.2: w_tilde normalization
        causal = self._causal_mask(seq_len, device).view(1, seq_len, 1, seq_len)
        pdf = pdf * causal
        
        # 应用 attention_mask 排除 Padding Token 对焦点概率的影响
        if attention_mask is not None:
            # attention_mask: (bsz, seq_len) -> (bsz, 1, 1, seq_len)
            mask = attention_mask.view(bsz, 1, 1, seq_len).to(pdf.dtype)
            pdf = pdf * mask
            
        probs = pdf / (pdf.sum(dim=-1, keepdim=True) + 1e-9)
        
        # 3.2: w_k,j = a_k * w_tilde
        # importance shape: (bsz, seq_len, k)
        weights = probs * importance.unsqueeze(-1)
        
        # 3.2: Chunk vectors C_k = sum(w_k,j * E_j)
        if self.chunk_attn is None:
            # Simple weighted pooling
            chunk_vectors = (weights.unsqueeze(-1) * base.unsqueeze(1).unsqueeze(2)).sum(dim=3)
        else:
            # Optional Chunk Self-Attention
            weighted_tokens = base.unsqueeze(1).unsqueeze(2) * probs.unsqueeze(-1)
            # RuntimeError: view size is not compatible with input tensor's size and stride
            # 使用 .reshape() 代替 .view()，或者在 .view() 前调用 .contiguous()
            attn_in = weighted_tokens.reshape(bsz * seq_len * self.config.k_focuses, seq_len, -1)
            attn_out, _ = self.chunk_attn(attn_in, attn_in, attn_in, need_weights=False)
            attn_out = attn_out.view(bsz, seq_len, self.config.k_focuses, seq_len, -1)
            # Apply importance magnitude after chunk attention
            chunk_vectors = (attn_out * weights.unsqueeze(-1)).sum(dim=3)
            
        # 3.3: Selection of M focuses (Refinement)
        topk_indices = torch.topk(importance, self.config.m_selection, dim=-1).indices
        topk_expand = topk_indices.unsqueeze(-1).expand(-1, -1, -1, chunk_vectors.shape[-1])
        selected_chunks = torch.gather(chunk_vectors, 2, topk_expand)
        
        # 3.3: Cross-Attention Fusion
        q = self.q_proj(h_final)
        k = self.k_proj(selected_chunks)
        v = self.v_proj(selected_chunks)
        
        # (bsz, seq_len, 1, h) * (bsz, seq_len, m, h) -> (bsz, seq_len, m)
        scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(h_final.shape[-1])
        attn = torch.softmax(scores, dim=-1)
        f_t = torch.einsum("bsm,bsmh->bsh", attn, v)
        
        # 3.4: Gated Integration
        if self.gate_mlp is not None:
            # g_t = sigma(w_g^T [H(L-1); C_mean] + b_g)
            c_mean = chunk_vectors.mean(dim=2) # Mean over K focuses
            gate_input = torch.cat([h_penultimate, c_mean], dim=-1)
            g_t = torch.sigmoid(self.gate_mlp(gate_input))
            
            # O_t = g_t * F_t + (1 - g_t) * S_t
            # S_t is the standard self-attention output (h_final)
            hidden = g_t * f_t + (1.0 - g_t) * h_final
        else:
            # Contrast test: Simple Addition (Add)
            hidden = h_final + f_t
            
        hidden = self.out_ln(hidden)
        
        # Generation
        lm_head = self.base_model.get_output_embeddings()
        logits_hidden = hidden.to(lm_head.weight.dtype)
        logits = lm_head(logits_hidden)
        
        output = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # 3.5: JS Divergence via Gauss-Legendre Quadrature (Continuous Space)
            diversity_loss = self._diversity_loss(focus_params, attention_mask=attention_mask)
            loss = lm_loss + self.config.diversity_lambda * diversity_loss
            output.update({
                "loss": loss,
                "lm_loss": lm_loss,
                "diversity_loss": diversity_loss,
            })
        return output
