import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18
import math

# -------------------------------
# Grouped Prompt 模块
# -------------------------------
class GroupedPrompt(nn.Module):
    """
    根据 MMSE 分组选择部分 prompt（x1），并拼接全局共享 prompt（x2）
    输出 shape: [B, group_len + shared_len, ctx_dim]
    """
    def __init__(self, num_groups=3, group_len=2, shared_len=3, ctx_dim=512, init_std=0.02, dtype=torch.float32):
        super().__init__()
        assert group_len + shared_len > 0

        # 保存为参数（float），在 forward 时 cast 到 clip 的 dtype
        self.group_prompt = nn.Parameter(torch.empty(num_groups, group_len, ctx_dim, dtype=dtype))
        nn.init.normal_(self.group_prompt, std=init_std)

        self.shared_prompt = nn.Parameter(torch.empty(shared_len, ctx_dim, dtype=dtype))
        nn.init.normal_(self.shared_prompt, std=init_std)

        self.num_groups = num_groups
        self.group_len = group_len
        self.shared_len = shared_len
        self.ctx_dim = ctx_dim

    def forward(self, group_ids, device=None, dtype=None):
        """
        group_ids: [B] long tensor
        returns: [B, group_len+shared_len, ctx_dim] with correct device/dtype
        """
        B = group_ids.size(0)
        # select group-specific prompts
        x1 = self.group_prompt[group_ids]  # [B, group_len, ctx_dim]
        x2 = self.shared_prompt.unsqueeze(0).expand(B, -1, -1)  # [B, shared_len, ctx_dim]

        out = torch.cat([x1, x2], dim=1)  # [B, group_len+shared_len, ctx_dim]

        if device is not None:
            out = out.to(device)
        if dtype is not None:
            out = out.to(dtype)

        return out


# -------------------------------
# ResNet3DWithCoOP
# -------------------------------
class ResNet3DWithCoOP(nn.Module):
    def __init__(self, model_clip, tokenizer, num_classes=2,
                 num_groups=3, group_len=2, shared_len=3):
        """
        model_clip: 已加载的 CLIP 模型（与 openai/clip 兼容）
        tokenizer: tokenizer，对应返回 token ids 的对象（例如 clip.simple_tokenizer）
        """
        super().__init__()

        # ---------------- backbone (3D ResNet)
        self.backbone = r3d_18(pretrained=True)
      
        self.backbone.stem[0] = nn.Conv3d(1, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False)
       
        self.backbone.fc = nn.Identity()

        self.feature_dim = 512

        # ---------------- CLIP 部分
        self.clip = model_clip
        self.tokenizer = tokenizer

        self.text_feature_dim = self.clip.ln_final.weight.shape[0]

        for param in self.clip.parameters():
            param.requires_grad = False

        self.prompt_module = GroupedPrompt(
            num_groups=num_groups,
            group_len=group_len,
            shared_len=shared_len,
            ctx_dim=self.text_feature_dim,
            dtype=torch.float32
        )

        # ---------------- 融合分类头
        fused_dim = self.feature_dim + self.text_feature_dim
        self.fc = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # 记录 n_ctx（可变）
        self.n_ctx = group_len + shared_len

        self.max_positional_len = self.clip.positional_embedding.shape[0]

    def forward(self, x, mm_score_tokens, mmse_values):
        """
        x: MRI input [B, 1, T, H, W]
        mm_score_tokens: tokenized text input (tensor of ids) [B, L]
        mmse_values: numeric MMSE values [B] 或 [B, 1] (float/int)
        """
        # ---------------- MRI 特征
        features = self.backbone(x)  # expect [B, feature_dim]
        assert features.dim() == 2, f"Backbone output must be [B, C], got {features.shape}"

        # 若 mmse_values 可能是 [B,1]，展平为 [B]
        mmse_values = mmse_values.view(-1)

        device = mm_score_tokens.device
        group_ids = torch.zeros_like(mmse_values, dtype=torch.long, device=device)
        group_ids[(mmse_values >= 10) & (mmse_values < 20)] = 1
        group_ids[(mmse_values >= 20)] = 2

        prompt = self.prompt_module(group_ids, device=device, dtype=self.clip.dtype)  # [B, n_ctx, ctx_dim]
        n_ctx = prompt.size(1)
        assert n_ctx == self.n_ctx, f"n_ctx mismatch, got {n_ctx} expected {self.n_ctx}"

        with torch.no_grad():
            base_embedding = self.clip.token_embedding(mm_score_tokens).type(self.clip.dtype)

        total_len = 1 + n_ctx + (base_embedding.shape[1] - 1)
        if total_len > self.max_positional_len:
            # 截断 base_embedding 尾部
            base_embedding = torch.cat([
                base_embedding[:, :1, :],
                base_embedding[:, 1:self.max_positional_len - n_ctx, :]
            ], dim=1)

        x_text = torch.cat([
            base_embedding[:, :1, :],  # keep start token
            prompt,                    # learned prompt: [B, n_ctx, D]
            base_embedding[:, 1:, :]   # rest tokens (including class & eos)
        ], dim=1)  # -> [B, L' (=1+n_ctx+L-1), D]

        pos = self.clip.positional_embedding[: x_text.size(1), :].unsqueeze(0).to(device).type(self.clip.dtype)
        x_text = x_text + pos  # broadcasting on batch

        x_text = x_text.permute(1, 0, 2)  # L, B, D
        x_text = self.clip.transformer(x_text)  # returns L,B,D (transformer params frozen)
        x_text = x_text.permute(1, 0, 2)  # B, L, D
        x_text = self.clip.ln_final(x_text).type(self.clip.dtype)  # B, L, D

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
          
            eos_mask = (mm_score_tokens == eos_token_id).to(torch.int32)  # B, L
           
            eos_any = eos_mask.sum(dim=1)
            eos_index = torch.where(eos_any > 0, eos_mask.argmax(dim=1), torch.tensor(mm_score_tokens.shape[1]-1, device=device))
        else:
            eos_index = mm_score_tokens.argmax(dim=-1)

        mm_score_features = x_text[torch.arange(x_text.size(0), device=device), eos_index]  # [B, D]
        if hasattr(self.clip, "text_projection") and self.clip.text_projection is not None:
            mm_score_features = mm_score_features @ self.clip.text_projection  # [B, C_v]
        else:
            pass

        # ------------- feature normalization (CoOp/CLIP 常用)
        features = F.normalize(features, dim=-1)
        mm_score_features = F.normalize(mm_score_features, dim=-1)

        # ------------- fuse 与分类
        fused = torch.cat((features, mm_score_features), dim=1)  # [B, feature_dim + text_feature_dim]
        out = self.fc(fused)  # [B, num_classes]
        return out
    
class ResNet3DWithCoOPGradCAM(ResNet3DWithCoOP):
    def __init__(self, model_clip, tokenizer, num_classes=2):
        super(ResNet3DWithCoOPGradCAM, self).__init__(model_clip, tokenizer, num_classes,num_groups=3, group_len=2, shared_len=3)
        
        # 存储特征图和梯度
        self.feature_maps = None
        self.gradients = None
        
        # 注册钩子以捕获特征图和梯度
        target_layer = self.backbone.layer4[1].conv2  # 选择最后一个卷积层
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)
    
    def forward_hook(self, module, input, output):
        # 保存前向传播的特征图
        self.feature_maps = output
    
    def backward_hook(self, module, grad_input, grad_output):
        # 保存反向传播的梯度
        self.gradients = grad_output[0]
    
    def get_feature_maps_and_gradients(self):
        return self.feature_maps, self.gradients
    
    def clear_hooks(self):
        self.feature_maps = None
        self.gradients = None

