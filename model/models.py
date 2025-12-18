import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import resnet18

# =========================================================================
# 1. Canonical MLP (Symbolic & Math)
# =========================================================================
class CanonicalMLP(nn.Module):
    """
    标准的 MLP，用于符号回归、稀疏多项式等任务。
    DLT 理论研究中最常用的 'Toy Model'，容易观察 Grokking。
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten()
        layers = []
        
        # Input Layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0: layers.append(nn.Dropout(dropout))
        
        # Hidden Layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0: layers.append(nn.Dropout(dropout))
            
        # Output Layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming Init 对 ReLU 网络至关重要，防止梯度消失/爆炸
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)


# =========================================================================
# 2. ResNet-18 Modified (Vision)
# =========================================================================
class ResNet18Small(nn.Module):
    """
    修改版的 ResNet-18，适配小尺寸图片 (32x32 或 64x64)。
    标准的 ResNet 第一层是 7x7 conv stride 2，这对于 CIFAR/dSprites 来说降采样太快了。
    """
    def __init__(self, input_channels, num_classes):
        super().__init__()
        # 加载标准结构，不预训练（我们要观察从零开始的 Grokking）
        self.net = resnet18(weights=None)
        
        # 修改第一层卷积：适应通道数 (1 for MNIST/dSprites)，去除下采样
        self.net.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        # 去掉第一层的 MaxPool，防止信息丢失过多
        self.net.maxpool = nn.Identity()
        
        # 修改全连接层适应输出
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x):
        return self.net(x)


# =========================================================================
# 3. Causal Transformer (NanoGPT) (Text & Sequence)
# =========================================================================
class CausalTransformer(nn.Module):
    """
    GPT 风格的 Decoder-only Transformer。
    用于生成式任务：TinyShakespeare, Dyck-2, String Reversal。
    特点：有因果掩码 (Causal Mask)，只能看过去，不能看未来。
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, output_dim, max_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=4*d_model, 
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_dim)
        self.d_model = d_model

    def forward(self, x):
        # x: [Batch, Seq_Len]
        seq_len = x.size(1)
        
        # 1. Embedding + Positional
        x = self.embedding(x.long()) * math.sqrt(self.d_model)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # 2. Causal Mask (Upper triangular -inf)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        
        # 3. Transformer
        x = self.transformer(x, mask=mask)
        
        # 4. Output: 取最后一个 Token 的输出用于分类/预测下一个词
        # 或者如果是 seq2seq，返回整个序列。这里为了通用性，我们取 Mean 或 Last。
        # 对于分类任务 (Sorting)，通常取 Mean；对于生成 (Next Token)，取 Last。
        # 这里默认取 Mean 用于分类准确率计算 (符合 data.py 的 label 设计)
        x = x.mean(dim=1) 
        return self.decoder(x)


# =========================================================================
# 4. Bidirectional Transformer (BERT-Tiny) (Logic & Reasoning)
# =========================================================================
class BidirectionalTransformer(nn.Module):
    """
    BERT 风格的 Encoder-only Transformer。
    用于逻辑推理任务：3-SAT, Graph Connectivity, Latin Squares。
    特点：全向注意力 (No Mask)，可以看到整个上下文进行推理。
    """
    def __init__(self, input_dim, d_model, num_heads, num_layers, output_dim, max_len=512):
        super().__init__()
        # 逻辑任务的输入可能是连续向量（Flattened Graph）或离散 Token
        self.is_discrete = False
        if input_dim > 500: # Heuristic: Large input dim implies discrete vocab
             self.is_discrete = True
             self.embedding = nn.Embedding(input_dim, d_model)
        else:
             self.embedding = nn.Linear(input_dim, d_model) # 投影层

        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=4*d_model, 
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: [Batch, Seq/Dim]
        if self.is_discrete:
            x = self.embedding(x.long())
        else:
            # 如果输入是 [Batch, Dim] (如 Flattened Graph)，unsqueeze 成 [Batch, 1, Dim]
            # 或者将其视为序列长度为 L 的向量。
            # 为了简单，对于 Graph Adj (Flattened)，我们先 Linear 投影成 d_model，然后当做一个长 Token 处理
            if x.dim() == 2: 
                x = x.unsqueeze(1) # [Batch, 1, Dim] -> project -> [Batch, 1, d_model]
            x = self.embedding(x)

        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # No Mask (Bidirectional)
        x = self.transformer(x)
        
        # Global Average Pooling for Reasoning Result
        x = x.mean(dim=1)
        return self.decoder(x)


# =========================================================================
# Model Factory
# =========================================================================
def get_model(modality, input_shape, output_dim, device='cuda'):
    """
    根据模态和输入形状自动分发 4 种架构
    """
    # 计算展平后的维度 (给 MLP 用)
    flat_dim = 1
    for d in input_shape: flat_dim *= d
    
    # -------------------------------------------
    # 1. Vision -> ResNet-18 (Modified)
    # -------------------------------------------
    if modality == 'vision':
        channels = input_shape[0] # [C, H, W]
        model = ResNet18Small(input_channels=channels, num_classes=output_dim)

    # -------------------------------------------
    # 2. Symbolic -> Canonical MLP
    # -------------------------------------------
    elif modality == 'symbolic':
        model = CanonicalMLP(
            input_dim=flat_dim,
            hidden_dim=512,
            output_dim=output_dim,
            num_layers=4
        )

    # -------------------------------------------
    # 3. Text -> Causal Transformer (GPT)
    # -------------------------------------------
    elif modality == 'text':
        # Text 任务 input_shape 通常是 [Seq_Len]
        # Vocab 大小我们假设一个足够大的值或者从 data 传入，这里设默认 10000
        # 如果是 Char-level (TinyShakespeare), vocab 约 65-100
        vocab_size = 10000 
        if output_dim > 100: vocab_size = output_dim # Heuristic
        
        model = CausalTransformer(
            vocab_size=vocab_size,
            d_model=128,
            num_heads=4,
            num_layers=2,
            output_dim=output_dim
        )

    # -------------------------------------------
    # 4. Logic -> Bidirectional Transformer (BERT)
    # -------------------------------------------
    elif modality == 'logic':
        # Logic 任务可能是离散的 (3-SAT) 也可能是连续/Flat的 (Graph)
        model = BidirectionalTransformer(
            input_dim=flat_dim if flat_dim < 5000 else 10000, # 简单阈值判断
            d_model=128,
            num_heads=4,
            num_layers=3,
            output_dim=output_dim
        )
        
    else:
        raise ValueError(f"Unknown modality: {modality}")

    return model.to(device)