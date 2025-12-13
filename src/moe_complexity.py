"""
MoE Complexity Analysis: 使用不同训练阶段的模型作为专家
观察在分布外数据上，路由器如何选择不同阶段的专家

实验设置：
- 3个专家：seed101在5000步、20000步、100000步的checkpoint（冻结权重，wd=1.0）
- 专家训练数据：P=97（原始训练分布）
- 路由器训练数据：P∈{83, 89}（分布外数据，OOD）
- 路由器测试数据：P∈{79, 73}（更小的质数，测试泛化）
- 只训练路由器网络，专家权重完全冻结

科学意义：
- 在OOD数据上，早期 vs 中期 vs 晚期模型哪个更有用？
- Grokking后的模型是否真的学到了更通用的算术规律？
- 不同训练阶段模型的知识如何在新任务上互补？
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

# ============================================================================
# 1. 模型定义（从grokking_multiseed.py复制）
# ============================================================================

class Block(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim), nn.Dropout(dropout)
        )
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, dim=128, num_layers=2, num_heads=4, num_tokens=97, seq_len=5, dropout=0.0):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList([Block(dim, num_heads, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer("causal_mask", mask)
        self.register_buffer("pos_ids", torch.arange(seq_len))
    def forward(self, x):
        h = self.token_embeddings(x) + self.position_embeddings(self.pos_ids[:x.size(1)])
        mask = self.causal_mask[:x.size(1), :x.size(1)]
        for layer in self.layers:
            h = layer(h, attn_mask=mask)
        return self.head(self.ln_f(h))

# ============================================================================
# 2. 加载预训练专家
# ============================================================================

def load_expert_model(checkpoint_path, device):
    """从checkpoint加载预训练模型作为专家"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 重建模型结构
    model = Decoder(dim=128, num_layers=2, num_heads=4, num_tokens=99, seq_len=5).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 冻结为评估模式
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    return model

# ============================================================================
# 3. MoE路由器
# ============================================================================

class Router(nn.Module):
    """学习如何分配权重给不同的专家"""
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts)
        )
    
    def forward(self, x):
        """返回专家权重 (batch_size, num_experts)"""
        return F.softmax(self.net(x), dim=-1)

class MoEModel(nn.Module):
    """混合专家模型"""
    def __init__(self, experts, input_dim=128):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.router = Router(input_dim, self.num_experts)
        
        # 冻结专家
        for expert in self.experts:
            expert.eval()
            for param in expert.parameters():
                param.requires_grad = False
    
    def forward(self, x, return_weights=False):
        """
        MoE前向传播：
        1. 原始输入token经过3个专家，得到3个输出logits
        2. 路由器根据输入特征决定每个专家的权重
        3. 对专家输出的概率分布进行加权求和
        
        Args:
            x: (batch_size, seq_len) token indices
            return_weights: 是否返回路由权重
            
        Returns:
            log_probs: (batch_size, seq_len, vocab_size) 加权后的对数概率
            router_weights (可选): (batch_size, num_experts) 路由权重
        """
        batch_size = x.size(0)
        
        # 步骤1: 所有专家处理相同的输入token
        expert_logits = []
        with torch.no_grad():  # 专家权重冻结，不需要梯度
            for expert in self.experts:
                logits = expert(x)  # (batch, seq_len, vocab_size)
                expert_logits.append(logits)
        
        # (num_experts, batch, seq_len, vocab_size)
        expert_logits = torch.stack(expert_logits, dim=0)
        
        # 步骤2: 路由器基于输入特征决定权重
        # 使用第一个专家的embedding层提取输入特征（所有专家的embedding应该相似）
        with torch.no_grad():
            expert_model = self.experts[0]
            token_emb = expert_model.token_embeddings(x)  # (batch, seq, dim)
            pos_ids = torch.arange(x.size(1), device=x.device)
            pos_emb = expert_model.position_embeddings(pos_ids)  # (seq, dim)
            input_features = token_emb + pos_emb.unsqueeze(0)  # (batch, seq, dim)
            
            # 使用序列的平均表示作为路由输入
            routing_input = input_features.mean(dim=1)  # (batch, dim=128)
        
        # 获取路由权重
        router_weights = self.router(routing_input)  # (batch, num_experts)
        
        # 步骤3: 加权组合专家的输出
        # 将logits转换为log概率，然后加权组合
        # router_weights: (batch, num_experts)
        # expert_logits: (num_experts, batch, seq_len, vocab_size)
        
        # 转换为 (batch, num_experts, seq_len, vocab_size)
        expert_logits = expert_logits.permute(1, 0, 2, 3)
        
        # 将logits转换为概率分布
        expert_probs = F.softmax(expert_logits, dim=-1)  # (batch, num_experts, seq_len, vocab_size)
        
        # 扩展router_weights的维度进行加权
        router_weights_expanded = router_weights.unsqueeze(-1).unsqueeze(-1)  # (batch, num_experts, 1, 1)
        
        # 加权求和概率
        mixed_probs = (expert_probs * router_weights_expanded).sum(dim=1)  # (batch, seq_len, vocab_size)
        
        # 转换为log概率（用于nll_loss）
        mixed_log_probs = torch.log(mixed_probs + 1e-10)  # (batch, seq_len, vocab_size)
        
        if return_weights:
            return mixed_log_probs, router_weights
        return mixed_log_probs

# ============================================================================
# 4. 生成训练数据（复用seed42的训练集）
# ============================================================================

def get_data(p, eq_token, op_token):
    """
    生成模运算数据：x-y mod p
    与grokking_multiseed.py中的逻辑一致
    """
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T
    
    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    result = (x - y) % p
    
    return torch.stack([x, op, y, eq, result]).T

def get_ood_data(p_list, eq_token, op_token):
    """
    生成分布外数据：使用不同于训练时的质数P
    
    Args:
        p_list: 质数列表，例如[83, 89]
        eq_token: 等号token
        op_token: 操作符token
    
    Returns:
        所有质数的数据合并
    """
    all_data = []
    for p in p_list:
        data = get_data(p, eq_token, op_token)
        all_data.append(data)
    
    return torch.cat(all_data, dim=0)

def get_ood_train_test_split(train_primes, test_primes, eq_token, op_token):
    """
    生成OOD训练集和测试集
    
    Args:
        train_primes: 用于训练的质数列表，如[83, 89]
        test_primes: 用于测试的质数列表，如[79, 73]
        eq_token: 等号token
        op_token: 操作符token
    
    Returns:
        train_data, test_data
    """
    train_data = get_ood_data(train_primes, eq_token, op_token)
    test_data = get_ood_data(test_primes, eq_token, op_token)
    
    return train_data, test_data

# ============================================================================
# 5. 训练和评估
# ============================================================================

def train_moe_router(experts, train_loader, test_loader, device, num_epochs=50):
    """训练MoE路由器"""
    moe_model = MoEModel(experts).to(device)
    optimizer = torch.optim.Adam(moe_model.router.parameters(), lr=1e-3)
    
    # 记录训练过程
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'expert_weights': []  # 每个epoch的平均专家权重
    }
    
    for epoch in range(num_epochs):
        # 训练
        moe_model.train()
        train_losses = []
        train_accs = []
        epoch_expert_weights = []
        
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            inp, target = batch_x[:, :-1], batch_x[:, -1]
            
            optimizer.zero_grad()
            log_probs, router_weights = moe_model(inp, return_weights=True)
            # log_probs现在是log概率，使用nll_loss
            loss = F.nll_loss(log_probs[:, -1, :], target)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            acc = (log_probs[:, -1, :].argmax(-1) == target).float().mean().item()
            train_accs.append(acc)
            
            # 记录路由权重
            epoch_expert_weights.append(router_weights.detach().cpu().numpy())
        
        # 评估
        moe_model.eval()
        test_losses = []
        test_accs = []
        
        with torch.no_grad():
            for (batch_x,) in test_loader:
                batch_x = batch_x.to(device)
                inp, target = batch_x[:, :-1], batch_x[:, -1]
                
                log_probs = moe_model(inp, return_weights=False)
                loss = F.nll_loss(log_probs[:, -1, :], target)
                acc = (log_probs[:, -1, :].argmax(-1) == target).float().mean().item()
                
                test_losses.append(loss.item())
                test_accs.append(acc)
        
        # 计算平均专家权重
        avg_expert_weights = np.concatenate(epoch_expert_weights, axis=0).mean(axis=0)
        
        history['epoch'].append(epoch)
        history['train_loss'].append(np.mean(train_losses))
        history['train_acc'].append(np.mean(train_accs))
        history['test_loss'].append(np.mean(test_losses))
        history['test_acc'].append(np.mean(test_accs))
        history['expert_weights'].append(avg_expert_weights)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Train Acc: {history['train_acc'][-1]:.3f} | "
                  f"Test Acc: {history['test_acc'][-1]:.3f} | "
                  f"Expert Weights: {avg_expert_weights}")
    
    return history, moe_model

# ============================================================================
# 6. 可视化和保存
# ============================================================================

def plot_moe_results(history, expert_steps, save_dir):
    """绘制MoE训练结果"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = history['epoch']
    expert_weights_array = np.array(history['expert_weights'])  # (num_epochs, num_experts)
    
    # 1. 准确率曲线
    ax1.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax1.plot(epochs, history['test_acc'], 'r-', label='Test Acc', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('MoE Router Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # 2. 损失曲线
    ax2.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax2.plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Curves')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 专家权重分配演化
    colors = ['green', 'blue', 'red']
    for i, (step, color) in enumerate(zip(expert_steps, colors)):
        ax3.plot(epochs, expert_weights_array[:, i], 
                color=color, label=f'Expert (Step {step:,})', linewidth=2)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Router Weight')
    ax3.set_title('Expert Selection Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.05, 1.05)
    
    plt.suptitle('MoE Analysis: Router Learning on Out-of-Distribution Data\\n'
                 'Task: x-y | Experts from Steps [1000, 10000, 100000]', 
                 fontsize=13, y=1.02)
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'moe_router_evolution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plot saved: {save_path}")

def save_results(history, expert_steps, save_dir):
    """保存训练历史到CSV"""
    csv_path = Path(save_dir) / 'moe_training_history.csv'
    
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc']
        for i, step in enumerate(expert_steps):
            fieldnames.append(f'expert_{step}_weight')
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(len(history['epoch'])):
            row = {
                'epoch': history['epoch'][i],
                'train_loss': history['train_loss'][i],
                'train_acc': history['train_acc'][i],
                'test_loss': history['test_loss'][i],
                'test_acc': history['test_acc'][i]
            }
            for j, step in enumerate(expert_steps):
                row[f'expert_{step}_weight'] = history['expert_weights'][i][j]
            
            writer.writerow(row)
    
    print(f"✓ Training history saved: {csv_path}")

# ============================================================================
# 7. 主函数
# ============================================================================

def evaluate_single_expert(expert, data_loader, device):
    """评估单个专家的准确率"""
    expert.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for (batch_x,) in data_loader:
            batch_x = batch_x.to(device)
            inp, target = batch_x[:, :-1], batch_x[:, -1]
            
            logits = expert(inp)  # (batch, seq, vocab)
            pred = logits[:, -1, :].argmax(-1)  # 预测最后一个位置
            
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return correct / total

def main():
    # 配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 专家checkpoint路径（x-y任务，WD=1.0，seed=101）
    checkpoint_dir = Path("/root/autodl-tmp/test/results/checkpoints/x-y/wd_1.0")
    expert_steps = [1000, 10000, 100000]  # 早期、中期、晚期三个阶段
    expert_checkpoints = [
        checkpoint_dir / f"seed101_step{step}.pt" for step in expert_steps
    ]
    
    # 检查文件是否存在
    for ckpt in expert_checkpoints:
        if not ckpt.exists():
            print(f"Error: Checkpoint not found: {ckpt}")
            return
    
    print("="*70)
    print("MoE Complexity Analysis Experiment")
    print("="*70)
    print("Configuration:")
    print(f"  - Task: x-y modular arithmetic")
    print(f"  - Experts: seed101 (wd=1.0) at steps {expert_steps}")
    print(f"  - Training: Experts trained on P=97")
    print(f"  - Testing: Router trained on OOD data (P ≠ 97)")
    print(f"  - Device: {device}")
    print("="*70)
    print("\nLoading expert models...")
    
    # 加载专家
    experts = []
    for step, ckpt_path in zip(expert_steps, expert_checkpoints):
        print(f"Loading Expert from Step {step:,}...")
        expert = load_expert_model(ckpt_path, device)
        experts.append(expert)
    
    print(f"✓ Loaded {len(experts)} experts")
    
    # 使用分布外数据（P ≠ 97）
    print("\nGenerating out-of-distribution data (P ≠ 97)...")
    train_primes = [83, 89]  # OOD训练集：较大质数
    test_primes = [79, 73]   # OOD测试集：较小质数
    
    train_data, test_data = get_ood_train_test_split(
        train_primes=train_primes,
        test_primes=test_primes,
        eq_token=97, 
        op_token=98
    )
    print(f"✓ Generated OOD data")
    print(f"  Train primes: {train_primes}")
    print(f"  Test primes:  {test_primes}")
    
    train_loader = DataLoader(TensorDataset(train_data), batch_size=256, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=256, shuffle=False)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # 先评估各个专家单独的准确率
    print("\n" + "="*70)
    print("Evaluating Individual Experts...")
    print("="*70)
    
    for i, (step, expert) in enumerate(zip(expert_steps, experts)):
        train_acc = evaluate_single_expert(expert, train_loader, device)
        test_acc = evaluate_single_expert(expert, test_loader, device)
        print(f"Expert {i+1} (Step {step:6,}): Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    # 训练路由器
    print("\n" + "="*70)
    print("Training MoE Router...")
    print("="*70)
    
    history, moe_model = train_moe_router(experts, train_loader, test_loader, device, num_epochs=50)
    
    # 保存结果
    output_dir = Path("/root/autodl-tmp/test/results/moe")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Saving results...")
    save_results(history, expert_steps, output_dir)
    plot_moe_results(history, expert_steps, output_dir)
    
    # 最终分析
    print("\n" + "="*70)
    print("Final Analysis:")
    print("="*70)
    final_weights = history['expert_weights'][-1]
    for step, weight in zip(expert_steps, final_weights):
        print(f"Expert (Step {step:6,}): {weight:.4f}")
    
    print(f"\nFinal Test Accuracy: {history['test_acc'][-1]:.4f}")
    print(f"\n✓ All results saved to: {output_dir}")

if __name__ == "__main__":
    main()

