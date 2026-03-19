import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.policies.act.modeling_act import ACTPolicy
# ================= 配置区 =================
CHECKPOINT_PATH = "/home/cyw/lerobot_s/outputs/train/act_2_3w_cyw_model/checkpoints/last/pretrained_model" # 替换为你的权重路径
DATASET_REPO = "/home/cyw/.cache/huggingface/lerobot/cyw_train/smolvla_2_v2" # 替换为你的数据集ID
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_task_label(episode_idx):
    """根据你的实验设计分配标签"""
    if episode_idx < 25:
        return "Grab Tissue (Single)", "blue"
    elif episode_idx < 50:
        return "Grab Tape (Single)", "red"
    else:
        return "Grab Tissue (Mixed)", "green"

def main():
    # 1. 加载数据集
    dataset = LeRobotDataset(DATASET_REPO)
    
    # 2. 加载模型
    # 注意：这里需要确保加载的是你修改后的 modeling_act_cyw.py 对应的策略
    policy = ACTPolicy.from_pretrained(CHECKPOINT_PATH)
    policy.to(DEVICE)
    policy.eval()

    all_mus = []
    all_labels = []
    all_colors = []

    print("正在提取隐变量 z (mu)...")
    print(dir(dataset))
    print(dataset.meta)
    # 我们从每个 Episode 中抽取中间帧进行分析（代表稳定的样式）
    # for ep_idx in tqdm(range(dataset.num_episodes)):
    #     # 获取该 Episode 的所有索引
    #     from_idx = dataset.episode_data_index['from'][ep_idx].item()
    #     to_idx = dataset.episode_data_index['to'][ep_idx].item()
    #     mid_idx = (from_idx + to_idx) // 2
        
    #     # 获取 Batch 数据
    #     batch = dataset[mid_idx]
# 预先获取所有帧的 episode 索引，避免在循环中重复读取
    # 这里的 dataset.hf_dataset 是底层的 Hugging Face Dataset 对象
    all_episode_indices = torch.tensor(dataset.hf_dataset["episode_index"])

    for ep_idx in tqdm(range(dataset.num_episodes)):
        # 找到属于当前 episode 的所有帧索引
        episode_mask = (all_episode_indices == ep_idx)
        indices = torch.where(episode_mask)[0]
        
        if len(indices) == 0:
            continue
            
        from_idx = indices[0].item()
        to_idx = indices[-1].item()
        mid_idx = (from_idx + to_idx) // 2
        
        # 获取 Batch 数据
        batch = dataset[mid_idx]
        # 增加 Batch 维度并移至设备
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].unsqueeze(0).to(DEVICE)
        
        # 3. 前向传播提取 mu
        # 根据 modeling_act_cyw.py，forward 返回 (actions, (mu, log_sigma_x2))
        with torch.no_grad():
            _, (mu, _) = policy.forward(batch)
            
        if mu is not None:
            all_mus.append(mu.squeeze().cpu().numpy())
            label, color = get_task_label(ep_idx)
            all_labels.append(label)
            all_colors.append(color)

    # 4. 降维分析 (PCA)
    mus_array = np.array(all_mus) # 形状通常为 [75, 32]
    pca = PCA(n_components=2)
    mus_2d = pca.fit_transform(mus_array)

    # 5. 绘图
    plt.figure(figsize=(10, 7))
    unique_labels = list(set(all_labels))
    for label in unique_labels:
        idxs = [i for i, l in enumerate(all_labels) if l == label]
        plt.scatter(mus_2d[idxs, 0], mus_2d[idxs, 1], 
                    c=all_colors[idxs[0]], label=label, alpha=0.7, edgecolors='w', s=100)

    plt.title("Latent Space Visualization (z-distribution)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_path = "z_distribution_analysis.png"
    plt.savefig(save_path)
    print(f"分析完成！结果已保存至: {save_path}")

if __name__ == "__main__":
    main()