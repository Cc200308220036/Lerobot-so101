# import pandas as pd
# import matplotlib.pyplot as plt

# # 1. 读取评估生成的 parquet 文件
# df_eval = pd.read_parquet("/home/cyw/.cache/huggingface/lerobot/cyw_train_smolvla/eval_test25k_v2_2/data/chunk-000/file-000.parquet")

# # 2. 读取一个训练集的 parquet 文件做对比 (找一个你之前的训练数据)
# df_train = pd.read_parquet("/home/cyw/.cache/huggingface/lerobot/cyw_train/smolvla_3/data/chunk-000/file-003.parquet")

# # 3. 绘制关节轨迹对比 (以第0个关节为例)
# plt.figure(figsize=(12, 6))

# # 画训练集的轨迹 (平滑，理想状态)
# plt.plot(df_train["observation.state"][:100].map(lambda x: x[0]), label="Training (Ideal)", color="green")

# # 画评估时的轨迹
# plt.plot(df_eval["observation.state"][:100].map(lambda x: x[0]), label="Evaluation (Real)", color="red", linestyle="--")

# plt.title("Joint 0 Trajectory Analysis")
# plt.legend()
# #plt.show()

# plt.savefig("trajectory_analysis.png")
# print("✅ 图片已保存为 trajectory_analysis.png，请在左侧文件列表查看。")


# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # 1. 定义路径 (建议精确到 .parquet 文件)
# # 注意：请根据你实际的文件名修改 'episode_000000.parquet'
# eval_path = "/home/cyw/.cache/huggingface/lerobot/cyw_train_smolvla/eval_test25k_v2_2/data/chunk-000/file-000.parquet"
# train_path = "/home/cyw/.cache/huggingface/lerobot/cyw_train/smolvla_3/data/chunk-000/file-000.parquet"

# # 2. 检查文件是否存在
# if not os.path.exists(eval_path):
#     print(f"❌ 错误：找不到评估文件: {eval_path}")
#     # 尝试列出目录下的文件帮忙定位
#     dir_path = os.path.dirname(eval_path)
#     if os.path.exists(dir_path):
#         print(f"   该目录下的文件有: {os.listdir(dir_path)[:5]}...")
#     exit()

# # 3. 读取数据
# print("正在读取数据...")
# try:
#     df_eval = pd.read_parquet(eval_path)
#     df_train = pd.read_parquet(train_path)
#     print(f"✅ 数据读取成功！评估集形状: {df_eval.shape}, 训练集形状: {df_train.shape}")
# except Exception as e:
#     print(f"❌ 读取 Parquet 失败: {e}")
#     exit()

# # 4. 绘图
# print("正在绘图...")
# plt.figure(figsize=(12, 6))

# # 获取第0个关节的数据 (假设 state 是列表列)
# # 注意：有的 Parquet 读取后 state 可能是字符串或 numpy array，这里做个防御性处理
# def get_joint_0(x):
#     return x[0] if len(x) > 0 else 0

# # 画训练集 (取前100帧)
# limit = 100
# plt.plot(df_train["observation.state"][:limit].map(get_joint_0), label="Training (Ideal)", color="green", alpha=0.7)

# # 画评估集 (取前100帧)
# plt.plot(df_eval["observation.state"][:limit].map(get_joint_0), label="Evaluation (Real)", color="red", linestyle="--")

# plt.title("Joint 0 Trajectory Analysis (First 100 Steps)")
# plt.xlabel("Time Steps")
# plt.ylabel("Joint Position (Rad)")
# plt.legend()
# plt.grid(True, alpha=0.3)

# # 5. 保存图片而不是显示
# save_name = "comparison_result.png"
# plt.savefig(save_name)
# print(f"✅ 绘图完成！请在左侧文件栏打开图片: {save_name}")



import pandas as pd
import matplotlib.pyplot as plt
import os

# 评估数据路径 
eval_path = "/home/cyw/.cache/huggingface/lerobot/cyw_train_smolvla/eval_test25k_v2_2/data/chunk-000/file-000.parquet"

# 训练数据路径 
train_path = "/home/cyw/.cache/huggingface/lerobot/cyw_train/smolvla_3/data/chunk-000/file-000.parquet"

# ==================== 2. 路径自检与读取 ====================
def read_data(path, label):
    if not os.path.exists(path):
        print(f"❌ [{label}] 文件不存在: {path}")
        # 尝试列出上级目录，帮你找路径
        parent = os.path.dirname(os.path.dirname(path))
        if os.path.exists(parent):
            print(f"   ℹ️  '{parent}' 下的文件夹有: {os.listdir(parent)}")
        else:
            print(f"   ℹ️  就连上级目录 '{parent}' 也不存在，请检查路径拼写。")
        return None
    
    try:
        df = pd.read_parquet(path, engine='pyarrow') # 显式指定引擎
        print(f"✅ [{label}] 读取成功！形状: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ [{label}] 读取出错: {e}")
        return None

df_eval = read_data(eval_path, "评估集")
df_train = read_data(train_path, "训练集")

if df_eval is None or df_train is None:
    print("⚠️ 程序因读取失败而终止，请根据上方提示修正路径。")
    exit()

# ==================== 3. 智能绘图 ====================
print("正在绘图...")
plt.figure(figsize=(12, 6))

# 自动寻找正确的 state 列名 (防止是 observation.state 或 state)
state_key = None
for key in ["observation.state", "state", "present_position"]:
    if key in df_train.columns:
        state_key = key
        break

if state_key is None:
    print(f"❌ 未找到状态数据，列名如下: {df_train.columns}")
    exit()

print(f"使用的状态列名: {state_key}")

# 提取第0个关节数据的函数
def get_joint_0(x):
    # 处理可能是 numpy array 或 list 的情况
    try:
        return float(x[1])
    except:
        return 0.0

# 截取前 150 帧进行对比
limit = 150

# 绘制训练集 (标准答案)
plt.plot(df_train[state_key][:limit].map(get_joint_0), 
         label="Training (Demonstration)", color="green", linewidth=2, alpha=0.7)

# 绘制评估集 (实际表现)
# 如果评估集很短，就画全部
eval_len = min(limit, len(df_eval))
plt.plot(df_eval[state_key][:eval_len].map(get_joint_0), 
         label="Evaluation (Real Deployment)", color="red", linestyle="--", linewidth=2)

plt.title(f"Joint 2 Trajectory: Sim-to-Real Gap Analysis ({state_key})")
plt.xlabel("Time Steps (Frames)")
plt.ylabel("Joint Position (Rad)")
plt.legend()
plt.grid(True, alpha=0.3)

# ==================== 4. 保存结果 ====================
save_name = "analysis_result_7.png"
plt.savefig(save_name)
print(f"\n✅✅✅ 分析完成！图片已保存为: {save_name}")
print("请在 VS Code 左侧文件列表中点击该图片查看。")