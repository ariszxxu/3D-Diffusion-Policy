import os
import h5py
import torch
import numpy as np
import zarr
from tqdm import tqdm
# from pyntcloud import PyntCloud
import pandas as pd
import re

# def random_downsample(point_cloud, num_points=512):
#     """ 使用随机采样对点云进行降采样 """
#     indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
#     return point_cloud[indices]

def process_point_cloud(h5_file, num_points=2048):
    """ 处理点云数据并降采样 """
    with h5py.File(h5_file, 'r') as f:
        group_names = list(f.keys())
        n_total_step = len(group_names) * 21
        point_clouds = np.zeros((n_total_step, num_points, 3), dtype=np.float32)
        groupname_list = []
        step_idx = 0
        
        for name in tqdm(group_names, desc="Processing Point Clouds"):
            data = f[name]['trajectories'][...]
            sampled_indices = np.random.choice(data.shape[0], num_points, replace=False)
            sampled_pc = data[sampled_indices, :, :]
            
            for t in range(21):
                point_clouds[step_idx] = sampled_pc[:, t, :]
                groupname_list.append(name)
                step_idx += 1
    
    return point_clouds, group_names, n_total_step, groupname_list


def fixed_description_embed(input_name, embed_dict):
    description_embed_list = []
    # for input_name in names:      # batch
    # print(input_name)
    type_mapping = {'D': 'Dress', 'T': 'Tops', 'P': 'Pants', 'S': 'Skirt'}
    property_mapping = {'S': 'Short', 'L': 'Long', 'H': 'Hooded', 'C': 'Collar', 'N': 'No-Collar'}
    sleeve_mapping = {'G': 'Gallus', 'T': 'Tube', 'L': 'Long-Sleeve', 'S': 'Short-Sleeve', 'N':'No-Sleeve'}
    extra_mapping = {'S': ' ', 'C': 'FrontClose', 'O': 'FrontOpen'}

    # Define the folding methods based on clothing types
    sleeveless_folding = ['DLG', 'DLNS', 'DLT', 'DSNS', 'SL', 'SS', 'TCNC', 'TCNO', 'THNC', 'TNNC']
    short_sleeve_folding = ['DLSS', 'DSSS', 'TCSC', 'TNSC']
    long_sleeve_folding = ['DLLS', 'DSLS', 'TCLC', 'TCLO', 'THLC', 'THLO', 'TNLC']
    pants_folding = ['PL', 'PS']

    # Extract information from the input string
    parts = input_name.split('_')
    cloth_code = parts[0]
    # cloth_type_name = parts[1]
    action = int(parts[-1].replace('action', ''))

    # Determine the type of clothing
    cloth_type = type_mapping.get(cloth_code[0], 'Unknown')
    description_parts = [cloth_type]

    is_mirror = False
    if parts[1] == 'L':
        is_mirror = False
    elif parts[1] == 'R':
        is_mirror = True
    else:
        assert 0, "assumpt the second letter indicate L/R"

    # Determine the properties of the clothing
    if len(cloth_code) > 1:
        description_parts.append(property_mapping.get(cloth_code[1], ''))

    if len(cloth_code) > 2:
        description_parts.append(sleeve_mapping.get(cloth_code[2], ''))

    if len(cloth_code) > 3:
        description_parts.append(extra_mapping.get(cloth_code[3], ''))

    if cloth_code in sleeveless_folding:
        description = "Fold the no-sleeve cloth bottom-up."
    elif cloth_code in short_sleeve_folding:
        if cloth_type == 'Dress':
            if action == 0:
                description = "Fold the short-sleeve dress from the left."
            elif action == 1:
                description = "Fold the short-sleeve dress from the right."
            elif action == 2:
                description = "Fold the short-sleeve dress bottom-up."
        elif cloth_type == 'Tops':
            if action == 0:
                description = "Fold the short-sleeve top from the left."
            elif action == 1:
                description = "Fold the short-sleeve top from the right."
            elif action == 2:
                description = "Fold the short-sleeve top bottom-up."
    elif cloth_code in long_sleeve_folding:
        if action == 0:
            description = "Fold the long-sleeve cloth from the left."
        elif action == 1:
            description = "Fold the long-sleeve cloth from the right."
        elif action == 2:
            description = "Fold the long-sleeve cloth bottom-up."
    elif cloth_code in pants_folding:
        if action == 0:
            description = "Fold the pants from the left."
        elif action == 1:
            description = "Fold the pants bottom-up."
    else:
        description = "unknown folding method"

    select_description = description
    
    if is_mirror:
        select_description = select_description.replace("left", "temp").replace("right", "left").replace("temp", "right")

    description_embed = embed_dict[select_description]

    return torch.tensor(description_embed)


def process_language_embeddings(group_names, embed_dict):
    """ 处理语言嵌入 """
    language_emb = np.zeros((len(group_names) * 21, 4096), dtype=np.float32)
    step_idx = 0
    
    for name in tqdm(group_names, desc="Processing Language Embeddings"):
        emb = fixed_description_embed(name, embed_dict)
        for _ in range(21):
            language_emb[step_idx] = emb
            step_idx += 1
    
    return language_emb

def mirror_action(action):
    """ 沿 yz 平面镜像 action """
    action[:, :, 0] *= -1  # 仅镜像 x 轴
    return action

def process_actions(group_names, action_base_path):
    """ 处理 action 数据 """
    action_list = []
    for name in tqdm(group_names, desc="Processing Actions"):
        parts = name.split('_')
        cloth_type, side, cloth_name_action = parts[0], parts[1], '_'.join(parts[2:])
        action_path = os.path.join(action_base_path, cloth_type, cloth_name_action, "keypoint_trajectory.npy")
        
        if not os.path.exists(action_path):
            raise FileNotFoundError(f"Missing action file: {action_path}")
        
        kp_traj = np.load(action_path)  # (21, num_kp, 3)
        num_kp = kp_traj.shape[1]  # 关键点数量
        
        if num_kp == 1:
            kp_traj = np.tile(kp_traj, (1, 2, 1))  # 复制关键点，使其满足 (21, 2, 3)
            num_kp = 2
        
        actions = np.zeros((21, num_kp, 6), dtype=np.float32)
        
        for t in range(21):
            pos_t = kp_traj[t, :, :]
            if t < 20:
                motion = kp_traj[t + 1, :, :] - pos_t
            else:
                motion = np.zeros_like(pos_t)
            actions[t] = np.concatenate([pos_t, motion], axis=-1)
        
        if side == 'R':
            actions = mirror_action(actions)
        
        actions = actions.reshape(21, num_kp*6)
        
        action_list.append(actions)
    
    return np.vstack(action_list)


def create_zarr_file(output_path, point_clouds, language_emb, actions, episode_ends, groupname_list):
    print("output_path: ", point_clouds.shape, language_emb.shape, actions.shape, len(episode_ends), len(groupname_list))

    """ 创建 .zarr 文件 """
    root = zarr.open(output_path, mode='w')
    root.create_group('data')
    root.create_group('meta')

    def safe_create_dataset(name, data, dtype, chunk_size=None):
        dset_size = data.nbytes  # 计算数据集的字节大小
        print(f"{name} dataset size: {dset_size / (1024 * 1024):.6f} MB")  # 转换为 MB 单位
        if dset_size == 0:
            print(f"Warning: {name} has size 0 bytes, skipping creation.")
            return
        if chunk_size is None:
            chunk_size = tuple(min(256, s) for s in data.shape)
        root.create_dataset(name, shape=data.shape, data=data, dtype=dtype, overwrite=True, chunks=chunk_size)

    # 逐个检查数据集
    safe_create_dataset('data/point_cloud', point_clouds, 'float32')
    safe_create_dataset('data/language_emb', language_emb, 'float32')
    safe_create_dataset('data/action', actions, 'float32')
    
    if len(episode_ends) > 0:
        print("creating episode_ends")
        root.create_dataset('meta/episode_ends', shape=(len(episode_ends),), data=np.array(episode_ends, dtype='int32'), dtype='int32', overwrite=True)
    else:
        print("Warning: episode_ends is empty, skipping dataset creation.")

    if len(groupname_list) > 0:
        print("creating group_names")
        chunk_size = min(256, len(groupname_list))  # 设置合理的 chunk size，防止 Zarr 计算 log10(0)
        root.create_dataset('meta/group_names', shape=(len(groupname_list),), data=groupname_list, dtype='S', overwrite=True, chunks=(chunk_size,))
    else:
        print("Warning: groupname_list is empty, skipping dataset creation.") 
    print("created ", output_path, " zarr")

def main():
    h5_file = "/data2/chaonan/points-traj-prediction/data/all_data_samedirection_mirrored_1121_traj2data.h5"
    embed_dict = torch.load("/data2/chaonan/points-traj-prediction/data/description_embeddings_mirrored.pt")
    action_base_path = "/data2/chaonan/cloth_traj_data_action_0128/"
    all_data_path = "/data2/xzhixuan/projects/3D-Diffusion-Policy/data/dp3_dataset.zarr"
    train_data_path = "/data2/xzhixuan/projects/3D-Diffusion-Policy/data/dp3_train_dataset.zarr"
    test_data_path = "/data2/xzhixuan/projects/3D-Diffusion-Policy/data/dp3_test_dataset.zarr"
    
    # 处理数据
    point_clouds, group_names, n_total_step, groupname_list = process_point_cloud(h5_file)
    language_emb = process_language_embeddings(group_names, embed_dict)
    actions = process_actions(group_names, action_base_path)
    episode_ends = [(i + 1) * 21 for i in range(len(group_names))]

    # print(point_clouds.shape, language_emb.shape, actions.shape, len(episode_ends))
    
    # 创建完整 Zarr 数据集
    create_zarr_file(all_data_path, point_clouds, language_emb, actions, episode_ends, groupname_list)
    
    # 按规则划分训练集和测试集
    train_indices = [i for i, name in enumerate(groupname_list) if re.search(r'(\d+)_action', name) and int(re.search(r'(\d+)_action', name).group(1)) % 5 != 0]
    test_indices = [i for i, name in enumerate(groupname_list) if not re.search(r'(\d+)_action', name) or int(re.search(r'(\d+)_action', name).group(1)) % 5 == 0]
    print(len(train_indices), len(test_indices), len(groupname_list), len(train_indices)+len(test_indices), len(train_indices) / 21, len(test_indices) / 21)
    
    create_zarr_file(train_data_path, point_clouds[train_indices], language_emb[train_indices], actions[train_indices], episode_ends=[(i + 1) * 21 for i in range(int(len(train_indices) / 21))], groupname_list=[groupname_list[i] for i in train_indices])
    create_zarr_file(test_data_path, point_clouds[test_indices], language_emb[test_indices], actions[test_indices], episode_ends=[(i + 1) * 21 for i in range(int(len(test_indices)  / 21))], groupname_list=[groupname_list[i] for i in test_indices])
    
    print("Zarr datasets saved: dp3_dataset.zarr, train_dataset.zarr, test_dataset.zarr")

if __name__ == "__main__":
    main()
