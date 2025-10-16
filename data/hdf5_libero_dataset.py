import os
import fnmatch
import json
import sys

import h5py
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

from configs.state_vec import STATE_VEC_IDX_MAPPING

# 添加utils路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.rotation_utils import convert_euler_to_6d_rotation, convert_6d_rotation_to_euler


class HDF5LIBERODataset:
    """
    This class is used to sample episodes from the LIBERO dataset
    stored in HDF5 files.
    """
    def __init__(self) -> None:
        # The path to the HDF5 dataset directory
        # Each HDF5 file contains multiple episodes
        self.HDF5_DIR = "data/datasets/libero_90/"
        self.DATASET_NAME = "libero_90"
        
        self.file_paths = []
        for root, _, files in os.walk(self.HDF5_DIR):
            for filename in fnmatch.filter(files, '*.hdf5'):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)
                
        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISTORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
    
        # Get each episode's len
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
    def __len__(self):
        return len(self.file_paths)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))
    
    def parse_hdf5_file(self, file_path):
        """Parse a LIBERO hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample.
        """
        with h5py.File(file_path, 'r') as f:
            # Get the first episode (LIBERO files contain multiple episodes)
            episode_key = list(f['data'].keys())[0]
            episode_data = f['data'][episode_key]
            
            # Get data shapes
            actions = episode_data['actions'][:]
            joint_states = episode_data['obs']['joint_states'][:]
            gripper_states = episode_data['obs']['gripper_states'][:]
            ee_pos = episode_data['obs']['ee_pos'][:]
            ee_ori = episode_data['obs']['ee_ori'][:]
            
            num_steps = actions.shape[0]
            
            # Drop too-short episodes
            if num_steps < 32:
                return False, None
            
            # Skip the first few still steps
            EPS = 1e-2
            joint_delta = np.abs(joint_states - joint_states[0:1])
            indices = np.where(np.any(joint_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                first_idx = 0
            
            # Randomly sample a timestep
            step_id = np.random.randint(max(first_idx-1, 0), num_steps)
            
            # Load the language instruction
            problem_info = json.loads(f['data'].attrs['problem_info'])
            instruction = problem_info['language_instruction']
            
            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }
            
            # Create LIBERO state vector with proper 6D rotation conversion
            # Convert ee_ori from 3D Euler angles to 6D rotation representation
            ee_ori_6d = convert_euler_to_6d_rotation(ee_ori)
            
            # Create state vector: [joint_states(7) + gripper_states(1) + ee_pos(3) + ee_ori_6d(6)]
            libero_states = np.concatenate([
                joint_states,                    # (T, 7)
                gripper_states[:, 0:1],         # (T, 1) - only first gripper state
                ee_pos,                         # (T, 3)
                ee_ori_6d                       # (T, 6) - 6D rotation representation
            ], axis=1)  # (T, 17)
            
            # Parse the state and action
            state = libero_states[step_id:step_id+1]  # (1, 17)
            state_std = np.std(libero_states, axis=0)
            state_mean = np.mean(libero_states, axis=0)
            state_norm = np.sqrt(np.mean(libero_states**2, axis=0))
            
            # Get action sequence
            actions_seq = actions[step_id:step_id+self.CHUNK_SIZE]
            if actions_seq.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions_seq = np.concatenate([
                    actions_seq,
                    np.tile(actions_seq[-1:], (self.CHUNK_SIZE - actions_seq.shape[0], 1))
                ], axis=0)
            
            # Fill state and action into unified vectors
            def fill_in_state(values):
                # LIBERO: single-arm robot, only use right arm portion
                # Map: joint_states(7) + gripper_states(1) + ee_pos(3) + ee_ori_6d(6) = 17 dims
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)  # joint positions
                ] + [
                    STATE_VEC_IDX_MAPPING["right_gripper_open"]  # gripper
                ] + [
                    STATE_VEC_IDX_MAPPING["right_eef_pos_x"],    # ee position x
                    STATE_VEC_IDX_MAPPING["right_eef_pos_y"],    # ee position y  
                    STATE_VEC_IDX_MAPPING["right_eef_pos_z"]     # ee position z
                ] + [
                    STATE_VEC_IDX_MAPPING["right_eef_angle_0"],  # ee orientation (6D representation)
                    STATE_VEC_IDX_MAPPING["right_eef_angle_1"],
                    STATE_VEC_IDX_MAPPING["right_eef_angle_2"],
                    STATE_VEC_IDX_MAPPING["right_eef_angle_3"],
                    STATE_VEC_IDX_MAPPING["right_eef_angle_4"],
                    STATE_VEC_IDX_MAPPING["right_eef_angle_5"]
                ]
                
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            
            def fill_in_action(values):
                """
                将LIBERO actions转换为RDT统一动作空间
                
                重要：按照RDT README IMPORTANT 3的要求，必须使用物理单位！
                "No physical quantities (except the gripper width) are normalized during pre-training.
                 Generally, we use the International System of Units."
                
                LIBERO原始actions: 7D [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, gripper]
                范围: [-1, 1] 归一化范围，需要转换为物理单位
                """
                
                # === 步骤1: 转换位置为物理单位（米） ===
                # LIBERO: [-1, 1] 对应实际物理增量
                # 通过数据分析，实际缩放因子约为 0.012，不是 0.05
                pos_normalized = values[:, 0:3]  # (T, 3) 归一化范围
                pos_meters = pos_normalized * 0.012  # 转换为米 (物理单位)
                # 现在范围: 约 [-0.012, 0.012] 米，符合实际测量
                
                # === 步骤2: 转换旋转为物理单位（弧度） ===
                # LIBERO: [-1, 1] 对应 [-0.5rad, 0.5rad] 的物理增量
                ori_normalized = values[:, 3:6]  # (T, 3) 欧拉角，归一化范围
                ori_radians = ori_normalized * 0.5  # 转换为弧度 (物理单位)
                # 现在范围: 约 [-0.5, 0.5] 弧度
                
                # === 步骤3: 转换为6D旋转表示 ===
                # 注意：从物理单位的弧度转换为6D表示
                ori_6d = convert_euler_to_6d_rotation(ori_radians)  # (T, 6)
                
                # === 步骤4: Gripper归一化（按README，这是唯一需要归一化的） ===
                # LIBERO gripper: [-1, 1] → RDT gripper: [0, 1]
                gripper_raw = values[:, 6:7]  # (T, 1)
                gripper_normalized = (gripper_raw + 1.0) / 2.0  # Map [-1,1] to [0,1]
                
                # === 步骤5: 组合为10D动作向量 ===
                action_10d = np.concatenate([
                    pos_meters,           # 位置：物理单位（米）
                    ori_6d,              # 旋转：6D表示（从物理单位的弧度转换）
                    gripper_normalized   # Gripper：归一化到[0, 1]
                ], axis=1)  # (T, 10)

                UNI_ACTION_INDICES = [
                    STATE_VEC_IDX_MAPPING["right_eef_pos_x"],    # position x
                    STATE_VEC_IDX_MAPPING["right_eef_pos_y"],    # position y
                    STATE_VEC_IDX_MAPPING["right_eef_pos_z"]     # position z
                ] + [
                    STATE_VEC_IDX_MAPPING["right_eef_angle_0"],  # orientation (6D)
                    STATE_VEC_IDX_MAPPING["right_eef_angle_1"],
                    STATE_VEC_IDX_MAPPING["right_eef_angle_2"],
                    STATE_VEC_IDX_MAPPING["right_eef_angle_3"],
                    STATE_VEC_IDX_MAPPING["right_eef_angle_4"],
                    STATE_VEC_IDX_MAPPING["right_eef_angle_5"]
                ] + [
                    STATE_VEC_IDX_MAPPING["right_gripper_open"]  # gripper
                ]

                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_ACTION_INDICES] = action_10d
                return uni_vec
            
            state = fill_in_state(state)
            state_indicator = fill_in_state(np.ones_like(state_std))
            state_std = fill_in_state(state_std)
            state_mean = fill_in_state(state_mean)
            state_norm = fill_in_state(state_norm)
            actions = fill_in_action(actions_seq)
            
            # Parse the images
            def parse_img(key):
                imgs = []
                for i in range(max(step_id-self.IMG_HISTORY_SIZE+1, 0), step_id+1):
                    img = episode_data['obs'][key][i]
                    imgs.append(img)
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISTORY_SIZE:
                    # Pad the images using the first image
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISTORY_SIZE-imgs.shape[0], 1, 1, 1)),
                        imgs
                    ], axis=0)
                return imgs
            
            # Get image history
            cam_high = parse_img('agentview_rgb')
            cam_right_wrist = parse_img('eye_in_hand_rgb')
            
            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - max(first_idx - 1, 0) + 1, self.IMG_HISTORY_SIZE)
            cam_high_mask = np.array(
                [False] * (self.IMG_HISTORY_SIZE - valid_len) + [True] * valid_len
            )
            cam_right_wrist_mask = cam_high_mask.copy()
            
            # Single-arm robot: left wrist is empty (following RDT's original design)
            cam_left_wrist = np.zeros((self.IMG_HISTORY_SIZE, 0, 0, 0))
            cam_left_wrist_mask = np.zeros(self.IMG_HISTORY_SIZE, dtype=bool)
            
            # Return the resulting sample
            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask,
            }
    
    def parse_hdf5_file_state_only(self, file_path):
        """Parse a LIBERO hdf5 file to get state information only.
        
        Args:
            file_path (str): the path to the hdf5 file
            
        Returns:
            valid (bool): whether the episode is valid
            dict: a dictionary containing only the state information
        """
        with h5py.File(file_path, 'r') as f:
            # Get the first episode
            episode_key = list(f['data'].keys())[0]
            episode_data = f['data'][episode_key]
            
            # Get data
            joint_states = episode_data['obs']['joint_states'][:]
            gripper_states = episode_data['obs']['gripper_states'][:]
            ee_pos = episode_data['obs']['ee_pos'][:]
            ee_ori = episode_data['obs']['ee_ori'][:]
            
            num_steps = joint_states.shape[0]
            
            # Drop too-short episodes
            if num_steps < 32:
                return False, None
            
            # Create LIBERO state vector with proper 6D rotation conversion
            ee_ori_6d = convert_euler_to_6d_rotation(ee_ori)
            
            # Create state vector: [joint_states(7) + gripper_states(1) + ee_pos(3) + ee_ori_6d(6)]
            libero_states = np.concatenate([
                joint_states,                    # (T, 7)
                gripper_states[:, 0:1],         # (T, 1) - only first gripper state
                ee_pos,                         # (T, 3)
                ee_ori_6d                       # (T, 6) - 6D rotation representation
            ], axis=1)  # (T, 17)
            
            # Fill into unified state vector
            def fill_in_state(values):
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
                ] + [
                    STATE_VEC_IDX_MAPPING["right_gripper_open"]
                ] + [
                    STATE_VEC_IDX_MAPPING["right_eef_pos_x"],
                    STATE_VEC_IDX_MAPPING["right_eef_pos_y"],
                    STATE_VEC_IDX_MAPPING["right_eef_pos_z"]
                ] + [
                    STATE_VEC_IDX_MAPPING["right_eef_angle_0"],
                    STATE_VEC_IDX_MAPPING["right_eef_angle_1"],
                    STATE_VEC_IDX_MAPPING["right_eef_angle_2"],
                    STATE_VEC_IDX_MAPPING["right_eef_angle_3"],
                    STATE_VEC_IDX_MAPPING["right_eef_angle_4"],
                    STATE_VEC_IDX_MAPPING["right_eef_angle_5"]
                ]
                
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            
            state = fill_in_state(libero_states)
            
            return True, {
                "state": state
            }


if __name__ == "__main__":
    # Test the dataset
    ds = HDF5LIBERODataset()
    print(f"Dataset name: {ds.get_dataset_name()}")
    print(f"Number of files: {len(ds)}")
    
    # Test getting a sample
    sample = ds.get_item()
    print(f"Sample keys: {sample.keys()}")
    print(f"State shape: {sample['state'].shape}")
    print(f"Actions shape: {sample['actions'].shape}")
    print(f"Instruction: {sample['meta']['instruction']}")
