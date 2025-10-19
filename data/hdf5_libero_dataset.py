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
    
    Supports two sampling strategies:
    1. Full-step enumeration (default, recommended): All steps from all episodes are enumerated
    2. Random sampling (legacy): Randomly sample one step from a random episode
    """
    def __init__(self, dataset_name: str = "libero_90", use_full_step_enumeration: bool = True) -> None:
        # The path to the HDF5 dataset directory
        # Each HDF5 file contains multiple episodes
        # Support environment variable for easy single-task testing
        default_dir = f"data/datasets/{dataset_name}/"
        self.HDF5_DIR = os.environ.get("LIBERO_DATASET_DIR", default_dir)
        self.use_full_step_enumeration = use_full_step_enumeration
        
        # 自动检测：如果数据集路径包含 "libero_single_task"，使用单任务统计
        # 这样可以确保loss weighting正确
        dataset_dir = os.environ.get("LIBERO_DATASET_DIR", "")
        if "libero_single_task" in dataset_dir or dataset_dir == "dataset_remote/":
            # 检查dataset_stat.json中是否有libero_single_task
            stat_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'dataset_stat.json')
            if os.path.exists(stat_path):
                with open(stat_path, 'r') as f:
                    stats = json.load(f)
                if 'libero_single_task' in stats:
                    self.DATASET_NAME = "libero_single_task"
                    print(f"🔍 检测到单任务训练，使用 libero_single_task 统计信息")
                    print(f"   数据集路径: {self.HDF5_DIR}")
                else:
                    self.DATASET_NAME = dataset_name
                    print(f"⚠️  警告：单任务训练但未找到 libero_single_task 统计，使用 {dataset_name}")
            else:
                self.DATASET_NAME = dataset_name
        else:
            self.DATASET_NAME = dataset_name
        
        self.file_paths = []
        for root, _, files in os.walk(self.HDF5_DIR):
            for filename in fnmatch.filter(files, '*.hdf5'):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)
                
        # Load the config (使用相对于当前文件的路径)
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'base.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISTORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
        
        # Load global dataset statistics for normalization (使用相对于当前文件的路径)
        stat_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'dataset_stat.json')
        with open(stat_path, 'r') as f:
            global_stats = json.load(f)
        self.action_std_global = np.array(global_stats[self.DATASET_NAME]['action_std'])
    
        # Initialize sampling strategy
        if self.use_full_step_enumeration:
            # Full-step enumeration: pre-process all episodes and enumerate all valid steps
            print(f"📊 使用全步骤枚举策略（Full-Step Enumeration）")
            self._build_full_step_index()
        else:
            # Random sampling: compute episode weights for random sampling
            print(f"🎲 使用随机采样策略（Random Sampling - Legacy）")
            episode_lens = []
            for file_path in self.file_paths:
                valid, res = self.parse_hdf5_file_state_only(file_path)
                _len = res['state'].shape[0] if valid else 0
                episode_lens.append(_len)
            self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
    def _build_full_step_index(self):
        """
        Build an index of all valid steps across all episodes.
        This implements the full-step enumeration strategy used in RDT's official pretrain/finetune.
        
        Each entry in the index is a tuple: (file_path, episode_key, step_id)
        """
        print(f"🔨 构建全步骤索引中...")
        self.all_steps = []
        
        for file_path in self.file_paths:
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'data' not in f:
                        continue
                    
                    # Iterate through all episodes in this file
                    episodes = list(f['data'].keys())
                    for episode_key in episodes:
                        episode_data = f['data'][episode_key]
                        
                        # Get the length of this episode
                        if 'actions' not in episode_data:
                            continue
                        
                        num_steps = len(episode_data['actions'])
                        
                        # For each valid starting position, create an entry
                        # A step is valid if there are at least CHUNK_SIZE future actions
                        for step_id in range(max(0, num_steps - self.CHUNK_SIZE + 1)):
                            self.all_steps.append({
                                'file_path': file_path,
                                'episode_key': episode_key,
                                'step_id': step_id
                            })
            
            except Exception as e:
                print(f"⚠️  警告：无法处理文件 {file_path}: {e}")
                continue
        
        print(f"✅ 全步骤索引构建完成！")
        print(f"   - HDF5文件数: {len(self.file_paths)}")
        print(f"   - 总训练样本数: {len(self.all_steps)}")
        print(f"   - 数据利用率: 100% (vs 随机采样的~30%)")
        
        if len(self.all_steps) == 0:
            raise ValueError(f"❌ 错误：未找到任何有效的训练样本！请检查数据集路径: {self.HDF5_DIR}")
    
    def __len__(self):
        """Return the number of training samples."""
        if self.use_full_step_enumeration:
            return len(self.all_steps)
        else:
            # Legacy mode: return number of episodes (not accurate for dataset size)
            return len(self.file_paths)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        """Get a training sample.
        
        In full-step enumeration mode: Returns the specific step indexed by 'index'
        In random sampling mode: Returns a random step from a random episode

        Args:
            index (int, optional): the index of the sample (full-step mode) or episode (random mode).
                If not provided, a random sample will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        if self.use_full_step_enumeration:
            # Full-step enumeration mode
            return self._get_item_full_step(index, state_only)
        else:
            # Random sampling mode (legacy)
            return self._get_item_random(index, state_only)
    
    def _get_item_full_step(self, index: int=None, state_only=False):
        """Get a specific training sample using full-step enumeration strategy."""
        if index is None:
            index = np.random.randint(0, len(self.all_steps))
        
        # Get the step info
        step_info = self.all_steps[index]
        file_path = step_info['file_path']
        episode_key = step_info['episode_key']
        step_id = step_info['step_id']
        
        # Parse the specific step from the HDF5 file
        valid, sample = self.parse_hdf5_file_at_step(
            file_path, episode_key, step_id, state_only=state_only
        )
        
        if not valid:
            # If this specific step is invalid, try a random one
            return self._get_item_full_step(index=None, state_only=state_only)
        
        return sample
    
    def _get_item_random(self, index: int=None, state_only=False):
        """Get a random training sample using random sampling strategy (legacy)."""
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
    
    def parse_hdf5_file_at_step(self, file_path, episode_key, step_id, state_only=False):
        """Parse a LIBERO hdf5 file to generate a training sample at a specific step.
        
        This method is used in full-step enumeration mode.
        
        Args:
            file_path (str): the path to the hdf5 file
            episode_key (str): the key of the episode (e.g., 'demo_0')
            step_id (int): the starting step index
            state_only (bool): whether to return only state information
        
        Returns:
            valid (bool): whether the sample is valid
            dict: a dictionary containing the training sample
        """
        try:
            with h5py.File(file_path, 'r') as f:
                if 'data' not in f or episode_key not in f['data']:
                    return False, None
                
                episode_data = f['data'][episode_key]
                
                # Get observations and actions
                obs = episode_data['obs']
                actions = episode_data['actions']
                num_steps = len(actions)
                
                # Validate step_id
                if step_id >= num_steps or step_id + self.CHUNK_SIZE > num_steps:
                    return False, None
                
                if state_only:
                    return self._parse_state_only_at_step(episode_data, step_id)
                else:
                    return self._parse_full_sample_at_step(episode_data, step_id, file_path)
        
        except Exception as e:
            print(f"⚠️  Error parsing {file_path}, episode {episode_key}, step {step_id}: {e}")
            return False, None
    
    def _parse_state_only_at_step(self, episode_data, step_id):
        """Helper method to parse state-only information at a specific step."""
        # Reuse parse_hdf5_file_state_only logic but for a specific episode
        # For now, return the full trajectory (this method is rarely used)
        try:
            joint_states = episode_data['obs']['joint_states'][:]
            gripper_states = episode_data['obs']['gripper_states'][:]
            ee_pos = episode_data['obs']['ee_pos'][:]
            ee_ori = episode_data['obs']['ee_ori'][:]
            
            # Convert orientation to 6D
            ee_ori_6d = convert_euler_to_6d_rotation(ee_ori)
            
            # Create state vector
            libero_states = np.concatenate([
                joint_states,
                gripper_states[:, 0:1],
                ee_pos,
                ee_ori_6d
            ], axis=1)
            
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
            
            return True, {"state": state}
        except Exception as e:
            print(f"⚠️  Error in _parse_state_only_at_step: {e}")
            return False, None
    
    def _parse_full_sample_at_step(self, episode_data, step_id, file_path):
        """Helper method to parse a full training sample at a specific step."""
        # This method essentially replicates parse_hdf5_file logic but for a specific step
        # Instead of random sampling, we use the provided step_id
        try:
            # Get data
            actions = episode_data['actions'][:]
            joint_states = episode_data['obs']['joint_states'][:]
            gripper_states = episode_data['obs']['gripper_states'][:]
            ee_pos = episode_data['obs']['ee_pos'][:]
            ee_ori = episode_data['obs']['ee_ori'][:]
            
            num_steps = actions.shape[0]
            
            # Validate step_id
            if step_id >= num_steps or step_id + self.CHUNK_SIZE > num_steps:
                return False, None
            
            # Get task instruction from file attributes
            try:
                # Try to get from episode_data parent (the HDF5 file's data group)
                problem_info = json.loads(episode_data.parent.attrs['problem_info'])
                instruction = problem_info['language_instruction']
            except:
                instruction = "unknown task"
            
            # Assemble meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }
            
            # Create state vector with 6D rotation
            ee_ori_6d = convert_euler_to_6d_rotation(ee_ori)
            libero_states = np.concatenate([
                joint_states,
                gripper_states[:, 0:1],
                ee_pos,
                ee_ori_6d
            ], axis=1)
            
            # Get state and actions at step_id
            state_17d = libero_states[step_id:step_id+1]
            state_std = np.std(libero_states, axis=0)
            state_mean = np.mean(libero_states, axis=0)
            
            # Get action sequence
            actions_seq = actions[step_id:step_id+self.CHUNK_SIZE]
            if actions_seq.shape[0] < self.CHUNK_SIZE:
                actions_seq = np.concatenate([
                    actions_seq,
                    np.tile(actions_seq[-1:], (self.CHUNK_SIZE - actions_seq.shape[0], 1))
                ], axis=0)
            
            # Fill into unified vectors
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
            
            def fill_in_action(values):
                # Convert actions to physical units (cm and radians)
                pos_normalized = values[:, 0:3]
                pos_cm = pos_normalized * 1.2  # cm
                
                ori_normalized = values[:, 3:6]
                ori_radians = ori_normalized * 0.5  # radians
                ori_6d = convert_euler_to_6d_rotation(ori_radians)
                
                gripper_normalized = values[:, 6:7]
                gripper_bin = (gripper_normalized + 1) / 2  # [0, 1]
                
                # Map to unified action space
                UNI_POS_INDICES = [
                    STATE_VEC_IDX_MAPPING["right_eef_pos_x"],
                    STATE_VEC_IDX_MAPPING["right_eef_pos_y"],
                    STATE_VEC_IDX_MAPPING["right_eef_pos_z"]
                ]
                UNI_ORI_INDICES = [
                    STATE_VEC_IDX_MAPPING["right_eef_angle_0"],
                    STATE_VEC_IDX_MAPPING["right_eef_angle_1"],
                    STATE_VEC_IDX_MAPPING["right_eef_angle_2"],
                    STATE_VEC_IDX_MAPPING["right_eef_angle_3"],
                    STATE_VEC_IDX_MAPPING["right_eef_angle_4"],
                    STATE_VEC_IDX_MAPPING["right_eef_angle_5"]
                ]
                UNI_GRIPPER_INDEX = STATE_VEC_IDX_MAPPING["right_gripper_open"]
                
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_POS_INDICES] = pos_cm
                uni_vec[..., UNI_ORI_INDICES] = ori_6d
                uni_vec[..., UNI_GRIPPER_INDEX] = gripper_bin.squeeze(-1)
                
                return uni_vec
            
            state = fill_in_state(state_17d)
            state_indicator = fill_in_state(np.ones_like(state_std))
            state_std_vec = fill_in_state(state_std)
            state_mean_vec = fill_in_state(state_mean)
            actions_vec = fill_in_action(actions_seq)
            
            state_norm = self.action_std_global + 1e-8
            
            # Parse images
            def parse_img(key):
                imgs = []
                for i in range(max(step_id-self.IMG_HISTORY_SIZE+1, 0), step_id+1):
                    img = episode_data['obs'][key][i]
                    imgs.append(img)
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISTORY_SIZE:
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISTORY_SIZE-imgs.shape[0], 1, 1, 1)),
                        imgs
                    ], axis=0)
                return imgs
            
            cam_high = parse_img('agentview_rgb')
            cam_right_wrist = parse_img('eye_in_hand_rgb')
            
            valid_len = min(step_id + 1, self.IMG_HISTORY_SIZE)
            cam_high_mask = np.array(
                [False] * (self.IMG_HISTORY_SIZE - valid_len) + [True] * valid_len
            )
            cam_right_wrist_mask = cam_high_mask.copy()
            
            cam_left_wrist = np.zeros((self.IMG_HISTORY_SIZE, 0, 0, 0))
            cam_left_wrist_mask = np.zeros(self.IMG_HISTORY_SIZE, dtype=bool)
            
            return True, {
                'meta': meta,
                'state': state,
                'actions': actions_vec,
                'state_indicator': state_indicator,
                'cam_high': cam_high,
                'cam_high_mask': cam_high_mask,
                'cam_right_wrist': cam_right_wrist,
                'cam_right_wrist_mask': cam_right_wrist_mask,
                'cam_left_wrist': cam_left_wrist,
                'cam_left_wrist_mask': cam_left_wrist_mask,
                'state_std': state_std_vec,
                'state_mean': state_mean_vec,
                'state_norm': state_norm
            }
        
        except Exception as e:
            print(f"⚠️  Error in _parse_full_sample_at_step: {e}")
            import traceback
            traceback.print_exc()
            return False, None
    
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
            # Randomly select one episode from the file (LIBERO files contain multiple episodes)
            episodes = list(f['data'].keys())
            episode_key = np.random.choice(episodes)
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
            state_17d = libero_states[step_id:step_id+1]  # (1, 17)
            state_std = np.std(libero_states, axis=0)
            state_mean = np.mean(libero_states, axis=0)
            
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
                
                # === 步骤1: 转换位置为物理单位（厘米） ===
                # LIBERO: [-1, 1] 对应实际物理增量
                # 关键修复：使用厘米而不是米，使Position与Orientation量级相同
                # 0.012米 * 100 = 1.2厘米，与Orientation 6D(~1.0)量级接近
                pos_normalized = values[:, 0:3]  # (T, 3) 归一化范围
                pos_cm = pos_normalized * 1.2  # 转换为厘米 (0.012米 = 1.2厘米)
                # 现在范围: 约 [-1.2, 1.2] 厘米，与Orientation量级匹配！
                
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
                    pos_cm,              # 位置：厘米（与Orientation量级匹配）
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
            
            state = fill_in_state(state_17d)
            state_indicator = fill_in_state(np.ones_like(state_std))
            state_std = fill_in_state(state_std)
            state_mean = fill_in_state(state_mean)
            actions = fill_in_action(actions_seq)
            
            # 重要：使用全局action_std作为state_norm（用于loss weighting）
            # README IMPORTANT 3: 不归一化物理量，保持物理意义
            state_norm = self.action_std_global + 1e-8  # (128,)
            
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
            # Randomly select one episode from the file
            episodes = list(f['data'].keys())
            episode_key = np.random.choice(episodes)
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
