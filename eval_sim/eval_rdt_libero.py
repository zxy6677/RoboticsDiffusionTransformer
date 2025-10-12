#!/usr/bin/env python3
"""
RDT在LIBERO任务上的推理评估脚本
基于RDT的推理流程，适配LIBERO环境
"""

import sys
import os
import numpy as np
import torch
from PIL import Image
import yaml
import argparse
from typing import Dict, List, Tuple, Optional
import cv2
from datetime import datetime

# 添加路径
sys.path.append('/home/ubuntu/LIBERO/libero')
sys.path.append('/home/ubuntu/LIBERO/libero/libero')
sys.path.append('/home/ubuntu/RoboticsDiffusionTransformer')

# 导入LIBERO模块
import libero
import benchmark
from libero.envs import OffScreenRenderEnv

# 导入RDT模块
import importlib.util
import sys

# 动态导入configs.state_vec
spec = importlib.util.spec_from_file_location("state_vec", "/home/ubuntu/RoboticsDiffusionTransformer/configs/state_vec.py")
state_vec_module = importlib.util.module_from_spec(spec)
sys.modules["state_vec"] = state_vec_module
spec.loader.exec_module(state_vec_module)
STATE_VEC_IDX_MAPPING = state_vec_module.STATE_VEC_IDX_MAPPING

from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner

class VideoRecorder:
    """视频录制器"""
    
    def __init__(self, output_path: str, fps: int = 30, width: int = 128, height: int = 128):
        """初始化视频录制器"""
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.frames = []
        
    def add_frame(self, image: np.ndarray):
        """添加帧"""
        if image is not None:
            # 确保图像尺寸正确
            if image.shape[:2] != (self.height, self.width):
                image = cv2.resize(image, (self.width, self.height))
            self.frames.append(image)
    
    def save_video(self):
        """保存视频"""
        if not self.frames:
            print("⚠️ 没有帧可以保存")
            return
            
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        for frame in self.frames:
            # 确保帧是BGR格式
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # 假设是RGB，转换为BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
        
        out.release()
        print(f"🎥 视频已保存到: {self.output_path}")
        
    def clear(self):
        """清空帧"""
        self.frames = []

class RDTLIBEROModel:
    """RDT模型在LIBERO上的推理包装器"""
    
    def __init__(self, config_path: str, pretrained_path: str, 
                 text_encoder_path: str, vision_encoder_path: str):
        """初始化RDT模型"""
        self.config_path = config_path
        self.pretrained_path = pretrained_path
        self.text_encoder_path = text_encoder_path
        self.vision_encoder_path = vision_encoder_path
        
        # 加载配置
        with open(config_path, "r") as fp:
            self.config = yaml.safe_load(fp)
        
        # 初始化编码器
        self._init_encoders()
        
        # 初始化RDT模型
        self._init_rdt_model()
        
    def _init_encoders(self):
        """初始化文本和视觉编码器"""
        # 文本编码器
        self.text_encoder = T5Embedder(
            device="cuda",
            from_pretrained=self.text_encoder_path,
            cache_dir=None,
            model_max_length=77,
        )
        
        # 视觉编码器
        self.vision_encoder = SiglipVisionTower(
            vision_tower=self.vision_encoder_path,
            args=self.config,
            delay_load=False,
        )
        
    def _init_rdt_model(self):
        """初始化RDT模型"""
        # 计算图像条件长度
        # SigLIP的patch数量计算: (image_size / patch_size)^2
        patch_size = self.vision_encoder.vision_tower.config.patch_size
        image_size = self.vision_encoder.vision_tower.config.image_size
        num_patches = (image_size // patch_size) ** 2
        
        img_cond_len = (self.config["common"]["img_history_size"] 
                       * self.config["common"]["num_cameras"] 
                       * num_patches)
        
        self.rdt_model = RDTRunner(
            action_dim=self.config["common"]["state_dim"],
            pred_horizon=self.config["common"]["action_chunk_size"],
            config=self.config["model"],
            lang_token_dim=self.config["model"]["lang_token_dim"],
            img_token_dim=self.config["model"]["img_token_dim"],
            state_token_dim=self.config["model"]["state_token_dim"],
            max_lang_cond_len=self.config["dataset"]["tokenizer_max_length"],
            img_cond_len=img_cond_len,
            img_pos_embed_config=[
                ("image", (self.config["common"]["img_history_size"], 
                    self.config["common"]["num_cameras"], 
                    -num_patches)),  
            ],
            lang_pos_embed_config=[
                ("lang", -self.config["dataset"]["tokenizer_max_length"]),
            ],
            dtype=torch.bfloat16,
        )
        
        # 加载预训练权重
        model_file = os.path.join(self.pretrained_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            checkpoint = torch.load(model_file, map_location="cpu")
            self.rdt_model.load_state_dict(checkpoint, strict=False)
            print(f"✅ 加载预训练模型: {model_file}")
        else:
            print(f"⚠️ 预训练模型不存在: {model_file}")
            
    def encode_instruction(self, instruction: str) -> torch.Tensor:
        """编码任务指令"""
        text_embeds, attention_mask = self.text_encoder.get_text_embeddings([instruction])
        return text_embeds
    
    def reset(self):
        """重置模型状态"""
        device = "cuda"
        weight_dtype = torch.bfloat16
        self.rdt_model.eval()
        self.text_encoder.model.eval()
        self.vision_encoder.vision_tower.eval()
        
        self.rdt_model = self.rdt_model.to(device, dtype=weight_dtype)
        self.text_encoder.model = self.text_encoder.model.to(device, dtype=weight_dtype)
        self.vision_encoder.vision_tower = self.vision_encoder.vision_tower.to(device, dtype=weight_dtype)
        
    def step(self, state: torch.Tensor, images: List[Image.Image], 
             text_embed: torch.Tensor) -> torch.Tensor:
        """执行一步推理"""
        device = "cuda"
        dtype = torch.bfloat16
        
        # 处理图像 - 参考maniskill_model.py的实现
        background_color = np.array([
            int(x*255) for x in self.vision_encoder.image_processor.image_mean
        ], dtype=np.uint8).reshape(1, 1, 3)
        background_image = np.ones((
            self.vision_encoder.image_processor.size["height"], 
            self.vision_encoder.image_processor.size["width"], 3), dtype=np.uint8
        ) * background_color
        
        image_tensor_list = []
        for image in images:
            if image is None:
                # 用背景图像替换
                image = Image.fromarray(background_image)
            
            # 预处理图像
            image = self.vision_encoder.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor_list.append(image)

        image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)
        
        # 获取图像特征 - 处理每个图像
        image_embeds_list = []
        for i in range(image_tensor.shape[0]):
            single_image = image_tensor[i:i+1]  # (1, 3, 384, 384)
            vision_output = self.vision_encoder.vision_tower(single_image)
            single_embeds = vision_output.last_hidden_state.detach()  # (1, 729, 1152)
            image_embeds_list.append(single_embeds)
        
        # 拼接所有图像特征
        image_embeds = torch.cat(image_embeds_list, dim=1)  # (1, 729*3, 1152)
        
        # 为了匹配训练时的格式，我们需要重复图像特征以模拟历史
        # 训练时使用 img_history_size=2，所以我们需要重复一次
        img_history_size = self.config["common"]["img_history_size"]
        if img_history_size > 1:
            # 重复图像特征以模拟历史
            image_embeds = image_embeds.repeat(1, img_history_size, 1)  # (1, 729*3*2, 1152)
        
        # 重塑为正确的形状: (batch_size, num_patches, hidden_size)
        image_embeds = image_embeds.reshape(1, -1, self.vision_encoder.vision_tower.config.hidden_size)
        
        # 处理状态
        joints = state.to(device).unsqueeze(0)   # (1, 128)
        states = joints.to(device, dtype=dtype).unsqueeze(1)  # (1, 1, 128)
        
        # 创建状态掩码
        state_elem_mask = torch.ones(
            (1, 1, self.config["model"]["state_token_dim"]),
            device=device, dtype=dtype
        )
        
        
        ctrl_freqs = torch.tensor([20]).to(device)  # LIBERO控制频率
        
        text_embeds = text_embed.to(device, dtype=dtype)
        
        with torch.no_grad():
            trajectory = self.rdt_model.predict_action(
                lang_tokens=text_embeds,
                lang_attn_mask=torch.ones(
                    text_embeds.shape[:2], dtype=torch.bool,
                    device=text_embeds.device),
                img_tokens=image_embeds,
                state_tokens=states,
                action_mask=state_elem_mask,  
                ctrl_freqs=ctrl_freqs
            )
        
        return trajectory.to(torch.float32)

def convert_libero_state_to_rdt(obs: Dict, state_dim: int = 128) -> torch.Tensor:
    """将LIBERO观察转换为RDT状态格式"""
    # 提取LIBERO状态
    joint_pos = obs["robot0_joint_pos"]  # (7,)
    gripper_pos = obs["robot0_gripper_qpos"]  # (2,)
    eef_pos = obs["robot0_eef_pos"]  # (3,)
    eef_quat = obs["robot0_eef_quat"]  # (4,)
    
    # 计算gripper状态
    gripper_state = np.mean(gripper_pos)
    
    # 将四元数转换为6D旋转表示
    def quat_to_6d_rotation(quat):
        """将四元数转换为6D旋转表示"""
        # 使用scipy进行精确的四元数到旋转矩阵转换
        from scipy.spatial.transform import Rotation
        r = Rotation.from_quat(quat)
        rot_matrix = r.as_matrix()
        # 取前两列作为6D表示
        return rot_matrix[:, :2].flatten()  # (6,)
    
    eef_ori_6d = quat_to_6d_rotation(eef_quat)
    
    # 构建17维LIBERO状态向量
    libero_state = np.concatenate([
        joint_pos,           # 7维
        [gripper_state],     # 1维
        eef_pos,            # 3维
        eef_ori_6d          # 6维 (正确的6D旋转)
    ])  # 总共17维
    
    # 映射到RDT的128维状态空间
    rdt_state = np.zeros(state_dim)
    
    # 使用右臂的索引映射
    right_arm_indices = [
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
    
    # 确保索引数量匹配
    min_len = min(len(libero_state), len(right_arm_indices))
    rdt_state[right_arm_indices[:min_len]] = libero_state[:min_len]
    
    # 加载数据集统计信息进行归一化
    import json
    with open('configs/dataset_stat.json', 'r') as f:
        stats = json.load(f)
    libero_stats = stats['libero_90']
    
    # 对状态进行归一化
    state_mean = np.array(libero_stats["state_mean"])
    state_std = np.array(libero_stats["state_std"])
    
    # 避免除零
    state_std = np.where(state_std == 0, 1.0, state_std)
    
    # 归一化
    rdt_state = (rdt_state - state_mean) / state_std
    
    return torch.from_numpy(rdt_state).float()

def convert_rdt_action_to_libero(rdt_action: torch.Tensor) -> np.ndarray:
    """将RDT动作转换为LIBERO动作格式"""
    # RDT输出128维动作，需要从正确的索引提取LIBERO动作
    action_128d = rdt_action[0, 0, :].cpu().numpy()  # (128,) - 先移到CPU再转numpy
    
    # 加载数据集统计信息进行反归一化
    import json
    with open('configs/dataset_stat.json', 'r') as f:
        stats = json.load(f)
    libero_stats = stats['libero_90']
    
    # 根据训练时的映射，从RDT的128维输出中提取LIBERO的7D动作
    # LIBERO动作: [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, gripper]
    # 注意：LIBERO使用增量控制，动作范围是[-1, 1]，输出范围是位置[-0.05, 0.05]，旋转[-0.5, 0.5]
    
    # 提取位置 (3D) - 需要反归一化
    pos_x_idx = STATE_VEC_IDX_MAPPING["right_eef_pos_x"]  # 索引30
    pos_y_idx = STATE_VEC_IDX_MAPPING["right_eef_pos_y"]  # 索引31
    pos_z_idx = STATE_VEC_IDX_MAPPING["right_eef_pos_z"]  # 索引32
    
    pos_x = action_128d[pos_x_idx] * libero_stats["state_std"][pos_x_idx] + libero_stats["state_mean"][pos_x_idx]
    pos_y = action_128d[pos_y_idx] * libero_stats["state_std"][pos_y_idx] + libero_stats["state_mean"][pos_y_idx]
    pos_z = action_128d[pos_z_idx] * libero_stats["state_std"][pos_z_idx] + libero_stats["state_mean"][pos_z_idx]
    
    # 提取完整的6D旋转表示 - 需要反归一化
    ori_indices = [
        STATE_VEC_IDX_MAPPING["right_eef_angle_0"],  # 索引33
        STATE_VEC_IDX_MAPPING["right_eef_angle_1"],  # 索引34
        STATE_VEC_IDX_MAPPING["right_eef_angle_2"],  # 索引35
        STATE_VEC_IDX_MAPPING["right_eef_angle_3"],  # 索引36
        STATE_VEC_IDX_MAPPING["right_eef_angle_4"],  # 索引37
        STATE_VEC_IDX_MAPPING["right_eef_angle_5"]   # 索引38
    ]
    
    ori_6d = np.array([
        action_128d[idx] * libero_stats["state_std"][idx] + libero_stats["state_mean"][idx]
        for idx in ori_indices
    ])
    
    # 将6D旋转表示转换为3D欧拉角
    def rotation_6d_to_euler(rot_6d):
        """将6D旋转表示转换为3D欧拉角"""
        # 重构旋转矩阵
        r1 = rot_6d[:3]  # 第一列
        r2 = rot_6d[3:]  # 第二列
        
        # 归一化
        r1 = r1 / (np.linalg.norm(r1) + 1e-8)
        r2 = r2 / (np.linalg.norm(r2) + 1e-8)
        
        # 计算第三列
        r3 = np.cross(r1, r2)
        r3 = r3 / (np.linalg.norm(r3) + 1e-8)
        
        # 重构旋转矩阵
        rot_matrix = np.column_stack([r1, r2, r3])
        
        # 转换为欧拉角
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(rot_matrix)
        euler = r.as_euler('xyz', degrees=False)
        return euler
    
    ori_3d = rotation_6d_to_euler(ori_6d)
    
    # 提取gripper状态 - 需要反归一化
    gripper_idx = STATE_VEC_IDX_MAPPING["right_gripper_open"]  # 索引10
    gripper_normalized = action_128d[gripper_idx] * libero_stats["state_std"][gripper_idx] + libero_stats["state_mean"][gripper_idx]
    
    # 将gripper从[0,1]映射回[-1,1]
    gripper = gripper_normalized * 2.0 - 1.0
    
    # 构建LIBERO动作向量 - 注意：LIBERO期望的是[-1, 1]范围内的增量控制
    # 位置增量范围：[-0.05, 0.05] 对应 [-1, 1]
    # 旋转增量范围：[-0.5, 0.5] 对应 [-1, 1]
    libero_action = np.array([pos_x, pos_y, pos_z, ori_3d[0], ori_3d[1], ori_3d[2], gripper])
    
    # 将动作缩放到LIBERO期望的[-1, 1]范围
    # 位置：从米单位缩放到[-1, 1]（对应[-0.05, 0.05]米）
    # 尝试反转某些轴的方向来修正运动方向
    libero_action[0] = np.clip(-libero_action[0] / 0.05, -1.0, 1.0)  # 反转X轴
    libero_action[1] = np.clip(libero_action[1] / 0.05, -1.0, 1.0)   # Y轴保持不变
    libero_action[2] = np.clip(-libero_action[2] / 0.05, -1.0, 1.0)  # 反转Z轴
    
    # 旋转：从弧度单位缩放到[-1, 1]（对应[-0.5, 0.5]弧度）
    libero_action[3:6] = np.clip(libero_action[3:6] / 0.5, -1.0, 1.0)
    
    # gripper已经在[-1, 1]范围内，保持不变
    
    return libero_action

def evaluate_rdt_on_libero(model: RDTLIBEROModel, 
                          benchmark_name: str = "libero_90",
                          num_tasks: int = 5,
                          max_steps: int = 100,
                          record_video: bool = False,
                          video_output_dir: str = "videos") -> Dict:
    """在LIBERO基准上评估RDT模型"""
    
    print(f"🚀 开始RDT在{benchmark_name}上的评估")
    
    # 创建视频输出目录
    if record_video:
        os.makedirs(video_output_dir, exist_ok=True)
        print(f"🎥 视频录制已启用，输出目录: {video_output_dir}")
    
    # 设置LIBERO环境
    libero.set_libero_default_path("/home/ubuntu/LIBERO/libero/libero")
    
    # 获取基准
    benchmark_dict = benchmark.get_benchmark_dict()
    libero_benchmark = benchmark_dict[benchmark_name]()
    
    results = {
        "total_tasks": min(num_tasks, len(libero_benchmark.get_task_names())),
        "successful_tasks": 0,
        "total_steps": 0,
        "task_results": []
    }
    
    # 评估指定数量的任务
    for task_idx in range(results["total_tasks"]):
        task_name = libero_benchmark.get_task_names()[task_idx]
        print(f"\n📋 评估任务 {task_idx+1}/{results['total_tasks']}: {task_name}")
        
        # 初始化视频录制器
        video_recorder = None
        if record_video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"task_{task_idx+1:02d}_{task_name[:30]}_{timestamp}.mp4"
            video_path = os.path.join(video_output_dir, video_filename)
            video_recorder = VideoRecorder(video_path, fps=10, width=128, height=128)
            print(f"  🎥 开始录制视频: {video_filename}")
        
        try:
            # 获取任务
            task = libero_benchmark.get_task(task_idx)
            
            # 创建环境
            bddl_files_path = libero.get_libero_path("bddl_files")
            bddl_file_path = os.path.join(bddl_files_path, task.problem_folder, task.bddl_file)
            
            env_args = {
                "bddl_file_name": bddl_file_path,
                "camera_heights": 128,
                "camera_widths": 128
            }
            
            env = OffScreenRenderEnv(**env_args)
            
            # 重置环境
            obs = env.reset()
            
            # 设置初始状态
            init_states = libero_benchmark.get_task_init_states(task_idx)
            if len(init_states) > 0:
                env.set_init_state(init_states[0])
            
            # 编码任务描述
            text_embed = model.encode_instruction(task_name)
            print(f"  📝 任务描述: {task_name}")
            print(f"  📝 文本嵌入形状: {text_embed.shape}")
            print(f"  📝 文本嵌入范围: [{text_embed.min():.3f}, {text_embed.max():.3f}]")
            
            # 重置模型
            model.reset()
            
            # 运行推理
            task_success = False
            task_steps = 0
            
            print(f"  🎯 开始推理循环，最大步数: {max_steps}")
            
            for step in range(max_steps):
                print(f"    📍 步骤 {step+1}:")
                
                # 设置随机种子以确保每次推理都有不同的随机性
                torch.manual_seed(task_idx * 1000 + step)
                np.random.seed(task_idx * 1000 + step)
                
                # 准备图像输入
                img = obs["agentview_image"]
                images = [Image.fromarray(img), None, None]  # [cam_high, cam_right_wrist, cam_left_wrist]
                print(f"      📷 图像形状: {img.shape}")
                
                # 录制视频帧
                if video_recorder is not None:
                    video_recorder.add_frame(img)
                
                # 转换状态
                rdt_state = convert_libero_state_to_rdt(obs)
                print(f"      🔧 RDT状态形状: {rdt_state.shape}")
                print(f"      🔧 关节位置: {obs['robot0_joint_pos'][:3]}")
                print(f"      🔧 末端执行器位置: {obs['robot0_eef_pos']}")
                
                # 模型推理
                print(f"      🧠 执行RDT推理...")
                rdt_actions = model.step(rdt_state, images, text_embed)
                print(f"      🧠 RDT输出形状: {rdt_actions.shape}")
                print(f"      🧠 RDT输出范围: [{rdt_actions.min():.3f}, {rdt_actions.max():.3f}]")
                
                # 检查RDT输出的变化
                if step == 0:
                    first_rdt_output = rdt_actions.clone()
                    # 保存第一个任务的状态和文本嵌入用于比较
                    if task_idx == 0:
                        first_task_state = rdt_state.clone()
                        first_task_text = text_embed.clone()
                else:
                    diff = torch.abs(rdt_actions - first_rdt_output).mean()
                    print(f"      🔍 RDT输出与第一步的差异: {diff:.6f}")
                    
                    # 如果是第二个任务的第一步，比较与第一个任务的差异
                    if task_idx == 1 and step == 0:
                        state_diff = torch.abs(rdt_state - first_task_state).mean()
                        text_diff = torch.abs(text_embed - first_task_text).mean()
                        print(f"      🔍 与任务1的状态差异: {state_diff:.6f}")
                        print(f"      🔍 与任务1的文本差异: {text_diff:.6f}")
                
                # 转换动作
                libero_action = convert_rdt_action_to_libero(rdt_actions)
                print(f"      ⚡ LIBERO动作: {libero_action}")
                print(f"      ⚡ 动作范围: [{libero_action.min():.3f}, {libero_action.max():.3f}]")
                
                # 执行动作
                print(f"      🎮 执行动作...")
                obs, reward, done, info = env.step(libero_action)
                task_steps += 1
                
                print(f"      📊 奖励: {reward:.3f}, 完成: {done}")
                if 'success' in info:
                    print(f"      🎯 成功状态: {info['success']}")
                
                if done:
                    task_success = info.get("success", False)
                    print(f"      🏁 Episode结束: 成功={task_success}")
                    break
            
            # 记录结果
            task_result = {
                "task_name": task_name,
                "success": task_success,
                "steps": task_steps,
                "reward": reward
            }
            results["task_results"].append(task_result)
            
            if task_success:
                results["successful_tasks"] += 1
                print(f"  ✅ 任务成功完成，步数: {task_steps}")
            else:
                print(f"  ❌ 任务失败，步数: {task_steps}")
            
            # 保存视频
            if video_recorder is not None:
                video_recorder.save_video()
            
            results["total_steps"] += task_steps
            
            env.close()
            
        except Exception as e:
            print(f"  ❌ 任务执行出错: {e}")
            results["task_results"].append({
                "task_name": task_name,
                "success": False,
                "steps": 0,
                "reward": 0,
                "error": str(e)
            })
    
    # 计算成功率
    results["success_rate"] = results["successful_tasks"] / results["total_tasks"]
    results["avg_steps"] = results["total_steps"] / results["total_tasks"]
    
    print(f"\n📊 评估结果:")
    print(f"  总任务数: {results['total_tasks']}")
    print(f"  成功任务数: {results['successful_tasks']}")
    print(f"  成功率: {results['success_rate']:.2%}")
    print(f"  平均步数: {results['avg_steps']:.1f}")
    
    return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RDT在LIBERO上的推理评估")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                       help="RDT配置文件路径")
    parser.add_argument("--pretrained", type=str, default="/home/ubuntu/rdt-1b",
                       help="预训练模型路径")
    parser.add_argument("--text_encoder", type=str, default="/home/ubuntu/t5-v1_1-xxl",
                       help="文本编码器路径")
    parser.add_argument("--vision_encoder", type=str, default="/home/ubuntu/siglip-so400m-patch14-384",
                       help="视觉编码器路径")
    parser.add_argument("--benchmark", type=str, default="libero_90",
                       help="LIBERO基准名称")
    parser.add_argument("--num_tasks", type=int, default=5,
                       help="评估任务数量")
    parser.add_argument("--max_steps", type=int, default=100,
                       help="每个任务最大步数")
    parser.add_argument("--record_video", action="store_true",
                       help="是否录制视频")
    parser.add_argument("--video_output_dir", type=str, default="videos",
                       help="视频输出目录")
    
    args = parser.parse_args()
    
    # 初始化模型
    print("🤖 初始化RDT模型...")
    model = RDTLIBEROModel(
        config_path=args.config,
        pretrained_path=args.pretrained,
        text_encoder_path=args.text_encoder,
        vision_encoder_path=args.vision_encoder
    )
    
    # 运行评估
    results = evaluate_rdt_on_libero(
        model=model,
        benchmark_name=args.benchmark,
        num_tasks=args.num_tasks,
        max_steps=args.max_steps,
        record_video=args.record_video,
        video_output_dir=args.video_output_dir
    )
    
    # 保存结果
    import json
    with open("libero_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 评估结果已保存到: libero_evaluation_results.json")

if __name__ == "__main__":
    main()