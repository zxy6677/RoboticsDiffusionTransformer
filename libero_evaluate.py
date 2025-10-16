#!/usr/bin/env python3
"""
RDT在LIBERO任务上的评估脚本
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import cv2
from PIL import Image
import json
from datetime import datetime

# 添加路径
sys.path.append('/home/ubuntu/LIBERO/libero')
sys.path.append('/home/ubuntu/LIBERO/libero/libero')
sys.path.append('.')

import libero
from libero import benchmark
from libero.envs import OffScreenRenderEnv

from models.rdt_runner import RDTRunner
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
import importlib.util

# 动态导入state_vec
spec = importlib.util.spec_from_file_location("state_vec", "configs/state_vec.py")
state_vec_module = importlib.util.module_from_spec(spec)
sys.modules["state_vec"] = state_vec_module
spec.loader.exec_module(state_vec_module)
STATE_VEC_IDX_MAPPING = state_vec_module.STATE_VEC_IDX_MAPPING

class VideoRecorder:
    """视频录制器"""
    
    def __init__(self, output_path: str, fps: int = 30, width: int = 128, height: int = 128):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.frames = []
        
    def add_frame(self, image: np.ndarray):
        if image is not None:
            if image.shape[:2] != (self.height, self.width):
                image = cv2.resize(image, (self.width, self.height))
            self.frames.append(image)
    
    def save_video(self):
        if not self.frames:
            print("⚠️ 没有帧可以保存")
            return
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        for frame in self.frames:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
        
        out.release()
        print(f"🎥 视频已保存到: {self.output_path}")

class RDTLIBEROModel:
    """RDT模型在LIBERO上的推理包装器"""
    
    def __init__(self, config_path: str, pretrained_path: str, 
                 text_encoder_path: str, vision_encoder_path: str):
        # 加载配置
        with open(config_path, "r") as fp:
            self.config = yaml.safe_load(fp)
        
        # 初始化编码器
        self._init_encoders(text_encoder_path, vision_encoder_path)
        
        # 初始化RDT模型
        self._init_rdt_model(pretrained_path)
        
    def _init_encoders(self, text_encoder_path: str, vision_encoder_path: str):
        # 文本编码器
        self.text_encoder = T5Embedder(
            device="cuda",
            from_pretrained=text_encoder_path,
            cache_dir=None,
            model_max_length=77,
        )
        
        # 视觉编码器
        self.vision_encoder = SiglipVisionTower(
            vision_tower=vision_encoder_path,
            args=self.config,
            delay_load=False,
        )
        
    def _init_rdt_model(self, pretrained_path: str):
        # 计算图像条件长度
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
        model_file = os.path.join(pretrained_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            checkpoint = torch.load(model_file, map_location="cpu")
            self.rdt_model.load_state_dict(checkpoint, strict=False)
            print(f"✅ 加载预训练模型: {model_file}")
        else:
            print(f"⚠️ 预训练模型不存在: {model_file}")
            
    def encode_instruction(self, instruction: str) -> torch.Tensor:
        text_embeds, attention_mask = self.text_encoder.get_text_embeddings([instruction])
        return text_embeds
    
    def reset(self):
        device = "cuda"
        weight_dtype = torch.bfloat16
        self.rdt_model.eval()
        self.text_encoder.model.eval()
        self.vision_encoder.vision_tower.eval()
        
        self.rdt_model = self.rdt_model.to(device, dtype=weight_dtype)
        self.text_encoder.model = self.text_encoder.model.to(device, dtype=weight_dtype)
        self.vision_encoder.vision_tower = self.vision_encoder.vision_tower.to(device, dtype=weight_dtype)
        
    def step(self, state: torch.Tensor, images: list, text_embed: torch.Tensor) -> torch.Tensor:
        device = "cuda"
        dtype = torch.bfloat16
        
        # 处理图像
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
                image = Image.fromarray(background_image)
            
            image = self.vision_encoder.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor_list.append(image)

        image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)
        
        # 获取图像特征
        image_embeds_list = []
        for i in range(image_tensor.shape[0]):
            single_image = image_tensor[i:i+1]
            vision_output = self.vision_encoder.vision_tower(single_image)
            single_embeds = vision_output.last_hidden_state.detach()
            image_embeds_list.append(single_embeds)
        
        image_embeds = torch.cat(image_embeds_list, dim=1)
        
        # 重复图像特征以模拟历史
        img_history_size = self.config["common"]["img_history_size"]
        if img_history_size > 1:
            image_embeds = image_embeds.repeat(1, img_history_size, 1)
        
        image_embeds = image_embeds.reshape(1, -1, self.vision_encoder.vision_tower.config.hidden_size)
        
        # 处理状态
        joints = state.to(device).unsqueeze(0)
        states = joints.to(device, dtype=dtype).unsqueeze(1)
        
        # 创建状态掩码
        state_elem_mask = torch.ones(
            (1, 1, self.config["model"]["state_token_dim"]),
            device=device, dtype=dtype
        )
        
        ctrl_freqs = torch.tensor([20]).to(device)
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

def convert_libero_state_to_rdt(obs: dict, state_dim: int = 128) -> torch.Tensor:
    """将LIBERO观察转换为RDT状态格式"""
    # 提取LIBERO状态
    joint_pos = obs["robot0_joint_pos"]
    gripper_pos = obs["robot0_gripper_qpos"]
    eef_pos = obs["robot0_eef_pos"]
    eef_quat = obs["robot0_eef_quat"]
    
    # 计算gripper状态
    gripper_state = np.mean(gripper_pos)
    
    # 使用修复后的四元数到6D旋转转换函数
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
    from utils.rotation_utils import convert_quaternion_to_6d_rotation
    
    eef_ori_6d = convert_quaternion_to_6d_rotation(eef_quat)
    
    # 构建17维LIBERO状态向量
    libero_state = np.concatenate([
        joint_pos,
        [gripper_state],
        eef_pos,
        eef_ori_6d
    ])
    
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
    
    min_len = min(len(libero_state), len(right_arm_indices))
    rdt_state[right_arm_indices[:min_len]] = libero_state[:min_len]
    
    # 加载数据集统计信息进行归一化
    with open('configs/dataset_stat.json', 'r') as f:
        stats = json.load(f)
    libero_stats = stats['libero_90']
    
    state_mean = np.array(libero_stats["state_mean"])
    state_std = np.array(libero_stats["state_std"])
    state_std = np.where(state_std == 0, 1.0, state_std)
    rdt_state = (rdt_state - state_mean) / state_std
    
    return torch.from_numpy(rdt_state).float()

def convert_rdt_action_to_libero(rdt_action: torch.Tensor) -> np.ndarray:
    """将RDT动作转换为LIBERO动作格式"""
    action_128d = rdt_action[0, 0, :].cpu().numpy()
    
    # 加载数据集统计信息进行反归一化
    with open('configs/dataset_stat.json', 'r') as f:
        stats = json.load(f)
    libero_stats = stats['libero_90']
    
    # 提取位置
    pos_x_idx = STATE_VEC_IDX_MAPPING["right_eef_pos_x"]
    pos_y_idx = STATE_VEC_IDX_MAPPING["right_eef_pos_y"]
    pos_z_idx = STATE_VEC_IDX_MAPPING["right_eef_pos_z"]
    
    pos_x = action_128d[pos_x_idx] * libero_stats["state_std"][pos_x_idx] + libero_stats["state_mean"][pos_x_idx]
    pos_y = action_128d[pos_y_idx] * libero_stats["state_std"][pos_y_idx] + libero_stats["state_mean"][pos_y_idx]
    pos_z = action_128d[pos_z_idx] * libero_stats["state_std"][pos_z_idx] + libero_stats["state_mean"][pos_z_idx]
    
    # 提取6D旋转
    ori_indices = [
        STATE_VEC_IDX_MAPPING["right_eef_angle_0"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_1"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_2"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_3"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_4"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_5"]
    ]
    
    ori_6d = np.array([
        action_128d[idx] * libero_stats["state_std"][idx] + libero_stats["state_mean"][idx]
        for idx in ori_indices
    ])
    
    # 使用修复后的6D旋转转换函数
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
    from utils.rotation_utils import convert_6d_rotation_to_euler
    
    ori_3d = convert_6d_rotation_to_euler(ori_6d)
    
    # 提取gripper状态
    gripper_idx = STATE_VEC_IDX_MAPPING["right_gripper_open"]
    gripper_normalized = action_128d[gripper_idx] * libero_stats["state_std"][gripper_idx] + libero_stats["state_mean"][gripper_idx]
    gripper = gripper_normalized * 2.0 - 1.0
    
    # 构建LIBERO动作向量
    libero_action = np.array([pos_x, pos_y, pos_z, ori_3d[0], ori_3d[1], ori_3d[2], gripper])
    
    # 将动作缩放到LIBERO期望的[-1, 1]范围
    # 修复：调整缩放参数以避免过度放大
    # 位置缩放：从0.05改为0.5，减少20倍放大
    libero_action[0] = np.clip(-libero_action[0] / 0.5, -1.0, 1.0)
    libero_action[1] = np.clip(libero_action[1] / 0.5, -1.0, 1.0)
    libero_action[2] = np.clip(-libero_action[2] / 0.5, -1.0, 1.0)
    # 旋转缩放：从0.5改为2.0，减少4倍放大
    libero_action[3:6] = np.clip(libero_action[3:6] / 2.0, -1.0, 1.0)
    
    return libero_action

def debug_benchmark_info(benchmark_name: str = "libero_90", num_tasks: int = 10):
    """调试函数：打印benchmark信息"""
    print(f"🔍 调试benchmark信息...")
    
    try:
        # 获取基准
        benchmark_dict = benchmark.get_benchmark_dict()
        print(f"📋 可用的benchmark: {list(benchmark_dict.keys())}")
        
        libero_benchmark = benchmark_dict[benchmark_name]()
        task_names = libero_benchmark.get_task_names()
        
        print(f"📊 {benchmark_name} 包含 {len(task_names)} 个任务")
        print(f"📋 前{min(num_tasks, len(task_names))}个任务:")
        
        for i in range(min(num_tasks, len(task_names))):
            task = libero_benchmark.get_task(i)
            print(f"  Task {i}: {task.name}")
            print(f"    - 问题文件夹: {task.problem_folder}")
            print(f"    - BDDL文件: {task.bddl_file}")
            
            # 检查初始状态
            init_states = libero_benchmark.get_task_init_states(i)
            print(f"    - 初始状态数量: {len(init_states)}")
            
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

def evaluate_rdt_on_libero(model: RDTLIBEROModel, 
                          benchmark_name: str = "libero_90",
                          num_tasks: int = 10,  # 默认评估前10个任务
                          max_steps: int = 100,
                          record_video: bool = False,
                          video_output_dir: str = "videos") -> dict:
    """在LIBERO基准上评估RDT模型 - 顺序评估前10个任务 (task0~task9)"""
    
    print(f"🚀 开始RDT在{benchmark_name}上的顺序评估 (前{num_tasks}个任务)")
    
    # 创建视频输出目录
    if record_video:
        os.makedirs(video_output_dir, exist_ok=True)
        print(f"🎥 视频录制已启用，输出目录: {video_output_dir}")
    
    # 设置LIBERO环境
    libero.set_libero_default_path("/home/ubuntu/LIBERO/libero/libero")
    
    # 调试：打印benchmark信息
    debug_benchmark_info(benchmark_name, num_tasks)
    
    # 获取基准 - 修复：正确获取benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    libero_benchmark = benchmark_dict[benchmark_name]()
    
    # 确保只评估前10个任务
    total_available_tasks = len(libero_benchmark.get_task_names())
    num_tasks_to_evaluate = min(num_tasks, total_available_tasks)
    
    results = {
        "total_tasks": num_tasks_to_evaluate,
        "successful_tasks": 0,
        "total_steps": 0,
        "task_results": [],
        "evaluation_time": datetime.now().isoformat(),
        "benchmark_name": benchmark_name
    }
    
    print(f"📊 将评估 {num_tasks_to_evaluate} 个任务 (task0~task{num_tasks_to_evaluate-1})")
    print(f"📋 可用任务列表: {libero_benchmark.get_task_names()[:num_tasks_to_evaluate]}")
    
    # 顺序评估前10个任务
    for task_idx in range(num_tasks_to_evaluate):
        task_name = libero_benchmark.get_task_names()[task_idx]
        print(f"\n📋 评估任务 {task_idx+1}/{results['total_tasks']}: {task_name}")
        
        # 初始化视频录制器
        video_recorder = None
        if record_video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"task_{task_idx:02d}_{task_name.replace(' ', '_').replace('/', '_')}_{timestamp}.mp4"
            video_path = os.path.join(video_output_dir, video_filename)
            video_recorder = VideoRecorder(video_path, fps=10, width=128, height=128)
            print(f"  🎥 开始录制视频: {video_filename}")
        
        try:
            # 获取任务 - 修复：确保获取正确的任务
            task = libero_benchmark.get_task(task_idx)
            print(f"  🎯 任务ID: {task_idx}, 任务名称: {task.name}")
            print(f"  📁 问题文件夹: {task.problem_folder}")
            print(f"  📄 BDDL文件: {task.bddl_file}")
            
            # 创建环境 - 修复：为每个任务创建独立的环境
            bddl_files_path = libero.get_libero_path("bddl_files")
            bddl_file_path = os.path.join(bddl_files_path, task.problem_folder, task.bddl_file)
            
            print(f"  📂 BDDL文件路径: {bddl_file_path}")
            
            env_args = {
                "bddl_file_name": bddl_file_path,
                "camera_heights": 128,
                "camera_widths": 128
            }
            
            env = OffScreenRenderEnv(**env_args)
            
            # 重置环境
            obs = env.reset()
            
            # 设置初始状态 - 修复：为每个任务设置正确的初始状态
            init_states = libero_benchmark.get_task_init_states(task_idx)
            if len(init_states) > 0:
                env.set_init_state(init_states[0])
                print(f"  🏁 设置初始状态: {len(init_states)} 个状态")
            
            # 编码任务描述 - 修复：使用正确的任务描述
            task_description = task.name  # 使用任务的实际名称
            text_embed = model.encode_instruction(task_description)
            print(f"  📝 任务描述: {task_description}")
            
            # 重置模型
            model.reset()
            
            # 运行推理
            task_success = False
            task_steps = 0
            
            print(f"  🎯 开始推理循环，最大步数: {max_steps}")
            
            for step in range(max_steps):
                print(f"    📍 步骤 {step+1}:")
                
                # 设置随机种子
                torch.manual_seed(task_idx * 1000 + step)
                np.random.seed(task_idx * 1000 + step)
                
                # 准备图像输入
                img = obs["agentview_image"]
                images = [Image.fromarray(img), None, None]
                
                # 录制视频帧
                if video_recorder is not None:
                    video_recorder.add_frame(img)
                
                # 转换状态
                rdt_state = convert_libero_state_to_rdt(obs)
                
                # 模型推理
                rdt_actions = model.step(rdt_state, images, text_embed)
                
                # 转换动作
                libero_action = convert_rdt_action_to_libero(rdt_actions)
                
                # 执行动作
                obs, reward, done, info = env.step(libero_action)
                task_steps += 1
                
                print(f"      📊 奖励: {reward:.3f}, 完成: {done}")
                print(f"      📋 信息: {info}")
                
                # 检查成功条件 - 修复：更准确的成功判断
                if 'success' in info:
                    task_success = info['success']
                    print(f"      🎯 成功状态: {task_success}")
                elif 'is_success' in info:
                    task_success = info['is_success']
                    print(f"      🎯 成功状态: {task_success}")
                else:
                    # 如果没有明确的成功标志，使用奖励判断
                    task_success = reward > 0.5  # 根据任务调整阈值
                    print(f"      🎯 基于奖励判断成功: {task_success} (奖励: {reward})")
                
                if done:
                    print(f"      🏁 Episode结束: 成功={task_success}, 奖励={reward:.3f}")
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
    parser = argparse.ArgumentParser(description="RDT在LIBERO上的评估 - 顺序评估前10个任务")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="RDT配置文件路径")
    parser.add_argument("--pretrained", type=str, default="checkpoints/rdt-1b", help="预训练模型路径")
    parser.add_argument("--text_encoder", type=str, default="google/t5-v1_1-xxl", help="文本编码器路径")
    parser.add_argument("--vision_encoder", type=str, default="google/siglip-so400m-patch14-384", help="视觉编码器路径")
    parser.add_argument("--benchmark", type=str, default="libero_90", help="LIBERO基准名称")
    parser.add_argument("--num_tasks", type=int, default=10, help="评估任务数量 (默认: 10个任务)")
    parser.add_argument("--max_steps", type=int, default=100, help="每个任务最大步数")
    parser.add_argument("--record_video", action="store_true", default=True, help="是否录制视频 (默认: 启用)")
    parser.add_argument("--video_output_dir", type=str, default="videos", help="视频输出目录")
    
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
    print(f"🎯 开始顺序评估前 {args.num_tasks} 个任务 (task0~task{args.num_tasks-1})")
    if args.record_video:
        print(f"📹 视频录制已启用，将保存到: {args.video_output_dir}")
    
    results = evaluate_rdt_on_libero(
        model=model,
        benchmark_name=args.benchmark,
        num_tasks=args.num_tasks,
        max_steps=args.max_steps,
        record_video=args.record_video,
        video_output_dir=args.video_output_dir
    )
    
    # 保存结果
    with open("libero_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 评估结果已保存到: libero_evaluation_results.json")

if __name__ == "__main__":
    main()


