# RDT在LIBERO上的评估实现详解

## 目录
1. [LIBERO评估机制分析](#libero评估机制分析)
2. [RDT与LIBERO的数据格式差异](#rdt与libero的数据格式差异)
3. [RDT评估代码实现](#rdt评估代码实现)
4. [关键函数详解](#关键函数详解)
5. [数据流转过程](#数据流转过程)

---

## 1. LIBERO评估机制分析

### 1.1 核心评估流程

LIBERO的评估流程在 `libero/lifelong/metric.py` 中实现，主要函数是 `evaluate_one_task_success`：

```python
def evaluate_one_task_success(cfg, algo, task, task_emb, task_id, sim_states=None, task_str=""):
    """
    评估单个任务的成功率
    
    参数:
    - cfg: 配置对象
    - algo: 算法对象（包含policy）
    - task: 任务对象（包含BDDL文件等）
    - task_emb: 任务嵌入（来自语言模型）
    - task_id: 任务ID
    - sim_states: 模拟状态记录（用于可视化）
    - task_str: 任务字符串标识
    """
```

**关键步骤：**

1. **环境创建**
   ```python
   env_args = {
       "bddl_file_name": os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
       "camera_heights": cfg.data.img_h,
       "camera_widths": cfg.data.img_w,
   }
   env = OffScreenRenderEnv(**env_args)
   ```

2. **初始状态设置**
   ```python
   init_states = torch.load(init_states_path)  # 加载固定的初始状态
   obs = env.set_init_state(init_states_)      # 设置环境初始状态
   ```

3. **评估循环**
   ```python
   while steps < cfg.eval.max_steps:
       # 1. 将原始观测转换为张量
       data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
       
       # 2. 获取动作
       actions = algo.policy.get_action(data)
       
       # 3. 执行动作
       obs, reward, done, info = env.step(actions)
       
       # 4. 检查是否成功
       if done:
           break
   ```

### 1.2 观测数据转换

`raw_obs_to_tensor_obs` 函数的作用是将环境返回的原始观测字典转换为模型需要的张量格式：

```python
def raw_obs_to_tensor_obs(obs, task_emb, cfg):
    """
    将原始观测转换为模型输入格式
    
    输入:
    - obs: 列表，包含每个环境的观测字典
        例如: obs[0] = {
            'agentview_image': (128, 128, 3),      # RGB图像
            'robot0_joint_pos': (7,),              # 关节位置
            'robot0_gripper_qpos': (2,),           # 夹爪位置
            'robot0_eef_pos': (3,),                # 末端执行器位置
            'robot0_eef_quat': (4,),               # 末端执行器四元数
            ...
        }
    - task_emb: 任务嵌入张量 (1, 512)
    - cfg: 配置对象
    
    输出:
    - data: 字典
        {
            'obs': {
                'agentview_rgb': Tensor (env_num, 3, 128, 128),
                'joint_states': Tensor (env_num, 7),
                'gripper_states': Tensor (env_num, 2),
                'ee_states': Tensor (env_num, 3),
                ...
            },
            'task_emb': Tensor (env_num, 512)
        }
    """
```

**关键处理：**
- 使用 `cfg.data.obs_key_mapping` 进行键名映射
- 使用 `ObsUtils.process_obs` 处理每个观测（例如图像归一化、维度调整）
- 将任务嵌入复制到每个环境

### 1.3 策略接口

LIBERO的策略类（如 `BCTransformerPolicy`）实现了标准接口：

```python
class BasePolicy(nn.Module):
    def get_action(self, data):
        """
        获取动作的API
        
        输入:
        - data: 字典，包含 'obs' 和 'task_emb'
        
        输出:
        - actions: numpy数组 (env_num, 7)
                  [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, gripper]
        """
        self.eval()
        with torch.no_grad():
            data = self.preprocess_input(data, train_mode=False)
            x = self.spatial_encode(data)      # 空间编码（图像、状态、语言）
            self.latent_queue.append(x)        # 维护时间序列
            x = torch.cat(self.latent_queue, dim=1)
            x = self.temporal_encode(x)        # 时间编码（Transformer）
            dist = self.policy_head(x[:, -1])  # 动作预测头
            actions = dist.sample().cpu().numpy()
        return actions
```

---

## 2. RDT与LIBERO的数据格式差异

### 2.1 模型架构差异

| 方面 | LIBERO | RDT |
|-----|--------|-----|
| 模型类型 | Transformer BC (行为克隆) | Diffusion Transformer |
| 输入格式 | 字典 {'obs': {...}, 'task_emb': ...} | 分离的 (state, images, text_embed) |
| 状态表示 | 多模态（joint, gripper, ee等） | 统一的128维向量 |
| 图像处理 | 使用robomimic的ObsUtils | 使用SigLIP预训练模型 |
| 文本编码 | 简单的嵌入向量 | T5 XXL预训练模型 |
| 动作输出 | 直接输出7维动作 | 输出128维，需要提取和转换 |

### 2.2 关键差异点

#### 2.2.1 状态维度

**LIBERO:**
- 原始状态：17维
  - joint_pos (7) + gripper_pos (2) + eef_pos (3) + eef_quat (4) = 17
- 模型内部可能使用不同的表示，但输入输出都是7维动作空间

**RDT:**
- 统一状态空间：128维
- 包含多个机械臂的状态（左臂、右臂）
- LIBERO的17维状态需要映射到RDT的128维空间中的特定位置

#### 2.2.2 旋转表示

**LIBERO:**
- 输入：四元数 (4维)
- 输出：欧拉角增量 (3维)

**RDT:**
- 统一使用6D旋转表示
- 需要进行四元数 ↔ 6D旋转 ↔ 欧拉角的转换

#### 2.2.3 图像编码

**LIBERO:**
```python
# 使用ResNet等编码器
image_encoder = ResNetEncoder(...)
img_features = image_encoder(images)  # (B, T, C, H, W) -> (B, T, 1, E)
```

**RDT:**
```python
# 使用SigLIP Vision Tower
vision_encoder = SiglipVisionTower(...)
img_features = vision_encoder(images)  # (B, 3, 384, 384) -> (B, 729, 1152)
# 729 = (384/14)^2 patches
# 1152 = hidden_size
```

#### 2.2.4 动作空间

**LIBERO动作空间:**
```python
action = [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, gripper]  # 7维
# 位置增量: [-1, 1] 对应 [-0.05m, 0.05m]
# 旋转增量: [-1, 1] 对应 [-0.5rad, 0.5rad]  
# 夹爪: [-1, 1] (-1=关闭, 1=打开)
```

**RDT动作空间:**
```python
action = [...]  # 128维
# 包含所有状态维度的预测
# 需要从特定索引提取LIBERO的7维动作
# 需要反归一化和尺度转换
```

---

## 3. RDT评估代码实现

### 3.1 总体架构

RDT的评估代码（`eval_sim/eval_rdt_libero.py`）主要包含以下组件：

```
RDTLIBEROModel (模型包装器)
├── _init_encoders()           # 初始化T5和SigLIP编码器
├── _init_rdt_model()          # 初始化RDT Diffusion模型
├── encode_instruction()       # 编码任务描述
├── reset()                    # 重置模型到GPU
└── step()                     # 执行推理步骤

数据转换函数
├── convert_libero_state_to_rdt()    # LIBERO obs -> RDT state (128D)
└── convert_rdt_action_to_libero()   # RDT action (128D) -> LIBERO action (7D)

评估函数
└── evaluate_rdt_on_libero()         # 主评估循环
```

### 3.2 模型初始化

```python
class RDTLIBEROModel:
    def __init__(self, config_path, pretrained_path, text_encoder_path, vision_encoder_path):
        # 1. 加载配置
        with open(config_path, "r") as fp:
            self.config = yaml.safe_load(fp)
        
        # 2. 初始化文本编码器 (T5 XXL)
        self.text_encoder = T5Embedder(
            device="cuda",
            from_pretrained=text_encoder_path,
            model_max_length=77,
        )
        
        # 3. 初始化视觉编码器 (SigLIP)
        self.vision_encoder = SiglipVisionTower(
            vision_tower=vision_encoder_path,
            args=self.config,
            delay_load=False,
        )
        
        # 4. 初始化RDT模型
        # 计算图像条件长度
        patch_size = 14  # SigLIP patch size
        image_size = 384 # SigLIP image size
        num_patches = (384 // 14) ** 2 = 729
        
        img_cond_len = (
            img_history_size=2 *     # 历史帧数
            num_cameras=3 *          # 摄像头数量
            num_patches=729          # 每个图像的patch数
        ) = 4374
        
        self.rdt_model = RDTRunner(
            action_dim=128,                    # RDT的统一动作维度
            pred_horizon=64,                   # 预测时域（action chunk）
            lang_token_dim=4096,               # T5 XXL输出维度
            img_token_dim=1152,                # SigLIP输出维度
            state_token_dim=128,               # 状态维度
            max_lang_cond_len=77,              # 最大文本长度
            img_cond_len=4374,                 # 图像条件长度
            ...
        )
        
        # 5. 加载预训练权重
        checkpoint = torch.load("pytorch_model.bin")
        self.rdt_model.load_state_dict(checkpoint)
```

---

## 4. 关键函数详解

### 4.1 状态转换: `convert_libero_state_to_rdt`

这是**最关键**的函数之一，负责将LIBERO的观测转换为RDT的输入格式。

```python
def convert_libero_state_to_rdt(obs: Dict, state_dim: int = 128) -> torch.Tensor:
    """
    将LIBERO观测转换为RDT状态格式
    
    输入:
    - obs: LIBERO环境观测字典
        {
            'robot0_joint_pos': (7,),      # 关节位置
            'robot0_gripper_qpos': (2,),   # 夹爪位置
            'robot0_eef_pos': (3,),        # 末端执行器位置
            'robot0_eef_quat': (4,),       # 末端执行器四元数
            'agentview_image': (128,128,3), # RGB图像
            ...
        }
    
    输出:
    - rdt_state: Tensor (128,) 归一化的RDT状态向量
    """
    
    # 步骤1: 提取LIBERO状态
    joint_pos = obs["robot0_joint_pos"]          # (7,)
    gripper_pos = obs["robot0_gripper_qpos"]     # (2,)
    eef_pos = obs["robot0_eef_pos"]              # (3,)
    eef_quat = obs["robot0_eef_quat"]            # (4,)
    
    # 步骤2: 计算gripper状态
    # LIBERO的gripper_qpos是两个关节的位置，取平均值得到开合状态
    gripper_state = np.mean(gripper_pos)  # 标量，范围约[0, 0.04]
    
    # 步骤3: 四元数转6D旋转
    # RDT使用6D旋转表示，需要转换
    from utils.rotation_utils import convert_quaternion_to_6d_rotation
    eef_ori_6d = convert_quaternion_to_6d_rotation(eef_quat)  # (6,)
    
    # 步骤4: 构建17维LIBERO状态向量
    libero_state = np.concatenate([
        joint_pos,           # 7维: 关节位置
        [gripper_state],     # 1维: 夹爪状态
        eef_pos,            # 3维: 末端执行器位置
        eef_ori_6d          # 6维: 末端执行器6D旋转
    ])  # 总共17维
    
    # 步骤5: 映射到RDT的128维状态空间
    rdt_state = np.zeros(state_dim)  # 初始化为0
    
    # 使用STATE_VEC_IDX_MAPPING定义的索引映射
    # 将LIBERO状态映射到"right arm"的位置
    right_arm_indices = [
        STATE_VEC_IDX_MAPPING["right_arm_joint_0_pos"],  # 索引3
        STATE_VEC_IDX_MAPPING["right_arm_joint_1_pos"],  # 索引4
        STATE_VEC_IDX_MAPPING["right_arm_joint_2_pos"],  # 索引5
        STATE_VEC_IDX_MAPPING["right_arm_joint_3_pos"],  # 索引6
        STATE_VEC_IDX_MAPPING["right_arm_joint_4_pos"],  # 索引7
        STATE_VEC_IDX_MAPPING["right_arm_joint_5_pos"],  # 索引8
        STATE_VEC_IDX_MAPPING["right_arm_joint_6_pos"],  # 索引9
        STATE_VEC_IDX_MAPPING["right_gripper_open"],      # 索引10
        STATE_VEC_IDX_MAPPING["right_eef_pos_x"],         # 索引30
        STATE_VEC_IDX_MAPPING["right_eef_pos_y"],         # 索引31
        STATE_VEC_IDX_MAPPING["right_eef_pos_z"],         # 索引32
        STATE_VEC_IDX_MAPPING["right_eef_angle_0"],       # 索引33
        STATE_VEC_IDX_MAPPING["right_eef_angle_1"],       # 索引34
        STATE_VEC_IDX_MAPPING["right_eef_angle_2"],       # 索引35
        STATE_VEC_IDX_MAPPING["right_eef_angle_3"],       # 索引36
        STATE_VEC_IDX_MAPPING["right_eef_angle_4"],       # 索引37
        STATE_VEC_IDX_MAPPING["right_eef_angle_5"],       # 索引38
    ]  # 17个索引
    
    # 填充RDT状态
    rdt_state[right_arm_indices] = libero_state
    
    # 步骤6: 归一化
    # 加载训练时的数据统计信息
    with open('configs/dataset_stat.json', 'r') as f:
        stats = json.load(f)
    libero_stats = stats['libero_90']
    
    state_mean = np.array(libero_stats["state_mean"])  # (128,)
    state_std = np.array(libero_stats["state_std"])    # (128,)
    
    # 避免除零
    state_std = np.where(state_std == 0, 1.0, state_std)
    
    # 标准化: (x - mean) / std
    rdt_state = (rdt_state - state_mean) / state_std
    
    return torch.from_numpy(rdt_state).float()
```

**关键要点：**

1. **状态映射**：LIBERO的17维状态被映射到RDT的128维空间的特定位置（right arm的索引）
2. **旋转转换**：四元数 → 6D旋转表示
3. **归一化**：使用训练时的均值和标准差进行标准化
4. **其余维度**：RDT的128维中未使用的部分保持为0（归一化后的0）

### 4.2 图像处理: `model.step` 中的图像部分

```python
def step(self, state: torch.Tensor, images: List[Image.Image], text_embed: torch.Tensor):
    """
    执行一步推理
    
    输入:
    - state: Tensor (128,) 归一化的状态
    - images: List[Image.Image] 3个图像 [cam_high, cam_right_wrist, cam_left_wrist]
    - text_embed: Tensor (1, 77, 4096) T5编码的任务描述
    
    输出:
    - trajectory: Tensor (1, 64, 128) 预测的64步动作轨迹
    """
    
    device = "cuda"
    dtype = torch.bfloat16
    
    # 步骤1: 创建背景图像（用于缺失的摄像头）
    background_color = np.array([
        int(x*255) for x in self.vision_encoder.image_processor.image_mean
    ], dtype=np.uint8).reshape(1, 1, 3)
    
    background_image = np.ones((384, 384, 3), dtype=np.uint8) * background_color
    
    # 步骤2: 预处理每个图像
    image_tensor_list = []
    for image in images:
        if image is None:
            # 使用背景图像
            image = Image.fromarray(background_image)
        
        # SigLIP预处理: resize到384x384, 归一化
        image = self.vision_encoder.image_processor.preprocess(
            image, return_tensors='pt'
        )['pixel_values'][0]  # (3, 384, 384)
        
        image_tensor_list.append(image)
    
    image_tensor = torch.stack(image_tensor_list, dim=0)  # (3, 3, 384, 384)
    image_tensor = image_tensor.to(device, dtype=dtype)
    
    # 步骤3: 逐个编码图像（获取patch features）
    image_embeds_list = []
    for i in range(image_tensor.shape[0]):
        single_image = image_tensor[i:i+1]  # (1, 3, 384, 384)
        vision_output = self.vision_encoder.vision_tower(single_image)
        single_embeds = vision_output.last_hidden_state.detach()  # (1, 729, 1152)
        image_embeds_list.append(single_embeds)
    
    # 步骤4: 拼接所有图像特征
    image_embeds = torch.cat(image_embeds_list, dim=1)  # (1, 729*3=2187, 1152)
    
    # 步骤5: 模拟历史帧（训练时使用img_history_size=2）
    img_history_size = self.config["common"]["img_history_size"]  # 2
    if img_history_size > 1:
        # 重复当前帧以模拟历史
        image_embeds = image_embeds.repeat(1, img_history_size, 1)  # (1, 4374, 1152)
    
    # 步骤6: 重塑为正确的形状
    image_embeds = image_embeds.reshape(1, -1, 1152)  # (1, 4374, 1152)
    
    # 步骤7: 处理状态
    states = state.to(device, dtype=dtype).unsqueeze(0).unsqueeze(1)  # (1, 1, 128)
    
    # 步骤8: 创建状态掩码（全1表示所有维度都有效）
    state_elem_mask = torch.ones((1, 1, 128), device=device, dtype=dtype)
    
    # 步骤9: 控制频率
    ctrl_freqs = torch.tensor([20]).to(device)  # LIBERO使用20Hz控制频率
    
    # 步骤10: 文本嵌入
    text_embeds = text_embed.to(device, dtype=dtype)  # (1, 77, 4096)
    
    # 步骤11: RDT推理
    with torch.no_grad():
        trajectory = self.rdt_model.predict_action(
            lang_tokens=text_embeds,                              # (1, 77, 4096)
            lang_attn_mask=torch.ones(text_embeds.shape[:2], 
                                     dtype=torch.bool, 
                                     device=device),              # (1, 77)
            img_tokens=image_embeds,                             # (1, 4374, 1152)
            state_tokens=states,                                  # (1, 1, 128)
            action_mask=state_elem_mask,                          # (1, 1, 128)
            ctrl_freqs=ctrl_freqs                                 # (1,)
        )
    
    return trajectory.to(torch.float32)  # (1, 64, 128)
```

**关键要点：**

1. **图像编码**：使用SigLIP将384x384图像编码为729个patch features
2. **多摄像头处理**：拼接3个摄像头的features
3. **历史模拟**：通过重复当前帧来模拟历史帧（因为评估时没有真实历史）
4. **维度对齐**：确保所有输入维度与训练时一致

### 4.3 动作转换: `convert_rdt_action_to_libero`

这是另一个**最关键**的函数，负责将RDT的输出转换为LIBERO可以执行的动作。

```python
def convert_rdt_action_to_libero(rdt_action: torch.Tensor) -> np.ndarray:
    """
    将RDT动作转换为LIBERO动作格式
    
    输入:
    - rdt_action: Tensor (1, 64, 128) RDT输出的动作轨迹
    
    输出:
    - libero_action: numpy array (7,) LIBERO动作
                    [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, gripper]
    """
    
    # 步骤1: 提取第一个时间步的动作
    action_128d = rdt_action[0, 0, :].cpu().numpy()  # (128,)
    
    # 步骤2: 加载数据集统计信息
    with open('configs/dataset_stat.json', 'r') as f:
        stats = json.load(f)
    libero_stats = stats['libero_90']
    
    # 步骤3: 提取位置 (3D) 并反归一化
    pos_x_idx = STATE_VEC_IDX_MAPPING["right_eef_pos_x"]  # 30
    pos_y_idx = STATE_VEC_IDX_MAPPING["right_eef_pos_y"]  # 31
    pos_z_idx = STATE_VEC_IDX_MAPPING["right_eef_pos_z"]  # 32
    
    # 反归一化: x_denorm = x_norm * std + mean
    pos_x = action_128d[pos_x_idx] * libero_stats["state_std"][pos_x_idx] + \
            libero_stats["state_mean"][pos_x_idx]
    pos_y = action_128d[pos_y_idx] * libero_stats["state_std"][pos_y_idx] + \
            libero_stats["state_mean"][pos_y_idx]
    pos_z = action_128d[pos_z_idx] * libero_stats["state_std"][pos_z_idx] + \
            libero_stats["state_mean"][pos_z_idx]
    
    # 步骤4: 提取6D旋转并反归一化
    ori_indices = [
        STATE_VEC_IDX_MAPPING["right_eef_angle_0"],  # 33
        STATE_VEC_IDX_MAPPING["right_eef_angle_1"],  # 34
        STATE_VEC_IDX_MAPPING["right_eef_angle_2"],  # 35
        STATE_VEC_IDX_MAPPING["right_eef_angle_3"],  # 36
        STATE_VEC_IDX_MAPPING["right_eef_angle_4"],  # 37
        STATE_VEC_IDX_MAPPING["right_eef_angle_5"]   # 38
    ]
    
    ori_6d = np.array([
        action_128d[idx] * libero_stats["state_std"][idx] + 
        libero_stats["state_mean"][idx]
        for idx in ori_indices
    ])  # (6,)
    
    # 步骤5: 6D旋转转欧拉角
    from utils.rotation_utils import convert_6d_rotation_to_euler
    ori_3d = convert_6d_rotation_to_euler(ori_6d)  # (3,) 弧度
    
    # 步骤6: 提取gripper状态并反归一化
    gripper_idx = STATE_VEC_IDX_MAPPING["right_gripper_open"]  # 10
    gripper_normalized = action_128d[gripper_idx] * \
                        libero_stats["state_std"][gripper_idx] + \
                        libero_stats["state_mean"][gripper_idx]
    
    # gripper从[0,1]映射到[-1,1]
    gripper = gripper_normalized * 2.0 - 1.0
    
    # 步骤7: 构建LIBERO动作向量
    libero_action = np.array([
        pos_x, pos_y, pos_z, 
        ori_3d[0], ori_3d[1], ori_3d[2], 
        gripper
    ])  # (7,)
    
    # 步骤8: 缩放到LIBERO的[-1, 1]控制范围
    # LIBERO使用OSC_POSE控制器:
    # - 位置增量范围: [-0.05, 0.05]米 对应 [-1, 1]
    # - 旋转增量范围: [-0.5, 0.5]弧度 对应 [-1, 1]
    
    # 位置: 米 -> [-1, 1]
    # 注意: 这里对某些轴进行了反转以修正坐标系差异
    libero_action[0] = np.clip(-libero_action[0] / 0.05, -1.0, 1.0)  # 反转X轴
    libero_action[1] = np.clip(libero_action[1] / 0.05, -1.0, 1.0)   # Y轴保持
    libero_action[2] = np.clip(-libero_action[2] / 0.05, -1.0, 1.0)  # 反转Z轴
    
    # 旋转: 弧度 -> [-1, 1]
    libero_action[3:6] = np.clip(libero_action[3:6] / 0.5, -1.0, 1.0)
    
    # gripper已经在[-1, 1]范围内
    
    return libero_action
```

**关键要点：**

1. **索引提取**：从128维向量的特定位置提取LIBERO需要的7个维度
2. **反归一化**：使用训练时的统计信息恢复原始尺度
3. **旋转转换**：6D旋转 → 欧拉角
4. **尺度映射**：将物理单位（米、弧度）映射到LIBERO的控制范围[-1, 1]
5. **坐标系修正**：通过反转某些轴来修正RDT和LIBERO之间的坐标系差异

### 4.4 评估循环: `evaluate_rdt_on_libero`

```python
def evaluate_rdt_on_libero(model, benchmark_name="libero_90", 
                          num_tasks=5, max_steps=100,
                          record_video=False, video_output_dir="videos"):
    """
    在LIBERO基准上评估RDT模型
    
    流程:
    1. 设置LIBERO环境
    2. 遍历每个任务
    3. 对每个任务运行推理循环
    4. 记录成功率和统计信息
    """
    
    # 1. 设置LIBERO路径
    libero.set_libero_default_path("../../LIBERO/libero/libero")
    
    # 2. 获取基准
    benchmark_dict = benchmark.get_benchmark_dict()
    libero_benchmark = benchmark_dict[benchmark_name]()
    
    results = {
        "total_tasks": min(num_tasks, len(libero_benchmark.get_task_names())),
        "successful_tasks": 0,
        "total_steps": 0,
        "task_results": []
    }
    
    # 3. 遍历每个任务
    for task_idx in range(results["total_tasks"]):
        task_name = libero_benchmark.get_task_names()[task_idx]
        task = libero_benchmark.get_task(task_idx)
        
        # 4. 创建环境
        bddl_files_path = libero.get_libero_path("bddl_files")
        bddl_file_path = os.path.join(
            bddl_files_path, 
            task.problem_folder, 
            task.bddl_file
        )
        
        env_args = {
            "bddl_file_name": bddl_file_path,
            "camera_heights": 128,
            "camera_widths": 128
        }
        
        env = OffScreenRenderEnv(**env_args)
        
        # 5. 重置环境并设置初始状态
        obs = env.reset()
        init_states = libero_benchmark.get_task_init_states(task_idx)
        if len(init_states) > 0:
            env.set_init_state(init_states[0])
        
        # 6. 编码任务描述
        text_embed = model.encode_instruction(task_name)  # (1, 77, 4096)
        
        # 7. 重置模型
        model.reset()
        
        # 8. 运行推理循环
        task_success = False
        task_steps = 0
        
        for step in range(max_steps):
            # 8.1 准备图像输入
            img = obs["agentview_image"]  # (128, 128, 3)
            images = [
                Image.fromarray(img),  # 主摄像头
                None,                   # 右手腕摄像头（LIBERO不提供）
                None                    # 左手腕摄像头（LIBERO不提供）
            ]
            
            # 8.2 转换状态
            rdt_state = convert_libero_state_to_rdt(obs)  # (128,)
            
            # 8.3 模型推理
            rdt_actions = model.step(rdt_state, images, text_embed)  # (1, 64, 128)
            
            # 8.4 转换动作
            libero_action = convert_rdt_action_to_libero(rdt_actions)  # (7,)
            
            # 8.5 执行动作
            obs, reward, done, info = env.step(libero_action)
            task_steps += 1
            
            # 8.6 检查是否完成
            if done:
                task_success = info.get("success", False)
                break
        
        # 9. 记录结果
        task_result = {
            "task_name": task_name,
            "success": task_success,
            "steps": task_steps,
            "reward": reward
        }
        results["task_results"].append(task_result)
        
        if task_success:
            results["successful_tasks"] += 1
        
        results["total_steps"] += task_steps
        
        env.close()
    
    # 10. 计算成功率
    results["success_rate"] = results["successful_tasks"] / results["total_tasks"]
    results["avg_steps"] = results["total_steps"] / results["total_tasks"]
    
    return results
```

---

## 5. 数据流转过程

### 5.1 完整数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                      LIBERO Environment                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        obs = {
            'agentview_image': (128, 128, 3),
            'robot0_joint_pos': (7,),
            'robot0_gripper_qpos': (2,),
            'robot0_eef_pos': (3,),
            'robot0_eef_quat': (4,),
            ...
        }
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              convert_libero_state_to_rdt()                       │
│                                                                   │
│  1. 提取: joint(7) + gripper(1) + eef_pos(3) + eef_quat(4)     │
│  2. 转换: quat -> 6D rotation                                   │
│  3. 映射: 17D -> 128D (填充到right arm索引)                     │
│  4. 归一化: (x - mean) / std                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        rdt_state: Tensor (128,) 归一化状态
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    model.step()                                  │
│                                                                   │
│  图像分支:                                                        │
│    image (128,128,3) -> resize(384,384)                         │
│                      -> SigLIP                                   │
│                      -> (729, 1152) patches                      │
│                      -> repeat for 3 cameras                     │
│                      -> repeat for history                       │
│                      -> (4374, 1152)                             │
│                                                                   │
│  文本分支:                                                        │
│    task_name -> T5 XXL -> (77, 4096)                           │
│                                                                   │
│  状态分支:                                                        │
│    rdt_state (128,) -> (1, 1, 128)                              │
│                                                                   │
│  RDT Diffusion Transformer:                                      │
│    [img_tokens, lang_tokens, state_tokens]                      │
│    -> Multi-head Attention                                       │
│    -> Diffusion Denoising                                        │
│    -> Action Prediction                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        rdt_actions: Tensor (1, 64, 128) 
        预测的64步轨迹，每步128维
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              convert_rdt_action_to_libero()                      │
│                                                                   │
│  1. 提取第一步: action[0, 0, :]                                  │
│  2. 从128维提取: pos(3) + ori_6d(6) + gripper(1)                │
│  3. 反归一化: x = x_norm * std + mean                            │
│  4. 转换旋转: 6D rotation -> euler angles                       │
│  5. 尺度映射:                                                     │
│     - pos (米) -> [-1,1] (控制信号)                             │
│     - ori (弧度) -> [-1,1] (控制信号)                           │
│     - gripper [0,1] -> [-1,1]                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        libero_action: numpy (7,)
        [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, gripper]
        范围: [-1, 1]
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LIBERO Environment                            │
│                    env.step(libero_action)                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        obs (next), reward, done, info
                              ↓
                        重复循环
```

### 5.2 维度变化追踪

| 阶段 | 数据 | 形状 | 数据类型 | 范围/单位 |
|-----|------|------|---------|----------|
| LIBERO obs | joint_pos | (7,) | numpy | 弧度 |
| | gripper_qpos | (2,) | numpy | 米 |
| | eef_pos | (3,) | numpy | 米 |
| | eef_quat | (4,) | numpy | 单位四元数 |
| | image | (128,128,3) | numpy | [0, 255] |
| 中间表示 | libero_state | (17,) | numpy | 混合单位 |
| | eef_ori_6d | (6,) | numpy | 6D旋转 |
| RDT输入 | rdt_state | (128,) | torch | 标准化 |
| | image_embeds | (1,4374,1152) | torch | bfloat16 |
| | text_embeds | (1,77,4096) | torch | bfloat16 |
| | state_tokens | (1,1,128) | torch | bfloat16 |
| RDT输出 | trajectory | (1,64,128) | torch | 标准化 |
| 提取动作 | action_128d | (128,) | numpy | 标准化 |
| 反归一化 | pos | (3,) | numpy | 米 |
| | ori_6d | (6,) | numpy | 6D旋转 |
| | ori_euler | (3,) | numpy | 弧度 |
| | gripper | (1,) | numpy | [0, 1] |
| LIBERO action | libero_action | (7,) | numpy | [-1, 1] |

---

## 6. 关键修复和优化

### 6.1 四元数到6D旋转的修复

**问题**：原始代码使用了不正确的6D旋转表示，导致旋转信息丢失。

**修复**：在 `utils/rotation_utils.py` 中实现了正确的6D旋转转换：

```python
def convert_quaternion_to_6d_rotation(quat):
    """
    将四元数转换为6D旋转表示
    
    6D旋转使用旋转矩阵的前两列（两个正交向量）
    这比欧拉角或四元数更适合神经网络学习
    
    参考: Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"
    """
    # quat: [w, x, y, z] 或 [x, y, z, w]
    # 需要检查数据格式
    
    # 转换为旋转矩阵
    R = quaternion_to_rotation_matrix(quat)  # (3, 3)
    
    # 提取前两列作为6D表示
    r6d = np.concatenate([R[:, 0], R[:, 1]])  # (6,)
    
    return r6d

def convert_6d_rotation_to_euler(r6d):
    """
    将6D旋转表示转换为欧拉角
    """
    # 重建旋转矩阵的前两列
    col1 = r6d[:3]  # 第一列
    col2 = r6d[3:]  # 第二列
    
    # 正交化第二列
    col2 = col2 - np.dot(col1, col2) * col1
    col2 = col2 / (np.linalg.norm(col2) + 1e-8)
    
    # 计算第三列（叉积）
    col3 = np.cross(col1, col2)
    
    # 重建旋转矩阵
    R = np.stack([col1, col2, col3], axis=1)  # (3, 3)
    
    # 转换为欧拉角
    euler = rotation_matrix_to_euler(R)
    
    return euler
```

### 6.2 坐标系修正

**问题**：RDT训练时使用的坐标系与LIBERO可能不同。

**修复**：在 `convert_rdt_action_to_libero` 中对某些轴进行反转：

```python
# 反转X轴和Z轴以修正坐标系差异
libero_action[0] = np.clip(-libero_action[0] / 0.05, -1.0, 1.0)  # 反转X
libero_action[1] = np.clip(libero_action[1] / 0.05, -1.0, 1.0)   # Y不变
libero_action[2] = np.clip(-libero_action[2] / 0.05, -1.0, 1.0)  # 反转Z
```

### 6.3 图像历史模拟

**问题**：RDT训练时使用历史帧（img_history_size=2），但评估时没有历史。

**修复**：通过重复当前帧来模拟历史：

```python
# 重复当前帧以模拟历史
if img_history_size > 1:
    image_embeds = image_embeds.repeat(1, img_history_size, 1)
```

### 6.4 数据统计加载

**问题**：归一化和反归一化需要使用训练时的统计信息。

**修复**：从 `configs/dataset_stat.json` 加载统计信息：

```python
with open('configs/dataset_stat.json', 'r') as f:
    stats = json.load(f)
libero_stats = stats['libero_90']

state_mean = np.array(libero_stats["state_mean"])
state_std = np.array(libero_stats["state_std"])
```

---

## 7. 总结

### 7.1 评估流程总结

1. **初始化阶段**
   - 加载RDT模型（T5文本编码器 + SigLIP视觉编码器 + Diffusion Transformer）
   - 加载数据统计信息（用于归一化/反归一化）
   - 创建LIBERO环境

2. **任务执行阶段**
   - 编码任务描述（T5）→ (1, 77, 4096)
   - 对每一步：
     - 观测转换：LIBERO obs → RDT state (128D)
     - 图像编码：RGB → SigLIP features → (4374, 1152)
     - 模型推理：[text, image, state] → RDT → trajectory (1, 64, 128)
     - 动作转换：RDT action (128D) → LIBERO action (7D)
     - 执行动作并获取下一个观测

3. **评估阶段**
   - 记录每个任务的成功/失败
   - 计算成功率和平均步数
   - 保存结果和视频

### 7.2 关键差异点

| 方面 | LIBERO原生评估 | RDT评估 |
|-----|---------------|--------|
| 模型类型 | BC Transformer | Diffusion Transformer |
| 输入格式 | 字典{'obs', 'task_emb'} | 分离的(state, image, text) |
| 状态维度 | 任务特定 | 统一128维 |
| 旋转表示 | 四元数/欧拉角 | 6D旋转 |
| 图像编码 | ResNet/ViT | SigLIP |
| 文本编码 | 简单嵌入 | T5 XXL |
| 动作输出 | 直接7维 | 128维需提取 |
| 数据处理 | robomimic工具 | 自定义转换 |

### 7.3 未来改进方向

1. **在线历史维护**：维护真实的图像历史而不是重复当前帧
2. **动作平滑**：使用action chunk的多步预测而不仅仅是第一步
3. **坐标系自适应**：自动学习坐标系转换而不是硬编码
4. **多任务泛化**：测试在不同LIBERO基准（spatial, goal, long horizon）上的表现
5. **微调优化**：使用LIBERO任务进行微调以提高性能

---

## 附录

### A. 状态向量索引映射

```python
STATE_VEC_IDX_MAPPING = {
    # Left arm (indices 0-2)
    "left_arm_joint_0_pos": 0,
    ...
    
    # Right arm (indices 3-9)
    "right_arm_joint_0_pos": 3,
    "right_arm_joint_1_pos": 4,
    "right_arm_joint_2_pos": 5,
    "right_arm_joint_3_pos": 6,
    "right_arm_joint_4_pos": 7,
    "right_arm_joint_5_pos": 8,
    "right_arm_joint_6_pos": 9,
    
    # Grippers (indices 10-11)
    "right_gripper_open": 10,
    "left_gripper_open": 11,
    
    # End effector positions (indices 30-35)
    "right_eef_pos_x": 30,
    "right_eef_pos_y": 31,
    "right_eef_pos_z": 32,
    "left_eef_pos_x": 33,
    "left_eef_pos_y": 34,
    "left_eef_pos_z": 35,
    
    # End effector 6D rotations (indices 36-47)
    "right_eef_angle_0": 36,
    "right_eef_angle_1": 37,
    "right_eef_angle_2": 38,
    "right_eef_angle_3": 39,
    "right_eef_angle_4": 40,
    "right_eef_angle_5": 41,
    ...
}
```

### B. LIBERO环境接口

```python
# 环境创建
env = OffScreenRenderEnv(
    bddl_file_name=bddl_file_path,
    camera_heights=128,
    camera_widths=128,
    control_freq=20,  # 20Hz控制频率
)

# 环境重置
obs = env.reset()  # 返回初始观测

# 设置初始状态
obs = env.set_init_state(init_state)  # 从保存的状态开始

# 执行动作
obs, reward, done, info = env.step(action)
# action: (7,) numpy array, [-1, 1]
# obs: 字典，包含图像和状态
# reward: 标量
# done: bool
# info: 字典，包含'success'键
```

### C. RDT模型接口

```python
# 模型创建
rdt_model = RDTRunner(
    action_dim=128,
    pred_horizon=64,
    config=config,
    ...
)

# 推理
trajectory = rdt_model.predict_action(
    lang_tokens=text_embeds,        # (B, L, D_text)
    lang_attn_mask=lang_mask,       # (B, L)
    img_tokens=image_embeds,        # (B, N_img, D_img)
    state_tokens=state_tokens,      # (B, 1, D_state)
    action_mask=action_mask,        # (B, 1, D_action)
    ctrl_freqs=ctrl_freqs           # (B,)
)
# trajectory: (B, T, D_action)
```

---

**文档版本**: 1.0  
**最后更新**: 2025-10-16  
**作者**: RDT Evaluation Team

