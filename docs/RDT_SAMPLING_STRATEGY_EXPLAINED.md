# RDT官方代码的数据采样策略详解

## 📚 基于代码分析的官方采样策略

### 核心发现：**全步骤采样策略（Full-Step Sampling）**

RDT官方代码**不是随机采样**，而是采用**全步骤枚举**的策略！

---

## 🔍 代码证据分析

### 1. Episode Transform阶段（data/episode_transform.py）

```python
# 第224-299行：flatten_episode函数
def flatten_episode(episode: dict) -> tf.data.Dataset:
    """
    Flatten the episode to a list of steps.
    """
    episode_dict = episode['episode_dict']
    dataset_name = episode['dataset_name']
    
    json_content, states, masks = generate_json_state(
        episode_dict, dataset_name
    )

    # 为每一步创建训练样本
    step_data = []
    for i in range(tf.shape(states)[0]):  # 遍历所有步骤！
        step_data.append({
            'step_id': episode['step_id'][i],
            'json_content': json_content,
            'state_chunk': past_states[i],      # 过去64步的state
            'action_chunk': future_states[i],    # 未来64步的action
            # ... 其他数据
        })
    
    return tf.data.Dataset.from_tensor_slices(step_data)
```

**关键点**：
- `for i in range(tf.shape(states)[0])` → **遍历episode的所有步骤**
- 每个step都创建一个训练样本
- 没有随机采样，而是全部转换

---

### 2. configs/base.yaml中的配置

```yaml
dataset:
  # 过滤掉长度小于32的episode
  epsd_len_thresh_low: 32
  
  # 对于超过2048步的episode，随机采样2048步
  epsd_len_thresh_high: 2048
  # to better balance the training datasets
```

**配置说明**：
- `epsd_len_thresh_low: 32` → 过滤太短的episode
- `epsd_len_thresh_high: 2048` → 限制太长episode的步数

**但是**：在现有代码中**没有找到实际使用这两个参数的地方**！

这说明：
1. 这些配置可能是为了预训练的大规模数据集设计的
2. 在我们能看到的代码中（data/episode_transform.py），**所有steps都被处理**

---

### 3. Producer-Consumer模式（data/producer.py + train/dataset.py）

#### Producer（生产者）

```python
# data/producer.py 第185-199行
for episode_steps in vla_dataset:
    for step in episode_steps:  # 遍历episode中的每个step
        if fill_up and fill_chunk_idx < chunk_end_idx:
            # 保存这个step到buffer
            save_sample(step, chunk_dir, fill_chunk_item_idx)
            # ...
```

**关键点**：
- Producer遍历episode的每一个step
- 每个step都会被保存到buffer

#### Consumer（消费者）

```python
# train/dataset.py 第204-246行
def _safe_load(self, index):
    read_chunk_idx = index // self.chunk_size
    
    # 从buffer中读取一个chunk
    read_chunk_dir = os.path.join(self.buffer_dir, f"chunk_{read_chunk_idx}")
    read_chunk_item_indices = get_clean_item(read_chunk_dir)
    
    # 根据index选择chunk内的item
    random_item_index = index % len(read_chunk_item_indices)
    read_chunk_item_index = read_chunk_item_indices[random_item_index]
    
    # 加载sample
    content, meta = self._load_data_from_chunk(read_chunk_dir, read_chunk_item_index)
    return (content, *meta)
```

**关键点**：
- Consumer根据DataLoader的index来读取steps
- `index % len(read_chunk_item_indices)` → 伪随机访问
- 但本质上buffer里存的是**所有episode的所有steps**

---

## 📊 RDT官方采样策略总结

### 策略特点

| 特性 | RDT官方策略 | 我们的LIBERO实现 |
|------|-----------|----------------|
| **采样方式** | 全步骤枚举 | 随机采样单步 |
| **数据覆盖** | 100%的steps | 每次只用1步 |
| **Episode使用** | 每个episode的每步都用 | 每次随机选1个episode |
| **训练样本数** | episode_len × num_episodes | num_epochs × batch_size |
| **数据多样性** | 自然顺序+DataLoader shuffle | 完全随机 |

---

### 详细对比

#### RDT官方策略（全步骤枚举）

```python
# 预处理阶段（离线）
for episode in dataset:
    for step_idx in range(len(episode)):
        # 每个step都创建一个训练样本
        sample = {
            'state': episode[step_idx],
            'actions': episode[step_idx:step_idx+64],  # 未来64步
            'images': episode[step_idx-2:step_idx],    # 历史2帧
            # ...
        }
        save_to_buffer(sample)

# 训练阶段（在线）
# DataLoader会shuffle这些samples
# 每个epoch会遍历所有samples
```

**优点**：
1. ✅ **数据利用率100%** - 每个step都被用到
2. ✅ **训练稳定** - 大量样本，更好的梯度估计
3. ✅ **覆盖完整** - 学习到轨迹的所有阶段
4. ✅ **适合大规模预训练** - 1M+ episodes → 数百万steps

**缺点**：
1. ⚠️ **需要大量存储** - buffer存储所有steps
2. ⚠️ **预处理时间长** - 需要离线处理所有数据
3. ⚠️ **内存占用大** - buffer可能需要400GB+

---

#### 我们的LIBERO实现（随机采样）

```python
# data/hdf5_libero_dataset.py
def get_item(self, index: int=None, state_only=False):
    # 1. 随机选择一个episode
    file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
    
    # 2. 随机选择episode中的一个起点
    episodes = list(f['data'].keys())
    episode_key = np.random.choice(episodes)
    episode_data = f['data'][episode_key]
    
    # 3. 随机选择起始step
    num_steps = len(actions)
    step_id = np.random.randint(0, num_steps - self.CHUNK_SIZE)
    
    # 4. 返回这个step的数据
    return sample
```

**优点**：
1. ✅ **不需要预处理** - 直接从HDF5读取
2. ✅ **内存占用小** - 只加载需要的数据
3. ✅ **灵活** - 容易修改采样策略

**缺点**：
1. ❌ **数据利用率低** - 每个epoch只用很少的steps
2. ❌ **可能采样不均匀** - 某些steps可能很少被采到
3. ❌ **关键时刻采样不足** - 任务转换点（如第120步）采样概率低

---

## 🎯 为什么RDT用全步骤枚举？

### 理由1: 大规模预训练需要

```
RDT-1B预训练数据：
- 46个数据集
- 1M+ episodes
- 平均每个episode: 100-500步
- 总steps: 100M-500M steps

采样策略：
- 全步骤枚举 → 100M-500M训练样本 ✅
- 随机采样 → 取决于训练epoch，可能只用10M samples ❌
```

**大规模预训练需要充分利用所有数据！**

### 理由2: 保证数据覆盖均匀

```python
# 全步骤枚举
for episode in dataset:
    for step in episode:
        # 每个step都会被用到
        # 包括：
        # - 任务开始（第0-10步）
        # - 任务中期（第50-100步）
        # - 任务转换（第100-120步）  ← 关键！
        # - 任务结束（第200-213步）
        save_sample(step)

# 随机采样
for epoch in range(num_epochs):
    step = random_sample()
    # 问题：
    # - 任务转换点（第100-120步）只占总步数的10%
    # - 被采样到的概率只有10%
    # - 如果训练不够久，可能学不到转换！
```

### 理由3: 训练稳定性

```python
# 有效batch size = 256
# 如果使用随机采样：
#   - 每个batch的256个samples可能来自同一个episode的不同位置
#   - 相邻steps的数据分布非常相似
#   - 梯度估计不够diverse

# 如果使用全步骤枚举 + DataLoader shuffle：
#   - 每个batch的256个samples来自不同episodes
#   - 数据分布更diverse
#   - 梯度估计更准确
```

---

## 🔬 configs/base.yaml中的epsd_len_thresh配置

### 配置的意图

```yaml
# 过滤太短的episode（信息量不足）
epsd_len_thresh_low: 32

# 对于太长的episode，随机采样固定数量的steps
epsd_len_thresh_high: 2048
# to better balance the training datasets
```

### 为什么需要thresh_high？

```python
# 问题：某些数据集的episode特别长
# 例如：
# - aloha dataset: 平均200步/episode
# - bridge dataset: 平均100步/episode  
# - robot_play: 平均5000步/episode ⚠️ 太长！

# 如果不限制：
for episode in robot_play:
    for step in episode:  # 5000 steps
        save(step)
# → robot_play贡献5000个samples
# → aloha只贡献200个samples
# → 训练会严重偏向robot_play！❌

# 如果限制到2048：
for episode in robot_play:
    sampled_steps = random.sample(episode, min(len(episode), 2048))
    for step in sampled_steps:  # 最多2048 steps
        save(step)
# → 每个episode最多贡献2048个samples
# → 数据集之间更平衡 ✅
```

### 但是代码中没有实现？

**可能的原因**：
1. **代码不完整** - 这个功能可能在其他文件中（我们没看到的部分）
2. **仅用于预训练** - 微调时不需要这个功能
3. **配置预留** - 计划实现但还没实现

**在我们的LIBERO微调中**：
- Episode长度都比较统一（~213步）
- 不需要thresh_high的限制
- 使用全步骤枚举就好

---

## 💡 对我们LIBERO训练的启示

### 当前问题

```python
# 我们的实现
def get_item(self):
    # 每次随机选1个episode的1个step
    return random_sample_one_step()

# 训练配置
batch_size = 4
gradient_accumulation = 1
GPUs = 8
effective_batch_size = 32

# 问题：
# - 50个episodes × 213步 = 10,650个可能的samples
# - 每个epoch只采样: 32 × num_batches
# - 如果num_batches=100，每个epoch只用3200个samples
# - 数据利用率：3200/10650 = 30% ❌
```

### RDT官方策略启示

```python
# 如果改用全步骤枚举：
# 预处理：把50个episodes的所有10,650步都存起来
# 训练：每个epoch遍历所有10,650个samples

# 好处：
# 1. 数据利用率100% ✅
# 2. 每个关键时刻都被学到 ✅
# 3. 训练更稳定（更多样本） ✅

# 配合大batch size（256）：
# - 梯度估计更准确
# - 收敛更快更稳定
# - 最终性能更好
```

---

## 🔧 改进建议

### 选项1: 实现全步骤枚举（推荐）⭐⭐⭐⭐⭐

```python
# 修改hdf5_libero_dataset.py
class HDF5LIBERODataset:
    def __init__(self, ...):
        # 预处理：展平所有episodes为steps
        self.all_samples = []
        for hdf5_file in self.file_paths:
            with h5py.File(hdf5_file, 'r') as f:
                for episode_key in f['data'].keys():
                    episode = f['data'][episode_key]
                    num_steps = len(episode['actions'])
                    
                    # 每个step都创建一个sample
                    for step_id in range(num_steps - CHUNK_SIZE):
                        sample = {
                            'file_path': hdf5_file,
                            'episode_key': episode_key,
                            'step_id': step_id
                        }
                        self.all_samples.append(sample)
        
        print(f"Total samples: {len(self.all_samples)}")
        # 对于50 demos: ~10,000 samples
    
    def __len__(self):
        return len(self.all_samples)
    
    def __getitem__(self, index):
        sample_info = self.all_samples[index]
        # 根据sample_info加载数据
        # ...
```

**优点**：
- 数据利用率100%
- 与RDT官方策略一致
- 训练更稳定

**缺点**：
- 需要重构代码
- 预处理时间稍长（但只需要一次）

---

### 选项2: 增加关键时刻采样权重 ⭐⭐⭐

```python
# 保持随机采样，但增加关键区域的权重
def sample_step_with_importance(num_steps, chunk_size):
    # 定义关键区域（任务转换点）
    transition_start = 100
    transition_end = 140
    
    # 创建采样权重
    weights = np.ones(num_steps - chunk_size)
    weights[transition_start:transition_end] *= 3.0  # 3倍权重
    weights = weights / weights.sum()
    
    # 加权随机采样
    step_id = np.random.choice(len(weights), p=weights)
    return step_id
```

**优点**：
- 简单，不需要大改代码
- 关键时刻学习更充分

**缺点**：
- 仍然不能保证100%覆盖
- 需要手动定义关键区域

---

### 选项3: 保持当前+增大有效batch size ⭐⭐⭐⭐

```python
# 不改采样策略，但增大有效batch size
# train_single_task_improved.sh

--train_batch_size=4 \
--gradient_accumulation_steps=8 \  # 从1改为8
# 有效batch size: 4 × 8 × 8 = 256

# 好处：
# - 虽然每次采样的samples少
# - 但大batch size让梯度估计更准确
# - 训练更稳定
# - 可能部分弥补采样不足的问题
```

**优点**：
- 最简单，只改配置
- 立即可用

**缺点**：
- 不能根本解决采样不足问题
- 但能显著改善训练稳定性

---

## 📊 策略对比总结

| 策略 | 数据利用率 | 实现复杂度 | 训练稳定性 | 推荐度 |
|------|----------|-----------|-----------|--------|
| **全步骤枚举** | 100% | 高（需重构） | 最高 | ⭐⭐⭐⭐⭐ |
| **关键时刻加权** | 30-50% | 低 | 中 | ⭐⭐⭐ |
| **大batch size** | 30% | 最低（改配置） | 高 | ⭐⭐⭐⭐ |
| **当前随机采样** | 30% | N/A | 低 | ⭐⭐ |

---

## 🎯 最终建议

### 短期方案（立即可行）

1. **增大batch size** ⭐⭐⭐⭐⭐
   ```bash
   # 使用train_single_task_improved.sh
   --gradient_accumulation_steps=8
   # 有效batch size: 256
   ```

2. **测试exec_horizon=16** ⭐⭐⭐⭐⭐
   ```bash
   bash test_dual_camera.sh
   ```

### 中期方案（如果需要重训）

3. **实现全步骤枚举** ⭐⭐⭐⭐⭐
   - 与RDT官方策略一致
   - 最大化数据利用
   - 预期效果最好

### 长期方案（进一步优化）

4. **添加课程学习**
   - 先训练简单阶段（关抽屉）
   - 再训练复杂阶段（抓碗）
   - 最后训练完整任务

---

## 📝 核心结论

**RDT官方采样策略 = 全步骤枚举（Full-Step Sampling）**

- **不是**从episode中随机采样某一步
- **而是**将每个episode的每一步都作为训练样本
- **优势**：数据利用率100%，训练稳定，覆盖完整
- **适用**：大规模预训练，需要充分利用所有数据

**我们的LIBERO实现 = 随机采样（Random Sampling）**

- **当前**：每次随机选1个episode的1个step
- **问题**：数据利用率低（~30%），关键时刻采样不足
- **改进**：实现全步骤枚举 或 增大batch size

**建议优先级**：
1. 🥇 立即测试：exec_horizon=16
2. 🥈 立即改进：gradient_accumulation_steps=8
3. 🥉 中期重训：实现全步骤枚举

---

**这就是为什么论文强调"We train on 1M+ episodes"——因为他们用了全步骤枚举，实际训练样本数是几百万到上亿个steps！** 🎯

