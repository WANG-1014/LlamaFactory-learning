# Lab W15: 风格条件化 LoRA 微调 Qwen2.5-0.5B-Instruct — 计划
使用 ModelScope 下载需先执行: export USE_MODELSCOPE_HUB=1  
```$env:USE_MODELSCOPE_HUB = "1" ```  
```llamafactory-cli train configs/qwen2.5_lora_sft.yaml```
```lmf chat configs/qwen2.5_lora_infer.yaml```
```lmf chat configs/qwen2.5_base_infer.yaml```

## 一、选择语言风格

- **数量要求**：选择 2–3 种风格
- **关键原则**：风格间需明显可区分，确保微调效果可观察

**风格示例（非穷举）**：

| 类别 | 示例 |
|------|------|
| 古典/文学 | 文言文、莎士比亚英语 |
| 作者风格 | 鲁迅、海明威 |
| 虚拟角色 | 二次元傲娇角色、海盗口吻 |
| 地域方言 | 东北话、粤语风格 |
| 领域语气 | 法律合同、学术论文、新闻主播 |

---

## 二、构建数据集

### 2.1 数据来源

- 修改并扩展 LLaMA-Factory 内置的 `data/identity.json`

### 2.2 格式要求

- **`alpaca`** 或 **`sharegpt`** 格式，兼容 LLaMA-Factory
- 在 `dataset_info.json` 中注册新数据集
- 这里我使用 **`alpaca`** 格式

### 2.3 核心规则

- 每种风格创建**足够的** `(instruction, output)` 样本对
- **显式标注目标风格**：在 instruction 中使用系统提示词或风格标签
- 确保模型学会**条件化生成**（按指定风格输出），而非坍缩为单一风格
- 记录每风格样本数量

### 2.4 数据生成方式

| 方式 | 说明 |
|------|------|
| 手写 | 自行编写，质量最高 |
| LLM 蒸馏 | 用更大的模型生成训练数据 |
| 网络爬取 | 从公开语料中收集 |

---

## 三、LoRA 微调

### 3.1 超参数配置

**LoRA 超参**：

| 参数 | 说明 |
|------|------|
| `lora_rank` (r) | LoRA 低秩矩阵的秩 |
| `lora_alpha` | 缩放系数 |
| `lora_target` | 目标模块（如 `all`） |
| `lora_dropout` | Dropout 比例 |

**训练超参**：

| 参数 | 说明 |
|------|------|
| `learning_rate` | 学习率 |
| `per_device_train_batch_size` | 每设备 batch size |
| `gradient_accumulation_steps` | 梯度累积步数 |
| `num_train_epochs` | 训练轮数 |
| `warmup_ratio` | 预热比例 |
| `lr_scheduler_type` | 调度器类型（如 `cosine`） |

### 3.2 运行训练

```bash
llamafactory-cli train configs/qwen2.5_lora_sft.yaml
```

### 3.3 训练过程记录

- 损失曲线截图
- 收敛性分析
- 过拟合判断

> **建议**：先在 50–100 样本上验证 pipeline 跑通，再扩大规模。

---

## 四、推理与对比

### 4.1 推理测试

- 加载 LoRA 适配器进行交互测试：`lmf chat`
- 准备测试 prompts，需：
  - 覆盖所有目标风格
  - 与训练数据中的 prompt 不同（检验泛化能力）
- 这里运行：
```bash
lmf chat configs/qwen2.5_lora_infer.yaml
```

### 4.2 对比实验

| 模型 | 说明 |
|------|------|
| 基座模型 | `Qwen2.5-0.5B-Instruct`（不加 LoRA） |
| 微调模型 | 基座模型 + LoRA 适配器 |

### 4.3 结果分析

- 识别**成功案例**并解释原因
- 识别**失败案例**并分析原因
- 制作**并排对比截图**（同一 Prompt → Base vs LoRA）

---

## 五、交付物

### 5.1 实验报告（PDF）

| 章节 | 内容要求 |
|------|---------|
| **第一节：数据构建与处理** | 所选风格、`identity.json` 修改说明、每风格样本数、样本示例、数据设计理由 |
| **第二节：训练超参与曲线分析** | 超参完整列表、损失曲线截图、收敛性分析、过拟合分析、调参决策说明 |
| **第三节：效果对比与案例研究** | Base vs LoRA 并排对比、典型成功案例、典型失败案例、总结与改进方向 |

> 好的报告不是堆砌截图，而是**分析**为什么某些做法有效或失败。

### 5.2 演示图片

- **对话截图**：每种风格的模型真实交互
- **前后对比图**：同一 Prompt 下 Base 模型 vs LoRA 模型输出

---

## 六、注意事项

### 6.1 目录组织

```
project/
├── data/       # 自定义数据集
├── configs/    # 训练 YAML 配置文件
├── outputs/    # LoRA 权重输出
└── report/     # 实验报告
```

### 6.2 防过拟合

- 0.5B 模型参数量小 + 自定义数据集规模有限 → 极易过拟合
- 密切观察损失曲线
- 通过调整 `num_train_epochs` 和 `lora_rank` 控制模型容量
- 先用小数据集（50–100 样本/风格）验证 pipeline

### 6.3 思维链处理

- Qwen 系列支持思考模式（`<think>` 标签）
- 训练和推理时注意 `enable_thinking` 参数保持一致
- 若数据不含思维链，框架会自动添加空思维链
