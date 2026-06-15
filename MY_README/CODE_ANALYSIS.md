# LLaMA-Factory Qwen2.5 LoRA 训练代码分析

本文分析以下命令在当前仓库中的实际执行流程：

```powershell
llamafactory-cli train configs/qwen2.5_lora_sft.yaml
```

分析范围是默认的经典版实现，即 `USE_V1` 未启用时使用的
`src/llamafactory` 代码。`src/llamafactory/v1` 是另一套仍在开发中的插件化实现，
不属于这条命令当前走过的主流程。

## 1. 本次训练要完成什么

配置文件 `configs/qwen2.5_lora_sft.yaml` 描述了一次监督微调任务：

- 基座模型：`Qwen/Qwen2.5-0.5B-Instruct`
- 训练阶段：SFT（Supervised Fine-Tuning）
- 参数高效微调方法：LoRA
- 数据集：4 个身份和语言风格数据集
- 训练精度：BF16 混合精度
- 训练轮数：5 epoch
- 输出：LoRA adapter、训练状态、检查点和损失曲线

训练并不会修改磁盘中的原始 Qwen2.5 权重文件。模型加载到内存后，基座参数被冻结，
训练主要更新插入到线性层中的 LoRA 低秩矩阵。

## 2. 总体调用链

```text
pyproject.toml 注册 llamafactory-cli
    |
    v
llamafactory.cli.main()
    |
    v
llamafactory.launcher.launch()
    |
    +-- 单卡：直接调用 run_exp()
    |
    +-- 多卡：使用 torchrun 重新启动，再进入 run_exp()
    |
    v
llamafactory.train.tuner.run_exp()
    |
    v
read_args() 读取并合并 YAML 和命令行覆盖参数
    |
    v
get_train_args() 转换为 5 组 dataclass 参数并校验
    |
    v
_training_function() 根据 stage 选择训练工作流
    |
    v
train.sft.workflow.run_sft()
    |
    +-- 加载 tokenizer/processor 和 qwen 模板
    +-- 加载、对齐、分词数据集
    +-- 加载 Qwen2.5 基座模型
    +-- 在目标线性层中插入 LoRA
    +-- 创建 DataCollator
    +-- 创建 CustomSeq2SeqTrainer
    |
    v
trainer.train()
    |
    +-- 前向传播
    +-- 计算仅覆盖 assistant 回答的交叉熵
    +-- 反向传播并累积梯度
    +-- 更新 LoRA 参数
    +-- 更新 cosine 学习率
    |
    v
保存 adapter、checkpoint、指标和损失曲线
```

关键代码入口：

| 阶段 | 文件 | 关键函数 |
| --- | --- | --- |
| CLI 注册 | `pyproject.toml` | `[project.scripts]` |
| CLI 入口 | `src/llamafactory/cli.py` | `main()` |
| 子命令和分布式启动 | `src/llamafactory/launcher.py` | `launch()` |
| 参数读取 | `src/llamafactory/hparams/parser.py` | `read_args()`、`get_train_args()` |
| 训练任务分派 | `src/llamafactory/train/tuner.py` | `run_exp()`、`_training_function()` |
| SFT 工作流 | `src/llamafactory/train/sft/workflow.py` | `run_sft()` |
| 数据加载 | `src/llamafactory/data/loader.py` | `get_dataset()` |
| SFT 数据编码 | `src/llamafactory/data/processor/supervised.py` | `SupervisedDatasetProcessor` |
| 模型加载 | `src/llamafactory/model/loader.py` | `load_model()` |
| LoRA 插入 | `src/llamafactory/model/adapter.py` | `_setup_lora_tuning()` |
| Trainer | `src/llamafactory/train/sft/trainer.py` | `CustomSeq2SeqTrainer` |

## 3. 第一步：命令如何进入 Python

`pyproject.toml` 中存在：

```toml
[project.scripts]
llamafactory-cli = "llamafactory.cli:main"
lmf = "llamafactory.cli:main"
```

因此安装项目后，执行 `llamafactory-cli` 等价于调用：

```python
llamafactory.cli.main()
```

`src/llamafactory/cli.py` 会检查 `USE_V1`：

```python
if is_env_enabled("USE_V1"):
    from .v1 import launcher
else:
    from . import launcher

launcher.launch()
```

当前配置没有启用 `USE_V1`，所以进入经典版 `src/llamafactory/launcher.py`。

## 4. 第二步：解析 `train` 子命令

命令启动时，参数近似为：

```python
sys.argv == [
    "llamafactory-cli",
    "train",
    "configs/qwen2.5_lora_sft.yaml",
]
```

`launcher.launch()` 取出子命令：

```python
command = sys.argv.pop(1)
```

取出后：

```python
command == "train"
sys.argv == [
    "llamafactory-cli",
    "configs/qwen2.5_lora_sft.yaml",
]
```

### 4.1 单卡情况

如果只有一张可见 GPU，且没有强制开启 `torchrun`、Ray、KTransformers 等后端，则执行：

```python
from .train.tuner import run_exp
run_exp()
```

### 4.2 多卡情况

如果检测到多张可见 GPU，启动器会构造 `torchrun` 命令，每张 GPU 启动一个训练进程。
相关环境变量包括：

| 变量 | 意义 |
| --- | --- |
| `NPROC_PER_NODE` | 每台机器启动的进程数，通常等于 GPU 数量 |
| `NNODES` | 参与训练的机器数量 |
| `NODE_RANK` | 当前机器编号 |
| `MASTER_ADDR` | 分布式主节点地址 |
| `MASTER_PORT` | 分布式通信端口 |
| `FORCE_TORCHRUN=1` | 即使单卡也强制使用 `torchrun` |

无论单卡还是多卡，最终都会进入同一个 `run_exp()`。

## 5. 第三步：读取 YAML 和构造参数对象

`run_exp()` 首先调用：

```python
args = read_args(args)
```

`read_args()` 发现第一个参数以 `.yaml` 结尾，于是通过 OmegaConf 加载配置：

```python
dict_config = OmegaConf.load(Path(sys.argv[1]).absolute())
override_config = OmegaConf.from_cli(sys.argv[2:])
args = OmegaConf.merge(dict_config, override_config)
```

这意味着 YAML 后面可以附加覆盖参数：

```powershell
llamafactory-cli train configs/qwen2.5_lora_sft.yaml `
  learning_rate=1e-4 `
  num_train_epochs=3
```

后面的参数优先级高于 YAML。

### 5.1 五组参数对象

`get_train_args()` 使用 `HfArgumentParser` 将字典转换成：

```python
model_args, data_args, training_args, finetuning_args, generating_args
```

| 参数对象 | 主要职责 |
| --- | --- |
| `ModelArguments` | 模型路径、tokenizer、量化、注意力实现、推理后端等 |
| `DataArguments` | 数据集、模板、截断、packing、验证集等 |
| `TrainingArguments` | batch size、学习率、epoch、日志、保存、分布式等 |
| `FinetuningArguments` | SFT/DPO 等阶段以及 LoRA/Freeze/Full 等方法 |
| `GeneratingArguments` | 生成时的 temperature、top-p、max length 等 |

### 5.2 参数解析后的派生值

根据当前配置，代码会额外设置：

```python
model_args.compute_dtype = torch.bfloat16
model_args.device_map = {"": current_device}
model_args.model_max_length = 1024
model_args.block_diag_attn = False
data_args.packing = False
training_args.label_names = ["labels"]
```

`transformers.set_seed(training_args.seed)` 还会设置随机种子。未显式指定 `seed` 时，使用
Transformers `TrainingArguments` 的默认值。

## 6. YAML 各参数的意义

原始配置位于 `configs/qwen2.5_lora_sft.yaml`。

### 6.1 模型参数

```yaml
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
trust_remote_code: true
```

#### `model_name_or_path`

基座模型的本地路径或 Hub ID。加载模型时会用于：

```python
AutoTokenizer.from_pretrained(model_name_or_path)
AutoConfig.from_pretrained(model_name_or_path)
AutoModelForCausalLM.from_pretrained(model_name_or_path)
```

如果设置 `USE_MODELSCOPE_HUB=1`，代码会尝试从 ModelScope 下载并把这个参数转换为缓存路径。

#### `trust_remote_code`

允许 Transformers 执行模型仓库中提供的自定义 Python 代码。

优点是可以加载未完全内置到 Transformers 的新模型结构；风险是模型仓库中的代码会在本机执行，
因此只应对可信仓库开启。

### 6.2 微调方法参数

```yaml
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_target: all
lora_dropout: 0.1
```

#### `stage: sft`

选择监督微调工作流。可用阶段包括：

| 值 | 意义 |
| --- | --- |
| `pt` | 增量预训练 |
| `sft` | 指令监督微调 |
| `rm` | 奖励模型训练 |
| `ppo` | PPO 强化学习 |
| `dpo` | DPO/ORPO/SimPO 等偏好优化 |
| `kto` | KTO 偏好优化 |

#### `do_train: true`

让 Trainer 执行训练。如果为 `false`，SFT 工作流仍可根据 `do_eval` 或 `do_predict` 执行评估或预测。

#### `finetuning_type: lora`

选择 LoRA 参数高效微调。其他值包括：

- `full`：全参数训练。
- `freeze`：只训练指定层或模块。
- `oft`：Orthogonal Fine-Tuning。

#### `lora_rank: 8`

LoRA 低秩矩阵的秩 `r`。秩越大，可训练参数越多、表达能力通常越强，同时显存和磁盘占用增加。

#### `lora_alpha: 16`

LoRA 缩放参数。标准 LoRA 的增量通常按 `alpha / rank` 缩放，本配置为：

```text
16 / 8 = 2
```

如果不填写，当前代码默认使用 `2 * lora_rank`。

#### `lora_target: all`

在所有合适的线性层中插入 LoRA，但排除 `lm_head` 等输出层。

在当前 Qwen2.5 模型中，实际得到的目标模块为：

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

也可以明确限制目标模块，例如：

```yaml
lora_target: q_proj,v_proj
```

这会减少参数量，但也降低 LoRA 对模型的可调整范围。

#### `lora_dropout: 0.1`

LoRA 支路输入的 dropout 概率。训练时随机丢弃 10% 的 LoRA 支路输入，用于缓解过拟合；推理时关闭。

### 6.3 数据参数

```yaml
dataset: identity,identity_wenyan,identity_cantonese,identity_news_anchor
template: qwen
cutoff_len: 1024
preprocessing_num_workers: 2
dataloader_num_workers: 0
```

#### `dataset`

这里填写的是 `data/dataset_info.json` 中的数据集逻辑名称，而不是任意文件路径。

对应关系为：

| 逻辑名称 | 文件 | 样本数 |
| --- | --- | ---: |
| `identity` | `data/identity.json` | 91 |
| `identity_wenyan` | `data/identity_wenyan.json` | 50 |
| `identity_cantonese` | `data/identity_Cantonese.json` | 50 |
| `identity_news_anchor` | `data/identity_news_anchor.json` | 50 |
| 合计 |  | 241 |

多个数据集默认使用 `concat` 方式拼接。

#### `template: qwen`

指定 Qwen ChatML 对话模板。单轮样本大致转换为：

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
用户问题<|im_end|>
<|im_start|>assistant
目标回答<|im_end|>
```

模板必须和模型匹配。错误模板会导致特殊 token、角色边界和停止标记不一致。

#### `cutoff_len: 1024`

一条样本分词后的最大 token 数。超过长度时，代码通过 `infer_seqlen()` 在 prompt 和 response 之间分配长度并截断。

它影响：

- 单样本最大显存占用。
- 长回答是否被截断。
- 每步计算量。
- 模型能学到的上下文范围。

#### `preprocessing_num_workers: 2`

数据格式转换和 tokenizer 预处理使用两个 worker。它影响训练前的数据准备速度，不改变梯度计算。

#### `dataloader_num_workers: 0`

DataLoader 在主训练进程中读取 batch，不创建额外 worker。Windows 下这样更稳定，可以避免多进程 pickle 问题，
代价是数据加载并行度较低。

### 6.4 输出和日志参数

```yaml
output_dir: outputs/qwen2.5-0.5b/lora_sft
logging_steps: 5
save_steps: 100
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none
```

#### `output_dir`

最终模型、adapter、指标、Trainer 状态和 checkpoint 的保存目录。

#### `logging_steps: 5`

每 5 个 optimizer step 输出一次 loss、learning rate、epoch 和进度信息。

这里的 step 是参数更新次数，不是每读取一个 micro-batch 就算一步。

#### `save_steps: 100`

每 100 个 optimizer step 保存一次 checkpoint。当前训练共 155 步，因此产生 `checkpoint-100`，
并在训练结束时保存最终状态。

#### `plot_loss: true`

读取 Trainer 日志并生成 `training_loss.png`。

#### `overwrite_output_dir: true`

允许使用已经存在且非空的输出目录。本配置同时设置 `resume_from_checkpoint: null`，所以重新执行时会从头训练，
而不是自动续训。

#### `save_only_model: false`

checkpoint 除模型/adapter 外，还保存 optimizer、scheduler、随机数状态等，因而能够完整恢复训练。

如果设为 `true`，checkpoint 更小，但通常无法从完全相同的优化器状态继续训练。

#### `report_to: none`

不向 TensorBoard、W&B、SwanLab 等外部实验平台上报指标，本地日志和结果文件仍会保存。

### 6.5 训练超参数

```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
num_train_epochs: 5.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
```

#### `per_device_train_batch_size: 2`

每个 GPU、每个 micro-step 同时处理 2 条样本。

#### `gradient_accumulation_steps: 4`

连续执行 4 次前向和反向传播后才执行一次 optimizer 更新。

单卡有效 batch size：

```text
2 * 4 = 8
```

多卡有效 batch size：

```text
per_device_train_batch_size
* gradient_accumulation_steps
* GPU 数量
```

梯度累积可以用更多计算时间换取较低的峰值显存。

#### `learning_rate: 2e-4`

LoRA 参数的最大学习率。LoRA 的可训练参数较少，通常允许使用比全参数训练更高的学习率。

#### `num_train_epochs: 5`

完整遍历训练集 5 次。数据量只有 241 条，5 轮存在记忆训练样本和过拟合的风险，因此最好配合验证集观察。

#### `lr_scheduler_type: cosine`

warmup 结束后，学习率按余弦曲线从最大值逐渐衰减到接近 0。

#### `warmup_ratio: 0.1`

总 optimizer step 的前 10% 用于学习率预热。本次共 155 步，约前 16 步进行 warmup。

#### `bf16: true`

模型主要前向和反向计算使用 BF16。BF16 相比 FP32 节省显存和带宽，并且指数范围比 FP16 更大。

当前 LoRA 实现默认会把可训练参数提升到 FP32，以提高小规模参数更新的稳定性；因此这里不是纯 BF16 训练。

#### `ddp_timeout`

分布式进程组通信超时时间，单位为秒。该值非常大，主要用于避免模型加载或数据预处理较慢时 DDP 过早超时。

单卡训练时基本不起作用。

#### `resume_from_checkpoint: null`

明确不从 checkpoint 恢复。若要续训，可以改成：

```yaml
resume_from_checkpoint: outputs/qwen2.5-0.5b/lora_sft/checkpoint-100
```

### 6.6 当前未启用的评估参数

```yaml
# val_size: 0.1
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 100
```

这些参数被注释，所以本次训练没有验证集，也不会产生 `eval_loss`。

若取消注释，`val_size: 0.1` 会从训练数据中划分 10% 作为验证集，并按指定步数评估。

## 7. 数据从 JSON 到模型输入的过程

### 7.1 查找数据集

`get_dataset_list()` 读取 `data/dataset_info.json`，把逻辑名称解析为文件路径。

随后 `_load_single_dataset()` 调用 Hugging Face Datasets：

```python
dataset = load_dataset(
    path="json",
    data_files=[...],
    split="train",
)
```

### 7.2 统一格式

当前数据是默认 Alpaca 格式：

```json
{
  "instruction": "你是谁？",
  "input": "",
  "output": "我是小问。"
}
```

`AlpacaDatasetConverter` 转换为内部统一结构：

```python
{
    "_prompt": [
        {"role": "user", "content": "你是谁？"}
    ],
    "_response": [
        {"role": "assistant", "content": "我是小问。"}
    ],
    "_system": "",
    "_tools": "",
    "_images": None,
    "_videos": None,
    "_audios": None,
}
```

统一格式使后续代码不需要关心原始数据来自 Alpaca、ShareGPT 还是 OpenAI messages。

### 7.3 套用 Qwen 模板

`Template.encode_multiturn()` 首先将角色消息格式化为字符串，再调用 tokenizer：

```python
tokenizer.encode(formatted_text, add_special_tokens=False)
```

最终每轮对话分成：

```python
source_ids  # system + user + assistant 起始标记
target_ids  # assistant 回答 + 结束标记
```

### 7.4 构造 labels

默认 `train_on_prompt=False`，所以：

```python
input_ids = source_ids + target_ids
labels = [-100] * len(source_ids) + target_ids
```

示意：

```text
input_ids: [system, user, assistant_prefix, 回, 答, eos]
labels:    [ -100, -100,            -100, 回, 答, eos]
```

PyTorch 的交叉熵默认忽略标签 `-100`。因此：

- system 和 user token 参与注意力计算，为回答提供上下文。
- system 和 user token 不直接贡献训练损失。
- assistant 回答 token 才是主要学习目标。

这正是指令监督微调的核心。

### 7.5 截断和 padding

单条样本超过 `cutoff_len=1024` 时会截断。进入 batch 后，DataCollator 将不同长度序列补齐到 batch 中的最大长度，
训练时还会尽量补到 8 的倍数。

padding 部分的 labels 同样使用 `-100`，不会产生损失。

## 8. 模型加载逻辑

`run_sft()` 的加载顺序是：

```python
tokenizer_module = load_tokenizer(model_args)
template = get_template_and_fix_tokenizer(tokenizer, data_args)
dataset_module = get_dataset(...)
model = load_model(tokenizer, model_args, finetuning_args, is_trainable=True)
```

模型必须在模板修复 tokenizer 后加载，因为模板可能增加或替换特殊 token。

`load_model()` 的主要步骤：

1. 加载 `AutoConfig`。
2. 根据模型类型选择 `AutoModelForCausalLM` 等自动模型类。
3. 调用 `from_pretrained()` 加载基座权重。
4. 修复 attention、RoPE、MoE、多模态等模型细节。
5. 开启 gradient checkpointing。
6. 关闭训练时 KV cache。
7. 调用 `init_adapter()` 插入 LoRA。

Gradient checkpointing 不保存所有中间激活，反向传播时重新计算部分前向结果，以计算时间换显存。

## 9. LoRA 原理

### 9.1 原始线性层

Transformer 中的线性层通常计算：

```text
y = W x
```

其中 `W` 可能是非常大的矩阵。全参数微调需要为整个 `W` 保存梯度和优化器状态。

### 9.2 低秩增量

LoRA 冻结原始矩阵 `W`，只学习增量：

```text
Delta_W = B A
```

其中：

```text
A: r x in_features
B: out_features x r
r << min(in_features, out_features)
```

前向传播变为：

```text
y = W x + scale * B(A(dropout(x)))
```

标准缩放近似为：

```text
scale = lora_alpha / lora_rank
```

当前配置中：

```text
r = 8
alpha = 16
scale = 2
```

### 9.3 参数量为什么小

原矩阵参数量：

```text
out_features * in_features
```

LoRA 参数量：

```text
r * in_features + out_features * r
= r * (in_features + out_features)
```

例如一个 `896 x 896` 的线性层：

```text
原始参数：896 * 896 = 802,816
LoRA 参数：8 * (896 + 896) = 14,336
```

LoRA 只需要原层约 1.8% 的新增参数。一个模型包含许多目标层，最终比例取决于层数和维度。

### 9.4 初始化行为

PEFT 的普通 LoRA 通常将一个低秩矩阵随机初始化，另一个初始化为零。因此训练开始时：

```text
Delta_W ~= 0
```

模型初始行为基本等同于基座模型，之后逐步学习任务增量。

### 9.5 训练和推理的区别

训练时：

```text
基座 W：requires_grad=False
LoRA A/B：requires_grad=True
```

未合并推理时：

```text
y = W x + scale * B A x
```

导出合并模型时：

```text
W_merged = W + scale * B A
```

合并后不再需要额外 LoRA 支路，但生成的是接近完整基座大小的模型权重。

## 10. LoRA 在代码中如何插入

### 10.1 选择训练方法

`init_adapter()` 根据 `finetuning_type` 分派：

```python
if finetuning_type == "full":
    _setup_full_tuning(...)
elif finetuning_type == "freeze":
    _setup_freeze_tuning(...)
elif finetuning_type in ["lora", "oft"]:
    model = _setup_lora_tuning(...)
```

### 10.2 查找目标线性层

因为 `lora_target == ["all"]`，调用：

```python
target_modules = find_all_linear_modules(model, freeze_vision_tower=True)
```

该函数遍历 `model.named_modules()`：

```python
if "Linear" in module.__class__.__name__ \
        and "Embedding" not in module.__class__.__name__:
    module_names.add(name.split(".")[-1])
```

同时排除 `lm_head`。当前实际 adapter 配置表明目标模块是：

```python
[
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
```

其中：

- `q_proj/k_proj/v_proj/o_proj` 属于自注意力。
- `gate_proj/up_proj/down_proj` 属于 MLP/FFN。

### 10.3 创建 PEFT 配置

代码构造：

```python
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    target_modules=target_modules,
    lora_alpha=16,
    lora_dropout=0.1,
    use_rslora=False,
    use_dora=False,
    modules_to_save=None,
)
```

随后：

```python
model = get_peft_model(model, peft_config)
```

PEFT 会包装目标线性层，在原始层旁边加入 `lora_A`、`lora_B` 和 dropout。

### 10.4 参数精度

本次不是 `pure_bf16`，也没有使用 BAdam 或 ZeRO-3，因此代码会：

```python
for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)
```

所以可以理解为：

- 大部分冻结基座计算使用 BF16。
- LoRA 可训练参数保留 FP32。
- 前向过程由 AMP/Accelerate 管理混合精度。

## 11. Trainer 和一次训练更新

`run_sft()` 创建 `CustomSeq2SeqTrainer`。它继承自 Transformers `Seq2SeqTrainer`，主要增加：

- 自定义优化器扩展入口。
- 自定义调度器扩展入口。
- 生成评估和预测结果保存。
- ASFT、DFT、EAFT 等可选损失。
- 多模态 processor 保存。

当前配置没有启用特殊损失或特殊优化器，因此主要使用 Transformers 默认的 causal language modeling loss 和 AdamW 类优化器。

### 11.1 一个 micro-step

DataCollator 给出：

```python
batch = {
    "input_ids": ...,       # [2, sequence_length]
    "attention_mask": ...,  # [2, sequence_length]
    "labels": ...,          # [2, sequence_length]
}
```

Trainer 近似执行：

```python
outputs = model(**batch)
loss = outputs.loss
loss = loss / gradient_accumulation_steps
loss.backward()
```

### 11.2 Causal LM 的错位预测

模型用当前位置之前的 token 预测下一个 token：

```python
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
```

损失近似为：

```text
Loss = mean(-log P(target_token | previous_tokens))
```

所有 label 为 `-100` 的位置被忽略。

### 11.3 梯度累积和参数更新

连续 4 个 micro-step 后才执行：

```python
optimizer.step()
lr_scheduler.step()
optimizer.zero_grad()
```

基座参数被冻结，所以 optimizer 主要持有 LoRA 参数及其一阶、二阶矩估计。

## 12. 本次训练步数计算

训练集有 241 条样本。

每个 micro-batch 处理 2 条：

```text
ceil(241 / 2) = 121 micro-batch / epoch
```

每 4 个 micro-batch 更新一次：

```text
ceil(121 / 4) = 31 optimizer step / epoch
```

训练 5 轮：

```text
31 * 5 = 155 optimizer step
```

这与现有 `trainer_state.json` 中的结果一致：

```text
global_step = 155
max_steps = 155
num_train_epochs = 5
```

注意最后一个梯度累积组可能不足 4 个 micro-batch，因此最后一次 optimizer step 的实际样本数可能较少。

## 13. 学习率变化

总步数为 155，`warmup_ratio=0.1`，预热步数约为：

```text
ceil(155 * 0.1) ~= 16
```

学习率大致经历：

```text
接近 0
  -> 前约 16 步线性升到 2e-4
  -> 剩余步骤按 cosine 衰减
  -> 训练结束时接近 0
```

warmup 可以减小训练刚开始时随机 LoRA 参数带来的不稳定更新。

## 14. 保存产物

训练结束后 `run_sft()` 执行：

```python
trainer.save_model()
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
plot_loss(...)
```

`outputs/qwen2.5-0.5b/lora_sft` 中主要文件：

| 文件 | 意义 |
| --- | --- |
| `adapter_model.safetensors` | LoRA 的 A/B 参数，约 17.6 MB |
| `adapter_config.json` | rank、alpha、dropout、目标模块和基座路径 |
| `trainer_state.json` | 当前 step、epoch、历史日志 |
| `trainer_log.jsonl` | LLaMA-Factory 格式的训练进度日志 |
| `train_results.json` | 最终 loss、耗时、吞吐等 |
| `training_args.bin` | Trainer 参数序列化结果 |
| `training_loss.png` | 损失曲线 |
| `tokenizer.json` | tokenizer 配置 |
| `chat_template.jinja` | 推理聊天模板 |
| `checkpoint-100` | 第 100 步检查点 |
| `checkpoint-155` | 最终检查点 |

当前已有训练结果：

```text
train_loss = 1.6858259016467678
train_runtime = 138.3362 秒
train_samples_per_second = 8.711
train_steps_per_second = 1.12
```

训练 loss 下降只说明模型越来越适合训练数据。由于当前没有验证集，不能仅根据该值判断泛化能力。

## 15. 推理时如何加载 LoRA

`configs/qwen2.5_lora_infer.yaml` 中指定：

```yaml
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
adapter_name_or_path: outputs/qwen2.5-0.5b/lora_sft
template: qwen
```

加载顺序为：

1. 加载原始 Qwen2.5 基座。
2. 读取 `adapter_config.json`。
3. 将 `adapter_model.safetensors` 加载到目标线性层。
4. 使用与训练相同的 `qwen` 模板生成输入。
5. 前向时同时执行基座支路和 LoRA 支路。

如果只移动 LoRA adapter 而目标机器没有基座模型，通常仍需要从 Hub 下载基座模型。

## 16. 当前配置的注意事项

### 16.1 `identity.json` 中的占位符不会自动替换

`data/identity.json` 的很多回答仍包含：

```text
{{name}}
{{author}}
```

数据转换器没有针对这两个字段做变量替换。模板中的 `{{content}}` 是 Qwen 模板内部占位符，
但数据内容里的 `{{name}}` 和 `{{author}}` 只是普通文本。

因此模型很可能会学习输出字面量 `{{name}}` 和 `{{author}}`。正式训练前应将它们替换为预期名称。

### 16.2 没有验证集

当前无法观察：

- `eval_loss`
- 未见问题上的身份一致性
- 各语言风格是否泛化
- 是否发生过拟合

建议至少启用 `val_size`，或准备独立 `eval_dataset`。

### 16.3 数据规模很小

241 条样本适合演示 LoRA 流程和身份注入，但不足以显著提升模型的通用能力。训练目标更接近：

- 修改自我介绍。
- 学习文言文、粤语、新闻主播等固定风格。
- 记忆少量固定表达。

### 16.4 这不是 QLoRA

配置没有 `quantization_bit: 4` 或 `quantization_bit: 8`，因此基座不是以 4/8 bit 形式加载。

当前是：

```text
BF16 基座模型 + FP32 LoRA 参数
```

QLoRA 通常是：

```text
4-bit 量化基座模型 + LoRA 参数
```

### 16.5 `overwrite_output_dir` 和续训

当前组合：

```yaml
overwrite_output_dir: true
resume_from_checkpoint: null
```

意味着再次执行命令会重新开始训练，并可能覆盖输出目录中的最终文件。若目标是续训，应明确指定 checkpoint。

## 17. 调试这条训练链路的方法

### 17.1 查看参数解析结果

重点在 `get_train_args()` 返回前打印：

```python
print(model_args)
print(data_args)
print(training_args)
print(finetuning_args)
```

### 17.2 查看分词后的样本

`_get_preprocessed_dataset()` 已经自动打印第一条样本：

```text
input_ids
inputs
label_ids
labels
```

其中 `labels` 解码结果应主要是 assistant 回答。

### 17.3 查看可训练参数

可在 YAML 中临时添加：

```yaml
print_param_status: true
```

正常情况下，大部分基座参数应显示 `trainable: false`，LoRA 参数应显示 `trainable: true`。

### 17.4 缩短调试训练

调试调用链时可以覆盖：

```powershell
llamafactory-cli train configs/qwen2.5_lora_sft.yaml `
  max_samples=8 `
  max_steps=2 `
  logging_steps=1 `
  output_dir=outputs/debug_lora
```

这样可以快速检查数据、模型和反向传播是否正常。

## 18. 一句话总结

这条命令的本质是：将 241 条身份和风格数据转换成 Qwen ChatML token，屏蔽 system/user 部分的损失，
在 Qwen2.5 注意力层和 MLP 层的线性变换旁插入 rank-8 LoRA 支路，然后通过 155 次 AdamW 参数更新，
只学习约 17.6 MB 的 adapter 权重，最后将 adapter、训练状态和日志保存到输出目录。
