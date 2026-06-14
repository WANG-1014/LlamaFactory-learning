# 实验报告
这是我的深度学习课程实验报告：Style-Conditioned LoRA Fine-Tuning of Qwen2.5-0.5B-Instruct（基于风格条件的LoRA对Qwen2.5-0.5B-Instruct的微调）

## 一、数据集构建与处理
- 我选择风格的原则：风格间需明显可区分，确保微调效果可观察
- 选择了3种风格：  
    | 类别 | 示例 |
    |------|------|
    | 古典/文学 | 文言文 |
    | 地域方言 | 粤语风格 |
    | 领域语气 | 新闻主播 |
- 构建数据集：采用大模型生成（DeepSeek）
- 数据集：[文言文数据（identity_wenyan.json）](/data/identity_wenyan.json) 、 [广东话数据（identity_Cantonese.jsonata）](/data/identity_Cantonese.json) 和 [新闻主播风格数据（identity_news_anchor.jsonata）](/data/identity_news_anchor.json)
- 修改配置文件：[dataset_info.json#L5-L13](/data/dataset_info.json#L5-L13)

## 二、实验过程（使用`llamafactory-cli`指令）
- 使用的是魔搭社区的模型：`$env:USE_MODELSCOPE_HUB = "1"`
### 2.1 使用`llamafactory-cli train configs/qwen2.5_lora_sft.yaml`指令训练
- 我使用WebUI训练会报错，WebUI默认使用16核：preprocessing_num_workers: 16  
  因此我用llamafactory-cli指令运行：`llamafactory-cli train configs/qwen2.5_lora_sft.yaml`
- 训练配置：[configs/qwen2.5_lora_sft.yaml](configs/qwen2.5_lora_sft.yaml)
### 2.2 与lora微调模型对话验证
- 用llamafactory-cli指令运行：`llamafactory-cli chat configs/qwen2.5_lora_sft.yaml`
- 对话验证配置：[configs/qwen2.5_lora_infer.yaml](configs/qwen2.5_lora_infer.yaml)
### 2.3 与原始模型对话对比
- 用llamafactory-cli指令运行：`llamafactory-cli chat configs/qwen2.5_base_infer.yaml`
- 对话对比配置：[configs/qwen2.5_base_infer.yaml](configs/qwen2.5_base_infer.yaml)
### 2.4 损失曲线图
- 图中origin曲线在切换风格训练时会出现loss的反弹，但是整体的趋势smoothed曲线看loss是持续下降的：  
    ![alt text](outputs/qwen2.5-0.5b/lora_sft/training_loss.png)

## 三、实验过程（使用webui配置）
- 运行指令打开UI界面：`llamafactory-cli webui`
- 修改runner.py配置的preprocessing_num_workers=2

## 三、训练超参数与曲线分析

## 四、效果比较与案例学习