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

## 二、实验过程（使用llamafactory-cli指令）
- 使用的是魔搭社区的模型：`$env:USE_MODELSCOPE_HUB = "1"`
- 我的指令config里保存模型会保存在outputs文件夹，与webUI不一样
### 2.1 使用模型训练
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
- 运行指令打开UI界面：`conda activate llama` `llamafactory-cli webui`
- 修改runner.py配置的preprocessing_num_workers=2
### 3.1 webui参数理解
- 最上面的部分是选择ui界面语言、模型、微调方法以及量化参数等。（使用Qwen2.5-0.5B-Instruct模型，不使用量化加速）  
    ![alt text](MY_README/imgs/webui_1.png)
- 中间的Train参数配置
    ![alt text](MY_README/imgs/webui_2.png)  
    - 训练阶段：选择训练方式（sft监督微调）
    - 数据集：选择要训练、微调的数据集
    - 学习率：每次参数更新的幅度（学习率越大，学习越快，但是容易损失原模型能力）
    - 训练轮数：完整遍历训练集的次数
    - 最大梯度范数：梯度裁剪阈值（防止梯度过大）
    - 最大样本数：每个数据集最多读取多少条（我的训练数据远小于该值，所以都使用）
    - 计算类型：模型训练使用的数值精度（Qwen2.5-0.5B-Instruct使用的是bf16）
    - 截断长度：每条样本最多保留的 token 数（我的数据比较短，建议使用短一点）
    - 批处理大小：每张 GPU 每次处理的样本数（训练一个批次用的数据量）
    - 梯度累积：累积多少个小批次后更新一次参数（相当于一次参数更新用到了 批处理大小*梯度累积 的数据）
    - 验证集比例：从训练数据中划分多少作为验证集（设置验证集，训练时会绘制对应的验证loss，可以对比查看会不会过拟合等）
    - 学习率调节器：学习率随训练进程变化的方式
- 下方其它配置（需要修改的是其它参数设置以及LoRA参数设置）
    - 其它参数设置
    ![alt text](MY_README/imgs/webui_3.png)
        - logging_steps：记录loss的步数，用于绘制loss曲线
        - save_steps：checkpoint的间隔步数
        - warmup_steps：预热步数
        - neftune_noise_alpha：在训练期间给模型的输入词嵌入向量添加随机噪声，相当于一种正则化方法（先不设置）
        - 额外参数：{"optim": "adamw_torch"}-使用 PyTorch AdamW 优化器，合理且兼容性好
        - packing：序列打包-把多条短样本拼成一个固定长度序列（先关闭）
        - neat_packing：无污染打包-打包后阻止不同样本互相注意（先关闭）
        - train_on_prompt：学习提示词-是否对用户输入部分也计算损失（先关闭）
        - mask_history：不学习历史对话（我的数据集都是单轮对话）
        - resize_vocab：更改词表大小（先关闭）
        - use_llama_pro：训练新增/扩展的模型层（先关闭）
        - enable_thinking：启用思考（Qwen2.5-0.5B-Instruct，它不是标准思考模型,先关闭）
        - report_to：使用其它数据平台（我使用wandb）
        - Trackio Setting：也是数据平台（先关闭）
    - LoRA参数设置
    ![alt text](MY_README/imgs/webui_4.png)
        - lora_rank：rank 越大，LoRA 能学习的变化越复杂，但参数量、显存和过拟合风险也越高。
        - lora_alpha：LoRA 缩放系数，常设为 rank 的两倍 16
        - lora_dropout：防止小数据集过拟合，设为 0.05
        - LoRA+ 学习率比例：LoRA+ 会让 B 矩阵使用更高学习率。例如设为 16，B 矩阵学习率约为基础学习率的 16 倍。（先设置0）
        - 新建适配器：也就是resume，这里不选择
        - 下方参数，先不设置






## 四、训练超参数与曲线分析

## 五、效果比较与案例学习