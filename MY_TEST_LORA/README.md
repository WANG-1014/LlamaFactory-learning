# 实验报告
这是我的深度学习课程实验报告：Style-Conditioned LoRA Fine-Tuning of Qwen2.5-0.5B-Instruct（基于风格条件的LoRA对Qwen2.5-0.5B-Instruct的微调）

## Data Construction & Process
- 我选择风格的原则：风格间需明显可区分，确保微调效果可观察
- 选择了3种风格：  
    | 类别 | 示例 |
    |------|------|
    | 古典/文学 | 文言文 |
    | 地域方言 | 粤语风格 |
    | 领域语气 | 新闻主播 |
- 构建数据集：采用大模型生成（DeepSeek）
- 数据集：