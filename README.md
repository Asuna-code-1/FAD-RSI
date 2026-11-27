# ATI-OPD 原生 FAD-RSI

正式实现位于 ATI-OPD/verl。这里的 train.py 调用原项目的 local/run_four_gpu_entry.py，默认只检查配置。早期 Transformers 参考模块不在运行路径上。

两组模型各有 opd、ati-opd、fad-opd、fixed、fad-rsi 五组。配置保存在 ATI-OPD/local/fad_configs/comparison-v1，可用 local/prepare_fad.py 从历史配置重建。opd 保留各模型的 vanilla OPD；ati-opd 从同一配置加入对应历史 ATI 干预和诊断参数；三个 FAD 组从 vanilla 加入 k3 定位和反馈。所有组从原学生权重开始，复用原训练加载器、vLLM、FSDP2、诊断和基准评估（含 step 0）。原生运行入口按历史配置使用 SwanLab。

FAD 仅在训练 rollout 上运行。原始回答由原 reward worker 评分，失败时用 k3 找首个超阈值段。教师参考原答和分歧段生成结构化诊断。学生逐 token 保留分歧段之前的原答，把诊断附在此前缀末尾，从分歧段起点生成新的推理后缀。检索到策略时先用诊断和策略续写，失败后再只用诊断续写。原 verifier 对保留前缀和新后缀拼成的完整答案判分。正确修订与错误原答组成偏好对，DPO 在共享的原题和保留前缀下比较两个后缀。诊断只用于引导采样，原答前缀和诊断均不进入 DPO 计分。正确原答及教师明确定位的首错之前、位于保留前缀内的 token 仍用原 top-k OPD。

每次仍采样 64 题 × 4 个原答，原答和修订行合并为一次优化器更新。更新前以冻结参考与当前 actor 的后缀平均 log-prob 计算 DPO margin，跨 microbatch 累加其两侧的导数。β=0.1，OPD 权重为 1，均为 FAD 专属默认值。每个 batch 后动态组更新策略，固定组持续使用初始库。经验记录于运行目录 fad_experiences，策略快照随原项目 checkpoint 保存。

在 FAD-RSI 目录可执行配置检查，不加载模型或启动训练：

```bash
source env.sh
python train.py --pair nemotron --mode fad-rsi
```

分布式运行需要原项目规定的 tmux 和集群运行环境。本次交付仅配置和 CPU 检查。FAD 需要冻结参考打分、额外师生生成和验证，实际显存、吞吐及效果需实测。

## 实现与复现实验

- [方法设计稿](docs/FAD-RSI.docx)、[反馈定位与修订](ATI-OPD/verl/experimental/agent_loop/fad_repair.py)、[动态策略库](ATI-OPD/verl/trainer/distillation/fad_memory.py)和[联合更新](ATI-OPD/verl/trainer/distillation/fad_update.py)。
- [两组模型的十套配置及 Hydra 解析文件](ATI-OPD/local/fad_configs/comparison-v1)和[配置生成脚本](ATI-OPD/local/prepare_fad.py)。
- [ATI-OPD 复现说明](ATI-OPD/local/论文分析与复现说明.txt)、[历史运行配置](ATI-OPD/local/runs)和[已有基准评测 CSV](experiments/results/README.md)。上游为 [jubgjf/ATI-OPD](https://github.com/jubgjf/ATI-OPD)，快照基于 `10db7986937784dffee5597cae7778770f81c471`，保留上游许可证、原有训练入口和本地改动。

`env.sh` 与历史配置保留原机的环境和数据路径。迁移时可激活自己的 Python 环境，按配置准备模型和数据，再运行 `python train.py --pair nemotron --mode fad-rsi`。默认检查会验证数据文件是否存在。分布式训练入口从环境变量读取 SwanLab 和 W&B 凭据。数据、模型权重、checkpoint、日志和本地依赖目录另行准备。
