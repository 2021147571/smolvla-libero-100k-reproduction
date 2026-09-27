# SmolVLA LIBERO 100k 复现记录

个人实验备份。模型权重：[laroi0124/SmolVLA_100k_test](https://huggingface.co/laroi0124/SmolVLA_100k_test)（上传状态以 Hugging Face 文件列表为准）。

## 各个模型怎么来的

| 模型 | 训练方式 | 当前用途 |
|---|---|---|
| **100k 父模型** | 从 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 初始化 VLM；不是从 LIBERO teacher 继续训练。冻结 VLM，训练动作专家及 state projection。使用 `HuggingFaceVLA/libero`，batch 64、seed 1000，训练 100,000 steps。 | 主要保留和发布的模型 |
| task0 子模型 | 从 100k 父模型出发，选择 38 条目标任务成功演示，额外训练 3,000 steps，batch 64，学习率从 1e-5 衰减至 1e-6，每 500 步保存。 | 早期 success-demo-only 实验，不当作最终改进模型 |
| 定向微调 v1 | 早期 teacher 引导的定向微调与子 checkpoint 筛选。 | 已废弃，原文件保留归档，不恢复 |
| 定向微调 v2（31d teacher） | 从父模型出发，将 teacher 纠正轨迹、目标任务成功演示、全短任务 replay 按 25%/25%/50% 采样；训练 2,000 steps，batch 64，学习率 2e-6→5e-7，每 250 步保存，共 8 个子 checkpoint。 | 失败实验：发现新数据统计量替换了父模型归一化统计量，不能作为有效提升结论 |
| v3 fixed-stats smoke | 保留父模型的归一化统计量，进行 2-step 冒烟测试，验证训练路径。 | 仅调试，不是完成的正式微调模型 |
| 官方 teacher `lerobot/smolvla_libero@31d453f7` | 下载的公开模型，保存的训练配置为 25k steps；不是本项目训练。 | 对照模型，不代表已确认的论文同款权重 |
| 2.2B teacher 候选 | 下载自 `HuggingFaceVLA/smolvla_libero_ckpts` 的 100000 checkpoint，进行了兼容性检查。 | 用户已暂停，不作为完成评测或有效 teacher 发布 |

## 父模型配置

- 数据版本：`86958911c0f959db2bbbdb107eb3e17c5f9c798e`。
- AdamW，初始学习率 `1e-4`，warmup 1,000，scheduler decay 30,000，最低学习率 `2.5e-6`；总训练仍为 100,000 steps。
- 两路图像，state 8 维，action 7 维；action chunk 长度 50，flow matching 积分 10 步。
- **执行 1/10/50 步才重新观察，是评测设置，不是训练了三个不同父模型。** 导出的原始配置默认执行 50 步。

## 已完成的评测

| 协议 | Spatial | Object | Goal | Long | 总计 |
|---|---:|---:|---:|---:|---:|
| 父模型，每执行 1 步重新观察 | 87/100 | 83/100 | 82/100 | 53/100 | 305/400；短任务 252/300 |
| teacher 31d，每执行 1 步重新观察 | 78/100 | 74/100 | 76/100 | 57/100 | 285/400；短任务 228/300 |
| 父模型，每执行 10 步重新观察 | 84/100 | 95/100 | 89/100 | 未测 | 268/300 |
| teacher 31d，每执行 10 步重新观察 | 75/100 | 88/100 | 87/100 | 未测 | 250/300 |
| 父模型，每执行 50 步重新观察（早期） | 69/100 | 75/100 | 71/100 | 未测 | 215/300 |

以上是本项目环境下的观察结果，不是官方论文成绩。正式表格使用每任务 10 个初始状态（0–9）、seed 1000、batch 1、AMP false、flow 积分 10 步。只在相同执行间隔下对比父模型和 teacher。不能声称模型已达到 300/300。

## 文件说明与限制

- `results/`：评测结果和汇总。
- `scripts/`：原实验脚本备份，包含实验机路径，使用前需调整；不是一键安装包。废弃微调脚本只作历史记录。
- `docs/`：代码审计与实验说明。
- 不包含密码、SSH 密钥、连接信息、原始敏感日志、数据集副本或第三方 teacher 权重。
- 评测环境和论文不保证完全一致；模拟器版本、随机性和已查看测试状态均可能影响解释。现有结果不能证明不存在训练/测试状态重叠。
- 废弃子模型权重仍保留在原实验存储中，不作为推荐模型上传到此代码仓库。

上游：[LeRobot](https://github.com/huggingface/lerobot)、[SmolVLA](https://arxiv.org/abs/2506.01844)、[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)。本仓库为个人复现资料，不是官方发布。

