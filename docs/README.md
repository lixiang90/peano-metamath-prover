# 文档导航 / Documentation

[English overview](../README.md) · [中文概览](../README-zh.md)

文档按共享形式系统、训练/搜索后端和实验状态组织，不再按 Git 分支拆分。
详细设计目前主要使用中文，历史实验报告保留原语言与原始日期。

## 上手与复现

- [训练与评估指南](training-and-evaluation.md)：安装迁移、接口矩阵、PA+、MCTS、
  HTPS 和 Scale 工作流，命令以统一主线为准。
- [系统架构](architecture.md)：三个模块的职责、共享可信边界与搜索后端差异。
- [贡献约定](../CONTRIBUTING.md)：测试、证明可靠性、文档与发布包要求。

## 形式系统与数据

- [基础数论定义](number-theory.md)：`peano-number-theory.mm` 的定义与目标边界。
- [PA+ 分层定义](pa-plus-definitions.md)：68个保守定义、35个非逻辑目标、编译与审计。
- [PA+ 随机生成](pa-plus-random-generation.md)：定义桥、有界项、目标引导、质量配额。
- [PA+ 神经管线](pa-plus-neural-training.md)：词表、候选损失、采样与推理接线；
  其中2026-08-31的 GPU 指标属于历史记录，旧桥动作语料须重建。

## 搜索、训练与认证

- [HTPS 设计](htps-design.md)：forward DAG、AND/OR 超图、受保护引理边与同步回放。
- [中间引理与连续潜在推理](lemma-and-latent-reasoning.md)：动作语义、模型升级、验收边界。
- [PA+ 认证与公平评估](pa-plus-certification.md)：确定性动作编码、环境指纹、外部门槛、消融。

## 结果、路线与设计提案

- [当前进展与研究路线](progress-and-roadmap.md)：当前工程状态、研究验收与下一步顺序。
- [形式系统演进设计](formal-system-evolution.md)：PA+ / Equations+ / Lean 等设计比较；
  不应把文中的未来方案视为均已实现。

以下只记录历史实验，不是当前 PA+/HTPS 的推荐配置或能力证明：

- [早期 Scale 基线](scale-baseline.md)。
- [2026-08-11 百万级训练与深度实验](first-large-scale-run.md)。
- [2026-08-24 RTX 5090 训练与 MCTS 闭环](first-rtx5090-closed-loop-run.md)。

阅读结果时区分：目标触达与目标证明、动作准确率与解题率、项目内回放与外部认证、
工程 smoke 与正式多种子研究验收。数据和模型位于被 Git 忽略的 `outputs/`，不随源码发布。
