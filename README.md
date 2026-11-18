# Agent Research

面向 LLM Agent 可靠性、工具调用、错误检测与恢复的研究工作区。

## Current project: DAV

**DAV — Dual-Budget Adaptive Verification and Dependency-Guided Re-computation**

在独立验证预算与恢复预算约束下，根据数据依赖、错误传播风险和预期恢复成本，自适应选择需要验证的中间结果；发现错误后，通过概率修复和依赖感知重计算恢复任务。

> 当前内容是研究模型与代码骨架 v0.1。算法有效性、概率假设及理论性质仍需通过推导和实验验证，不代表已获得理论保证或实证结论。

## Repository structure

- `docs/dav/`: 研究问题、数学模型、算法、实验计划
- `src/dav/`: Python 实现
- `configs/`: 合成 DAG 与消融实验配置
- `scripts/`: 实验入口
- `tests/`: 单元测试
- `notes`: 既有论文阅读笔记（保留）

## First milestones

1. 明确概率依赖假设，构建小规模 DAG 的精确枚举 oracle。
2. 实现节点状态、版本追踪、概率验证和概率修复。
3. 实现 Monte Carlo 期望成本估计器与 AMCR 候选策略。
4. 对比 Random、Uncertainty、Impact、Static Risk 等基线。
5. 完成消融实验，再接入有日志可追踪的 Agent benchmark。

## Research safeguards

- 区分节点自身错误与上游错误传播。
- 验证结果更新后验信念，不代表绝对正确。
- 只使真正消费了旧版本的下游结果失效。
- 共享下游节点只重算一次。
- 验证预算与恢复预算独立记账，并在每个动作前执行硬预算检查。
- INVALID、STALE、BLOCKED 节点不得作为可信输入。
- 不在未经证明时宣称目标函数次模或贪心算法具有近似保证。
- 第一版不处理不可逆副作用工具，除非有幂等或补偿语义。

## Quick start

Python package and runnable experiment entry points will be added incrementally. See [DAV research overview](docs/dav/research_overview.md).
