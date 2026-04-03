# Multi-Agent Reflection - 多智能体反思循环

> **日期**: 2026-04-03 | **难度**: 进阶 | **类别**: Agent 架构 / 多智能体系统

---

## 一句话总结

不依赖单个 LLM 完成所有工作，而是让 **多个 LLM 扮演不同角色**（生成者、审查者、改进者），通过多轮 "生成 → 反馈 → 修正" 的循环，利用多样性换取更好的最终输出。这比单模型的 Self-Reflection 更强，因为每个 Agent 有不同的视角和 Prompt。

---

## 从 Agent 进化看 Multi-Agent

```
LLM Evolution:
  
  v1. Prompt → LLM → Response           # 单次生成
        ↓
  v2. Prompt → LLM → Self-Critique → Fix  # 单体自我反思 (Reflexion)
        ↓
  v3. Generator ──→ Critic ──→ Refiner    # 两阶段流水线
           ↑___________│ (多轮)
        ↓
  v4. Planner → Executor → Reviewer       # 分工明确的多 Agent
             ↓              ↑
          Verifier ────────┘
        ↓
  v5. 多个 Expert Agent 并行生成，
      一个 Judge Agent 聚合结果
      (Committee / Debate 架构)
```

---

## 经典架构: Reflexion

Reflexion 是最著名的单体 Agent 反思循环，也是所有 Multi-Agent 架构的基础。

```python
def reflexion_agent(model, problem, max_reflections=3):
    """
    Reflexion: 生成 → 评估 → 反思 → 再尝试
    
    关键: 不是简单让模型改答案，而是让模型生成
    一份 "经验教训" (Reflection)，包含在下次尝试的 prompt 中。
    这样 Agent 能从过去的错误中 "学习"。
    """
    # 经验库: 记录过去的反思
    reflections = []
    
    for attempt in range(max_reflections + 1):
        # 构建 prompt (包含过去的反思)
        prompt = f"Problem: {problem}\n"
        if reflections:
            prompt += f"\n过去的经验教训:\n"
            for r in reflections:
                prompt += f"- {r}\n"
        prompt += "\nSolution: "
        
        # 生成
        solution = model.generate(prompt, stop=["\nEvaluation:"])
        
        # 评估（可以是代码执行也可以是模型自评）
        evaluation = _evaluate(solution, problem)
        
        if evaluation.passed:
            return solution
        
        # 反思: 为什么错了？下次应该怎么做？
        reflection = model.generate(
            f"Problem: {problem}\n"
            f"My solution: {solution}\n"
            f"Error: {evaluation.error}\n"
            f"反思: 我哪里做错了？下次应该怎么改进？"
        )
        reflections.append(reflection)
    
    return solution  # 超过最大尝试次数
```

---

## Multi-Agent 架构对比

### 架构 1: Critique-Refine 流水线 (两 Agent)

```
    ┌────────┐   生成   ┌────────┐   批评   ┌────────┐
    │Generator│ ──────→ │  Critic │ ──────→ │Refiner │
    │          │         │         │          │       │
    └────────┘         └────────┘         └───┬───┘
                                                  │
                              ┌──────────────────┘
                              │ (修正)
                              ▼
                         ┌────────┐
                         │ Output │
                         └────────┘
```

```python
def critique_refine_pipeline(generator, critic, refiner, input_text, max_rounds=3):
    output = generator.generate(f"Generate: {input_text}")
    
    for round in range(max_rounds):
        critique = critic.generate(
            f"Review this output for errors, gaps, and improvements:\n{output}"
        )
        
        if "no issues found" in critique.lower() or round == max_rounds - 1:
            break
        
        output = refiner.generate(
            f"Original output: {output}\nCritique: {critique}\n"
            f"Revise the output addressing all issues."
        )
    
    return output
```

### 架构 2: Debate / 辩论架构 (多 Agent)

```
        Topic
          │
      ┌───┴───┐
      │       │
  Agent A  Agent B          # 对立视角
  ("支持X") ("反对X")
      │       │
   论点1    论点1
      │       │
      └───┬───┘
          │ 交叉质疑 & 回应 (多轮)
      ┌───┴───┐
      │       │
  Agent A  Agent B
  论点2    论点2
      │       │
      └───┬───┘
          ▼
     ┌─────────┐
     │  Judge  │   # 第三方 Agent 评估谁的最终论点更强
     └─────────┘
```

**为什么 Debate 比单模型强？**

- **强制探索对立观点**: 如果不强制 Agent A 只支持 X，模型倾向于说 "两边都有道理"
- **消除幻觉**: Agent B 专门攻击 Agent A 的错误
- **更深的覆盖**: 每轮辩论都推进一层深度

```python
def debate_system(agent_a, agent_b, judge, topic, rounds=3):
    """Multi-Agent Debate System"""
    # 初始化: 两方各生成初始立场
    a_argument = agent_a.generate(f"你支持以下论点，请详细论证:\n{topic}")
    b_argument = agent_b.generate(f"你反对以下论点，请逐一反驳:\n{topic}")
    
    for round in range(rounds):
        # 互相质疑
        a_rebuttal = agent_a.generate(
            f"Topic: {topic}\n"
            f"Opponent's argument: {b_argument}\n"
            f"请逐一回应并指出对方论证中的漏洞:"
        )
        b_rebuttal = agent_b.generate(
            f"Topic: {topic}\n"
            f"Opponent's argument: {a_rebuttal}\n"
            f"请逐一回应并指出对方论证中的漏洞:"
        )
        a_argument = a_rebuttal
        b_argument = b_rebuttal
    
    # Judge 裁决
    verdict = judge.generate(
        f"Topic: {topic}\n"
        f"Side A's final argument: {a_argument}\n"
        f"Side B's final argument: {b_argument}\n"
        f"综合考虑，哪一方的论点更有说服力？为什么？"
    )
    
    return {
        "side_a": a_argument,
        "side_b": b_argument, 
        "verdict": verdict
    }
```

### 架构 3: Planner-Executor-Verifier (PEV)

这是 **最接近人类解决问题方式** 的多 Agent 架构。

```
                    Problem
                       │
                  ┌────┴────┐
                  │ Planner │  # 分析任务，拆解子任务
                  └────┬────┘
                       │ 计划 (steps: [s1, s2, ..., sn])
              ┌────────┼────────┐
              ▼        ▼        ▼
        ┌─────────┐┌───────┐┌───────┐
        │Executor1││Exec 2 ││Exec 3 │  # 并行执行子任务
        └────┬────┘└───┬───┘└──┬────┘
             │         │       │
             ▼         ▼       ▼
        ┌─────────────────────────┐
        │       Verifier          │  # 检查每个子任务的输出
        │  ┌───pass/fail───┐      │
        │  │ 如果不通过      │      │
        │  ▼               │      │
        │  要求重做/修正     │      │
        └────────┬─────────────────┘
                 │ 汇总通过的结果
                 ▼
           Final Answer
```

```python
class PlannerExecutorVerifier:
    def __init__(self, planner, executor_model, verifier, 
                 max_retries=2):
        self.planner = planner
        self.executor = executor_model
        self.verifier = verifier
        self.max_retries = max_retries
    
    def solve(self, problem):
        # Step 1: Planner 拆分解题步骤
        plan = self.planner.generate(
            f"Problem: {problem}\n"
            f"Break this down into independent subtasks. "
            f"Each subtask should have clear input and expected output."
        )
        
        # Step 2: 解析计划为结构化步骤 (实际中会用 JSON parsing)
        subtasks = self._parse_plan(plan)
        
        results = {}
        for step in subtasks:
            # Step 3: Executor 执行
            result = self.executor.generate(step.prompt)
            results[step.id] = result
            
            # Step 4: Verifier 验证
            for retry in range(self.max_retries):
                verdict = self.verifier.generate(
                    f"Subtask: {step.description}\n"
                    f"Result: {result}\n"
                    f"Does this result correctly and completely "
                    f"address the subtask? Answer YES or NO with reasons."
                )
                
                if verdict.startswith("YES"):
                    break
                
                # 修正
                result = self.executor.generate(
                    f"Subtask: {step.description}\n"
                    f"Previous result: {result}\n"
                    f"Verifier feedback: {verdict}\n"
                    f"Fix the issues and regenerate."
                )
                results[step.id] = result
        
        # Step 5: Planner 汇总
        final = self.planner.generate(
            f"Problem: {problem}\n"
            f"Subtask results: {results}\n"
            f"Synthesize these results into a final answer."
        )
        
        return final
```

---

## 深度思考: 为什么 Multi-Agent 比 Single Agent 强?

### 1. 信息多样性 (Information Diversity)

同一问题，不同 prompt 会触发模型不同的 "思维路径"。强制 Agent A 支持 X、Agent B 反对 X，保证覆盖了单一 Agent 不会同时产生的两面视角。

### 2. 错误消除 (Error Cancellation)

如果两个独立 Agent 的幻觉率各 10%，它们恰好犯同样幻觉的概率 < 1%。Critic/Verifier 的存在可以显著降低输出中的错误。

### 3. 角色扮演降低 "和稀泥" 倾向

LLM 天生倾向于 "两边都有道理"。强制角色分配（"你是辩护律师"/"你是检察官"）打破了这种倾向。

### 4. 什么时候 Multi-Agent 不一定更好?

| 场景 | Multi-Agent | 原因 |
|------|------------|------|
| 事实查询 | 不推荐 | 直接查就好，不需要多 Agent 辩论 |
| 创意写作 | 看情况 | 多样性可能破坏 coherence |
| 复杂编程 | 强烈推荐 | PEV 架构在编程任务效果最明显 |
| 数学证明 | 强烈推荐 | Verifier 可以用代码执行器 |
| 长文档分析 | 推荐 | 多 Agent 可并行分析不同章节 |

---

## 扩展阅读

- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [Chain of Verification (CoVe)](https://arxiv.org/abs/2309.11495) -- 生成 → 验证计划 → 执行验证 → 修正
- [LLM Debates](https://arxiv.org/abs/2305.14333) -- 辩论式推理
- [AutoGen](https://microsoft.github.io/autogen/) -- Microsoft 的多 Agent 框架
- [MetaGPT](https://arxiv.org/abs/2308.00352) -- 用 SOP 编码多 Agent 流程

---

_上一个: [Day 4 - Test-Time Compute](04-test-time-compute.md)_

_这是第一期的内容。仓库每天更新，明天见!_
