# Day 05: Multi-Agent Reflection
# 第 05 天: 多智能体反思系统

> **Date**: 2026-04-03 | **Difficulty**: Advanced | **Category**: Agent Architecture 智能体架构

> **Watch the animation**: ![Multi-Agent Reflection Animation](../gifs/05-multi-agent-reflection.gif)

---

## One-Line Summary

Instead of relying on a single LLM to do everything, have **multiple LLMs play different roles** (generator, critic, refiner, verifier) in a multi-round "generate → critique → refine" loop. This is stronger than single-model self-reflection because each agent has a distinct perspective and prompt.

不依赖单个 LLM 完成所有工作，而是让多个 LLM 扮演不同角色，通过多轮"生成 → 批评 → 修正"循环提升最终输出质量。比单体自我反思更强，因为每个 Agent 有不同的视角和 Prompt。

---

## The Evolution | 进化路径

```
v1  Prompt → LLM → Response
               单次生成 (简单但质量不可控)
    ↓
v2  Prompt → LLM → Self-Critique → Fix
               单体自我反思 (Reflexion，但 LLM 倾向于"和稀泥")
    ↓
v3  Generator → Critic → Refiner
               角色分离 (Critic 更客观)
    ↓
v4  Planner → Executor → Verifier
               PEV 架构 (最接近人类解决问题的方式)
    ↓
v5  Multiple Experts → Judge
               委员会/辩论 (强制探索对立观点)
```

---

## Classic Architecture: Reflexion

The foundation of all multi-agent reflection:

```python
def reflexion(model, problem, max_attempts=4):
    """
    Generate → Evaluate → Reflect → Retry
    Key: Reflections are carried forward as context,
         so the agent "learns from its mistakes"
    """
    reflections = []  # lessons learned

    for attempt in range(max_attempts):
        prompt = f"Problem: {problem}\n"
        for r in reflections:
            prompt += f"  Lesson: {r}\n"
        prompt += "Solution: "

        solution = model.generate(prompt)
        eval_result = _evaluate(solution, problem)
        if eval_result.passed:
            return solution

        # Reflect on the error
        reflection = model.generate(
            f"Problem: {problem}\n"
            f"My solution: {solution}\n"
            f"Error: {eval_result.error}\n"
            f"What did I do wrong? How to improve?"
        )
        reflections.append(reflection)

    return solution
```

---

## The PEV Architecture | PEV 架构

This is the **most powerful** pattern for complex tasks:

```
                     Problem
                        │
                   ┌────┴────┐
                   │Planner  │  # Decompose, plan, synthesize
                   │  规划器  │
                   └────┬────┘
                        │ Subtasks [s1, s2, s3]
              ┌─────────┼─────────┐
              ▼         ▼         ▼
        ┌─────────┐ ┌──────┐ ┌──────┐
        │Exec-A   │ │Exec-B│ │Exec-C│  # Parallel execution
        │子任务A  │ │子任务B│ │子任务C│
        └────┬────┘ └──┬───┘ └──┬───┘
             │         │        │
             ▼         ▼        ▼
        ┌──────────────────────────┐
        │       Verifier           │  # Check each output
        │        验证器             │  # Retry if failed
        │   ✗ A: pass              │
        │   ✗ B: pass              │
        │   ✓ C: fail → retry     │
        └─────────┬────────────────┘
                  │ Verified results
                  ▼
            Final Answer
```

```python
class PEVSystem:
    def __init__(self, planner, executor, verifier, max_retries=2):
        self.planner   = planner
        self.executor  = executor
        self.verifier  = verifier
        self.max_retries = max_retries

    def solve(self, problem):
        # Step 1: Decompose
        plan = self.planner.generate(
            f"Problem: {problem}\n"
            f"Break into independent subtasks with clear I/O."
        )
        subtasks = self._parse_plan(plan)

        results = {}
        for step in subtasks:
            result = self.executor.generate(step.prompt)

            # Step 2: Verify (with retry loop)
            for retry in range(self.max_retries):
                verdict = self.verifier.generate(
                    f"Task: {step.desc}\nResult: {result}\n"
                    f"Correct and complete? YES/NO with reasons."
                )
                if verdict.startswith("YES"):
                    break
                result = self.executor.generate(
                    f"Task: {step.desc}\nPrevious: {result}\n"
                    f"Feedback: {verdict}\nFix and regenerate."
                )
            results[step.id] = result

        # Step 3: Synthesize
        return self.planner.generate(
            f"Problem: {problem}\nResults: {results}\n"
            f"Synthesize into a final answer."
        )
```

---

## Why Multi-Agent > Single Agent? | 为什么更强？

### 1. Information Diversity | 信息多样性

Different prompts trigger different "reasoning paths" in the model. Forcing Agent A to support X and Agent B to oppose X covers perspectives a single agent would never explore simultaneously.

### 2. Error Cancellation | 错误消除

If two independent agents each have a 10% hallucination rate, the chance they hallucinate the *same* thing is < 1%. The Critic/Verifier catches errors the Generator missed.

### 3. Role Assignment Breaks "Agreement Bias" | 角色扮演打破"和稀泥"

LLMs naturally say "both sides have merit." Forcing roles ("you're the defense attorney" / "you're the prosecutor") breaks this tendency.

---

## When to Use | 何时使用

| Scenario | Recommended | Why |
|----------|:---:|-----|
| Factual queries | NO | Direct lookup is enough |
| Creative writing | CASE-DEPENDS | Diversity may hurt coherence |
| Complex coding | YES | PEV architecture excels |
| Math proofs | YES | Verifier can use code execution |
| Long docs | YES | Parallel section analysis |

---

## Debate Architecture | 辩论架构

```
        Topic
          │
      ┌───┴───┐
  Pro-Agent  Con-Agent     # Opposing perspectives
  (支持方)    (反对方)
      │        │
  Round 1   Round 1
  Arguments Arguments
      │        │
  ┌───┴────────┴───┐
  │  Cross-Examination (N rounds)  │
  │  Rebuttal + Counter-Rebuttal   │
  └───┬─────────────┬───┘
      │             │
  ┌───┴─────────────┴───┐
  │        Judge         │  # Third-party evaluation
  │       裁决者          │
  └──────────┬────────────┘
          Verdict
```

---

## Further Reading

- [Reflexion](https://arxiv.org/abs/2303.11366)
- [Chain of Verification (CoVe)](https://arxiv.org/abs/2309.11495)
- [LLM Debates](https://arxiv.org/abs/2305.14333)
- [AutoGen](https://microsoft.github.io/autogen/)
- [MetaGPT](https://arxiv.org/abs/2308.00352)

---

_Prev: [Day 04 - Test-Time Compute](04-test-time-compute.md)_
_上一篇: [Day 04 - 推理时计算](04-test-time-compute.md)_

_This completes the first batch. More tutorials coming daily!_
_第一期完成。每日持续更新！_
