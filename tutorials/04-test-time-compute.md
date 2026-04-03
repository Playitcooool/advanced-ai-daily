# Day 04: Test-Time Compute / Inference-Time Scaling
# 第 04 天: 推理时计算扩展

> **Date**: 2026-04-03 | **Difficulty**: Advanced | **Category**: Inference Strategy 推理策略

> **Watch the animation**: ![Test-Time Compute Animation](../gifs/04-test-time-compute.gif)

---

## One-Line Summary

Instead of spending 100x cost on training to make a model better, let the model "think longer" at inference -- generate multiple candidates, self-evaluate, iteratively refine. **Same model, more compute budget = significantly better outputs.**

与其花 100 倍训练成本让模型变强，不如在推理时让模型多"思考"。**同一个模型，给它更多计算时间，性能显著提升。**

---

## Two Scaling Laws | 两种扩展法则

### Training-Time Scaling | 训练时扩展

```
More data + More params + More FLOPs → Better model

│  Performance
│  /
│ /
│/
└───────────  Training FLOPs  →
```

Dominated AI development for 3 years (GPT-3 → GPT-4 → Claude 3). But **diminishing returns** are kicking in.

过去三年主导 AI 发展，但边际效益递减。

### Test-Time Scaling | 推理时扩展

```
More sampling + More thinking + More search → Better output

│  Performance
│  /
│ /
│/
└───────────  Inference token budget  →
```

Key insight: **The model already knows many "right answers," it just needs to be pushed to find them.**

模型已经知道很多"正确答案"，只是没被"逼"到那个点。

---

## Core Methods | 核心方法

### Method Overview | 方法总览

```
            Test-Time Compute 方法分类
            ════════════════════════
                    │
      ┌─────────────┼─────────────┐
      │             │             │
  Sample &      Iterative     Tree/Graph
  Select        Refinement     Search
  采样后选择     迭代修正        树/图搜索
      │             │             │
  Best-of-N    Self-Correct    Tree of Thoughts
  Majority      Self-Critique  Graph of Thoughts
  Verifier      Reflexion      MCTS
```

### 1. Best-of-N

```python
def best_of_n(model, prompt, reward_fn, n=64):
    """Generate N candidates, pick the best via a scoring function."""
    candidates = []
    for _ in range(n):
        response = model.generate(prompt, temperature=0.7)
        score = reward_fn(response)
        candidates.append((score, response))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0]  # best score, best response
```

**Why it works**: LLM generation is stochastic. Same prompt × 10 runs might yield 3 good, 5 medium, 2 bad. If you only run once (greedy), you might hit medium. Run 10 and pick, you're more likely to find one of the 3 good ones.

### 2. Self-Correction / Reflexion

```python
def reflexion(model, problem, max_rounds=3):
    """Generate → Evaluate → Reflect → Retry"""
    reflections = []  # learned lessons carry across attempts

    for attempt in range(max_rounds + 1):
        # Build prompt with past reflections
        prompt = f"Problem: {problem}\n"
        for r in reflections:
            prompt += f"Lesson learned: {r}\n"
        prompt += "\nSolution: "

        solution = model.generate(prompt)
        evaluation = _evaluate(solution, problem)
        if evaluation.passed:
            return solution

        # Reflect: What went wrong? What should I do differently?
        reflection = model.generate(
            f"Problem: {problem}\n"
            f"My solution: {solution}\n"
            f"Error: {evaluation.error}\n"
            f"Reflection: Where did I go wrong? How to fix next time?"
        )
        reflections.append(reflection)

    return solution
```

Key difference from naive retry: the **Reflection** (经验教训) carries forward. The model learns from its mistakes.

### 3. Tree of Thoughts

Structured search over reasoning paths:

```
                    Problem
                      │
              ┌───────┼───────┐
          Thought_A Thought_B Thought_C   (branch K=3)
          │         │         │
      ┌───┼   ┌─────┼     ┌──┘
     AA  AB  BA    BB    CA
     │   │
   EVAL EVAL   ← Score leaves with verifier
   (Validator)  (验证器)
```

```python
def tree_of_thoughts(model, root_prompt, K=3, depth=4, beam_width=2):
    """
    Branch: generate K candidate thoughts per node
    Evaluate: score each candidate
    Select: keep top-B nodes
    Repeat until depth D or solution found
    """
    nodes = [(1.0, "")]  # (score, accumulated_text)
    for step in range(depth):
        candidates = []
        for score, text in nodes:
            for _ in range(K):
                thought = model.generate(
                    f"{root_prompt}\nCurrent reasoning: {text}\n"
                    f"Next thinking step:"
                )
                eval_score = _evaluate_thought(model, root_prompt, thought)
                candidates.append((score * eval_score, text + "\n" + thought))
        candidates.sort(key=lambda x: x[0], reverse=True)
        nodes = candidates[:beam_width]
    return nodes[0][1]
```

### 4. Verifier-Guided Search (AlphaCode Style)

For problems with **objective verification** (coding, math):

```python
def verifier_search(model, problem, test_cases, n=128):
    """
    Generate N candidates → verify against test cases → return correct ones.
    AlphaCode used this to reach top 50% on Codeforces.
    """
    for _ in range(n):
        code = model.generate(f"Problem: {problem}\nWrite Python solution:")
        if all(_run_test(code, inp, exp) for inp, exp in test_cases):
            return code  # First solution passing all tests
    return _best_partial(model, problem, test_cases)
```

---

## Compute Budget vs Performance | 计算预算 vs 性能

Experiments (DeepSeek, OpenAI o1 series):

| Inference Token Budget | Hard Problem Accuracy | Method |
|-----------------------|:-:|------|
| 1x (standard greedy) | ~30% | Single generation |
| 10x | ~45% | Best-of-N |
| 100x | ~60% | CoT + Search |
| 1000x | ~72% | MCTS deep search |
| 10000x | ~75% | Diminishing returns |

**Key**: easy problems need no extra compute; hard problems need budget to unlock the model's latent capability.

---

## Limitations | 局限性

1. **Not a cure-all**: if the model's knowledge is wrong/missing, more compute doesn't help
2. **Costly**: 100x compute = 100x GPU time and cost
3. **Evaluation bottleneck**: without objective validators, picking the best output is subjective
4. **Latency**: users won't wait 30 seconds for a response

---

## Further Reading

- [AlphaCode](https://arxiv.org/abs/2203.07814)
- [Chain of Thought](https://arxiv.org/abs/2201.11903)
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
- [OpenAI o1](https://openai.com/index/learning-to-reason-with-llms/)
- [STaR: Self-Taught Reasoner](https://arxiv.org/abs/2203.14465)

---

_Prev: [Day 03 - Speculative Decoding](03-speculative-decoding.md)  |  Next: [Day 05 - Multi-Agent Reflection](05-multi-agent-reflection.md)_
_上一篇: [Day 03 - 投机解码](03-speculative-decoding.md)  |  下一篇: [Day 05 - 多智能体反思](05-multi-agent-reflection.md)_
