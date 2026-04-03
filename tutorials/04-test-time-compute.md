# Test-Time Compute / 推理时计算扩展

> **日期**: 2026-04-03 | **难度**: 进阶 | **类别**: 推理策略 / 自我改进

---

## 一句话总结

与其花 100x 成本在训练时（pretrain/SFT/RL）让模型变强，不如在推理时让模型多 "思考" -- 生成多个候选、自己评估、迭代修正。**同样的模型，给它更多计算时间，性能可以显著提升。**

---

## 两种 Scaling 范式

### 训练时扩展 (Training-Time Scaling)

```
更多的数据 + 更多的参数量 + 更多的训练步数 → 更好的模型

    │  性能
    │  /
    │ /
    │/
    └─────────────── 训练 FLOPs
         →
```

这在过去三年主导了 AI 发展（GPT-3 → GPT-4 → Claude 3）。但边际效益在递减。

### 推理时扩展 (Test-Time Compute Scaling)

```
更多的采样 + 更多的思考步骤 + 更多的自我反思 → 更好的输出

    │  性能
    │  /
    │ /
    │/
    └─────────────── 推理 token 预算
         →
```

**关键洞察: 模型已经知道很多 "正确答案"，只是没被 "逼" 到那个点。** 给它足够的推理预算，它自己能找到。

---

## 核心方法分类

### 流程图

```
                    Test-Time Compute 方法分类
                    ════════════════════════

                            │
            ┌───────────────┼───────────────┐
            │               │               │
     采样后选择        迭代修正         树/图搜索
     (Sample &        (Iterative       (Tree/Graph
      Select)          Refinement)       Search)
            │               │               │
     ┌──────┴──────┐  ┌────┴────┐    ┌──────┴────────┐
     │ Best of N   │  │ Self    │    │  Tree of      │
     │ 采样 N 条   │  │ Refine  │    │  Thoughts     │
     │ 挑最好的    │  │ 自己改   │    │  树搜索       │
     └─────────────┘  └─────────┘    └───────────────┘
     Majority Vote   Self-Critique   Graph of Thoughts
     投票选多数      自批评+修正     图搜索
                                          │
                                    ┌─────┴──────┐
                                    │  MCTS      │
                                    │  蒙特卡洛   │
                                    │  树搜索     │
                                    └────────────┘
```

### 1. Best-of-N (采样后选择)

```python
def best_of_n(model, prompt, reward_fn, n=64):
    """
    生成 N 个候选答案，用奖励/评分函数选最好的。
    
    深度:
    - N 越大，找到好答案的概率越高
    - 但 N 越大，边际收益递减
    - 关键瓶颈: 怎么评估哪个答案更好
    """
    candidates = []
    for _ in range(n):
        response = model.generate(prompt, temperature=0.7)
        score = reward_fn(response)
        candidates.append((score, response))
    
    # 排序返回最好的
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0]  # (best_score, best_response)
```

**为什么 Best-of-N 有效？**

LLM 的生成是随机采样的。同一个 prompt 生成 10 次，可能有 3 次是好的、5 次中等、2 次差。如果只生成 1 次（greedy），你可能正好采样到中等的那个。生成 10 次并选择，你就更可能找到那 3 个好的。

### 2. Self-Correction / 迭代修正

```python
def self_correct(model, prompt, max_rounds=3):
    """
    让模型自己检查、批评、修正答案。
    
    深度 - 为什么有时候有效、有时候无效:
    - 当模型 "知道" 正确答案（知识存在但不确定性高）时，修正有效
    - 当模型 "不知道" 正确答案（知识缺失）时，修正无用，甚至变差
    - 关键: 需要一个靠谱的自我评估 prompt
    """
    # 第一轮: 初始回答
    answer = model.generate(prompt + "\n请给出你的答案:")
    
    for round in range(max_rounds):
        # 自我评估
        critique = model.generate(
            f"问题: {prompt}\n"
            f"我的回答: {answer}\n"
            f"这个回答正确吗？有哪些错误？请仔细分析。"
        )
        
        # 如果自我评估认为无误，停止
        if "正确" in critique or "无误" in critique:
            break
        
        # 修正
        answer = model.generate(
            f"问题: {prompt}\n"
            f"我的回答: {answer}\n"
            f"批评: {critique}\n"
            f"请修正你的回答。"
        )
    
    return answer
```

### 3. Tree of Thoughts / 思维树搜索

这比单纯的 Self-Correction 更结构化：

```
                    Problem
                      │
              ┌───────┼───────┐
          Thought_A Thought_B Thought_C
          │         │         │
      ┌───┼───┐  ┌──┼──┐  ┌──┘
     AA  AB  AC  BA  BB   CA...
     │   │   │   │   │
    AAA AAB ... BAA BAB
     │   │
   EVAL EVAL    ← 在叶子节点用评估器打分
   (验证器) (验证器)
```

**算法步骤:**
1. **分支**: 对每个当前节点，生成 K 个候选思维
2. **评估**: 每个候选用启发式或验证器打分
3. **选择**: 保留 Top-B 个最好的节点
4. **继续**: 重复直到达到深度 D 或找到解

```python
def tree_of_thoughts(model, root_prompt, branch_factor=3, 
                     depth=3, beam_width=2):
    """
    beam search over "thoughts" -- LLM generated reasoning steps.
    
    Args:
        branch_factor K: 每个节点生成多少分支
        depth D: 树有多深
        beam_width B: 保留多少个最好的节点
    """
    # 根节点
    nodes = [(1.0, "")]  # (score, text)
    
    for step in range(depth):
        candidates = []
        for score, text in nodes:
            # 分支: 生成 K 个候选思维
            for k in range(branch_factor):
                new_thought = model.generate(
                    f"{root_prompt}\n"
                    f"当前思路: {text}\n"
                    f"请给出下一步思考:"
                )
                # 评估: 用模型自我评估
                eval_score = _evaluate_thought(model, root_prompt, new_thought)
                candidates.append((score * eval_score, text + "\n" + new_thought))
        
        # 选择: 保留 top-B
        candidates.sort(key=lambda x: x[0], reverse=True)
        nodes = candidates[:beam_width]
        
        if nodes[0][0] > 0.95:  # 足够好了
            break
    
    return nodes[0][1]  # 最好的完整思路
```

### 4. 验证器辅助搜索 (AlphaCode 风格)

对于 **可以用程序验证** 的问题（编程、数学）:

```python
def verifier_guided_search(model, problem, test_cases, n_candidates=128):
    """
    生成大量候选方案，用测试用例验证正确性。
    AlphaCode 用这个方法在 Codeforces 达到人类中上水平。
    
    关键: 这不是让模型 "自己想谁对" -- 而是用代码执行器
    做 objective 验证，完全不依赖模型的主观判断。
    """
    correct_solutions = []
    
    for _ in range(n_candidates):
        code = model.generate(
            f"Problem: {problem}\n"
            f"Write Python code to solve this."
        )
        
        # 用测试用例进行客观验证
        all_pass = True
        for inp, expected in test_cases:
            try:
                result = _exec_code(code, inp)
                if result != expected:
                    all_pass = False
                    break
            except Exception:
                all_pass = False
                break
        
        if all_pass:
            # 通过所有测试！
            correct_solutions.append(code)
    
    if correct_solutions:
        # 选最简洁的那个
        return min(correct_solutions, key=len)
    
    # 没有全部通过的，选通过测试最多的
    return _best_by_partial(model, problem, test_cases)
```

---

## 计算预算 vs 性能

实验表明 (DeepSeek, OpenAI o1 系列):

```
推理 token 预算 (log scale)    │  难题正确率
     1x (标准生成)             │  30%
    10x (Best-of-N)            │  45%
   100x (CoT + 搜索)           │  60%
  1000x (MCTS 深度搜索)        │  70%
 10000x                        │  75%  ← 边际效益递减
```

**关键结论**: 对简单问题不需要额外计算（一次就对），对难题需要大量计算预算才有效。

---

## Test-Time Compute 的局限

1. **不是万能药**: 如果模型的知识本身就是错的/缺的，再多计算也没用
2. **成本高昂**: 100 倍的计算意味着 100 倍的 GPU 时间和费用
3. **评估难题**: 如果问题没有客观验证器（开放性问题、创意写作），选择哪个候选就是主观的
4. **延迟不可接受**: 用户通常等不了 30 秒出一个回答

---

## 扩展阅读

- [AlphaCode](https://arxiv.org/abs/2203.07814) -- 用大规模采样+验证器在编程竞赛中取得突破
- [Chain of Thought Prompting](https://arxiv.org/abs/2201.11903) -- 让模型逐步思考
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601) -- 结构化思维搜索
- [OpenAI o1: Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/) -- RL 训练推理时扩展
- [STaR: Self-Taught Reasoner](https://arxiv.org/abs/2203.14465) -- 自我训练推理能力

---

_上一个: [Day 3 - Speculative Decoding](03-speculative-decoding.md) | 下一个: [Day 5 - Multi-Agent Reflection](05-multi-agent-reflection.md)_
