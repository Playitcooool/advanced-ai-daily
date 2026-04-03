# 第四天：测试时计算 / 推理时扩展

## 目录

- 概述
- 为什么测试时计算很重要
- 策略一：N选优采样（Best-of-N）
- 策略二：自我纠正 / 反思（Self-Correction / Reflexion）
- 策略三：思维树（Tree of Thoughts）
- 策略四：验证器引导解码（Verifier-Guided Decoding）
- 对比表
- 代码示例
- ASCII 流程图
- 延伸阅读
- 练习题

---

## 概述

**测试时计算**（**Test-Time Compute**，也称**推理时扩展 Inference-Time Scaling**）是一种在**生成阶段**而非训练阶段投入额外计算以提升大语言模型输出质量的策略。与其依赖单次前向传播生成结果，我们可以为采样多个答案、批判和精炼回答、搜索解空间或使用学习到的验证器引导生成过程分配额外算力。

**核心洞察：** 更多的训练成本高昂且往往存在收益递减。而测试时计算更加灵活、针对具体任务，且通常能在推理任务上带来显著的性能提升。

---

## 为什么测试时计算很重要

| 维度 | 预训练 | 微调 | 测试时计算 |
|------|--------|------|------------|
| 成本 | 极高 | 高 | 随每次查询而变化 |
| 灵活性 | 权重固定 | 每个任务固定 | 每次查询可调 |
| 推理增益 | 通用 | 特定领域 | 特定任务 |
| 关键技术 | 扩大数据/模型 | SFT, DPO | N选优, 思维树, 反思 |
| 延迟影响 | 无 | 无 | 更高（需权衡） |

**为什么有效：** 语言模型不仅仅是文本生成器——它们是概率推理器。通过在测试时投入更多算力，我们可以：

1. **采样多样性：** 生成多个候选解并选择最优。
2. **深思熟虑：** 让模型通过链接推理步骤"思考更久"。
3. **验证：** 使用独立流程评估和排序输出。
4. **搜索：** 探索可能的推理路径所构成的树或图。

---

## 策略一：N选优采样（Best-of-N）

最简单且最广泛使用的测试时计算策略。

### 工作原理

1. 给定一个提示，生成 N 个独立的回答（通常使用较高的 temperature）。
2. 使用奖励模型、验证器或启发式规则为每个回答打分。
3. 选择得分最高的回答作为最终输出。

### ASCII 流程图

```
        ┌─────────────┐
        │    提示词    │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  采样 N 个回答│  （temperature > 0）
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  逐一打分    │  （奖励模型 / 验证器）
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  选择最高分   │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │    输出      │
        └─────────────┘
```

### 数学直觉

给定语言模型 P 和奖励模型 R：

```
输出 = argmax_{y_i ~ P(y|x)} R(y_i, x)，其中 i = 1..N
```

随着 N 的增加，找到高奖励输出的概率也随之增加。然而，**收益递减规律**适用——超过某个 N 值后，额外样本很少能显著改善质量。

### 实践建议

- 使用 **temperature 0.7–1.0** 以鼓励多样性。
- N = 5–25 对大多数应用场景来说较为实用。
- 超过 N=64 时，边际收益迅速减少。
- 奖励模型可以是从简单的正则表达式检查器到训练好的偏好模型的任何东西。

---

## 策略二：自我纠正 / 反思（Self-Correction / Reflexion）

与独立采样不同，模型首先生成草稿、自我批判然后精炼——模仿人类写作和编辑的过程。

### 工作原理

1. **生成：** 产出初步答案。
2. **反思：** 让同一个（或另一个）模型批判该答案——识别错误、遗漏或改进点。
3. **修订：** 根据批判生成改进后的答案。
4. 可选择性地重复步骤 2–3，进行多轮迭代。

### ASCII 流程图

```
  ┌─────────────────────────────┐
  │          提示词              │
  └──────────────┬──────────────┘
                 │
       ┌─────────▼──────────┐
       │     生成草稿        │
       └─────────┬──────────┘
                 │
       ┌─────────▼──────────┐
       │   自我反思：        │
       │  "哪里有问题？"     │
       └─────────┬──────────┘
                 │
       ┌─────────▼──────────┐
       │     有错误吗？       │──无──►  最终输出
       └─────────┬──────────┘
                有
       ┌─────────▼──────────┐
       │     修订答案        │
       └─────────┬──────────┘
                 │  （循环回去，最多 K 轮）
                 │
        ┌────────┴────────┐
        ▼                 │
  （回到反思步骤）          │
```

### 关键变体

| 变体 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| 单轮反思 | 一次批判，一次修订 | 快速、简单 | 改进有限 |
| 多轮反思 | 迭代 K 次 | 更深入的纠正 | 计算成本更高 |
| 外部批判者 | 使用更强的模型作为批判者 | 批判质量更高 | API 成本、延迟 |
| 反思标记 | 训练特殊的反思标记 | 集成式、高效 | 需要微调 |

### 常见提示模式

```
你是一个严谨的问题解决者。

步骤1：对以下问题给出你认为最佳的答案。
问题：{问题}
回答：{初步回答}

步骤2：仔细审查你的答案。推理、计算或逻辑中是否有任何错误？找出具体的问题。
批判：{自我批判}

步骤3：根据你的批判，提供一个修订后改进的答案。
修订回答：{修订回答}
```

---

## 策略三：思维树（Tree of Thoughts，ToT）

思维树通过显式地在可能推理步骤构成的**树**上搜索，扩展了思维链方法。

### 工作原理

1. **思维分解：** 将问题分解为连续的思维步骤。
2. **思维生成：** 在每个步骤，提出 K 个候选的下一思维。
3. **状态评估：** 使用启发式规则或学习到的评估器为每个部分解（状态）打分。
4. **搜索：** 使用广度优先搜索（BFS）或深度优先搜索（DFS）探索最有希望的分支。
5. **回溯：** 如果一个分支走入死胡同，则返回并探索替代方案。

### ASCII 流程图

```
                      根节点（问题）
                          │
          ┌───────────────┼───────────────┐
          │               │               │
       思维1a          思维1b          思维1c
          │               │               │
      ┌───┼───┐          ...             ...
      │   │   │
   2a  2b  2c
   │
 ┌─┼─┐
3a 3b 3c   ... 评估叶节点，选择最佳路径
```

### 搜索策略

| 策略 | 工作原理 | 适用场景 |
|------|---------|----------|
| 广度优先搜索（BFS） | 逐层探索所有分支 | 需要广泛覆盖；浅层树 |
| 深度优先搜索（DFS） | 先深入一个分支 | 深度推理问题 |
| 束搜索（Beam Search） | 每层仅保留前 K 个状态 | 探索与成本之间的平衡 |

### Python 骨架代码

```python
class TreeNode:
    def __init__(self, text, parent=None, evaluation=0.0):
        self.text = text
        self.parent = parent
        self.children = []
        self.evaluation = evaluation
        self.value = None  # 叶节点设置

class TreeOfThoughts:
    def __init__(self, model, evaluator, k=5, beam_width=3, max_depth=10):
        self.model = model      # 用于生成思维的 LLM
        self.evaluator = evaluator  # 评分函数
        self.k = k              # 每步的候选数
        self.beam_width = beam_width
        self.max_depth = max_depth

    def generate_candidates(self, prompt):
        """生成 K 个候选的下一思维。"""
        responses = self.model.generate(prompt, n=self.k)
        return responses

    def solve(self, problem):
        root = TreeNode(text=problem)
        beam = [root]
        
        for depth in range(self.max_depth):
            next_beam = []
            for node in beam:
                candidates = self.generate_candidates(self._build_prompt(node))
                for cand_text in candidates:
                    child = TreeNode(text=cand_text, parent=node)
                    child.evaluation = self.evaluator(cand_text, problem)
                    node.children.append(child)
                    next_beam.append(child)
            
            beam = sorted(next_beam, key=lambda n: n.evaluation, reverse=True)
            beam = beam[:self.beam_width]
        
        # 叶节点评估
        best = sorted(beam, key=lambda n: n.evaluation, reverse=True)[0]
        return self._extract_trace(best)

    def _build_prompt(self, node):
        trace = self._extract_trace(node)
        return f"求解：{trace}\n下一步思考："

    def _extract_trace(self, node):
        if node.parent is None:
            return node.text
        return self._extract_trace(node.parent) + "\n" + node.text
```

---

## 策略四：验证器引导解码（Verifier-Guided Decoding）

使用训练好的验证器（奖励模型）为每一步——或至少是关键决策点——的生成进行评分和引导。

### 工作原理

1. 在标注数据上训练一个**验证器**（或奖励模型），将解评分为正确/错误或给出置信度分数。
2. 在生成过程中，采样多个候选回答。
3. 将每个候选通过验证器运行。
4. 基于验证器分数进行选择或重新排序。

### 不同粒度的验证

| 粒度 | 描述 | 示例 | 计算成本 |
|------|------|------|----------|
| 输出级 | 仅验证最终答案 | 数学题答案检查 | 低 |
| 步骤级 | 验证每个推理步骤 | 证明中的每个方程 | 中 |
| 词元级 | 为单个词元评分 | 每个生成步骤都使用验证器 | 高 |

### 代码示例

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Verifier:
    """用于评估解的奖励模型评分器。"""
    
    def __init__(self, model_name="OpenAI/summarize_from_feedback"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
    
    def score(self, question, answer):
        """为给定问题的回答打分（0-1）。"""
        text = f"问：{question} 答：{answer}"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            score = torch.sigmoid(outputs.logits).item()
        return score

def verifier_rerank(question, model, verifier, n=10):
    """生成 N 个回答并按验证器分数重新排序。"""
    answers = model.generate(question, n=n, temperature=0.8)
    
    scored = []
    for ans in answers:
        s = verifier.score(question, ans)
        scored.append((s, ans))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], scored  # 最佳答案，所有评分回答
```

---

## 对比表

| 策略 | 计算成本 | 最适合 | 核心优势 | 核心劣势 |
|------|---------|--------|----------|----------|
| N选优采样 | 低–中 | 通用问答、代码生成 | 简单、可靠 | 收益递减 |
| 自我纠正 | 中 | 推理、写作 | 模仿人类编辑过程 | 可能强化模型偏差 |
| 思维树 | 中–高 | 复杂规划、数学 | 结构化探索 | 成本高，实现复杂 |
| 验证器引导 | 中–高 | 数学、代码、逻辑 | 客观评分 | 需要训练验证器 |

---

## 代码示例

### 完整的 N选优 流水线

```python
class BestOfN:
    def __init__(self, generator, scorer, n=10):
        self.generator = generator
        self.scorer = scorer
        self.n = n
    
    def solve(self, question):
        """生成 n 个回答，返回最佳。"""
        results = []
        for i in range(self.n):
            answer = self.generator(question, temperature=0.7)
            score = self.scorer(question, answer)
            results.append({"answer": answer, "score": score, "index": i})
        
        # 按分数降序排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    def solve_detailed(self, question):
        """求解并返回详细统计信息。"""
        results = self.solve(question)
        return {
            "best": results[0],
            "best_score": results[0]["score"],
            "avg_score": sum(r["score"] for r in results) / len(results),
            "score_spread": results[0]["score"] - results[-1]["score"],
            "all_results": results
        }
```

### 反思循环

```python
def reflexion_loop(generator, question, max_rounds=3):
    """运行反思循环：生成、批判、修订。"""
    current_answer = generator(question)
    history = []
    
    for round_num in range(max_rounds):
        # 生成反思
        critique_prompt = (
            f"审查下面的解法，找出任何错误或弱点。\n"
            f"问题：{question}\n"
            f"解法：{current_answer}\n"
            f"审查： "
        )
        critique = generator(critique_prompt)
        
        # 检查是否需要修订
        revision_prompt = (
            f"根据以下批判，改进你的解法。\n"
            f"问题：{question}\n"
            f"原始解法：{current_answer}\n"
            f"批判：{critique}\n"
            f"修订解法： "
        )
        revised = generator(revision_prompt)
        
        history.append({
            "round": round_num + 1,
            "answer": current_answer,
            "critique": critique,
        })
        
        current_answer = revised
    
    return current_answer, history
```

---

## 延伸阅读

1. **《推理扩展定律》（Inference Scaling Laws）** — Wang 等 (2024) — https://arxiv.org/abs/2403.05530
   关于测试时计算如何随模型性能扩展的基础性论文。

2. **《思维树：用大语言模型进行有意识的问题求解》（Tree of Thoughts）** — Yao 等 (2023) — https://arxiv.org/abs/2305.10601
   引入在推理路径上进行系统搜索的原始思维树论文。

3. **《自我精炼：通过自我反馈进行迭代精炼》（Self-Refine）** — Madaan 等 (2023) — https://arxiv.org/abs/2303.17651
   展示了模型如何通过自我批判来精炼自身输出。

4. **《ReST：语言模型的强化自训练》（ReST）** — Gulcehre 等 (2023) — https://arxiv.org/abs/2308.08998
   将自生成数据与测试时扩展结合，以提升推理能力。

5. **《让我们一步步验证》（Let's Verify Step by Step）** — Lightman 等 (2023) — https://arxiv.org/abs/2305.20050
   证明了过程级验证优于结果级验证。

6. **Anthropic 的宪法式 AI（Constitutional AI）** — https://arxiv.org/abs/2212.08073
   使用受原则引导的自我批判来实现对齐。

---

## 练习题

1. **实现 N选优：** 编写一个脚本，对同一道数学题生成 20 个回答，并使用基于正则表达式的简单验证器（检查最终数字是否正确）进行排序。

2. **构建反思循环：** 创建一个用于代码生成的两轮反思系统——首先生成代码，然后批判其中的 bug，然后重新生成。

3. **4x4 数独的思维树：** 为 4x4 数独谜题实现一个思维树求解器。使用计算约束违反次数的启发式评估器。

4. **消融实验：** 在 10 道推理题上比较单次生成、N选优-5 和 N选优-20。绘制准确度曲线。

5. **验证器设计：** 在 GSM8K 解（正确/错误）上训练一个简单的二分类器，并用它来重新排序模型输出。

---

*第四天教程 — 高级 AI 日报*
