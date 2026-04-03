# Day 04: Test-Time Compute / Inference-Time Scaling

## Table of Contents

- Overview
- Why Test-Time Compute Matters
- Strategy 1: Best-of-N Sampling
- Strategy 2: Self-Correction / Reflexion
- Strategy 3: Tree of Thoughts (ToT)
- Strategy 4: Verifier-Guided Decoding
- Comparison Table
- Code Examples
- ASCII Flowcharts
- Further Reading
- Exercises

---

## Overview

**Test-time compute** (also called **inference-time scaling**) refers to the strategy of spending additional computation during the *generation* phase — rather than during training — to improve the quality of a language model's output. Instead of relying solely on a single forward pass, we allocate extra compute to sample multiple answers, critique and refine them, search over solution spaces, or use learned verifiers to guide generation.

**Key insight:** More training is expensive and often has diminishing returns. More compute at test time is flexible, task-specific, and often yields dramatic improvements on reasoning tasks.

---

## Why Test-Time Compute Matters

| Dimension | Pre-Training | Fine-Tuning | Test-Time Compute |
|-----------|-------------|-------------|--------------------|
| Cost | Extremely high | High | Scales with each query |
| Flexibility | Fixed weights | Fixed per task | Adjustable per query |
| Reasoning gains | General | Domain-specific | Task-specific |
| Key techniques | Scale data/model | SFT, DPO | Best-of-N, ToT, Reflexion |
| Latency impact | None | None | Higher (trade-off) |

**Why it works:** Language models are not just text generators — they are probabilistic reasoners. By spending more compute at test time, we can:

1. **Sample diversity:** Generate multiple candidate solutions and pick the best.
2. **Deliberation:** Let the model "think longer" by chaining reasoning steps.
3. **Verification:** Use a separate process to evaluate and rank outputs.
4. **Search:** Explore trees or graphs of possible reasoning paths.

---

## Strategy 1: Best-of-N Sampling

The simplest—and most widely used—test-time compute strategy.

### How It Works

1. Given a prompt, generate N independent responses (often with higher temperature).
2. Score each response using a reward model, verifier, or heuristic.
3. Select the highest-scoring response as the final output.

### ASCII Flowchart

```
        ┌─────────────┐
        │   Prompt    │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  Sample N x │  (temperature > 0)
        │  Responses  │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  Score Each │  (reward model / verifier)
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  Select Max │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │   Output    │
        └─────────────┘
```

### Mathematical Intuition

Given a language model P and a reward model R:

```
output = argmax_{y_i ~ P(y|x)} R(y_i, x) for i = 1..N
```

As N increases, the probability of finding a high-reward output increases. However, **diminishing returns** apply — after a certain N, additional samples rarely improve quality significantly.

### Practical Tips

- Use **temperature 0.7–1.0** to encourage diversity.
- N = 5–25 is practical for most applications.
- Beyond N=64, marginal gains shrink rapidly.
- A reward model can be anything from a simple regex checker to a trained preference model.

---

## Strategy 2: Self-Correction / Reflexion

Instead of sampling independently, the model generates a draft, critiques it, and refines it — mimicking the human process of writing and editing.

### How It Works

1. **Generate:** Produce an initial answer.
2. **Reflect:** Have the same (or another) model critique the answer — identify errors, gaps, or improvements.
3. **Revise:** Generate an improved answer based on the critique.
4. Optionally repeat steps 2–3 for multiple rounds.

### ASCII Flowchart

```
  ┌─────────────────────────────┐
  │           Prompt            │
  └──────────────┬──────────────┘
                 │
       ┌─────────▼──────────┐
       │   Generate Draft   │
       └─────────┬──────────┘
                 │
       ┌─────────▼──────────┐
       │  Self-Reflection:  │
       │  "What's wrong?"   │
       └─────────┬──────────┘
                 │
       ┌─────────▼──────────┐
       │       Error?       │────No──►  Final Output
       └─────────┬──────────┘
                Yes
       ┌─────────▼──────────┐
       │   Revise Answer    │
       └─────────┬──────────┘
                 │  (loop back, max K rounds)
                 │
        ┌────────┴────────┐
        ▼                 │
  (back to Reflect)       │
```

### Key Variants

| Variant | Description | Pros | Cons |
|---------|-------------|------|------|
| Single-pass reflection | One critique, one revision | Fast, simple | Limited improvement |
| Multi-round Reflexion | Iterate K times | Deeper correction | Higher compute cost |
| External critic | Use a stronger model as critic | Better critique | API cost, latency |
| Reflection tokens | Train special reflection tokens | Integrated, efficient | Requires fine-tuning |

### Common Prompt Pattern

```
You are a careful problem solver.

Step 1: Provide your best answer to the following question.
Question: {question}
Answer: {initial_answer}

Step 2: Review your answer carefully. Are there any errors in reasoning, 
calculation, or logic? Identify specific issues.
Critique: {self_critique}

Step 3: Based on your critique, provide a revised and improved answer.
Revised Answer: {revised_answer}
```

---

## Strategy 3: Tree of Thoughts (ToT)

Tree of Thoughts extends chain-of-thought by explicitly searching over a **tree** of possible reasoning steps.

### How It Works

1. **Thought decomposition:** Break the problem into sequential thought steps.
2. **Thought generation:** At each step, propose K candidate next thoughts.
3. **State evaluation:** Score each partial solution (state) using a heuristic or learned evaluator.
4. **Search:** Use BFS (breadth-first search) or DFS (depth-first search) to explore the most promising branches.
5. **Backtrack:** If a branch leads to a dead end, return and explore alternatives.

### ASCII Flowchart

```
                      Root (Problem)
                          │
          ┌───────────────┼───────────────┐
          │               │               │
      Thought 1a      Thought 1b      Thought 1c
          │               │               │
      ┌───┼───┐          ...             ...
      │   │   │
   2a  2b  2c
   │
 ┌─┼─┐
3a 3b 3c   ... evaluate leaf nodes, pick best path
```

### Search Strategies

| Strategy | How It Works | When to Use |
|----------|-------------|-------------|
| BFS | Explore all branches level-by-level | Need broad coverage; shallow trees |
| DFS | Go deep on one branch first | Deep reasoning problems |
| Beam Search | Keep only top-K states at each level | Balance exploration vs. cost |

### Python Skeleton

```python
class TreeNode:
    def __init__(self, text, parent=None, evaluation=0.0):
        self.text = text
        self.parent = parent
        self.children = []
        self.evaluation = evaluation
        self.value = None  # set for leaf nodes

class TreeOfThoughts:
    def __init__(self, model, evaluator, k=5, beam_width=3, max_depth=10):
        self.model = model      # LLM for generating thoughts
        self.evaluator = evaluator  # function to score states
        self.k = k              # candidates per step
        self.beam_width = beam_width
        self.max_depth = max_depth

    def generate_candidates(self, prompt):
        """Generate K candidate next thoughts."""
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
        
        # Leaf evaluation
        best = sorted(beam, key=lambda n: n.evaluation, reverse=True)[0]
        return self._extract_trace(best)

    def _build_prompt(self, node):
        trace = self._extract_trace(node)
        return f"Solve: {trace}\nNext thinking step:"

    def _extract_trace(self, node):
        if node.parent is None:
            return node.text
        return self._extract_trace(node.parent) + "\n" + node.text
```

---

## Strategy 4: Verifier-Guided Decoding

Use a trained verifier (reward model) to score and guide generation at every step — or at least at key decision points.

### How It Works

1. Train a **verifier** (or reward model) on labeled data to score solutions as correct/incorrect or with a confidence score.
2. During generation, sample multiple candidate responses.
3. Run each candidate through the verifier.
4. Select or re-rank based on verifier scores.

### Verification at Different Granularities

| Granularity | Description | Example | Compute Cost |
|------------|-------------|---------|--------------|
| Output-level | Verify the final answer only | Math problem answer check | Low |
| Step-level | Verify each reasoning step | Each equation in a proof | Medium |
| Token-level | Score individual tokens | Verifier at each generation step | High |

### Code Example

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Verifier:
    """A reward model scorer for solution evaluation."""
    
    def __init__(self, model_name="OpenAI/summarize_from_feedback"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
    
    def score(self, question, answer):
        """Score an answer for a given question (0-1)."""
        text = f"Q: {question} A: {answer}"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            score = torch.sigmoid(outputs.logits).item()
        return score

def verifier_rerank(question, model, verifier, n=10):
    """Generate N answers and rerank by verifier score."""
    answers = model.generate(question, n=n, temperature=0.8)
    
    scored = []
    for ans in answers:
        s = verifier.score(question, ans)
        scored.append((s, ans))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], scored  # best answer, all scored answers
```

---

## Comparison Table

| Strategy | Compute Cost | Best For | Key Strength | Key Weakness |
|----------|-------------|----------|--------------|--------------|
| Best-of-N | Low–Medium | General QA, code gen | Simple, reliable | Diminishing returns |
| Self-Correction | Medium | Reasoning, writing | Mimics human editing | May reinforce model biases |
| Tree of Thoughts | Medium–High | Complex planning, math | Structured exploration | Expensive, complex to implement |
| Verifier-Guided | Medium–High | Math, code, logic | Objective scoring | Requires training a verifier |

---

## Code Examples

### Complete Best-of-N Pipeline

```python
class BestOfN:
    def __init__(self, generator, scorer, n=10):
        self.generator = generator
        self.scorer = scorer
        self.n = n
    
    def solve(self, question):
        """Generate n answers, return the best."""
        results = []
        for i in range(self.n):
            answer = self.generator(question, temperature=0.7)
            score = self.scorer(question, answer)
            results.append({"answer": answer, "score": score, "index": i})
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    def solve_detailed(self, question):
        """Solve and return detailed statistics."""
        results = self.solve(question)
        return {
            "best": results[0],
            "best_score": results[0]["score"],
            "avg_score": sum(r["score"] for r in results) / len(results),
            "score_spread": results[0]["score"] - results[-1]["score"],
            "all_results": results
        }
```

### Reflexion Loop

```python
def reflexion_loop(generator, question, max_rounds=3):
    """Run a reflexion loop: generate, critique, revise."""
    current_answer = generator(question)
    history = []
    
    for round_num in range(max_rounds):
        # Generate reflection
        critique_prompt = (
            f"Review this solution and identify any errors or weaknesses.\n"
            f"Problem: {question}\n"
            f"Solution: {current_answer}\n"
            f"Review: "
        )
        critique = generator(critique_prompt)
        
        # Check if revision is needed
        revision_prompt = (
            f"Based on the following critique, improve your solution.\n"
            f"Problem: {question}\n"
            f"Original Solution: {current_answer}\n"
            f"Critique: {critique}\n"
            f"Revised Solution: "
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

## Further Reading

1. **"Inference Scaling Laws"** — Wang et al. (2024) — https://arxiv.org/abs/2403.05530
   Foundational paper on how test-time compute scales with model performance.

2. **"Tree of Thoughts: Deliberate Problem Solving with Large Language Models"** — Yao et al. (2023) — https://arxiv.org/abs/2305.10601
   Original ToT paper introducing systematic search over reasoning paths.

3. **"Self-Refine: Iterative Refinement with Self-Feedback"** — Madaan et al. (2023) — https://arxiv.org/abs/2303.17651
   Shows how models can refine their own outputs through self-critique.

4. **"ReST: Reinforced Self-Training for Language Models"** — Gulcehre et al. (2023) — https://arxiv.org/abs/2308.08998
   Combines self-generated data with test-time scaling for improved reasoning.

5. **"Let's Verify Step by Step"** — Lightman et al. (2023) — https://arxiv.org/abs/2305.20050
   Demonstrates that process-level verification beats outcome-level verification.

6. **Anthropic's Constitutional AI** — https://arxiv.org/abs/2212.08073
   Uses self-critique guided by principles for alignment.

---

## Exercises

1. **Implement Best-of-N:** Write a script that generates 20 responses to a math problem and ranks them with a simple regex-based verifier (check if the final number is correct).

2. **Build a Reflexion Loop:** Create a two-round reflexion system for code generation — first generate code, then critique it for bugs, then regenerate.

3. **Tree of Thoughts for Sudoku:** Implement a ToT solver for 4x4 Sudoku puzzles. Use a heuristic evaluator that counts constraint violations.

4. **Ablation Study:** Compare single-pass generation vs. Best-of-5 vs. Best-of-20 on 10 reasoning problems. Plot the accuracy curve.

5. **Verifier Design:** Train a simple binary classifier on GSM8K solutions (correct/incorrect) and use it to rerank model outputs.

---

*Day 04 Tutorial — Advanced AI Daily*
