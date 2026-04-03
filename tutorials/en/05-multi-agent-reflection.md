# Day 05: Multi-Agent Reflection

## Table of Contents

- Overview
- Why Multi-Agent > Single-Agent
- Reflexion: The Foundation
- PEV Architecture (Planner-Executor-Verifier)
- Debate Architecture
- Comparison: Architecture Patterns
- Code Examples
- ASCII Flowcharts
- Practical Design Patterns
- Further Reading
- Exercises

---

## Overview

**Multi-agent reflection** takes the self-correction idea from Day 04 and distributes it across multiple specialized agents, each playing a distinct role. Instead of one model trying to generate, critique, and revise its own work, we deploy a team of agents: one to plan, one to execute, one to verify, one to debate — and let them collaborate, challenge, and refine each other.

**Key insight:** Multi-agent systems consistently outperform single-agent approaches on complex tasks because they:

1. **Separate concerns** — each agent specializes in one cognitive function.
2. **Avoid self-bias** — an independent critic catches errors the generator missed.
3. **Enable structured disagreement** — debate surfaces flaws that consensus would hide.
4. **Scale compute compositionally** — add agents only where the task is hardest.

---

## Why Multi-Agent > Single-Agent

The core hypothesis driving multi-agent research is that **division of cognitive labor** outperforms asking a single model to do everything.

| Dimension | Single Agent | Multi-Agent | Why Multi-Agent Wins |
|-----------|-------------|-------------|---------------------|
| Role confusion | One prompt must handle all roles | Each agent has a focused prompt | Focused prompts = fewer errors |
| Self-bias | Model evaluates its own output | Independent critic provides detachment | External eyes catch blind spots |
| Error propagation | Errors compound silently | Verification stops bad branches | Checkpoints prevent cascade failures |
| Scalability | One pass of N tokens = fixed compute | Add agents to scale compute where needed | Targeted resource allocation |
| Perspectival diversity | Single viewpoint | Multiple viewpoints via debate | Adversarial testing improves robustness |
| Auditability | Black box pipeline | Each agent's output is visible | Easier debugging and analysis |

### The "Blind Spot" Problem

A single LLM has coherent but systematic biases. When asked to critique its own answer, it often:
- Fails to notice its own errors (self-consistent but wrong).
- Hallucinates problems that don't exist.
- Is overly lenient on its own reasoning.

A **separate critic agent**, especially with a different system prompt or different model variant, does not share the same reasoning path and therefore catches different error types.

---

## Reflexion: The Foundation

Reflexion is the simplest multi-agent architecture — and the bridge between single-agent self-correction and full multi-agent systems.

### Architecture

```
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │   Generator   │────►│   Critic      │────►│   Refiner    │
  │   (Agent 1)   │     │   (Agent 2)   │     │   (Agent 3)  │
  └──────────────┘     └──────────────┘     └──────┬───────┘
        ▲                                          │
        │        ┌──────────────┐                  │
        └────────│   Done?      │◄─────────────────┘
                 └──────┬───────┘
                    Yes │
                        ▼
                 ┌──────────────┐
                 │   Output     │
                 └──────────────┘
```

### Key Design Decisions

| Decision | Options | Trade-offs |
|----------|---------|------------|
| Same model vs. different model | Use identical LLM with different prompts, or different LLMs entirely | Different models = more diverse critique |
| Shared memory | All agents access a shared context, or pass messages through a controller | Shared = richer context, but more tokens used |
| Termination | Fixed rounds, agreement-based, or confidence threshold | Agreement = better quality, but risk of infinite loops |
| Memory of past failures | Store critique history for future use, or start fresh each time | History = prevents repeated mistakes |

---

## PEV Architecture (Planner-Executor-Verifier)

The **Planner-Executor-Verifier (PEV)** pattern is the most widely used multi-agent workflow for complex tasks. It decomposes a problem into three distinct phases, each handled by a specialized agent.

### ASCII Flowchart

```
                    ┌────────────────┐
                    │     Task       │
                    └───────┬────────┘
                            │
               ┌────────────▼────────────┐
               │                         │
               │   PLANNER (Agent 1)     │
               │   "How should I solve   │
               │    this? What are the   │
               │    sub-steps?"          │
               │                         │
               └────────────┬────────────┘
                            │
                    ┌───────▼───────┐
                    │   Decomposed  │
                    │    Plan       │
                    └───────┬───────┘
                            │
               ┌────────────▼────────────┐
               │                         │
               │  EXECUTOR (Agent 2)     │
               │   "I'll carry out each  │
               │    step and collect     │
               │    intermediate results │
               │    and the final answer │
               │                         │
               └────────────┬────────────┘
                            │
                    ┌───────▼───────┐
                    │  Raw Result   │
                    │  + Trace      │
                    └───────┬───────┘
                            │
               ┌────────────▼────────────┐
               │                         │
               │   VERIFIER (Agent 3)    │
               │   "Is this correct?     │
               │    Does the logic hold? │
               │    Any gaps in the      │
               │    reasoning chain?"    │
               │                         │
               └────────────┬────────────┘
                            │
                    ┌───────▼───────┐
                    │  Verdict:     │
                    │  PASS / FAIL  │
                    └───────┬───────┘
               PASS ────────┘ ├────── FAIL
                              │
                    ┌─────────▼─────────┐
                    │  Plan sent back   │
                    │  to PLANNER for   │
                    │  revision         │
                    └─────────┬─────────┘
                              │
           (loop until PASS or max iterations)
```

### Agent Roles

| Agent | System Prompt Focus | Input | Output |
|-------|-------------------|-------|--------|
| Planner | "Break down complex problems into executable steps. Be systematic." | Raw task | Step-by-step plan |
| Executor | "Follow the plan precisely. Collect all intermediate results." | Plan | Executed result + trace |
| Verifier | "Check every step for logical consistency. Identify specific errors." | Plan + result + trace | Pass/Fail + detailed critique |

### Code Example

```python
class PlannerAgent:
    def __init__(self, model):
        self.model = model
    
    def plan(self, task):
        prompt = (
            "You are a planning expert. Break this task into clear, "
            "sequential steps. Each step should be independently executable.\n\n"
            f"Task: {task}\n\n"
            "Steps:\n"
            "1. "
        )
        response = self.model.complete(prompt)
        # Parse numbered steps
        steps = [line.strip() for line in response.split("\n") 
                 if line.strip() and line.strip()[0].isdigit()]
        return steps

class ExecutorAgent:
    def __init__(self, model):
        self.model = model
    
    def execute(self, task, plan):
        results = []
        for i, step in enumerate(plan):
            prompt = (
                f"Task: {task}\n"
                f"Current step ({i+1}/{len(plan)}): {step}\n"
                f"Previous results: {results}\n\n"
                f"Execute this step and return your result:\n"
            )
            result = self.model.complete(prompt)
            results.append(result)
        return {
            "steps": plan,
            "results": results,
            "final_answer": results[-1] if results else None
        }

class VerifierAgent:
    def __init__(self, model):
        self.model = model
    
    def verify(self, task, plan, execution):
        prompt = (
            "You are a rigorous verifier. Review this solution critically.\n\n"
            f"Task: {task}\n"
            f"Plan: {plan}\n"
            f"Execution trace: {execution}\n\n"
            "For each step, check:\n"
            "1. Is the step logically sound?\n"
            "2. Are the calculations correct?\n"
            "3. Does the final answer follow from the steps?\n\n"
            "Respond with PASS or FAIL, followed by your reasoning.\n"
            "Verdict: "
        )
        response = self.model.complete(prompt)
        
        verdict = "PASS" if response.upper().startswith("PASS") else "FAIL"
        critique = response.split("\n", 1)[1].strip() if "\n" in response else response
        
        return verdict, critique

class PEVSystem:
    def __init__(self, planner, executor, verifier, max_iterations=3):
        self.planner = planner
        self.executor = executor
        self.verifier = verifier
        self.max_iterations = max_iterations
    
    def solve(self, task):
        plan = self.planner.plan(task)
        iteration = 0
        
        while iteration < self.max_iterations:
            execution = self.executor.execute(task, plan)
            verdict, critique = self.verifier.verify(task, plan, execution)
            
            if verdict == "PASS":
                return execution
        
            if verdict == "FAIL":
                # Revise plan based on critique
                plan = self._revise_plan(task, plan, critique)
            
            iteration += 1
        
        raise RuntimeError(f"Failed after {self.max_iterations} iterations")
    
    def _revise_plan(self, task, old_plan, critique):
        prompt = (
            f"Revise the plan based on this critique.\n"
            f"Task: {task}\n"
            f"Current Plan: {old_plan}\n"
            f"Critique: {critique}\n\n"
            "Revised numbered plan:\n1. "
        )
        response = self.planner.model.complete(prompt)
        return [line.strip() for line in response.split("\n")
                if line.strip() and line.strip()[0].isdigit()]
```

---

## Debate Architecture

Debate introduces **adversarial reasoning** — two or more agents argue opposing positions, and a judge agent determines the winner. This forces each agent to strengthen its arguments and exposes weak reasoning.

### ASCII Flowchart

```
  ┌──────────────────────────────────────────────────────────┐
  │                     QUESTION / TASK                      │
  └─────────────────────────┬────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
      ┌───────▼──────┐     ...   ┌───────▼──────┐
      │   Arguer A   │           │   Arguer B   │
      │   (Position  │           │   (Position  │
      │    X)        │           │    Y)        │
      └───────┬──────┘           └───────┬──────┘
              │                          │
      ┌───────▼──────┐           ┌───────▼──────┐
      │  Counter A   │  ◄─────►  │  Counter B   │
      │  (rebuttal)  │           │  (rebuttal)  │
      └───────┬──────┘           └───────┬──────┘
              │      K rounds of          │
              │      cross-examination    │
              └──────────┬───────────────┘
                         │
               ┌─────────▼──────────┐
               │      JUDGE         │
               │  (evaluates both   │
               │   argument chains, │
               │   picks winner)    │
               └─────────┬──────────────┘
                         │
              ┌──────────▼──────────────┐
              │   Winning Argument      │
              │   = Final Answer        │
              └─────────────────────────┘
```

### Debate Flow

1. **Setup:** The question is presented to two (or more) agents.
2. **Opening statements:** Each agent presents its initial position.
3. **Rebuttals:** Agents counter each other's arguments for K rounds.
4. **Closing:** Agents give final summaries.
5. **Judgment:** A judge agent reads all arguments and selects the most convincing.

### When Debate Works Best

| Scenario | Why Debate Helps |
|----------|-----------------|
| Multiple plausible answers | Forces comparison of alternatives |
| Ambiguous tasks | Different perspectives reveal hidden assumptions |
| Safety-critical decisions | Adversarial testing reduces false confidence |
| Factual disputes | Cross-checking claims catches hallucinations |

### Code Example

```python
class Debate:
    def __init__(self, model, agent_a_system=None, 
                 agent_b_system=None, judge_system=None,
                 rounds=3):
        self.model = model
        self.agent_a_system = agent_a_system or "Present the strongest argument for one side."
        self.agent_b_system = agent_b_system or "Present the strongest counter-argument."
        self.judge_system = judge_system or "Objectively evaluate both arguments. Pick the better one."
        self.rounds = rounds
    
    def run(self, question):
        log = []
        
        # Opening statements
        arg_a = self._get_response(self.agent_a_system, question, log)
        arg_b = self._get_response(self.agent_b_system, question, log)
        log.append({"speaker": "A", "argument": arg_a, "round": "opening"})
        log.append({"speaker": "B", "argument": arg_b, "round": "opening"})
        
        # Debate rounds
        for r in range(1, self.rounds + 1):
            rebut_a = self._get_response(
                self.agent_a_system, question, log
            )
            rebut_b = self._get_response(
                self.agent_b_system, question, log
            )
            log.append({"speaker": "A", "argument": rebut_a, "round": r})
            log.append({"speaker": "B", "argument": rebut_b, "round": r})
        
        # Judgment
        verdict = self._judge(question, log)
        
        return {
            "verdict": verdict,
            "winner": verdict["winner"],
            "log": log
        }
    
    def _get_response(self, system_prompt, question, history):
        context = "\n".join(
            f"[{entry['speaker']} Round {entry['round']}]: {entry['argument']}"
            for entry in history
        )
        prompt = f"{system_prompt}\n\nQuestion: {question}\n\nHistory:\n{context}\n\nYour response:"
        return self.model.complete(prompt)
    
    def _judge(self, question, log):
        transcript = "\n".join(
            f"[Speaker {entry['speaker']} Round {entry['round']}]: {entry['argument']}"
            for entry in log
        )
        prompt = (
            f"{self.judge_system}\n\n"
            f"Question: {question}\n\n"
            f"Full transcript:\n{transcript}\n\n"
            "Which speaker presented the stronger argument?\n"
            "Respond with 'A' or 'B' followed by your reasoning.\n"
            "Verdict: "
        )
        response = self.model.complete(prompt)
        
        winner = "A" if response.strip().upper().startswith("A") else "B"
        reasoning = response.strip()
        
        return {"winner": winner, "reasoning": reasoning}
```

### Debate Variants

| Variant | Description | Use Case |
|---------|-------------|----------|
| Fixed-side debate | Agents are assigned positions (Pro/Con) | Factual verification |
| Free-sides debate | Agents independently form positions | Open-ended reasoning |
| N-agent debate | 3+ agents with different viewpoints | Complex policy questions |
| Expertise-weighted | Each agent has a different domain system prompt | Cross-domain problems |
| Self-debate | Same model, two different system prompts | When only one model is available |

---

## Comparison: Architecture Patterns

| Pattern | Agents | Communication | Best For | Complexity |
|---------|--------|-------------|----------|------------|
| Reflexion | 2 (Generator + Critic) | Sequential | Code review, writing | Low |
| PEV | 3 (Planner + Executor + Verifier) | Sequential with feedback loop | Complex multi-step tasks | Medium |
| Debate | 2+ Agents + Judge | Bidirectional rebuttals | Ambiguous questions, fact-checking | Medium |
| Multi-Agent Team | 4+ with specialized roles | Mixed (sequential + parallel) | Full pipeline (research → write → review → publish) | High |
| AutoGen-style Conversational | N agents | Free-form conversation | Open-ended collaboration | High |

---

## Code Examples

### Multi-Agent Orchestrator with Shared Memory

```python
class Agent:
    def __init__(self, name, system_prompt, model):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
    
    def respond(self, prompt):
        return self.model.complete(
            f"{self.system_prompt}\n\n{prompt}"
        )

class SharedMemory:
    def __init__(self):
        self.messages = []
    
    def add(self, agent_name, content):
        self.messages.append({
            "agent": agent_name,
            "content": content,
        })
    
    def get_history(self):
        return "\n".join(
            f"[{m['agent']}]: {m['content']}" for m in self.messages
        )
    
    def last(self, n=1):
        return self.messages[-n:]

class MultiAgentCollaboration:
    def __init__(self, agents, memory=None, max_rounds=5):
        self.agents = {a.name: a for a in agents}
        self.memory = memory or SharedMemory()
        self.max_rounds = max_rounds
    
    def run(self, initial_task):
        self.memory.add("SYSTEM", initial_task)
        round_num = 0
        
        while round_num < self.max_rounds:
            for name, agent in self.agents.items():
                context = self.memory.get_history()
                response = agent.respond(
                    f"Context:\n{context}\n\n"
                    f"Your turn to respond to: {initial_task}"
                )
                self.memory.add(name, response)
                
                if self._should_terminate(response):
                    return self._summarize()
            
            round_num += 1
        
        return self._summarize()
    
    def _should_terminate(self, latest_response):
        # Simple termination: agent says "I agree" or "No further changes needed"
        lower = latest_response.lower()
        return ("no further changes" in lower or 
                "i agree" in lower or 
                "agreed" in lower)
    
    def _summarize(self):
        return self.memory.get_history()
```

---

## ASCII Flowcharts

### Complete Multi-Agent System with Reflexion Loop

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                          TASK INPUT                                 │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │        PLANNER              │
                  │   "Decompose the problem"   │
                  └──────────────┬──────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
    ┌───────▼──────┐    ┌───────▼──────┐    ┌───────▼──────┐
    │  Sub-task 1  │    │  Sub-task 2  │    │  Sub-task 3  │
    │  EXECUTOR    │    │  EXECUTOR    │    │  EXECUTOR    │
    └───────┬──────┘    └───────┬──────┘    └───────┬──────┘
            │                   │                   │
            └────────────────────┼───────────────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │        VERIFIER             │
                  │   "Check all solutions"     │
                  └──────────────┬──────────────┘
                                 │
                    ┌────────────┼────────────┐
                FAIL │         PASS            │
                    ▼            │             │
           ┌──────────────┐     │             │
           │   REFLECT:   │     │             │
           │  What went   │     │             │
           │  wrong? Why? │     │             │
           └──────┬───────┘     │             │
                  │ send back   │             │
           ┌──────▼──────┐      │             │
           │   PLANNER   │      │             │
           │  revision   │      │             │
           └─────────────┘      │             │
                                ▼             ▼
                  ┌───────────────┬──────────────┐
                  │         FINAL OUTPUT          │
                  └───────────────────────────────┘
```

---

## Practical Design Patterns

### Pattern 1: Critic-Only Multi-Agent

```
Task → Generator → Critic → [Fix?] 
                            ├── Yes → Output
                            ├── No  → Generator (revised)
```

**When to use:** Code review, essay grading, translation quality checks.

### Pattern 2: Pipeline with Checkpoints

```
Step 1 (Agent A) → Verify → Step 2 (Agent B) → Verify → Step 3 (Agent C) → Output
                       │                           │                          │
                    [FAIL]                      [FAIL]                    [FAIL]
                       │                           │                          │
                    Agent A retry            Agent B retry             Agent C retry
```

**When to use:** Data processing pipelines, multi-step reasoning, software development.

### Pattern 3: Consensus Voting

```
Task → [Agent 1] ──┐
     → [Agent 2] ──┤ → Majority vote → Output
     → [Agent 3] ──┘     (or weighted)
```

**When to use:** Classification, factual QA, when you need robustness through diversity.

### Pattern 4: Adversarial Debate

```
Task → [Pro Agent] ←→ [Con Agent] → Judge → Winning argument → Output
```

**When to use:** Decision-making with tradeoffs, policy analysis, ethical reasoning.

---

## Further Reading

1. **"Reflexion: Language Agents with Verbal Reinforcement Learning"** — Shinn et al. (2023) — https://arxiv.org/abs/2303.11366
   Foundational paper introducing self-reflection with verbal reward signals for RL in LLM agents.

2. **"Multi-Agent Consensus: Improving LLM Reasoning through Group Collaboration"** — Liang et al. (2023) — https://arxiv.org/abs/2305.14325
   Demonstrates that multi-agent consensus outperforms single-agent reasoning.

3. **"ChatDev: Communicative Agents for Software Development"** — Qian et al. (2023) — https://arxiv.org/abs/2307.07924
   Full multi-agent pipeline for software engineering with specialized roles.

4. **"AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation"** — Wu et al. (2023) — https://arxiv.org/abs/2308.08155
   Framework for building multi-agent conversational systems.

5. **"The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery"** — Lu et al. (2024) — https://arxiv.org/abs/2408.06292
   Multi-agent system for automated scientific research.

6. **"Judging LLM-as-a-Judge with MT-Bench"** — Zheng et al. (2023) — https://arxiv.org/abs/2306.05685
   Research on using LLMs as evaluators in multi-agent systems.

7. **"Language Models as Zero-Shot Planners"** — Huang et al. (2022) — https://arxiv.org/abs/2201.07207
   Foundational work on LLMs as planners in multi-step reasoning.

---

## Exercises

1. **Build a PEV System:** Create a Planner-Executor-Verifier pipeline for solving word math problems. The planner should decompose the problem, the executor should solve it step-by-step, and the verifier should check if the solution is mathematically sound.

2. **Debate on Ambiguous Questions:** Set up a two-agent debate on an intentionally ambiguous questions, such as "Is social media good for society?" Use a third judge agent to evaluate. Compare results with a single-agent response.

3. **Implement Reflexion for Code:** Create a generator-critic loop for code generation. The generator writes Python code, the critic reviews it for bugs and edge cases, and the generator revises. Test on LeetCode Easy problems.

4. **Consensus Benchmark:** Generate 5 independent answers to the same question using 5 agents with slightly different system prompts. Compare consensus (majority vote) versus single-agent accuracy on a set of 20 trivia questions.

5. **Agent Memory Experiment:** Run a multi-agent system once with shared memory and once without. Compare how well the system handles follow-up questions in both configurations.

6. **Cost Analysis:** Calculate the token cost for each architecture (single-agent, reflexion, PEV, debate) on the same task. Plot accuracy versus cost to find the Pareto frontier.

---

*Day 05 Tutorial — Advanced AI Daily*
