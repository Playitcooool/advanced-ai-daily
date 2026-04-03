# 第五天：多智能体反思

## 目录

- 概述
- 为什么多智能体优于单智能体
- 反思（Reflexion）：基础架构
- PEV 架构（规划者-执行者-验证者）
- 辩论架构（Debate Architecture）
- 架构模式对比
- 代码示例
- ASCII 流程图
- 实用设计模式
- 延伸阅读
- 练习题

---

## 概述

**多智能体反思**将第四天的自我纠正理念推广到多个专业化智能体的协作框架中。与其让一个模型同时承担生成、批判和修订的所有角色，我们部署一个智能体团队——一个负责规划，一个负责执行，一个负责验证，一个负责辩论——让它们相互协作、质疑和精炼。

**核心洞察：** 在复杂任务上，多智能体系统始终优于单智能体方法，因为：

1. **关注点分离**——每个智能体专精于一种认知功能。
2. **避免自我偏差**——独立的批判者能捕捉到生成者遗漏的错误。
3. **支持结构化分歧**——辩论能暴露共识所掩盖的缺陷。
4. **组合式扩展算力**——仅在最困难的任务节点上增加智能体。

---

## 为什么多智能体优于单智能体

多智能体研究的核心假设是：**认知劳动的分工**优于让单个模型包揽一切。

| 维度 | 单智能体 | 多智能体 | 为什么多智能体更优 |
|------|---------|---------|-------------------|
| 角色混淆 | 一个提示词须处理所有角色 | 每个智能体有聚焦的提示词 | 聚焦提示 = 更少错误 |
| 自我偏差 | 模型评估自身的输出 | 独立批判者提供超然视角 | 外部目光能发现盲点 |
| 错误传播 | 错误静默累积 | 验证阻止错误分支 | 检查点防止级联失败 |
| 可扩展性 | 单次 N 词元 = 固定算力 | 在需要的地方添加智能体 | 针对性资源分配 |
| 视角多样性 | 单一视角 | 通过辩论产生多重视角 | 对抗测试提升鲁棒性 |
| 可审计性 | 黑盒流水线 | 每个智能体的输出可见 | 更易于调试和分析 |

### "盲点"问题

单个大语言模型有连贯但系统性的偏差。当要求它批判自身答案时，它往往：
- 无法发现自身的错误（自洽但错误）。
- 虚构不存在的问题。
- 对自身的推理过于宽容。

一个**独立的批判智能体**，尤其是使用不同的系统提示词或不同模型变体的智能体，不会走相同的推理路径，因此能捕捉到不同类型的错误。

---

## 反思（Reflexion）：基础架构

反思是最简单的多智能体架构——也是单智能体自我纠正与完整多智能体系统之间的桥梁。

### 架构流程图

```
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │   生成器      │────►│   批判者      │────►│   修订者      │
  │  （智能体1）   │     │  （智能体2）   │     │  （智能体3）   │
  └──────────────┘     └──────────────┘     └──────┬───────┘
        ▲                                          │
        │        ┌──────────────┐                  │
        └────────│   完成了吗？   │◄─────────────────┘
                 └──────┬───────┘
                    是  │
                        ▼
                 ┌──────────────┐
                 │    输出       │
                 └──────────────┘
```

### 关键设计决策

| 决策 | 选项 | 权衡 |
|------|------|------|
| 相同模型 vs. 不同模型 | 用相同大模型但不同提示词，或完全不同的模型 | 不同模型 = 更多样化的批判 |
| 共享记忆 | 所有智能体访问共享上下文，或通过控制器传递消息 | 共享 = 更丰富上下文，但消耗更多词元 |
| 终止条件 | 固定轮次、达成共识、或置信度阈值 | 共识 = 更高质量，但有无限循环风险 |
| 过去失败记忆 | 存储批判历史供未来使用，或每次重新开始 | 历史 = 防止重复犯错 |

---

## PEV 架构（规划者-执行者-验证者）

**规划者-执行者-验证者（PEV）** 模式是处理复杂任务最广泛使用的多智能体工作流。它将问题分解为三个不同的阶段，每个阶段由专门的智能体处理。

### ASCII 流程图

```
                    ┌────────────────┐
                    │      任务       │
                    └───────┬────────┘
                            │
               ┌────────────▼────────────┐
               │                         │
               │    规划者（智能体1）      │
               │   "我该如何求解？        │
               │    子步骤有哪些？"       │
               │                         │
               └────────────┬────────────┘
                            │
                    ┌───────┴───────┐
                    │    分解后的    │
                    │      计划      │
                    └───────┬───────┘
                            │
               ┌────────────▼────────────┐
               │                         │
               │   执行者（智能体2）       │
               │   "我将逐步执行         │
               │    并收集中间结果和       │
               │    最终答案"             │
               │                         │
               └────────────┬────────────┘
                            │
                    ┌───────┴───────┐
                    │    原始结果    │
                    │    + 追踪记录   │
                    └───────┬───────┘
                            │
               ┌────────────▼────────────┐
               │                         │
               │   验证者（智能体3）       │
               │   "这是否正确？          │
               │    逻辑是否成立？        │
               │    推理链中是否有漏洞？"  │
               │                         │
               └────────────┬────────────┘
                            │
                    ┌───────┴───────┐
                    │   裁决结果：    │
                    │   通过 / 不通过  │
                    └───────┬───────┘
               通过 ────────┘ ├────── 不通过
                              │
                    ┌─────────▼─────────┐
                    │  将计划送回规划者  │
                    │  进行修订          │
                    └─────────┬─────────┘
                              │
           （循环直至通过或达到最大迭代次数）
```

### 智能体角色

| 智能体 | 系统提示词重点 | 输入 | 输出 |
|--------|---------------|------|------|
| 规划者 | "将复杂问题分解为可执行的步骤。要有条不紊。" | 原始任务 | 分步计划 |
| 执行者 | "严格按照计划执行。收集中间结果。" | 计划 | 执行结果 + 追踪记录 |
| 验证者 | "检查每一步的逻辑一致性。找出具体错误。" | 计划 + 结果 + 追踪记录 | 通过/不通过 + 详细批判 |

### 代码示例

```python
class PlannerAgent:
    def __init__(self, model):
        self.model = model
    
    def plan(self, task):
        prompt = (
            "你是规划专家。将任务分解为清晰的、顺序的步骤。"
            "每个步骤应该是可独立执行的。\n\n"
            f"任务：{task}\n\n"
            "步骤：\n"
            "1. "
        )
        response = self.model.complete(prompt)
        # 解析编号步骤
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
                f"任务：{task}\n"
                f"当前步骤（{i+1}/{len(plan)}）：{step}\n"
                f"之前的结果：{results}\n\n"
                f"执行这一步并返回结果：\n"
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
            "你是一个严谨的验证者。严格审查这个解法。\n\n"
            f"任务：{task}\n"
            f"计划：{plan}\n"
            f"执行追踪：{execution}\n\n"
            "对每一步检查：\n"
            "1. 该步骤逻辑是否合理？\n"
            "2. 计算是否正确？\n"
            "3. 最终答案是否由步骤推导得出？\n\n"
            "回答通过或不通过，并附上你的理由。\n"
            "裁决： "
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
                # 根据批判修订计划
                plan = self._revise_plan(task, plan, critique)
            
            iteration += 1
        
        raise RuntimeError(f"在 {self.max_iterations} 次迭代后失败")
    
    def _revise_plan(self, task, old_plan, critique):
        prompt = (
            f"根据以下批判修订计划。\n"
            f"任务：{task}\n"
            f"当前计划：{old_plan}\n"
            f"批判：{critique}\n\n"
            "修订后的编号计划：\n1. "
        )
        response = self.planner.model.complete(prompt)
        return [line.strip() for line in response.split("\n")
                if line.strip() and line.strip()[0].isdigit()]
```

---

## 辩论架构（Debate Architecture）

辩论引入了**对抗性推理**——两个或多个智能体就对立立场进行辩论，由一个评判智能体裁定胜负。这迫使每个智能体加强自己的论据并暴露薄弱的推理。

### ASCII 流程图

```
  ┌──────────────────────────────────────────────────────────┐
  │                      问题 / 任务                          │
  └─────────────────────────┬────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
      ┌───────▼──────┐     ...   ┌───────▼──────┐
      │  辩手A        │           │  辩手B        │
      │  （立场X）     │           │  （立场Y）     │
      └───────┬──────┘           └───────┬──────┘
              │                          │
      ┌───────▼──────┐           ┌───────▼──────┐
      │ A的反驳      │  ◄─────►  │ B的反驳      │
      └───────┬──────┘           └───────┬──────┘
              │       K 轮交叉质询         │
              └──────────┬───────────────┘
                         │
               ┌─────────▼──────────┐
               │      评判者         │
               │  （评估双方的论证   │
               │   链，选出胜者）     │
               └─────────┬──────────────┘
                         │
              ┌──────────▼──────────────┐
              │   获胜的论证            │
              │   = 最终答案            │
              └─────────────────────────┘
```

### 辩论流程

1. **设置：** 问题提交给两个（或更多）智能体。
2. **开场陈述：** 每个智能体展示其初始立场。
3. **反驳：** 智能体在 K 轮中互相反驳对方的论点。
4. **总结：** 智能体给出最终总结。
5. **裁决：** 评判智能体阅读所有论据并选择更有说服力的一方。

### 辩论最有效的场景

| 场景 | 为什么辩论有帮助 |
|------|-----------------|
| 存在多个合理答案 | 迫使比较各种替代方案 |
| 模棱两可的任务 | 不同视角揭示隐藏假设 |
| 安全关键型决策 | 对抗测试降低虚假信心 |
| 事实争议 | 交叉核实论据能抓住幻觉 |

### 代码示例

```python
class Debate:
    def __init__(self, model, agent_a_system=None, 
                 agent_b_system=None, judge_system=None,
                 rounds=3):
        self.model = model
        self.agent_a_system = agent_a_system or "为一方的立场提出最强有力的论证。"
        self.agent_b_system = agent_b_system or "提出最强有力的反驳论证。"
        self.judge_system = judge_system or "客观评估双方的论证。选出更好的一方。"
        self.rounds = rounds
    
    def run(self, question):
        log = []
        
        # 开场陈述
        arg_a = self._get_response(self.agent_a_system, question, log)
        arg_b = self._get_response(self.agent_b_system, question, log)
        log.append({"speaker": "A", "argument": arg_a, "round": "opening"})
        log.append({"speaker": "B", "argument": arg_b, "round": "opening"})
        
        # 辩论轮次
        for r in range(1, self.rounds + 1):
            rebut_a = self._get_response(
                self.agent_a_system, question, log
            )
            rebut_b = self._get_response(
                self.agent_b_system, question, log
            )
            log.append({"speaker": "A", "argument": rebut_a, "round": r})
            log.append({"speaker": "B", "argument": rebut_b, "round": r})
        
        # 裁决
        verdict = self._judge(question, log)
        
        return {
            "verdict": verdict,
            "winner": verdict["winner"],
            "log": log
        }
    
    def _get_response(self, system_prompt, question, history):
        context = "\n".join(
            f"[{entry['speaker']} 第{entry['round']}轮]: {entry['argument']}"
            for entry in history
        )
        prompt = f"{system_prompt}\n\n问题：{question}\n\n历史：\n{context}\n\n你的回应："
        return self.model.complete(prompt)
    
    def _judge(self, question, log):
        transcript = "\n".join(
            f"[辩手 {entry['speaker']} 第{entry['round']}轮]: {entry['argument']}"
            for entry in log
        )
        prompt = (
            f"{self.judge_system}\n\n"
            f"问题：{question}\n\n"
            f"完整记录：\n{transcript}\n\n"
            "哪位辩手提出了更有力的论证？\n"
            "回答'A'或'B'，然后附上你的理由。\n"
            "裁决： "
        )
        response = self.model.complete(prompt)
        
        winner = "A" if response.strip().upper().startswith("A") else "B"
        reasoning = response.strip()
        
        return {"winner": winner, "reasoning": reasoning}
```

### 辩论变体

| 变体 | 描述 | 使用场景 |
|------|------|---------|
| 固定立场辩论 | 智能体被分配固定立场（正方/反方） | 事实验证 |
| 自由立场辩论 | 智能体自主形成立场 | 开放性推理 |
| N 智能体辩论 | 3个以上智能体持不同观点 | 复杂政策问题 |
| 专家加权辩论 | 每个智能体有不同的领域系统提示词 | 跨领域问题 |
| 自我辩论 | 同一个模型，两个不同的系统提示词 | 仅有一个模型可用时 |

---

## 架构模式对比

| 模式 | 智能体数 | 通信方式 | 最适合 | 复杂度 |
|------|---------|---------|--------|--------|
| 反思 | 2（生成器 + 批判者） | 顺序 | 代码审查、写作 | 低 |
| PEV | 3（规划者 + 执行者 + 验证者） | 顺序带反馈循环 | 复杂多步骤任务 | 中 |
| 辩论 | 2+ 智能体 + 评判者 | 双向反驳 | 模棱两可的问题、事实核查 | 中 |
| 多智能体团队 | 4+ 专业化角色 | 混合（顺序 + 并行） | 完整流水线（研究→撰写→审查→发布） | 高 |
| AutoGen风格对话式 | N 个智能体 | 自由形式对话 | 开放式协作 | 高 |

---

## 代码示例

### 带共享记忆的多智能体协调器

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
                    f"上下文：\n{context}\n\n"
                    f"轮到你回应：{initial_task}"
                )
                self.memory.add(name, response)
                
                if self._should_terminate(response):
                    return self._summarize()
            
            round_num += 1
        
        return self._summarize()
    
    def _should_terminate(self, latest_response):
        # 简单终止条件：智能体说"我同意"或"无需进一步修改"
        lower = latest_response.lower()
        return ("无需进一步修改" in lower or 
                "我同意" in lower or 
                "一致同意" in lower or
                "no further changes" in lower or 
                "i agree" in lower or 
                "agreed" in lower)
    
    def _summarize(self):
        return self.memory.get_history()
```

---

## ASCII 流程图

### 带反思循环的完整多智能体系统

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                            任务输入                                  │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │          规划者              │
                  │    "分解问题"               │
                  └──────────────┬──────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
    ┌───────▼──────┐    ┌───────▼──────┐    ┌───────▼──────┐
    │   子任务1     │    │   子任务2     │    │   子任务3     │
    │    执行者     │    │    执行者     │    │    执行者     │
    └───────┬──────┘    └───────┬──────┘    └───────┬──────┘
            │                   │                   │
            └────────────────────┼───────────────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │          验证者              │
                  │    "检查所有解"              │
                  └──────────────┬──────────────┘
                                 │
                    ┌────────────┼────────────┐
                不通过 │         通过            │
                    ▼            │             │
           ┌──────────────┐     │             │
           │   反思：      │     │             │
           │  哪里出错了？  │     │             │
           │  原因是什么？  │     │             │
           └──────┬───────┘     │             │
                  │ 送回修订    │             │
           ┌──────▼──────┐      │             │
           │   规划者     │      │             │
           │  修订计划    │      │             │
           └─────────────┘      │             │
                                ▼             ▼
                  ┌───────────────┬──────────────┐
                  │          最终输出             │
                  └───────────────────────────────┘
```

---

## 实用设计模式

### 模式一：仅批判者多智能体

```
任务 → 生成器 → 批判者 → [修复？] 
                        ├── 是 → 输出
                        └── 否 → 生成器（修订版）
```

**何时使用：** 代码审查、论文评审、翻译质量检查。

### 模式二：带检查点的流水线

```
步骤1（智能体A）→ 验证 → 步骤2（智能体B）→ 验证 → 步骤3（智能体C）→ 输出
                       │                           │                          │
                    [失败]                      [失败]                    [失败]
                       │                           │                          │
                   智能体A重试                智能体B重试                智能体C重试
```

**何时使用：** 数据处理流水线、多步骤推理、软件开发。

### 模式三：共识投票

```
任务 → [智能体1] ──┐
     → [智能体2] ──┤ → 多数投票 → 输出
     → [智能体3] ──┘   （或加权投票）
```

**何时使用：** 分类、事实性问答、需要通过多样性实现鲁棒性时。

### 模式四：对抗性辩论

```
任务 → [正方智能体] ←→ [反方智能体] → 评判者 → 获胜论证 → 输出
```

**何时使用：** 带权衡的决策、政策分析、伦理推理。

---

## 延伸阅读

1. **《Reflexion：带有语言强化学习的大语言智能体》（Reflexion: Language Agents with Verbal Reinforcement Learning）** — Shinn 等 (2023) — https://arxiv.org/abs/2303.11366
   引入自我反思与语言奖励信号以实现大语言智能体强化学习的基础性论文。

2. **《多智能体共识：通过群体协作提升大语言模型推理》（Multi-Agent Consensus）** — Liang 等 (2023) — https://arxiv.org/abs/2305.14325
   证明了多智能体共识优于单智能体推理。

3. **《ChatDev：面向软件开发的通信智能体》（ChatDev）** — Qian 等 (2023) — https://arxiv.org/abs/2307.07924
   具有专业化角色的软件工程完整多智能体流水线。

4. **《AutoGen：通过多智能体对话启用下一代大语言模型应用》（AutoGen）** — Wu 等 (2023) — https://arxiv.org/abs/2308.08155
   构建多智能体对话系统的框架。

5. **《AI科学家：迈向完全自动化的开放式科学发现》（The AI Scientist）** — Lu 等 (2024) — https://arxiv.org/abs/2408.06292
   用于自动化科学研究的多智能体系统。

6. **《MT-Bench：用MT-Bench评测大语言模型作为评判者》（Judging LLM-as-a-Judge with MT-Bench）** — Zheng 等 (2023) — https://arxiv.org/abs/2306.05685
   关于在多智能体系统中使用大语言模型作为评估器的研究。

7. **《语言模型作为零样本规划者》（Language Models as Zero-Shot Planners）** — Huang 等 (2022) — https://arxiv.org/abs/2201.07207
   关于大语言模型在多步骤推理中作为规划者的基础性工作。

---

## 练习题

1. **构建 PEV 系统：** 创建规划者-执行者-验证者流水线来求解文字数学题。规划者应分解问题，执行者应逐步求解，验证者应检查解法是否在数学上合理。

2. **在模糊问题上的辩论：** 设置一个针对刻意模糊问题的双智能体辩论，例如"社交媒体对社会有益吗？"使用第三个评判智能体进行评估。将结果与单智能体的回答进行比较。

3. **为代码实现反思：** 创建一个用于代码生成的生成器-批判者循环。生成器编写 Python 代码，批判者检查其中的 bug 和边界情况，生成器进行修订。在 LeetCode 简单题上测试。

4. **共识基准测试：** 使用 5 个具有略微不同系统提示词的智能体，对同一个问题生成 5 个独立回答。在 20 道常识题上比较共识（多数投票）与单智能体的准确率。

5. **智能体记忆实验：** 分别在使用共享记忆和不使用共享记忆的情况下运行多智能体系统。比较两种配置下系统处理后续问题的能力。

6. **成本分析：** 计算每种架构（单智能体、反思、PEV、辩论）在同一任务上的词元成本。绘制准确率与成本的关系图，找出帕累托前沿。

---

*第五天教程 — 高级 AI 日报*
