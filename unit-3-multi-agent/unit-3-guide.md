# Unit 3: Multi-Agent Orchestration, Graph Pipelines, and Cost-Bounding Systems

## Progress Summary
- **Current step:** _Not started_
- **Last updated:** _YYYY-MM-DD_
- **Notes/blockers:** _None_

## Main Topics
* Multi-Agent Collaboration Patterns (Orchestrator / Worker)
* State Management in Distributed Graphs (LangGraph)
* Supervisor Pattern & Shared Ephemeral Memory
* Async Task Orchestration & Parallel Node Execution
* Human-in-the-Loop Interruption & Verification Checkpoints
* Token Tracking & Budget-Bounding Middleware
* Multi-Agent Observability & Cross-Node Latency Tracing
* Automated Quality Grading (LLM-as-a-Judge)

## Short Description of Goals
Build a coding team simulation where a planner agent breaks a feature request into subtasks, a coder agent implements each subtask, and a reviewer agent critiques and requests changes. Ensure all agents share a working memory (code files, decisions, test results). Define every agent's structured output (task plan, code block, review verdict) as a Pydantic model so agents cannot miscommunicate format. Build a report generation pipeline where a researcher agent gathers info (web + RAG), a writer drafts sections in parallel, and an editor merges and checks consistency. Use LangGraph to define the workflow as a graph with conditional edges. Add state-based counter mechanisms into the graph loop to track token usage and financial cost, automatically cutting off execution if a multi-agent loop runs out of bounds or spirals. Add a human-in-the-loop step where you can inject corrections mid-run. Add an LLM-as-judge eval at the end: a separate model call scores the final report on coherence, groundedness, and completeness — log scores to Langfuse alongside the run. After a few runs, look at your Langfuse dashboard to determine which agent step has the highest latency and which fails most often. Understand orchestrator/worker patterns, shared state, inter-agent communication, DAG-based orchestration, end-to-end evaluation, using observability data to improve systems, and cost bounding.

---

## Concrete Projects (2 Hours/Day Practice)

### Project 1: State-Bounded Issue-to-PR Engineering Workflow
Build a LangGraph workflow that simulates an autonomous engineering team resolving simple software bugs.

* **What you will do:**

  **Week 1 — Graph skeleton, agents, and shared state**

  - [ ] **Set up the playground repo.** Fork or copy a small open-source repo with a known, simple bug (e.g. an off-by-one error or missing null check). Write a one-line issue description that describes the bug.
     > **Theory:** Scope control is the hardest part of coding agents. A one-line bug in a 200-line repo is tractable; "build me a feature" is not. You're testing the *orchestration pattern*, not the agent's coding genius.
     > **Trade-offs:** Bug you planted yourself (known fix, tests the workflow) vs real open-source good-first-issue (realistic, but the agent may fail for reasons unrelated to your orchestration). Plant your own bug for the first end-to-end run.

  - [ ] **Define shared graph state.** Create a `GraphState` TypedDict: `issue`, `task_plan`, `code_files`, `review_feedback`, `iteration_count`, `token_count`, `total_cost`. This is the single source of truth all agents read and write.
     > **Theory:** Multi-agent systems need a shared data structure — the same way microservices need a database or message bus. In LangGraph, state is a TypedDict that each node reads from and returns partial updates to. This is fundamentally different from Unit 2's conversation history: agents don't pass messages to each other, they read/write shared fields.
     > **Trade-offs:** Centralized state dict (simple, can become a god object) vs message-passing between agents (decoupled, harder to debug). LangGraph's shared state is the right pattern here — it's explicit about what each agent can see.

  - [ ] **Define Pydantic output schemas per agent.** `TaskPlan(steps: list[str])` for Planner, `CodePatch(file: str, content: str)` for Coder, `ReviewVerdict(approved: bool, feedback: str)` for Reviewer. Use `instructor` in each node.
     > **Theory:** Inter-agent contracts are API contracts. The Planner doesn't know or care how the Coder works — it just needs to produce a `TaskPlan` that the Coder can consume. Pydantic schemas at every handoff prevent the "telephone game" problem where meaning degrades across agents.
     > **Trade-offs:** Strict schemas (reliable, may reject valid but unusual outputs) vs free-text messages between agents (flexible, format drift). Always schema the handoffs. Keep individual agent prompts flexible.

  - [ ] **Build the Planner node.** Takes the issue description, outputs a `TaskPlan`. Log input/output to Langfuse. Test in isolation before connecting the graph.
     > **Theory:** Test each agent node like a unit test — feed it a known input, check the structured output. Don't debug the whole graph when one node is broken. The Planner is the simplest node: one LLM call, one structured output, no tool use.
     > **Trade-offs:** Same model for all agents (simpler, cheaper) vs specialized models per role (Planner uses a reasoning model, Coder uses a code model). Same model is fine for learning; specialization is a production optimization.

  - [ ] **Build the Coder node.** Reads `task_plan` and `review_feedback` from state, outputs a `CodePatch`. Test with a hardcoded plan.
     > **Theory:** The Coder is the highest-risk node — it writes to the filesystem. Test it with a hardcoded `TaskPlan` so you know any failure is in the Coder, not the Planner. On retry loops, the Coder receives `review_feedback` from the Reviewer — this is how iterative refinement works.
     > **Trade-offs:** Return full file content in `CodePatch` (simple, overwrites entire file) vs unified diff/patch format (precise, harder for LLMs to generate correctly). Full file content is more reliable with current models.

  - [ ] **Build the Reviewer node.** Reads the `CodePatch` and original issue, outputs a `ReviewVerdict`. Test with a deliberately bad patch to confirm it rejects.
     > **Theory:** The Reviewer is your quality gate. Test it adversarially: feed it a patch that doesn't fix the bug, one that introduces a new bug, and one that's correct. If it approves everything, your loop is theater — the agent never actually iterates.
     > **Trade-offs:** LLM reviewer (flexible, may be too lenient or too harsh) vs deterministic checks (run pytest, approve if tests pass — more reliable for code). Ideally use both: pytest pass is necessary, LLM review is advisory. Start with LLM-only; add pytest gate in step 7.

  **Week 2 — Conditional routing, cost caps, and end-to-end runs**

  - [ ] **Wire the LangGraph.** Connect nodes: Planner → Coder → Reviewer. Add a conditional edge: if `ReviewVerdict.approved == False` and `iteration_count < 3`, route back to Coder with feedback; otherwise, end.
     > **Theory:** A LangGraph is a state machine, not a linear pipeline. Conditional edges are `if/else` branches in your workflow — the graph decides what runs next based on state. This is strictly more expressive than Unit 2's `while` loop, and the graph structure is visible and debuggable.
     > **Trade-offs:** Max 3 review iterations (prevents infinite loops, may not be enough for hard bugs) vs unlimited (dangerous — see step 9). Three is a reasonable default. Log when you hit the cap — it means either the bug is too hard or an agent is broken.

  - [ ] **Add cost-tracking middleware.** After every LLM call, increment `token_count` and `total_cost` in state (use your provider's token counts and pricing). Print running cost to the terminal on each step.
     > **Theory:** Multi-agent systems multiply cost — 3 agents × 3 review loops × 2K tokens each adds up fast. Cost tracking in state is the AI equivalent of timing your database queries: you need to know which step is expensive before you can optimize it. This is FinOps for LLM applications.
     > **Trade-offs:** Track tokens (precise, provider-specific pricing) vs track API calls (simpler, less granular). Track both tokens and estimated dollar cost. Log per-node, not just per-run.

  - [ ] **Add the budget exit node.** If `total_cost > $1.00` or `iteration_count > 5`, route to a `GracefulExit` node that saves partial progress and logs an abort reason. Test by setting the cap to $0.01 to force a trigger.
     > **Theory:** This is the circuit breaker pattern — a standard resilience technique from distributed systems. Without it, a stuck review loop can burn $10+ in minutes. Graceful exit means saving state and logging *why* you stopped, not just crashing.
     > **Trade-offs:** Dollar cap (intuitive, varies by model) vs token cap (model-agnostic, less intuitive). Use both. Test the breaker by forcing it — if you've never seen it trigger, you don't know it works.

  - [ ] **Run end-to-end on your bug.** Execute the full graph on your playground issue. Let it loop through review cycles. Inspect the final code change and Langfuse trace.
      > **Theory:** Integration test for the entire system. Expect failure on the first run — that's data, not defeat. The Langfuse trace shows you the exact node and iteration where things went wrong.
      > **Trade-offs:** If it fails, resist the urge to tweak prompts randomly. Read the trace, identify the failing node, fix that node in isolation, then re-run the graph.

  - [ ] **Add Langfuse cross-node tracing.** Each graph node becomes a Langfuse span nested under a parent trace. After 3 runs, identify which node has the highest latency and which fails most often.
      > **Theory:** Cross-node tracing turns "the agent is slow" into "the Reviewer node averages 8 seconds because it receives the entire file content as context." This is how you prioritize optimization — same as profiling a slow API endpoint.
      > **Trade-offs:** Latency (optimize with smaller context, faster model) vs quality (larger context, better model). The trace tells you which node justifies the trade-off.

  - [ ] **Document failure modes.** Write a short `FAILURES.md`: what caused infinite loops, what made the reviewer too lenient or too harsh, where cost spiked. This is interview material.
      > **Theory:** Postmortems are a core engineering practice. Writing down what broke and why — even in a solo project — builds the habit of systematic improvement over vibes-based debugging. Interviewers love candidates who can articulate failure modes.
      > **Trade-offs:** Be specific ("Reviewer approved patches that didn't modify the failing test") not vague ("the agent didn't work"). Specificity shows you actually ran the system.

### Project 2: Human-in-the-Loop Research Synthesis Pipeline
Create an automated report generator that leverages web search and your Unit 1 RAG database to assemble a structured whitepaper.

* **What you will do:**

  **Week 1 — Multi-agent nodes and parallel writing**

  - [ ] **Pick a research topic.** Choose a narrow technical topic covered by both your Unit 1 document corpus and publicly searchable sources (e.g. "hybrid search in RAG systems").
     > **Theory:** Topic scope determines pipeline success. Too broad ("machine learning") → shallow sections and hallucinations. Too narrow ("BM25 k1 parameter tuning") → not enough sources. Pick something your Unit 1 corpus already covers so you can compare RAG vs web sources.
     > **Trade-offs:** Topic you know well (easy to spot hallucinations) vs unfamiliar topic (tests the system more honestly, harder to evaluate). Know the topic well for your first run.

  - [ ] **Define graph state and schemas.** `GraphState`: `topic`, `sources`, `outline`, `sections`, `final_draft`, `judge_scores`. Schemas: `Source(url, title, excerpt)`, `Outline(sections: list[str])`, `Section(title, content)`, `JudgeScore(coherence, groundedness, completeness)`.
     > **Theory:** Same shared-state pattern as Project 1, but the data is documents instead of code. Each schema represents a stage in the pipeline: sources are raw material, outline is the plan, sections are drafts, final_draft is the merged output.
     > **Trade-offs:** Store full source text in state (simple, blows up context) vs store source IDs and re-fetch (compact, adds complexity). Store excerpts (first 500 chars) in state for now; full text lives in your RAG DB.

  - [ ] **Build the Researcher node.** Query your Unit 1 RAG system and a web search API. Collect 5–10 sources into `state.sources`. Test in isolation.
     > **Theory:** The Researcher combines two retrieval backends with different strengths. Your RAG system has deep, trusted content from your corpus. Web search has breadth and recency. Using both is standard in production research agents.
     > **Trade-offs:** RAG only (grounded, limited to your corpus) vs web only (broad, noisier, may hit paywalls) vs both (best coverage, dedup needed). Use both; deduplicate by URL/title before passing to the next node.

  - [ ] **Build the Outline generator.** Researcher output feeds into a node that proposes a document outline (`Outline` schema). This is what the human will review.
     > **Theory:** Planning before execution — the same principle as the Planner in Project 1. A bad outline produces bad sections no matter how good the Writers are. This node is cheap (one LLM call) relative to the Writers (one call per section).
     > **Trade-offs:** LLM-generated outline (fast, may miss important angles) vs template-based outline (predictable, inflexible). LLM-generated is fine when a human reviews it at the interrupt point.

  - [ ] **Add the LangGraph interrupt.** Insert an `interrupt_before=["writer"]` breakpoint. When the graph pauses, print the proposed outline to the terminal and wait for keyboard input: approve, or type edits.
     > **Theory:** Human-in-the-loop (HITL) is a checkpoint, not a crutch. The interrupt lets you catch bad plans before expensive downstream work runs. LangGraph persists graph state at the interrupt — you can walk away and resume later. This is critical for production workflows where a human must approve before publishing.
     > **Trade-offs:** HITL on every run (high quality, doesn't scale) vs HITL only during development (train your confidence, then automate) vs no HITL (fast, risky). Use HITL while building; remove it once judge scores are consistently high.

  - [ ] **Build the Writer nodes.** For each section in the approved outline, spawn a parallel writer (async Python or LangGraph `Send` API). Each writer receives relevant sources and returns a `Section`.
     > **Theory:** Parallel execution is the main reason to use a graph framework over a script. Three sections written sequentially = 3× latency. Parallel writers run simultaneously, each with only the sources relevant to their section. LangGraph's `Send` API fans out work to multiple nodes dynamically.
     > **Trade-offs:** Parallel writers (fast, sections may contradict each other) vs sequential writers with shared context (slower, more consistent). Parallel + Editor merge (step 7) is the standard pattern. The Editor exists to fix cross-section inconsistencies.

  **Week 2 — Editing, judging, and observability**

  - [ ] **Build the Editor node.** Takes all `Section` objects, merges them into a single Markdown document, and checks for contradictions between sections. Output `final_draft`.
     > **Theory:** The Editor is a reduce step — it combines parallel outputs into a coherent whole. This is MapReduce logic: Writers map (parallel), Editor reduces (merge). The Editor should check for factual contradictions (section 2 says X, section 4 says not-X) and tonal inconsistencies.
     > **Trade-offs:** LLM editor (catches subtle issues, may introduce its own edits) vs programmatic merge (concatenate sections, no quality check). LLM editor is worth it — this is where parallel-writing inconsistencies get caught.

  - [ ] **Build the LLM-as-Judge node.** A separate model call receives `final_draft` + original `sources` and returns a `JudgeScore`. Prompt it to flag specific sentences that look hallucinated.
     > **Theory:** LLM-as-judge uses a model to evaluate another model's output. It's not ground truth — judges have their own biases and blind spots — but it automates quality scoring at scale. Ask for specific flagged sentences, not just a number, so you can verify the judge's reasoning.
     > **Trade-offs:** LLM judge (scalable, biased toward verbose/fluent text) vs human eval (gold standard, doesn't scale) vs Ragas metrics (standardized, may not fit long-form reports). Use LLM judge for long-form; Ragas for Q&A. Always spot-check 2–3 judge verdicts manually.

  - [ ] **Wire the full LangGraph.** Researcher → Outline → **[interrupt]** → Writers (parallel) → Editor → Judge → END. Run once without the interrupt to verify the graph executes.
     > **Theory:** Wire the happy path first without the interrupt to confirm the graph topology is correct. Then add the interrupt. This is standard integration testing — validate structure before adding complexity.
     > **Trade-offs:** If the graph fails without the interrupt, the problem is in node wiring, not HITL logic. Debug topology first.

  - [ ] **Run with human approval.** Execute the full pipeline. When interrupted, modify the outline (e.g. add or remove a section), then resume. Compare the final draft with and without your edits.
      > **Theory:** This A/B test quantifies the value of human oversight. If your edit significantly improves the judge score, HITL is earning its keep. If the score is the same, your outline generator is already good enough.
      > **Trade-offs:** Document the delta — it's evidence for when to keep or remove the interrupt in production.

  - [ ] **Log everything to Langfuse.** Parent trace for the run, child spans per agent node, judge scores as metadata on the final span. Verify you can see which writer was slowest.
      > **Theory:** Multi-agent pipelines have combinatorial failure modes. Tracing lets you answer: "Was the Researcher slow, or did one Writer hang?" Without per-node spans, you're debugging a black box.
      > **Trade-offs:** Tag parallel writer spans with `section_title` metadata so you can filter/compare them in the Langfuse UI.

  - [ ] **Run the judge eval 3 times.** Execute the pipeline on 3 different topics. Record `JudgeScore` for each. Note which step produces the lowest groundedness score — that's your improvement target.
      > **Theory:** Three runs reveal variance — LLM outputs are non-deterministic. If groundedness is consistently low, the Researcher isn't finding good sources. If coherence is low, the Editor isn't merging well. The lowest-scoring dimension tells you which node to fix.
      > **Trade-offs:** Three topics (minimum for pattern spotting) vs one topic run 3 times (measures variance, not generalization). Different topics test generalization; same topic tests consistency. Do different topics.
