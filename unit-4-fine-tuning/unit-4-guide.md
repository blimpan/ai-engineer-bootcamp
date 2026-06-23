# Unit 4: Fine-Tuning Basics, Local Inference & Serving Optimization

## Progress Summary
- **Current step:** _Not started_
- **Last updated:** _YYYY-MM-DD_
- **Notes/blockers:** _None_

## Main Topics
* Parameter-Efficient Fine-Tuning (LoRA / QLoRA)
* Dataset Curation & Synthetic ChatML/JSONL Formatting
* Local Training Infrastructures (Unsloth / HuggingFace PEFT)
* Quantization & Model Serialization
* Local Weight Serving (Ollama / vLLM Engines)
* Hardware Optimization (Context Caching & Speculative Decoding)
* Latency Benchmarking (Time-to-First-Token / TTFT)
* Comparative Evaluation (Pre- vs Post-Tune Benchmarks)

## Short Description of Goals
Conduct a task-specific LoRA fine-tune by picking a narrow task your agent system does badly — e.g. matching a highly specific behavioral persona, or writing in a specific style. Curate ~200–500 examples in chat format (instruction / response pairs). Fine-tune Llama 3 8B or Mistral 7B using QLoRA via Unsloth (free on Colab T4). Run your unit 1 eval harness against the fine-tuned model before and after — use the same Ragas metrics so the comparison is apples-to-apples. Log fine-tuning runs to Langfuse or W&B so you can correlate training decisions with eval outcomes. Slot the fine-tuned model into your agent system by replacing one API call in your unit 3 system with your model served locally via Ollama or vLLM. Benchmark latency, cost, and quality vs the API to find where fine-tuning wins. Optimize local inference parameters, experimenting with Context Caching and Speculative Decoding in vLLM to minimize Time-To-First-Token (TTFT) during agent loops. Verify that your structured output layer from unit 2 works unchanged and the fine-tuned model still respects the Pydantic schemas. Try merging your LoRA adapter back into the base model for a single deployable file. Understand LoRA adapters, data formatting, overfitting, how to measure improvement rigorously, local inference, adapter merging, inference latency optimization, and when fine-tuning is actually worth it.

---

## Concrete Projects (2 Hours/Day Practice)

### Project 1: Fine-Tuning a Custom Persona Chatbot
Train a small open-source model to adopt a highly specific communication style and behavioral persona without relying on massive system prompts.

* **Why it fits the timeframe:** `Unsloth` abstracts away the boilerplate of raw HuggingFace training scripts and cuts fine-tuning runtime on a T4 GPU to under 30 minutes.

* **What you will do:**

  **Week 1 — Dataset curation and baseline evaluation**

  - [ ] **Choose your persona.** Pick a distinct voice with clear linguistic markers (e.g. a terse senior engineer, a specific historical figure, a strict API documentation writer). Write 5 example exchanges by hand that capture the style.
     > **Theory:** Fine-tuning bakes behavior into weights; prompting instructs at runtime. Fine-tuning wins when you need consistent style across thousands of calls and system prompts become too long or too easy to override. Persona is a good learning exercise because results are easy to judge by eye.
     > **Trade-offs:** Persona fine-tune (easy to evaluate, limited production value) vs task-specific fine-tune on a failure mode from Unit 3 (harder, more portfolio-worthy). Persona is fine for learning mechanics; consider switching to a real failure mode if you have time. Always try prompting first — if 3 examples in a system prompt achieve 80% of the result, fine-tuning may not be worth the effort.

  - [ ] **Define the dataset schema.** Each training example is a multi-turn ChatML JSONL record: `{ "messages": [{"role": "user", ...}, {"role": "assistant", ...}] }`. Write a validator script that checks format before training.
     > **Theory:** Format is not cosmetic — training frameworks expect exact token templates. ChatML wraps messages in special tokens (`<|im_start|>user\n...`) so the model learns which tokens are instructions vs responses. A single malformed record can corrupt a training batch.
     > **Trade-offs:** ChatML format (standard for Llama/Mistral) vs Alpaca format (`### Instruction: ... ### Response: ...`) vs ShareGPT format. Match the format to your base model's training template. Unsloth handles ChatML for Llama 3 — use that.

  - [ ] **Generate synthetic training data.** Write a script that calls a frontier model (GPT-4o or Claude) with your 5 hand-written examples as few-shot prompts. Generate 200–500 diverse conversations. Save to `train.jsonl`.
     > **Theory:** Synthetic data generation (teaching a small model using a large model's outputs) is called distillation. The quality ceiling of your fine-tuned model is set by the teacher model's outputs. Diversity matters more than volume — 200 varied conversations beat 500 near-duplicates.
     > **Trade-offs:** Synthetic via frontier model (fast, inherits teacher's style, may inherit teacher's mistakes) vs hand-written (highest quality, doesn't scale) vs curated from real logs (best if available, you don't have them yet). Generate synthetic, then manually review and cull the worst 10%.

  - [ ] **Split and inspect the data.** Hold out 20 examples as a test set (`test.jsonl`). Manually read 10 random training examples — do they actually sound like the persona? Remove any that don't.
     > **Theory:** Data quality beats data quantity in fine-tuning. One bad example teaching the wrong style can pull the model in the wrong direction. The held-out test set must never appear in training — contamination inflates your eval scores artificially.
     > **Trade-offs:** 20-example test set (small but usable for persona) vs 50+ (better statistics, harder to hand-curate). 20 is fine for a persona task. For production, you'd want 100+.

  - [ ] **Establish a baseline.** Run your 20-example test set against the base model (Llama 3 8B via API or Ollama) with no fine-tuning. Score with an LLM-as-judge: "Does this response match the persona on a 1–5 scale?" Save baseline scores.
     > **Theory:** You cannot claim fine-tuning helped without a before measurement. This is the same eval discipline from Unit 1 — baseline first, change second, compare third. Also try a strong system prompt on the base model; if prompting alone scores 4/5, fine-tuning has less room to improve.
     > **Trade-offs:** LLM-as-judge (fast, subjective) vs human eval (gold standard, slow) vs perplexity on test set (objective, doesn't measure persona quality). Use LLM judge for iteration; spot-check 5 examples yourself before claiming success.

  - [ ] **Set up the training environment.** Create a Google Colab notebook with `Unsloth`, load `Llama 3 8B` (or `Mistral 7B`) in 4-bit quantization. Confirm the model loads on a T4 without OOM.
     > **Theory:** QLoRA (Quantized LoRA) loads the base model in 4-bit precision (quarter the VRAM) and trains small adapter matrices on top. An 8B model needs ~16GB in fp16 but only ~5GB in 4-bit — that's why it fits on a free Colab T4 (16GB). Unsloth optimizes the training kernels for 2–5× speedup over raw HuggingFace.
     > **Trade-offs:** Unsloth (fast, opinionated, great for learning) vs raw HuggingFace PEFT (more control, more boilerplate) vs Axolotl (config-driven, good for serious training). Unsloth for this project. Colab free tier (T4, may disconnect) vs Colab Pro (A100, reliable) vs local GPU. Free tier is fine for 8B QLoRA.

  **Week 2 — Training, evaluation, and integration**

  - [ ] **Configure QLoRA.** Set LoRA rank, alpha, target modules (attention layers), learning rate, and epochs. Use Unsloth defaults as a starting point. Log hyperparameters to W&B or a local `training_log.json`.
     > **Theory:** LoRA injects small trainable matrices (rank *r*) into attention layers while freezing the base model. Rank controls capacity: low rank (8–16) = subtle style changes; high rank (64–128) = more capacity but risks overfitting on small datasets. Alpha scales the adapter's learning rate relative to rank. You are *not* retraining the whole model — just nudging its behavior.
     > **Trade-offs:** Rank 16 (safe default for 200 examples) vs rank 64 (more expressive, overfits on small data). Learning rate 2e-4 (Unsloth default) vs 1e-4 (safer, slower convergence). Start with defaults; only tune if loss doesn't decrease. Log everything — you'll want to know what you changed when comparing runs.

  - [ ] **Train the adapter.** Run fine-tuning on your `train.jsonl`. Monitor loss curve — it should decrease steadily. If it flatlines immediately, check your data format. Training should finish in under 30 minutes on a T4.
     > **Theory:** Training loss measures how well the model predicts the next token in your examples. Steady decrease = learning. Flat loss = broken data format or learning rate too low. Loss near zero = overfitting — the model is memorizing, not generalizing. For persona tasks, some overfitting is acceptable; for factual tasks, it's dangerous.
     > **Trade-offs:** 1 epoch (safe, may underfit) vs 3 epochs (better learning, overfitting risk on small data). Start with 1 epoch; add a second only if eval scores are below baseline.

  - [ ] **Evaluate post-tune.** Run the same 20-example test set against the fine-tuned model. Score with the same LLM-as-judge prompt. Compare average persona scores: baseline vs fine-tuned.
     > **Theory:** Same test set, same judge, same scoring prompt — this is a controlled experiment. Improvement on persona score is your primary success metric. Also check for regressions: does the model still answer factual questions correctly, or did persona training break general knowledge?
     > **Trade-offs:** If persona score improved but general QA degraded, you've overfit. Reduce rank, reduce epochs, or add general examples to the training mix.

  - [ ] **Run Unit 1 Ragas eval.** Swap the fine-tuned model into your Unit 1 RAG generation step. Run your 25-question eval set. Compare faithfulness and answer relevancy against your Unit 1 baseline scores.
      > **Theory:** This tests whether fine-tuning for persona/style degraded your model's ability to do useful work. A model that sounds great but hallucinates more in RAG is a net loss. Cross-task eval catches catastrophic forgetting — when training on one task silently breaks another.
      > **Trade-offs:** Acceptable regression: persona score +15%, faithfulness -2%. Unacceptable: persona score +15%, faithfulness -20%. Define your threshold before looking at results.

  - [ ] **Test structured output compatibility.** Pass the fine-tuned model through your Unit 2 `instructor` + Pydantic pipeline. Verify it still produces valid `AnalysisResult` or `ReviewVerdict` objects. Note any regressions.
      > **Theory:** Fine-tuning changes token probabilities globally. A model fine-tuned on conversational persona text may become worse at outputting strict JSON — it "wants" to chat instead of conforming to schema. This is why you test downstream compatibility, not just the fine-tune task itself.
      > **Trade-offs:** If structured output breaks, options: fine-tune on a mix of persona + structured examples, use a separate non-fine-tuned model for structured tasks, or add schema examples to the training data. Mixing is the most robust.

  - [ ] **Merge and export.** Merge the LoRA adapter back into the base model weights. Export as a single Safetensors or GGUF file. Confirm you can load it in Ollama with one command.
      > **Theory:** LoRA adapters are small delta files (tens of MB) that modify a base model's behavior. Merging bakes the adapter into the base weights, producing a single standalone model file you can deploy without carrying two files. GGUF is a quantized inference format (used by Ollama); Safetensors is the full-precision training format.
      > **Trade-offs:** Merged model (single file, easy to deploy, can't swap adapters) vs adapter-only (flexible, need base model at runtime). Merge for deployment. GGUF Q4 quantization (small, fast, slight quality loss) vs full Safetensors (large, best quality). Export both if disk space allows.

### Project 2: High-Throughput Local Inference Optimization Pipeline
Set up a local inference engine to serve your fine-tuned model and optimize it for multi-step agent loops.

* **What you will do:**

  **Week 1 — Serving setup and baseline benchmarks**

  - [ ] **Choose your serving engine.** Pick `vLLM` (preferred for benchmarking) or `Ollama` (simpler setup). Install locally or on a cloud GPU instance with enough VRAM for your 8B model.
     > **Theory:** Serving engines sit between your application and the model weights, handling batching, KV-cache management, and request scheduling. Raw `model.generate()` works for one request but can't handle concurrent agent loops efficiently.
     > **Trade-offs:** vLLM (high throughput, production-grade, complex setup, best for benchmarking) vs Ollama (one-command start, lower throughput, great for development) vs TGI (HuggingFace's server, middle ground). vLLM for Project 2's benchmarks; Ollama is fine if vLLM setup fights you.

  - [ ] **Serve the merged model.** Load your exported model from Project 1. Confirm you can send a basic chat completion request via HTTP and get a response.
     > **Theory:** An OpenAI-compatible HTTP API (`/v1/chat/completions`) is the de facto standard. If your local server speaks this format, you can swap it into any code that uses the OpenAI SDK — just change the `base_url`. This is how Unit 3's LangGraph nodes can use a local model with zero code changes.
     > **Trade-offs:** OpenAI-compatible API (drop-in replacement) vs custom API (more work to integrate). Always choose OpenAI-compatible servers.

  - [ ] **Write a benchmark script.** A Python script that sends N identical requests (start with N=10) and records per-request: time-to-first-token (TTFT), total latency, and tokens/second. Save results to `benchmark_baseline.json`.
     > **Theory:** TTFT measures how long until the first token arrives (perceived responsiveness). Total latency measures end-to-end time. Tokens/second measures generation speed. These three metrics tell different stories — an agent loop with long system prompts cares most about TTFT; batch processing cares most about throughput.
     > **Trade-offs:** Measure at the HTTP level (includes network overhead, realistic) vs at the Python SDK level (cleaner, excludes network). HTTP level is more realistic for production.

  - [ ] **Benchmark with agent-style prompts.** Instead of short queries, send multi-turn conversation histories (system prompt + 3 tool-call turns + final question) mimicking your Unit 2 ReAct loop. Record TTFT — this is the realistic number.
     > **Theory:** Short-prompt benchmarks lie. Agent prompts are 2–10K tokens of system prompt + conversation history. TTFT on a 50-token "hello" is meaningless when your agent sends 4K tokens of context per turn. Prefix length dramatically affects TTFT because the model must process the entire context before generating.
     > **Trade-offs:** Always benchmark with production-realistic prompts. Keep the short-prompt baseline for comparison, but make decisions based on agent-style numbers.

  - [ ] **Benchmark parallel requests.** Fire 5 concurrent requests using `asyncio`. Record per-request latency and throughput. Note if latency degrades under load.
     > **Theory:** Production agents don't run one at a time. vLLM's core advantage is continuous batching — it processes multiple requests simultaneously on the GPU. Under load, per-request latency may increase but total throughput goes up. The question is whether latency degradation is acceptable.
     > **Trade-offs:** 5 concurrent (modest, fits one GPU) vs 20+ (stress test, needs more VRAM). Start with 5. If latency doubles at 5 concurrent, you need a bigger GPU or quantization before adding more users.

  - [ ] **Establish cost comparison.** Calculate cost-per-1K-tokens for your local setup (GPU hourly rate / tokens processed) vs your API provider. Document the crossover point where local is cheaper.
     > **Theory:** Local inference has high fixed cost (GPU) and low marginal cost (per token). API inference has zero fixed cost and higher marginal cost. The crossover depends on volume: at 1M tokens/month, API is usually cheaper; at 100M tokens/month, local wins. Agent loops are token-hungry, which shifts the crossover lower.
     > **Trade-offs:** Cloud GPU ($0.50–2.00/hr) vs owned hardware (upfront cost, cheaper long-term) vs API ($0.01–0.06/1K tokens). Calculate your actual crossover with real numbers, not rules of thumb.

  **Week 2 — Optimization, integration, and final comparison**

  - [ ] **Enable context caching in vLLM.** Configure prefix caching so repeated system prompts across agent turns are not re-computed. Re-run the agent-style benchmark from step 4.
     > **Theory:** In agent loops, the system prompt and early conversation turns are identical across iterations — only the latest message changes. Prefix caching stores the KV-cache (attention key/value tensors) for the shared prefix, so only new tokens are processed. This can cut TTFT by 50–80% on multi-turn agent conversations.
     > **Trade-offs:** Prefix caching (huge win for agent loops, uses more GPU memory) vs no caching (simpler, re-processes everything). Enable it for agent workloads. Memory cost is usually worth it.

  - [ ] **Enable speculative decoding.** Configure a smaller draft model for speculative decoding in vLLM. Re-run the same benchmark. Compare TTFT and total latency against step 7 results.
     > **Theory:** Speculative decoding uses a small "draft" model to predict several tokens ahead, then a larger "verification" model checks them in parallel. Accepted tokens are essentially free — you get multiple tokens of output for one forward pass of the large model. It improves tokens/second without changing output quality.
     > **Trade-offs:** Speculative decoding (faster generation, needs a compatible draft model, more VRAM) vs larger batch sizes (simpler, less speedup). Try speculative decoding if tokens/second is your bottleneck; skip if TTFT (prefix processing) is the bottleneck — caching helps more there.

  - [ ] **Plot the results.** Generate a simple bar chart or table: baseline vs context caching vs speculative decoding, for both single and parallel request modes. Save to `benchmark_optimized.json`.
     > **Theory:** Optimization without measurement is guessing. A simple table showing TTFT and throughput for each configuration is portfolio-grade evidence. It shows you understand the performance characteristics of your system, not just that you enabled a flag.
     > **Trade-offs:** Focus on the metric that matters for your use case. Agent loops → TTFT. Batch processing → throughput. Don't optimize the wrong metric.

  - [ ] **Slot into Unit 3.** Replace one LLM call in your Unit 3 LangGraph (e.g. the Writer node) with your local endpoint. Run the research pipeline end-to-end on local inference.
      > **Theory:** Integration test for the full 8-week stack. Change the `base_url` in your OpenAI client config — the LangGraph, Pydantic schemas, and Langfuse tracing should all work unchanged. If they don't, you've found a compatibility gap worth documenting.
      > **Trade-offs:** Replace the cheapest node first (Writer — many calls, shorter prompts) vs the most expensive node (Researcher — fewer calls, longer context). Writer is the safer first swap.

  - [ ] **Compare quality on the full pipeline.** Run the Unit 3 pipeline once on API and once on local. Compare `JudgeScore` results and total wall-clock time. Note where quality dropped (if at all).
      > **Theory:** Local models are typically weaker than frontier APIs. The question isn't "is local better?" — it's "is local good enough for this specific node, and does the cost savings justify the quality trade-off?" A 10% groundedness drop on the Writer may be acceptable; a 30% drop is not.
      > **Trade-offs:** Hybrid deployment (frontier model for hard nodes, local for easy nodes) is the production answer. You don't have to go all-local or all-API. Document which nodes can be local.

  - [ ] **Write a summary decision doc.** A short `LOCAL_VS_API.md`: for which tasks local inference wins on cost, which on latency, where quality regressed, and when you'd still use an API. This closes the 8-week loop.
      > **Theory:** Architecture decision records (ADRs) are how senior engineers communicate trade-offs to their team. This doc is a solo ADR: "We chose local inference for Writer nodes because TTFT with prefix caching is 200ms vs 800ms API, and judge scores dropped only 5%." That's a hireable conclusion.
      > **Trade-offs:** Be honest about where local loses. "Local is cheaper but hallucinates more on research tasks" is more credible than "local is better at everything."
