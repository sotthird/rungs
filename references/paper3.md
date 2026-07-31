
https://arxiv.org/html/2603.26557v1


MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference
=================================================================

Joris Köster1, Zixuan Liu2, Siavash Khajavi3, Zizhan Zheng2,  
1 Department of Computer Science, Aalto University, Finland,  
2Department of Computer Science, Tulane University, LA, USA,  
3Department of Industrial Engineering and Management, Aalto University, Finland,  
{joris.koster,siavash.khajavi}@aalto.fi, {zliu41,zzheng3}@tulane.edu

###### Abstract

Large Language Models (LLMs) deliver strong performance but incur high inference cost in real-world services, especially under workloads with repeated or near-duplicate queries across users and sessions. In this work, we propose MemBoost, a memory-boosted LLM serving framework that enables a lightweight model to reuse previously generated answers and retrieve relevant supporting information for cheap inference, while selectively escalating difficult or uncertain queries to a stronger model. Unlike standard retrieval-augmented generation, which primarily grounds a single response, MemBoost is designed for interactive settings by supporting answer reuse, continual memory growth, and cost-aware routing. Experiments across multiple models under simulated workloads show that MemBoost substantially reduces expensive large-model invocations and overall inference cost, while maintaining high answer quality comparable to the strong model baseline.

MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference

Joris Köster1, Zixuan Liu2, Siavash Khajavi3, Zizhan Zheng2, 1 Department of Computer Science, Aalto University, Finland, 2Department of Computer Science, Tulane University, LA, USA, 3Department of Industrial Engineering and Management, Aalto University, Finland, {joris.koster,siavash.khajavi}@aalto.fi, {zliu41,zzheng3}@tulane.edu

1 Introduction
--------------

Large Language Models (LLMs) Hurst et al. ([2024](#bib.bib4 "Gpt-4o system card")); Team et al. ([2023](#bib.bib5 "Gemini: a family of highly capable multimodal models")) have demonstrated strong capabilities in natural language understanding, instruction following Chung et al. ([2024](#bib.bib10 "Scaling instruction-finetuned language models")); Ouyang et al. ([2022](#bib.bib8 "Training language models to follow instructions with human feedback")), coding Gao et al. ([2023](#bib.bib6 "Pal: program-aided language models")); Ni et al. ([2024](#bib.bib7 "L2ceval: evaluating language-to-code generation capabilities of large language models")), and decision-making Anil et al. ([2023](#bib.bib9 "Palm 2 technical report")); Wei et al. ([2022](#bib.bib11 "Chain-of-thought prompting elicits reasoning in large language models")). However, deploying frontier-scale models in real-world services remains expensive, as inference often requires multiple high-end GPUs, especially for long-form reasoning and explanation-heavy responses Kwon et al. ([2023](#bib.bib15 "Efficient memory management for large language model serving with pagedattention")); Park et al. ([2025](#bib.bib17 "A survey on inference engines for large language models: perspectives on optimization and efficiency")). These costs are amplified in production settings where many user queries are repeated or near duplicates, causing redundant computation. A widely adopted direction to mitigate these costs is to shift part of the workload to smaller models with retrieval-augmented generation (RAG) Lewis et al. ([2020](#bib.bib12 "Retrieval-augmented generation for knowledge-intensive nlp tasks")); Guu et al. ([2020](#bib.bib13 "Retrieval augmented language model pre-training")); Borgeaud et al. ([2022](#bib.bib14 "Improving language models by retrieving from trillions of tokens")). RAG allows a model to externalize knowledge into a searchable index and condition generation on retrieved evidence, reducing the need to store everything in parameters Izacard et al. ([2023](#bib.bib1 "Atlas: few-shot learning with retrieval augmented language models")); Fan et al. ([2025](#bib.bib16 "Minirag: towards extremely simple retrieval-augmented generation")). For example, Atlas Izacard et al. ([2023](#bib.bib1 "Atlas: few-shot learning with retrieval augmented language models")) demonstrates that the RAG model with far fewer parameters can be competitive with, and in some regimes outperform, much larger models on knowledge-intensive tasks.

However, most existing RAG work primarily targets knowledge grounding, i.e., improving the answer to a single query by retrieving external documents Lewis et al. ([2020](#bib.bib12 "Retrieval-augmented generation for knowledge-intensive nlp tasks")); Guu et al. ([2020](#bib.bib13 "Retrieval augmented language model pre-training")). In contrast, interactive LLM services exhibit additional properties that are not addressed by standard RAG alone. First, many queries that are semantically equivalent are repeatedly requested across users and sessions. Recomputing these same answers wastes GPU time. This motivates semantic caching, which retrieves a previously generated response when a new query is semantically similar Gill et al. ([2025](#bib.bib2 "MeanCache: user-centric semantic caching for llm web services")); Yu et al. ([2025](#bib.bib18 "Smartcache: context-aware semantic cache for efficient multi-turn llm inference")); Liu et al. ([2025](#bib.bib19 "Semantic caching for low-cost llm serving: from offline learning to online adaptation")). However, making this reliable requires careful control. Second, deployed LLMs continuously generate new high-quality answers. When these answers are valuable, they should be written back into the retrieval memory so that a semantic cache can serve future similar queries cheaply. However, this write-back mechanism raises practical questions, such as what/when to store it. Finally, although small models augmented with retrieval can be competitive on knowledge-intensive tasks, their capabilities remain limited on more challenging requests. A practical system should therefore enable the small model to escalate to a stronger model when retrieval is insufficient, aligning with recent work on routing across multiple LLMs Ong et al. ([2024](#bib.bib3 "Routellm: learning to route llms with preference data")); Zhang et al. ([2025](#bib.bib20 "Router-r1: teaching llms multi-round routing and aggregation via reinforcement learning")); Moslem and Kelleher ([2026](#bib.bib21 "Dynamic model routing and cascading for efficient llm inference: a survey")) to balance quality and cost.

These observations suggest a missing system-level paradigm: a unified approach that enables a small model to cheaply reuse supporting knowledge or prior answers from semantic memory, selectively defer to a stronger but costly model when retrieval is insufficient, and continuously write back useful new information into the retrieval memory, while explicitly balancing answer quality and inference cost. In this paper, we introduce MemBoost, a memory-boosted architecture with three components: (1) an Associative Memory Engine (AME) that performs fast semantic retrieval and supports write-back of newly generated answers; (2) a high-capability Large-LLM Oracle that provides an accurate fallback when memory is insufficient; and (3) a lightweight Meta Controller, which is a small LLM that composes the final response by either reusing from memory or routing the query to the oracle, and support write back mechanism for future reuse, thereby balancing quality and cost. Together, MemBoost turns inference into a “retrieve-or-escalate” decision problem, substantially reducing expensive large-model calls while preserving answer quality. Experiments on the MMLU-Pro dataset Wang et al. ([2024](#bib.bib34 "MMLU-pro: a more robust and challenging multi-task language understanding benchmark")) under a simulated workload with repeated and near-duplicate queries show that MemBoost substantially reduces calls to the costly large LLM while largely preserving the answer quality of the oracle model.

2 MemBoost: Memory-Boosted LLM Serving Framework
------------------------------------------------

### 2.1 Problem Setup

Unlike standard single-turn question answering (QA) tasks studied in prior work, we consider a setting with continuous interaction between multiple users and an LLM service, where incoming queries may be exact duplicates or semantic near-duplicates over time. At each time step tt, the system receives a user query xtx\_{t} and produces an answer yty\_{t}. We use a ground-truth quality signal r​(xt,yt)r(x\_{t},y\_{t}) to measure whether the response is helpful, correct, and aligned with the user’s intent. Our objective is to maximize the average response quality over a horizon TT, max⁡1T​∑t\=1Tr​(xt,yt),\\max\\;\\frac{1}{T}\\sum\_{t=1}^{T}r(x\_{t},y\_{t}), while minimizing the overall inference cost, as formalized in Section [2.3](#S2.SS3 "2.3 System Cost ‣ 2 MemBoost: Memory-Boosted LLM Serving Framework ‣ MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference").

### 2.2 Overview of MemBoost

As discussed above, in production settings, repeatedly invoking a frontier model to answer semantically redundant queries wastes substantial compute. To mitigate this inefficiency while balancing quality and cost, we propose MemBoost, a memory-boosted LLM serving system composed of three components: Associative Memory Engine, Large-LLM Oracle and Meta Controller, as illustrated in Figure [1](#S2.F1 "Figure 1 ‣ 2.2 Overview of MemBoost ‣ 2 MemBoost: Memory-Boosted LLM Serving Framework ‣ MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference").

![Refer to caption](2603.26557v1/x1.png)

Figure 1: Overview of MemBoost. For each incoming query, the AME retrieves a small set of relevant memory entries. The MC then either composes an answer from the retrieved results or escalates to the Oracle. When the Oracle is used, the MC decides whether to write the new answer back into the AME for future reuse.

#### Associative Memory Engine (AME).

AME maintains an external memory that stores auxiliary knowledge as well as previously answered queries and their associated responses, and supports fast semantic retrieval (details in Appendix [B](#A2 "Appendix B Additional Implementation Details ‣ MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference")). Concretely, given a new query xtx\_{t} at time tt, AME retrieves a small set of KK candidate memory entries ℳt\={(x(i),y(i),m(i))}i\=1K,\\mathcal{M}\_{t}=\\{(x^{(i)},y^{(i)},m^{(i)})\\}\_{i=1}^{K}, where m(i)m^{(i)} denotes metadata (e.g., query category). The AME stores the question-answer pair for retrieval together with the qid (a unique identifier provided by the dataset), the category and the timestamp. In addition, AME supports write-back by storing newly generated high-quality entries (xt,yt,mt)(x\_{t},y\_{t},m\_{t}) when instructed by the Meta Controller.

#### Large-LLM Oracle.

The oracle is a high-capability model that produces high-quality answers but incurs high inference cost. We use it as a fallback: the system queries the oracle when retrieval from AME is missing, ambiguous, or deemed unreliable by the Meta Controller. We denote the oracle’s response to query xtx\_{t} as yt⋆\=Oracle​(xt).y\_{t}^{\\star}=\\mathrm{Oracle}(x\_{t}).

#### Meta-Controller (MC).

MC is a lightweight LLM that orchestrates the interaction among the user, AME, and the oracle. At each step, MC decides whether to answer using information retrieved from AME or to escalate to the oracle. When escalation occurs, MC also determines whether the newly generated query answer pair should be stored in AME for future reuse. Concretely, given a query xtx\_{t}, MC first requests retrieval from AME and obtains ℳt\\mathcal{M}\_{t}. Conditioned on xtx\_{t} and ℳt\\mathcal{M}\_{t}, MC either returns an answer directly, yt\=MC​(xt,ℳt)y\_{t}=\\mathrm{MC}(x\_{t},\\mathcal{M}\_{t}), or calls the oracle to obtain yt⋆\=Oracle​(xt)y\_{t}^{\\star}=\\mathrm{Oracle}(x\_{t}) and returns yt\=yt⋆y\_{t}=y\_{t}^{\\star}. In the latter case, MC may optionally write (xt,yt⋆,mt)(x\_{t},y\_{t}^{\\star},m\_{t}) back into AME, where mtm\_{t} denotes metadata inferred by MC. This “retrieve →\\rightarrow decide →\\rightarrow escalate (if needed) →\\rightarrow write-back (if needed)” loop reduces expensive oracle calls under repeated queries while preserving high answer quality when retrieval is insufficient.

### 2.3 System Cost

To model system efficiency, let cO​(xt)c\_{O}(x\_{t}) denote the cost of an oracle call at time tt (e.g., GPU time, energy, or monetary cost), cM​(xt)c\_{M}(x\_{t}) the cost of running the lightweight MC, and cR​(xt)c\_{R}(x\_{t}) denote the cost of retrieval (e.g., CPU time). In typical deployments, cO​(xt)c\_{O}(x\_{t}) is much larger than cM​(xt)+cR​(xt)c\_{M}(x\_{t})+c\_{R}(x\_{t}), since frontier-model inference dominates GPU compute, whereas retrieval is primarily CPU-bound and inexpensive relative to large-model generation. Let It∈{0,1}I\_{t}\\in\\{0,1\\} indicate whether the retrieval information from AME is used at time tt (It\=1I\_{t}=1 if memory is used, and It\=1I\_{t}=1 if the system escalates to the oracle). The total cost over TT steps is

CT\=∑t\=1T(cM​(xt)+cR​(xt))+∑t\=1T(1−It)​cO​(xt).C\_{T}=\\sum\_{t=1}^{T}\\bigl(c\_{M}(x\_{t})+c\_{R}(x\_{t})\\bigr)+\\sum\_{t=1}^{T}(1-I\_{t})\\,c\_{O}(x\_{t}).

By contrast, an oracle-only baseline would cost approximately CToracle\=∑t\=1TcO​(xt)C\_{T}^{\\text{oracle}}=\\sum\_{t=1}^{T}c\_{O}(x\_{t}). So our framework achieves cost savings whenever CT<CToracleC\_{T}<C\_{T}^{\\text{oracle}}. Equivalently, this condition can be written as

∑t\=1TIt​cO​(xt)\>∑t\=1T(cM​(xt)+cR​(xt)).\\sum\_{t=1}^{T}I\_{t}\\,c\_{O}(x\_{t})>\\sum\_{t=1}^{T}\\bigl(c\_{M}(x\_{t})+c\_{R}(x\_{t})\\bigr).

Putting quality and cost together, our goal is to achieve oracle-level quality with significantly fewer oracle invocations:

1T​∑t\=1Tr​(xt,yt)≈1T​∑t\=1Tr​(xt,yt⋆),CT<CToracle\\frac{1}{T}\\sum\_{t=1}^{T}r(x\_{t},y\_{t})\\approx\\frac{1}{T}\\sum\_{t=1}^{T}r\\bigl(x\_{t},y\_{t}^{\\star}\\bigr),C\_{T}<C\_{T}^{\\text{oracle}}

3 Experiments
-------------

In this section, we evaluate MemBoost in terms of both (i) reducing inference cost and (ii) retaining high response quality.

### 3.1 Experiment Setup

We evaluate the proposed MemBoost framework on the MMLU-Pro dataset Wang et al. ([2024](#bib.bib34 "MMLU-pro: a more robust and challenging multi-task language understanding benchmark")), a widely used and challenging benchmark that covers diverse disciplines and is designed to more rigorously assess LLM capabilities. To emulate real-world LLM-serving workloads where many requests are repeated or near-duplicated, we generate a query stream by sampling MMLU-Pro questions according to a Zipf distribution Zipf ([1949](#bib.bib48 "Human behavior and the principle of least effort: an introduction to human ecology")). This produces a heavy-tailed access pattern in which a small number of questions occur frequently while most questions appear rarely, capturing the repetition behavior commonly observed in practice. We vary the Zipf parameter (α\\alpha) to obtain different repetition rates and study how workload skew affects MemBoost. For the lightweight Meta Controller (MC), we evaluate several small-scale LLMs, including Qwen-3.5-2B Qwen Team ([2026](#bib.bib47 "Qwen3.5-2b")), Ministral-3-3B-Instruct-2512 Liu et al. ([2026](#bib.bib37 "Ministral 3")), and Qwen3-4B-Instruct-2507-FP8 Yang et al. ([2025](#bib.bib43 "Qwen3 technical report")). For the Large-LLM Oracle, we use Qwen3-14B-FP8-dynamic Yang et al. ([2025](#bib.bib43 "Qwen3 technical report")); Micikevicius et al. ([2022](#bib.bib46 "FP8 formats for deep learning")). To support fast retrieval in the Associative Memory Engine (AME), we embed stored query-answer pairs using all-MiniLM-L6-v2 Wang et al. ([2020](#bib.bib39 "MiniLM: deep self-attention distillation for task-agnostic compression of pre-trained transformers")); Reimers and Gurevych ([2019](#bib.bib40 "Sentence-bert: sentence embeddings using siamese bert-networks")) and perform approximate nearest-neighbor search with a FAISS cosine-similarity index Douze et al. ([2024](#bib.bib41 "The faiss library")). Additional implementation details can be found in the Appendix  [B](#A2 "Appendix B Additional Implementation Details ‣ MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference").

### 3.2 Results

Model

Accuracy (%)

Zipf 0.8

Zipf 1.1

Zipf 1.4

Baselines

Qwen3.5-2B

50.0

43.5

37.1

Ministral-3-3B-Instruct-2512

53.8

46.4

38.2

Qwen3-4B-Instruct-2507-FP8

74.5

75.6

80.5

Qwen3-14B-FP8-dynamic (Oracle)

76.4

79.9

85.0

MemBoost (ours)

MemBoost (Qwen3.5-2B)

76.7

81.8

87.4

MemBoost (Ministral-3-3B-Instruct-2512)

76.2

79.7

85.0

MemBoost (Qwen3-4B-Instruct-2507-FP8)

76.1

79.8

85.0

Table 1: Accuracy (%) of different methods on MMLU-Pro under Zipf-sampled query streams with varying repetition rates (Zipf α\\alpha).

Table [1](#S3.T1 "Table 1 ‣ 3.2 Results ‣ 3 Experiments ‣ MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference") reports accuracy (%) against the MMLU-Pro ground-truth labels for different methods under Zipf-sampled query streams with varying repetition rates (Zipf exponent α\\alpha). Across all settings, MemBoost consistently improves over the corresponding small-model baselines and achieves performance comparable to the oracle. Notably, MemBoost with Qwen3.5-2B even outperforms the oracle in all three workloads. We attribute this to MemBoost’s ability to reuse previously generated high-quality answers through semantic memory: once a question (or a near-duplicate) has been answered correctly and written back, subsequent occurrences can be served directly from memory, avoiding additional generation errors. Finally, MemBoost exhibits a clear improvement as the workload becomes more skewed (larger α\\alpha), since higher repetition rates lead to more memory hits, thereby increasing the fraction of queries answered using stored correct responses.

![Refer to caption](2603.26557v1/x2.png)

Figure 2: Average memory-use rate I¯t\\overline{I}\_{t} (200-step window) over a 5,000-step Zipf-sampled query stream. Higher I¯t\\overline{I}\_{t} indicates more queries served from AME and fewer oracle calls, implying lower total inference cost.

To quantify the serving cost of MemBoost, we report the average memory-use rate over a 200-step window, i.e., I¯t\=1200​∑s\=t−199tIs,\\overline{I}\_{t}\\;=\\;\\frac{1}{200}\\sum\_{s=t-199}^{t}I\_{s}, where Is\=1I\_{s}=1 indicates that the system answers using information retrieved from AME (and Is\=0I\_{s}=0 indicates escalation to the oracle). Under our cost model, where the oracle cost cOc\_{O} is assumed to dominate the Meta Controller and retrieval overhead cM+cRc\_{M}+c\_{R}, a larger I¯t\\overline{I}\_{t} corresponds to fewer oracle calls and thus lower total cost. Figure [2](#S3.F2 "Figure 2 ‣ 3.2 Results ‣ 3 Experiments ‣ MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference") plots I¯t\\overline{I}\_{t} over the 5,000-step workload for different MC choices and Zipf distributions. In all settings, I¯t\\overline{I}\_{t} increases over time as AME is progressively populated via the write-back mechanism, indicating that an increasing fraction of queries are served from memory rather than escalated to the oracle. Under our cost model, this implies that MemBoost achieves lower total cost than an oracle-only baseline. Moreover, memory usage is consistently higher under more skewed workloads (larger Zipf α\\alpha), where repeated queries occur more frequently, further reducing the number of expensive oracle calls.

In addition to accuracy and system cost, we also report the latency of MemBoost. Figure [3](#S3.F3 "Figure 3 ‣ 3.2 Results ‣ 3 Experiments ‣ MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference") shows the average response time over the previous 100 steps during the 5,000-step Zipf-sampled workload. As MemBoost increasingly answers queries using AME over time, the average latency steadily decreases and remains well below the oracle-only baseline.

Overall, these results indicate that MemBoost is particularly effective for repeat-heavy workloads, preserving the strong answer quality of the larger oracle model while significantly reducing system cost and response latency.

![Refer to caption](2603.26557v1/x3.png)

Figure 3: Response latency over time under Zipf-sampled workloads (average over the previous 100 steps). MemBoost reduces latency relative to the oracle-only baseline as an increasing fraction of queries are served from AME.

4 Conclusion
------------

In this paper, we introduced MemBoost, a memory-boosted architecture for efficient LLM serving under interactive, repeat-heavy workloads. By combining an Associative Memory Engine, a high-capability Large-LLM Oracle, and a lightweight Meta-Controller for routing and response composition, MemBoost reduces redundant large-model inference while preserving answer quality. We hope MemBoost provides a practical foundation for building next-generation LLM services that jointly optimize quality, cost, and responsiveness.

Limitations
-----------

While MemBoost shows promising accuracy and reduced inference cost on MMLU-Pro, the framework has not yet been evaluated in more diverse and demanding settings, such as open-ended long-form question answering or coding tasks, where outputs are longer, and error modes may differ. In addition, although Zipf sampling is used to simulate repeat-heavy workloads, the current simulation largely reflects repeated questions drawn from a fixed benchmark. In real deployments, queries are often not exact duplicates but only semantically similar, which can increase the risk of retrieval errors. A more realistic evaluation would therefore include workloads with paraphrased or semantically overlapping queries to stress-test robustness, particularly with respect to potential false hits in AME retrieval.



Appendix A Related Work
-----------------------

### A.1 Retrieval-Augmented Language Modeling

Retrieval-augmented generation (RAG) is a common way to improve language models by letting them look up relevant information from an external collection of documents, and then using that retrieved text to help produce an answer Lewis et al. ([2020](#bib.bib12 "Retrieval-augmented generation for knowledge-intensive nlp tasks")); Liu et al. ([2024](#bib.bib27 "How much can rag help the reasoning of llm?")); Du et al. ([2024](#bib.bib28 "Vul-rag: enhancing llm-based vulnerability detection via knowledge-level rag")); Mansurova et al. ([2024](#bib.bib29 "QA-rag: exploring llm reliance on external knowledge")). Early work explored different ways to combine retrieval with generation, including training the retriever together with the language model Guu et al. ([2020](#bib.bib13 "Retrieval augmented language model pre-training")), using retrieval to support open-domain question answering and other knowledge-intensive tasks Lewis et al. ([2020](#bib.bib12 "Retrieval-augmented generation for knowledge-intensive nlp tasks")); Izacard et al. ([2023](#bib.bib1 "Atlas: few-shot learning with retrieval augmented language models")), and scaling retrieval to very large datastores to improve language modeling and downstream performance Borgeaud et al. ([2022](#bib.bib14 "Improving language models by retrieving from trillions of tokens")). A key advantage of these approaches is that knowledge can be updated by changing the document index, without having to retrain the model, and retrieval can improve factual coverage without simply making the model larger Guu et al. ([2020](#bib.bib13 "Retrieval augmented language model pre-training")); Lewis et al. ([2020](#bib.bib12 "Retrieval-augmented generation for knowledge-intensive nlp tasks")); Izacard et al. ([2023](#bib.bib1 "Atlas: few-shot learning with retrieval augmented language models")). More recent work studies how to make retrieval easier to use in practice. For example, some methods encourage the model to retrieve only when needed and to check whether the retrieved evidence actually supports the response Asai et al. ([2023](#bib.bib22 "Self-rag: learning to retrieve, generate, and critique through self-reflection")). Overall, however, the main goal of RAG remains the same: improving the quality of a single response by grounding it in external evidence. As a result, standard RAG typically does not focus on reusing previously generated answers across users or sessions, nor does it explicitly support escalating to a stronger model when retrieval is insufficient.

### A.2 Semantic Caching

Semantic caching extends classical caching to LLM services by reusing previous answers not only for exact duplicate queries, but also for queries that are semantically similar. In practice, a system stores past query-response pairs, represents queries with embeddings, and returns a cached response when a new query is close enough to an existing one, avoiding a full model call Bang ([2023](#bib.bib23 "GPTCache: an open-source semantic cache for LLM applications enabling faster answers and cost savings")); Regmi and Pun ([2024](#bib.bib24 "Gpt semantic cache: reducing llm costs and latency via semantic embedding caching")); Li et al. ([2024](#bib.bib25 "SCALM: towards semantic caching for automated chat services with large language models")); Wang et al. ([2025a](#bib.bib26 "Category-aware semantic caching for heterogeneous llm workloads")). Recent work has shown that this can significantly reduce latency and cost in real deployments, since many user requests concentrate on a small set of recurring intents Bang ([2023](#bib.bib23 "GPTCache: an open-source semantic cache for LLM applications enabling faster answers and cost savings")); Li et al. ([2024](#bib.bib25 "SCALM: towards semantic caching for automated chat services with large language models")). At the same time, semantic caching introduces new reliability challenges. Because a false hit can directly harm correctness (e.g., returning an answer for a different but similar-looking question), while a false miss loses potential savings Gill et al. ([2025](#bib.bib2 "MeanCache: user-centric semantic caching for llm web services")). This has motivated work on improving similarity matching and cache policies, including user-centric designs Gill et al. ([2025](#bib.bib2 "MeanCache: user-centric semantic caching for llm web services")), context-aware caching for multi-turn interactions Yu et al. ([2025](#bib.bib18 "Smartcache: context-aware semantic cache for efficient multi-turn llm inference")), and more principled formulations of cache management and eviction under unknown workloads Liu et al. ([2025](#bib.bib19 "Semantic caching for low-cost llm serving: from offline learning to online adaptation")). Overall, prior semantic caching research typically focuses on when to reuse cached outputs and how to manage the cache, but it often treats the fallback generation model and the caching layer as loosely coupled components Bang ([2023](#bib.bib23 "GPTCache: an open-source semantic cache for LLM applications enabling faster answers and cost savings")); Li et al. ([2024](#bib.bib25 "SCALM: towards semantic caching for automated chat services with large language models")).

### A.3 Routing in LLMs

A complementary line of work studies routing across multiple LLMs to reduce inference cost Chen et al. ([2023](#bib.bib30 "Frugalgpt: how to use large language models while reducing cost and improving performance")); Ong et al. ([2024](#bib.bib3 "Routellm: learning to route llms with preference data")); Zhang et al. ([2025](#bib.bib20 "Router-r1: teaching llms multi-round routing and aggregation via reinforcement learning")); Wang et al. ([2025b](#bib.bib31 "Mixllm: dynamic routing in mixed large language models")); Fedus et al. ([2022](#bib.bib32 "Switch transformers: scaling to trillion parameter models with simple and efficient sparsity")). The core idea is to maintain a pool of models with different cost-quality trade-offs and decide, for each query, which model to use. Many systems follow a cascade pattern: attempt a query with a cheaper model first and escalate to a stronger model only when needed, guided by a confidence or quality estimate Chen et al. ([2023](#bib.bib30 "Frugalgpt: how to use large language models while reducing cost and improving performance")). More recent approaches learn routing policies directly from preference data so that routing decisions better match human judgments of quality while reducing expensive model calls Ong et al. ([2024](#bib.bib3 "Routellm: learning to route llms with preference data")). Beyond single-step routing, recent work also explores more adaptive and sequential routing strategies. For example, some methods treat routing as a multi-round decision process and train routers that interleave reasoning with routing actions Zhang et al. ([2025](#bib.bib20 "Router-r1: teaching llms multi-round routing and aggregation via reinforcement learning")). Other work considers routing in heterogeneous model pools where different models have complementary strengths, requiring robust routing strategies under distribution shift Wang et al. ([2025b](#bib.bib31 "Mixllm: dynamic routing in mixed large language models")). While routing methods are effective at reducing calls to the strongest model, they typically assume that each query is still handled by some model generation pass and do not explicitly leverage cross-request answer reuse as a primary mechanism for efficiency Chen et al. ([2023](#bib.bib30 "Frugalgpt: how to use large language models while reducing cost and improving performance")); Ong et al. ([2024](#bib.bib3 "Routellm: learning to route llms with preference data")). Separately, there is also extensive work on routing within a single model, such as mixture-of-experts architectures, which improves efficiency by activating sparse components but addresses a different level of the system stack Fedus et al. ([2022](#bib.bib32 "Switch transformers: scaling to trillion parameter models with simple and efficient sparsity")).

Component

Hyperparameter

Value

Traffic simulation

Workload distribution

Zipfian

Zipf exponent (α\\alpha)

{0.8,1.1,1.4}\\{0.8,1.1,1.4\\}

Number of requests (NN)

5,000

Random seed

1

Associative Memory Engine

Embedding model

all-MiniLM-L6-v2

Similarity metric

Cosine (FAISS inner product)

Similarity threshold (τ\\tau)

0.95

Retrieval top-kk

3

LLM generation (MC)

Temperature

0.0

Max generation tokens

4096

Frequency / presence penalty

0.0

Chat template kwargs

"enable\_thinking"=false

LLM generation (Oracle)

Temperature

0.0

Max generation tokens

4096

Frequency / presence penalty

0.0

Environment

Python version

3.10

NVIDIA driver

573.57

CUDA version

12.8

Table 2: System and hyperparameter configuration for the continuous serving simulation.

Appendix B Additional Implementation Details
--------------------------------------------

To ensure reproducibility of our experiment runs, we attach the exact hyperparameters as found in Table [2](#A1.T2 "Table 2 ‣ A.3 Routing in LLMs ‣ Appendix A Related Work ‣ MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference") which we used across all experiments. All models were served using the vLLM library to optimize for throughput and latency Kwon et al. ([2023](#bib.bib15 "Efficient memory management for large language model serving with pagedattention")). To ensure fair inference time comparisons, the Meta-Controller including the small LLM and the Associative Memory Engine and the Large LLM Oracle (solver) were each deployed on a dedicated NVIDIA A100 80GB GPU.

Both the Meta-Controller and the Large LLM Oracle are configured with Temperature = 0.00.0 to maintain deterministic responses. In addition the Oracle and the Router model had a 4,096 max tokens setting to allow for 5-shot Chain-of-Thought reasoning.

We use the Business category of the MMLU-Pro dataset and ground truth label answers as our benchmark for comparing the accuracy of the meta-controller (MC) with the Large LLM (Oracle). As the Business category contains 768 examples it is large enough to fit 5000 requests of the chosen Zipf distributions.












































https://arxiv.org/html/2508.12491v2


 Cost-Aware Contrastive Routing for LLMs   

1.  [1 Introduction](https://arxiv.org/html/2508.12491v2#S1 "In Cost-Aware Contrastive Routing for LLMs")
2.  [2 Related Work](https://arxiv.org/html/2508.12491v2#S2 "In Cost-Aware Contrastive Routing for LLMs")
    1.  [2.1 LLM Routing](https://arxiv.org/html/2508.12491v2#S2.SS1 "In 2 Related Work ‣ Cost-Aware Contrastive Routing for LLMs")
        1.  [Non-Predictive Routing.](https://arxiv.org/html/2508.12491v2#S2.SS1.SSS0.Px1 "In 2.1 LLM Routing ‣ 2 Related Work ‣ Cost-Aware Contrastive Routing for LLMs")
        2.  [Predictive Routing.](https://arxiv.org/html/2508.12491v2#S2.SS1.SSS0.Px2 "In 2.1 LLM Routing ‣ 2 Related Work ‣ Cost-Aware Contrastive Routing for LLMs")
    2.  [2.2 Routing within MoE and Hybrid Architectures](https://arxiv.org/html/2508.12491v2#S2.SS2 "In 2 Related Work ‣ Cost-Aware Contrastive Routing for LLMs")
    3.  [2.3 Model Fusion, Merging and Cascading](https://arxiv.org/html/2508.12491v2#S2.SS3 "In 2 Related Work ‣ Cost-Aware Contrastive Routing for LLMs")
3.  [3 Method](https://arxiv.org/html/2508.12491v2#S3 "In Cost-Aware Contrastive Routing for LLMs")
    1.  [3.1 Model Fingerprints](https://arxiv.org/html/2508.12491v2#S3.SS1 "In 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")
        1.  [3.1.1 Logit–Footprint Descriptors for Transparent LLMs](https://arxiv.org/html/2508.12491v2#S3.SS1.SSS1 "In 3.1 Model Fingerprints ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")
            1.  [Why logits?](https://arxiv.org/html/2508.12491v2#S3.SS1.SSS1.Px1 "In 3.1.1 Logit–Footprint Descriptors for Transparent LLMs ‣ 3.1 Model Fingerprints ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")
        2.  [3.1.2 Perplexity Fingerprints for Black‑Box or API‑Only LLMs](https://arxiv.org/html/2508.12491v2#S3.SS1.SSS2 "In 3.1 Model Fingerprints ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")
            1.  [Why perplexity?](https://arxiv.org/html/2508.12491v2#S3.SS1.SSS2.Px1 "In 3.1.2 Perplexity Fingerprints for Black‑Box or API‑Only LLMs ‣ 3.1 Model Fingerprints ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")
            2.  [Unified metric space.](https://arxiv.org/html/2508.12491v2#S3.SS1.SSS2.Px2 "In 3.1.2 Perplexity Fingerprints for Black‑Box or API‑Only LLMs ‣ 3.1 Model Fingerprints ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")
    2.  [3.2 Cost‑Spectrum Contrastive Router](https://arxiv.org/html/2508.12491v2#S3.SS2 "In 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")
    3.  [3.3 Background: The Classic InfoNCE Loss.](https://arxiv.org/html/2508.12491v2#S3.SS3 "In 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")
    4.  [3.4 Cost‑Spectrum InfoNCE.](https://arxiv.org/html/2508.12491v2#S3.SS4 "In 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")
        1.  [Why Incorporate Cost:](https://arxiv.org/html/2508.12491v2#S3.SS4.SSS0.Px1 "In 3.4 Cost‑Spectrum InfoNCE. ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")
        2.  [Banded positives.](https://arxiv.org/html/2508.12491v2#S3.SS4.SSS0.Px2 "In 3.4 Cost‑Spectrum InfoNCE. ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")
        3.  [Cost‑dependent temperature.](https://arxiv.org/html/2508.12491v2#S3.SS4.SSS0.Px3 "In 3.4 Cost‑Spectrum InfoNCE. ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")
        4.  [Negative cost penalty.](https://arxiv.org/html/2508.12491v2#S3.SS4.SSS0.Px4 "In 3.4 Cost‑Spectrum InfoNCE. ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")
    5.  [3.5 Inference Router](https://arxiv.org/html/2508.12491v2#S3.SS5 "In 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")
4.  [4 Experiments](https://arxiv.org/html/2508.12491v2#S4 "In Cost-Aware Contrastive Routing for LLMs")
    1.  [4.1 Experimental Settings](https://arxiv.org/html/2508.12491v2#S4.SS1 "In 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        1.  [Baselines](https://arxiv.org/html/2508.12491v2#S4.SS1.SSS0.Px1 "In 4.1 Experimental Settings ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        2.  [Datasets & Benchmarks](https://arxiv.org/html/2508.12491v2#S4.SS1.SSS0.Px2 "In 4.1 Experimental Settings ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        3.  [Training](https://arxiv.org/html/2508.12491v2#S4.SS1.SSS0.Px3 "In 4.1 Experimental Settings ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        4.  [Evaluation](https://arxiv.org/html/2508.12491v2#S4.SS1.SSS0.Px4 "In 4.1 Experimental Settings ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
    2.  [4.2 Results](https://arxiv.org/html/2508.12491v2#S4.SS2 "In 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        1.  [4.2.1 Generalization to New LLMs](https://arxiv.org/html/2508.12491v2#S4.SS2.SSS1 "In 4.2 Results ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        2.  [4.2.2 Generalization to Out-of-Distribution Prompts](https://arxiv.org/html/2508.12491v2#S4.SS2.SSS2 "In 4.2 Results ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
    3.  [4.3 Ablative Studies](https://arxiv.org/html/2508.12491v2#S4.SS3 "In 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        1.  [4.3.1 Descriptor Choice](https://arxiv.org/html/2508.12491v2#S4.SS3.SSS1 "In 4.3 Ablative Studies ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        2.  [4.3.2 Cost-Aware Training](https://arxiv.org/html/2508.12491v2#S4.SS3.SSS2 "In 4.3 Ablative Studies ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        3.  [4.3.3 Cost-Spectrum Granularity](https://arxiv.org/html/2508.12491v2#S4.SS3.SSS3 "In 4.3 Ablative Studies ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
5.  [5 Theoretical Analysis](https://arxiv.org/html/2508.12491v2#S5 "In Cost-Aware Contrastive Routing for LLMs")
    1.  [Set‑up.](https://arxiv.org/html/2508.12491v2#S5.SS0.SSS0.Px1 "In 5 Theoretical Analysis ‣ Cost-Aware Contrastive Routing for LLMs")
    2.  [5.1 Excess‑risk of cost–spectrum k‑NN](https://arxiv.org/html/2508.12491v2#S5.SS1 "In 5 Theoretical Analysis ‣ Cost-Aware Contrastive Routing for LLMs")
    3.  [5.2 Consistency of Cost–Spectrum InfoNCE](https://arxiv.org/html/2508.12491v2#S5.SS2 "In 5 Theoretical Analysis ‣ Cost-Aware Contrastive Routing for LLMs")
    4.  [5.3 Discussion](https://arxiv.org/html/2508.12491v2#S5.SS3 "In 5 Theoretical Analysis ‣ Cost-Aware Contrastive Routing for LLMs")
6.  [6 Conclusion](https://arxiv.org/html/2508.12491v2#S6 "In Cost-Aware Contrastive Routing for LLMs")
7.  [7 Acknowledgments](https://arxiv.org/html/2508.12491v2#S7 "In Cost-Aware Contrastive Routing for LLMs")
8.  [A Related Work](https://arxiv.org/html/2508.12491v2#A1 "In Cost-Aware Contrastive Routing for LLMs")
    1.  [A.1 Enhancing and Optimizing LLMs](https://arxiv.org/html/2508.12491v2#A1.SS1 "In Appendix A Related Work ‣ Cost-Aware Contrastive Routing for LLMs")
        1.  [Single-LLM Techniques.](https://arxiv.org/html/2508.12491v2#A1.SS1.SSS0.Px1 "In A.1 Enhancing and Optimizing LLMs ‣ Appendix A Related Work ‣ Cost-Aware Contrastive Routing for LLMs")
        2.  [Model Fusion and Merging.](https://arxiv.org/html/2508.12491v2#A1.SS1.SSS0.Px2 "In A.1 Enhancing and Optimizing LLMs ‣ Appendix A Related Work ‣ Cost-Aware Contrastive Routing for LLMs")
        3.  [Cascading.](https://arxiv.org/html/2508.12491v2#A1.SS1.SSS0.Px3 "In A.1 Enhancing and Optimizing LLMs ‣ Appendix A Related Work ‣ Cost-Aware Contrastive Routing for LLMs")
    2.  [A.2 LLM Routing](https://arxiv.org/html/2508.12491v2#A1.SS2 "In Appendix A Related Work ‣ Cost-Aware Contrastive Routing for LLMs")
        1.  [Non-Predictive Routing.](https://arxiv.org/html/2508.12491v2#A1.SS2.SSS0.Px1 "In A.2 LLM Routing ‣ Appendix A Related Work ‣ Cost-Aware Contrastive Routing for LLMs")
        2.  [Predictive Routing.](https://arxiv.org/html/2508.12491v2#A1.SS2.SSS0.Px2 "In A.2 LLM Routing ‣ Appendix A Related Work ‣ Cost-Aware Contrastive Routing for LLMs")
        3.  [Theoretical Foundations and Robustness.](https://arxiv.org/html/2508.12491v2#A1.SS2.SSS0.Px3 "In A.2 LLM Routing ‣ Appendix A Related Work ‣ Cost-Aware Contrastive Routing for LLMs")
    3.  [A.3 Routing as Recommendation](https://arxiv.org/html/2508.12491v2#A1.SS3 "In Appendix A Related Work ‣ Cost-Aware Contrastive Routing for LLMs")
    4.  [A.4 Scaling Laws and Architecture Trends](https://arxiv.org/html/2508.12491v2#A1.SS4 "In Appendix A Related Work ‣ Cost-Aware Contrastive Routing for LLMs")
    5.  [A.5 Routing within MoE and Hybrid Architectures](https://arxiv.org/html/2508.12491v2#A1.SS5 "In Appendix A Related Work ‣ Cost-Aware Contrastive Routing for LLMs")
9.  [B Proofs of Theoretical Results](https://arxiv.org/html/2508.12491v2#A2 "In Cost-Aware Contrastive Routing for LLMs")
    1.  [B.1 Notation and Preliminaries](https://arxiv.org/html/2508.12491v2#A2.SS1 "In Appendix B Proofs of Theoretical Results ‣ Cost-Aware Contrastive Routing for LLMs")
    2.  [B.2 Proof of Theorem 5.2](https://arxiv.org/html/2508.12491v2#A2.SS2 "In Appendix B Proofs of Theoretical Results ‣ Cost-Aware Contrastive Routing for LLMs")
        1.  [Term (A): Lipschitz bias.](https://arxiv.org/html/2508.12491v2#A2.SS2.SSS0.Px1 "In B.2 Proof of Theorem 5.2 ‣ Appendix B Proofs of Theoretical Results ‣ Cost-Aware Contrastive Routing for LLMs")
        2.  [Term (B): Estimation variance.](https://arxiv.org/html/2508.12491v2#A2.SS2.SSS0.Px2 "In B.2 Proof of Theorem 5.2 ‣ Appendix B Proofs of Theoretical Results ‣ Cost-Aware Contrastive Routing for LLMs")
    3.  [B.3 Proof of Lemma 5.3](https://arxiv.org/html/2508.12491v2#A2.SS3 "In Appendix B Proofs of Theoretical Results ‣ Cost-Aware Contrastive Routing for LLMs")
10.  [C Method](https://arxiv.org/html/2508.12491v2#A3 "In Cost-Aware Contrastive Routing for LLMs")
    1.  [C.1 Logit-Footprint Descriptors](https://arxiv.org/html/2508.12491v2#A3.SS1 "In Appendix C Method ‣ Cost-Aware Contrastive Routing for LLMs")
        1.  [Why take the most frequent tokens?](https://arxiv.org/html/2508.12491v2#A3.SS1.SSS0.Px1 "In C.1 Logit-Footprint Descriptors ‣ Appendix C Method ‣ Cost-Aware Contrastive Routing for LLMs")
        2.  [Are frequent tokens trivial?](https://arxiv.org/html/2508.12491v2#A3.SS1.SSS0.Px2 "In C.1 Logit-Footprint Descriptors ‣ Appendix C Method ‣ Cost-Aware Contrastive Routing for LLMs")
        3.  [Shared token set vs. per-model selection.](https://arxiv.org/html/2508.12491v2#A3.SS1.SSS0.Px3 "In C.1 Logit-Footprint Descriptors ‣ Appendix C Method ‣ Cost-Aware Contrastive Routing for LLMs")
        4.  [Beyond raw frequency.](https://arxiv.org/html/2508.12491v2#A3.SS1.SSS0.Px4 "In C.1 Logit-Footprint Descriptors ‣ Appendix C Method ‣ Cost-Aware Contrastive Routing for LLMs")
    2.  [C.2 The Step from Equation 3 to Equation 5](https://arxiv.org/html/2508.12491v2#A3.SS2 "In Appendix C Method ‣ Cost-Aware Contrastive Routing for LLMs")
    3.  [C.3 Band-Specific Temperatures and Smoother Gradients](https://arxiv.org/html/2508.12491v2#A3.SS3 "In Appendix C Method ‣ Cost-Aware Contrastive Routing for LLMs")
    4.  [C.4 Dense Human Annotations](https://arxiv.org/html/2508.12491v2#A3.SS4 "In Appendix C Method ‣ Cost-Aware Contrastive Routing for LLMs")
11.  [D Experiments](https://arxiv.org/html/2508.12491v2#A4 "In Cost-Aware Contrastive Routing for LLMs")
    1.  [D.1 Experimental Settings](https://arxiv.org/html/2508.12491v2#A4.SS1 "In Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        1.  [Baselines](https://arxiv.org/html/2508.12491v2#A4.SS1.SSS0.Px1 "In D.1 Experimental Settings ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        2.  [Datasets, Benchmarks, and Evaluation](https://arxiv.org/html/2508.12491v2#A4.SS1.SSS0.Px2 "In D.1 Experimental Settings ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        3.  [Training](https://arxiv.org/html/2508.12491v2#A4.SS1.SSS0.Px3 "In D.1 Experimental Settings ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
    2.  [D.2 Results](https://arxiv.org/html/2508.12491v2#A4.SS2 "In Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        1.  [D.2.1 Out-of-Distribution Prompt Experiments](https://arxiv.org/html/2508.12491v2#A4.SS2.SSS1 "In D.2 Results ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
    3.  [D.3 Ablation of Cost Bands](https://arxiv.org/html/2508.12491v2#A4.SS3 "In Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        1.  [D.3.1 Ablation of Number of Cost bands](https://arxiv.org/html/2508.12491v2#A4.SS3.SSS1 "In D.3 Ablation of Cost Bands ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        2.  [D.3.2 Ablation of the Number of Neighbors](https://arxiv.org/html/2508.12491v2#A4.SS3.SSS2 "In D.3 Ablation of Cost Bands ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        3.  [D.3.3 Ablation of Negative Cost Penalty](https://arxiv.org/html/2508.12491v2#A4.SS3.SSS3 "In D.3 Ablation of Cost Bands ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        4.  [D.3.4 Ablation of Band-Specific Temperature Slope](https://arxiv.org/html/2508.12491v2#A4.SS3.SSS4 "In D.3 Ablation of Cost Bands ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        5.  [D.3.5 Ablation of Cheapest Cost Band](https://arxiv.org/html/2508.12491v2#A4.SS3.SSS5 "In D.3 Ablation of Cost Bands ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
    4.  [D.4 Larger Encoder Size](https://arxiv.org/html/2508.12491v2#A4.SS4 "In Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
    5.  [D.5 Qualitative Insights and Interpretability of Routing](https://arxiv.org/html/2508.12491v2#A4.SS5 "In Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        1.  [D.5.1 Selection Profiles](https://arxiv.org/html/2508.12491v2#A4.SS5.SSS1 "In D.5 Qualitative Insights and Interpretability of Routing ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
        2.  [D.5.2 Routing Error Breakdown](https://arxiv.org/html/2508.12491v2#A4.SS5.SSS2 "In D.5 Qualitative Insights and Interpretability of Routing ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs")
    6.  [D.6 Statistical Significance](https://arxiv.org/html/2508.12491v2#A4.SS6 "In Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs")

Cost-Aware Contrastive Routing for LLMs
=======================================

Reza Shirkavand  
Department of Computer Science  
University of Maryland - College Park  
[\[email protected\]](/cdn-cgi/l/email-protection)  
&Shangqian Gao  
Department of Computer Science  
Florida State University  
[\[email protected\]](/cdn-cgi/l/email-protection)  
Peiran Yu  
Department of Computer Science and Engineering  
University of Texas at Arlington  
[\[email protected\]](/cdn-cgi/l/email-protection)  
&Heng Huang  
Department of Computer Science  
University of Maryland - College Park  
[\[email protected\]](/cdn-cgi/l/email-protection) This work was partially supported by NSF IIS 2347592, 2348169, DBI 2405416, CCF 2348306, CNS 2347617, RISE 2536663.

###### Abstract

We study cost-aware routing for large language models across diverse and dynamic pools of models. Existing approaches often overlook prompt-specific context, rely on expensive model profiling, assume a fixed set of experts, or use inefficient trial-and-error strategies. We introduce Cost-Spectrum Contrastive Routing (CSCR), a lightweight framework that maps both prompts and models into a shared embedding space to enable fast, cost-sensitive selection. CSCR uses compact, fast-to-compute _logit footprints_ for open-source models and _perplexity fingerprints_ for black-box APIs. A contrastive encoder is trained to favor the cheapest accurate expert within adaptive cost bands. At inference time, routing reduces to a single kk‑NN lookup via a FAISS index, requiring no retraining when the expert pool changes and enabling microsecond latency. Across multiple benchmarks, CSCR consistently outperforms baselines, improving the accuracy–cost tradeoff by up to 25%, while generalizing robustly to unseen LLMs and out-of-distribution prompts.

1 Introduction
--------------

After a burst of reinforcement‑learning and specialized finetuning, the Large Language Model (LLM)  \[[65](https://arxiv.org/html/2508.12491v2#bib.bib65), [66](https://arxiv.org/html/2508.12491v2#bib.bib66), [6](https://arxiv.org/html/2508.12491v2#bib.bib6), [86](https://arxiv.org/html/2508.12491v2#bib.bib86), [17](https://arxiv.org/html/2508.12491v2#bib.bib17)\] ecosystem has fractured: code models excel at generating code but hallucinate outside programming contexts, math‑tuned variants solve AIME \[[46](https://arxiv.org/html/2508.12491v2#bib.bib46)\] yet mishandle open‑ended dialogue, and instruction chatbots trade being precise for fluency. Production systems therefore host a pool of models with different sizes, licenses, and domain strengths, and decide at run time which one to call for every user prompt, or worse: burden the users with picking the model they need.

A router (a.k.a. model selector, mixture‑gate) adjudicates that choice online. It dynamically selects the most appropriate LLM from a pool for each input. Without it, operators either over‑pay by defaulting to the largest model or risk quality regressions by pinning to cheaper ones. Current routers \[[33](https://arxiv.org/html/2508.12491v2#bib.bib33), [32](https://arxiv.org/html/2508.12491v2#bib.bib32), [51](https://arxiv.org/html/2508.12491v2#bib.bib51), [76](https://arxiv.org/html/2508.12491v2#bib.bib76)\] for a pool of LLMs fall into two broad camps.

Parametric routing methods, such as softmax-based “one-head-per-model” classifiers \[[24](https://arxiv.org/html/2508.12491v2#bib.bib24)\], optimize exclusively for top-1 accuracy without explicit consideration of inference costs. Consequently, they tend to default to selecting expensive models and require full retraining whenever new models are introduced.

Recent approaches, such as UMR \[[40](https://arxiv.org/html/2508.12491v2#bib.bib40)\], enhance generalization by routing across a joint space of prompt clusters and model footprints. On the other hand, bandit-based methods \[[61](https://arxiv.org/html/2508.12491v2#bib.bib61), [34](https://arxiv.org/html/2508.12491v2#bib.bib34), [48](https://arxiv.org/html/2508.12491v2#bib.bib48)\], including Thompson sampling \[[85](https://arxiv.org/html/2508.12491v2#bib.bib85), [2](https://arxiv.org/html/2508.12491v2#bib.bib2)\] and UCB \[[3](https://arxiv.org/html/2508.12491v2#bib.bib3)\], track simplified quality–cost metrics that disregard detailed prompt characteristics, resulting in slower convergence and reduced effectiveness, especially with heterogeneous workloads. All these methods remain cost-agnostic during training, solely relying on post-hoc hyperparameter tuning to achieve an effective balance between cost and accuracy.

In our view, routing boils down to _similarity search_. If we can embed both prompts and experts in one metric space where cosine distance trades off {quality, cost}, the job reduces to a microsecond Nearest Neighbor query - no brittle softmax gate, no retraining when the pool changes.

![Refer to caption](figures/deferral_embedllm.png)

(a) EmbedLLM

![Refer to caption](figures/deferral_mix-instruct.png)

(b) MixInstruct

![Refer to caption](figures/deferral_routerbench.png)

(c) RouterBench

Figure 1: Accuracy–cost/size deferral curves on three expert pools. Across all benchmarks, our Cost‑Spectrum Contrastive Router (blue) consistently dominates the Pareto frontier—achieving higher accuracy at lower model size and latency (left, middle) and reduced cost (right).

In this paper we introduce:

*   •
    
    Universal ultra-compact descriptors: Two lightweight and fast to compute fingerprints that work across the full spectrum of LLMs. (1) Logit footprints that require only <10 forward passes through an open‑weights model. (2) Perplexity fingerprints that score any black‑box API with a small public LM, enabling vendor‑agnostic routing.
    
*   •
    
    Cost-Spectrum InfoNCE: A novel contrastive objective that (1) selects correct positives within adaptive cost bands, (2) temperature‑scales each band separately, and (3) down‑weights negatives in proportion to their price. This aligns the learned metric with the accuracy‑cost Pareto frontier.
    
*   •
    
    Routing Efficiency: A shared space turns routing into a single kk‑NN lookup, eliminating brittle softmax gates and retraining whenever the pool changes. Then a lightweight FAISS index makes routing to a microsecond lookup.
    
*   •
    
    Comprehensive Evaluation: We evaluate our method on three routing benchmarks spanning both open-source checkpoints and proprietary APIs. It achieves up to 25% higher accuracy–cost efficiency on a fixed pool of LLMs and demonstrates strong robustness to unseen models and out-of-distribution prompts at inference time.
    

2 Related Work
--------------

### 2.1 LLM Routing

##### Non-Predictive Routing.

Non-predictive methods generate outputs from one or more models before making a selection. FrugalGPT \[[10](https://arxiv.org/html/2508.12491v2#bib.bib10)\] uses a sequential strategy and a response quality threshold to minimize cost. Other works adopt layered inference architectures to escalate hard queries to more powerful models \[[91](https://arxiv.org/html/2508.12491v2#bib.bib91)\], or leverage cascades with self-verification \[[54](https://arxiv.org/html/2508.12491v2#bib.bib54), [98](https://arxiv.org/html/2508.12491v2#bib.bib98), [43](https://arxiv.org/html/2508.12491v2#bib.bib43), [51](https://arxiv.org/html/2508.12491v2#bib.bib51)\].

##### Predictive Routing.

In contrast, predictive routing aims to select the best model before any inference is performed. Strategies include supervised learning \[[76](https://arxiv.org/html/2508.12491v2#bib.bib76)\], reward-model-based routing \[[32](https://arxiv.org/html/2508.12491v2#bib.bib32)\], and meta-models trained to predict LLM performance given an input \[[71](https://arxiv.org/html/2508.12491v2#bib.bib71)\]. Router models vary widely in implementation, including neural networks \[[19](https://arxiv.org/html/2508.12491v2#bib.bib19), [88](https://arxiv.org/html/2508.12491v2#bib.bib88), [11](https://arxiv.org/html/2508.12491v2#bib.bib11), [1](https://arxiv.org/html/2508.12491v2#bib.bib1)\], kk\-nearest neighbors \[[34](https://arxiv.org/html/2508.12491v2#bib.bib34), [76](https://arxiv.org/html/2508.12491v2#bib.bib76), [80](https://arxiv.org/html/2508.12491v2#bib.bib80), [43](https://arxiv.org/html/2508.12491v2#bib.bib43)\], matrix factorization \[[62](https://arxiv.org/html/2508.12491v2#bib.bib62), [105](https://arxiv.org/html/2508.12491v2#bib.bib105), [48](https://arxiv.org/html/2508.12491v2#bib.bib48)\], and graph neural networks \[[25](https://arxiv.org/html/2508.12491v2#bib.bib25)\]. Others incorporate model-specific tokens or train across multiple domains \[[18](https://arxiv.org/html/2508.12491v2#bib.bib18), [7](https://arxiv.org/html/2508.12491v2#bib.bib7)\].

Academic routers usually assumed the expert set is static. Recently routing with a dynamic pool of experts has been explored \[[40](https://arxiv.org/html/2508.12491v2#bib.bib40), [48](https://arxiv.org/html/2508.12491v2#bib.bib48)\]. UMR \[[40](https://arxiv.org/html/2508.12491v2#bib.bib40)\] clusters probes and stores coarse capability footprints but still rebuilds them offline whenever the model pool changes. LLM‑Bandit \[[48](https://arxiv.org/html/2508.12491v2#bib.bib48)\] optimizes cost but ignores prompt semantics and offers no cold-start prior for unseen experts.

### 2.2 Routing within MoE and Hybrid Architectures

Routing LLMs can be viewed as a coarse-grained MoE, where each expert is a full LLM. Routing is a central mechanism in MoE models \[[36](https://arxiv.org/html/2508.12491v2#bib.bib36), [41](https://arxiv.org/html/2508.12491v2#bib.bib41), [75](https://arxiv.org/html/2508.12491v2#bib.bib75)\], where expert modules are dynamically activated based on input. While classical MoEs involved equally-sized sub-models, modern approaches like Switch Transformer \[[24](https://arxiv.org/html/2508.12491v2#bib.bib24)\] and Mixtral \[[38](https://arxiv.org/html/2508.12491v2#bib.bib38)\] employ sparse activation to minimize cost. Approaches like UltraFuser \[[20](https://arxiv.org/html/2508.12491v2#bib.bib20)\] highlight recent advances in combining model specialization and flexibility.

### 2.3 Model Fusion, Merging and Cascading

Fusion strategies synthesize outputs from multiple LLMs to improve output quality \[[68](https://arxiv.org/html/2508.12491v2#bib.bib68), [39](https://arxiv.org/html/2508.12491v2#bib.bib39), [29](https://arxiv.org/html/2508.12491v2#bib.bib29), [89](https://arxiv.org/html/2508.12491v2#bib.bib89), [52](https://arxiv.org/html/2508.12491v2#bib.bib52)\]. Fusion approaches often rely on unsupervised metrics \[[99](https://arxiv.org/html/2508.12491v2#bib.bib99), [73](https://arxiv.org/html/2508.12491v2#bib.bib73), [97](https://arxiv.org/html/2508.12491v2#bib.bib97)\] or ensemble voting to determine the final output \[[44](https://arxiv.org/html/2508.12491v2#bib.bib44)\]. A related but distinct technique is model merging \[[50](https://arxiv.org/html/2508.12491v2#bib.bib50)\], where weights from multiple pre-trained or fine-tuned models are combined, either directly via methods like weight averaging \[[92](https://arxiv.org/html/2508.12491v2#bib.bib92)\], Task Arithmetic \[[35](https://arxiv.org/html/2508.12491v2#bib.bib35)\], or Fisher merging \[[56](https://arxiv.org/html/2508.12491v2#bib.bib56)\]. In contrast, cascading invokes models sequentially (often ordered by computational cost) and halts once a satisfactory output is generated \[[10](https://arxiv.org/html/2508.12491v2#bib.bib10), [98](https://arxiv.org/html/2508.12491v2#bib.bib98), [30](https://arxiv.org/html/2508.12491v2#bib.bib30)\].

To the best of our knowledge, no prior work simultaneously (_i_) embeds both prompts and arbitrary experts into a unified metric space, (_ii_) incorporates inference cost explicitly into its learning objective, and (_iii_) generalizes effectively to new LLMs and out-of-distribution prompts using simple, efficiently computable descriptors.

3 Method
--------

This section formalizes our _Cost–Spectrum Contrastive Router_ (CSCR) and its two drop‑in, model‑agnostic descriptors: _logit fingerprints_ and _perplexity fingerprints_. CSCR is trained once on a fixed pool of LLMs and deployed without modification on any subset of that pool. At inference time it performs a kk\-NN lookup in a FAISS \[[21](https://arxiv.org/html/2508.12491v2#bib.bib21)\] index 111We use a FAISS IndexFlatIP. [https://github.com/facebookresearch/faiss/wiki/Faiss-indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) to return the kk most cost–effective experts for a prompt. Throughout, let ℋ\={h(1),…,h(M)}\\mathcal{H}\\!=\\!\\{h^{(1)},\\dots,h^{(M)}\\} denote the available LLMs, c​(h)c(h) their normalized cost, and Φ​(𝐱)∈ℝD\\Phi(\\mathbf{x})\\!\\in\\!\\mathbb{R}^{D} our frozen query encoder with a trainable MLP head gθ(.)g\_{\\theta}(.).

### 3.1 Model Fingerprints

We map every LLM to a compact, task–independent vector 𝐝h∈ℝD′\\mathbf{d}\_{h}\\!\\in\\!\\mathbb{R}^{D^{\\prime}}. Both descriptors are gradient‑free; they can be computed off‑line, cached, and shipped without IP‑sensitive weights.

#### 3.1.1 Logit–Footprint Descriptors for Transparent LLMs

Let Sprobe\={x(i)}i\=1NS\_{\\text{probe}}\\!=\\!\\{x^{(i)}\\}\_{i=1}^{N} be a fixed set of short, diverse prompts shared across all experts. For an autoregressive LLM hh, denote by

ph(v|x,t)\=softmax(logitsh(x)t)v,v∈𝒱,t≥1p\_{h}\\!\\left(v\\,\\middle|\\,x,t\\right)\\;=\\;\\mathrm{softmax}\\hskip 1.0pt\\!\\bigl(\\operatorname{logits}\_{h}(x)\_{t}\\bigr)\_{v},\\quad v\\in\\mathcal{V},\\;t\\geq 1

(1)

the probability that hh emits vocabulary token vv at generation step tt conditioned on the prompt prefix xx. We compress these probabilities into a fixed–length logit footprint:

𝐝logit​(h)\=1N​T​∑i\=1N∑t\=1T\[ph​(vk∣x(i),t)\]k\=1K∈ℝK,\\mathbf{d}\_{\\text{logit}}(h)\\;=\\;\\frac{1}{N\\,T}\\sum\_{i=1}^{N}\\sum\_{t=1}^{T}\\bigl\[p\_{h}\\!\\bigl(v\_{k}\\mid x^{(i)},t\\bigr)\\bigr\]\_{k=1}^{K}\\in\\mathbb{R}^{K},

(2)

where TT is a small horizon, and {vk}k\=1K\\{v\_{k}\\}\_{k=1}^{K} are the KK most frequent tokens across all probes. We ℓ2\\ell\_{2}–normalize 𝐝logit​(h)\\mathbf{d}\_{\\text{logit}}(h) to live on the unit hypersphere, after which cosine similarity is a proxy for KL divergence between the first–token distributions.

##### Why logits?

Equation ([2](https://arxiv.org/html/2508.12491v2#S3.E2 "In 3.1.1 Logit–Footprint Descriptors for Transparent LLMs ‣ 3.1 Model Fingerprints ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")) directly samples the model’s internal predictive distribution ph​(⋅)p\_{h}(\\cdot)—the very quantity trained by maximum–likelihood objective −∑log⁡pθ\-\\!\\sum\\log p\_{\\theta} \[[6](https://arxiv.org/html/2508.12491v2#bib.bib6)\]. It therefore encodes both topical preference and generation style while remaining inexpensive (only N×TN{\\times}T forward passes with greedy decoding). Thus, we use ([2](https://arxiv.org/html/2508.12491v2#S3.E2 "In 3.1.1 Logit–Footprint Descriptors for Transparent LLMs ‣ 3.1 Model Fingerprints ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")) as the primary descriptor whenever logits are available. See Appendix [C.1](https://arxiv.org/html/2508.12491v2#A3.SS1 "C.1 Logit-Footprint Descriptors ‣ Appendix C Method ‣ Cost-Aware Contrastive Routing for LLMs") for a more detailed discussion.

#### 3.1.2 Perplexity Fingerprints for Black‑Box or API‑Only LLMs

Closed‑source APIs (e.g., GPT‑o3 \[[37](https://arxiv.org/html/2508.12491v2#bib.bib37)\], Gemini-2.5 \[[69](https://arxiv.org/html/2508.12491v2#bib.bib69)\]) expose responses but hide almost all logits. For such models we adopt a per–prompt cross‑entropy fingerprint:

ℓh​(x)\\displaystyle\\ell\_{h}(x)

\=−1Lx​∑j\=1Lxlog⁡ph​(wj∣w<j),\\displaystyle\\;=\\;-\\frac{1}{L\_{x}}\\sum\_{j=1}^{L\_{x}}\\log p\_{h}\\!\\bigl(w\_{j}\\mid w\_{<j}\\bigr),

(3)

𝐝PPL​(h)\\displaystyle\\mathbf{d}\_{\\mathrm{PPL}}(h)

\=normalize⁡(\[ℓh​(x(i))\]i\=1N)∈ℝN,\\displaystyle\\;=\\;\\operatorname{normalize}\\!\\Bigl(\\bigl\[\\ell\_{h}\\bigl(x^{(i)}\\bigr)\\bigr\]\_{i=1}^{N}\\Bigr)\\in\\mathbb{R}^{N},

(4)

where w1:Lxw\_{1{:}L\_{x}} are the gold target tokens (ground‑truth answers if available, or probe continuations); normalize⁡(⋅)\\operatorname{normalize}(\\cdot) denotes mean‑centering and unit–variance scaling. Practically, we approximate ([3](https://arxiv.org/html/2508.12491v2#S3.E3 "In 3.1.2 Perplexity Fingerprints for Black‑Box or API‑Only LLMs ‣ 3.1 Model Fingerprints ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")) with a lightweight open LM that scores the API output y^h​(x)\\hat{y}\_{h}(x) instead of inaccessible php\_{h}:

ℓ~h​(x)\=−1|y^h​(x)|​∑jlog⁡pgpt2​(y^h,j∣y^h,<j).\\tilde{\\ell}\_{h}(x)=-\\tfrac{1}{|\\hat{y}\_{h}(x)|}\\sum\_{j}\\log p\_{\\text{gpt2}}\\!\\bigl(\\hat{y}\_{h,j}\\mid\\hat{y}\_{h,<j}\\bigr).

(5)

See Appendix [C.2](https://arxiv.org/html/2508.12491v2#A3.SS2 "C.2 The Step from Equation 3 to Equation 5 ‣ Appendix C Method ‣ Cost-Aware Contrastive Routing for LLMs") for a more detailed discussion.

##### Why perplexity?

![Refer to caption](figures/routerbench_cosine_similarity.png)

Figure 2: Cosine similarity of perplexity descriptors for RouterBench LLMs. Despite using a shared scorer, the descriptors distinctly separate the experts.

Cross‑entropy is proportional to KL(ptrue∥php\_{\\text{true}}\\!\\parallel\\,p\_{h}) plus entropy of the data distribution; it therefore quantifies text fit and has long been a proxy for LM quality \[[58](https://arxiv.org/html/2508.12491v2#bib.bib58), [66](https://arxiv.org/html/2508.12491v2#bib.bib66)\]. Prior work has shown that perplexity can serve as an effective metric for distinguishing between human-generated text and LLM-generated output \[[31](https://arxiv.org/html/2508.12491v2#bib.bib31)\]. Moreover, although ℓ~h\\tilde{\\ell}\_{h} is an approximation, the descriptor vector in ([4](https://arxiv.org/html/2508.12491v2#S3.E4 "In 3.1.2 Perplexity Fingerprints for Black‑Box or API‑Only LLMs ‣ 3.1 Model Fingerprints ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")) still captures how hard each prompt is for a given expert; routing on these vectors recovers much of the benefit of logit footprints while remaining viable for black‑box LLMs. Figure [2](https://arxiv.org/html/2508.12491v2#S3.F2 "Figure 2 ‣ Why perplexity? ‣ 3.1.2 Perplexity Fingerprints for Black‑Box or API‑Only LLMs ‣ 3.1 Model Fingerprints ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs") shows that, despite using the same model to compute the perplexity of text generated by different LLMs, we still observe a clear separation between the expert descriptors.

##### Unified metric space.

We can make both descriptors reside on the unit sphere 𝕊K−1\\mathbb{S}^{K-1} by setting N\=KN=K (i.e., the number of top tokens equals the number of prompts) with cosine similarity σ​(𝐝1,𝐝2)\=𝐝1⊤​𝐝2\\sigma(\\mathbf{d}\_{1},\\mathbf{d}\_{2})=\\mathbf{d}\_{1}^{\\top}\\mathbf{d}\_{2}. A single kk‑NN router can thus mix transparent experts (logit fingerprints) and opaque experts (perplexity fingerprints) without altering downstream training loss; only the descriptor extraction pipeline changes (See Table [4](https://arxiv.org/html/2508.12491v2#S4.T4 "Table 4 ‣ 4.3.2 Cost-Aware Training ‣ 4.3 Ablative Studies ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs")) .

### 3.2 Cost‑Spectrum Contrastive Router

A contrastive router learns a shared embedding space where each query vector is pulled toward the descriptor of the right‑sized expert and pushed away from less suitable ones; this yields three key pay‑offs. First, because routing reduces to a nearest‑neighbor lookup in that space, inference is a microsecond operation that adds virtually no latency to serving large expert pools. Second, contrastive objectives let the router exploit implicit supervision (“expert X solved this prompt while expert Y failed” or “X is cheaper than an equally accurate Y”) so it can be trained with only correctness or cost signals, no dense human annotations. Third, the geometry learned by contrastive learning naturally generalizes: queries that look semantically or structurally similar land near the same expert regions, which improves robustness to distribution shift and unseen prompts, a behavior long noted in contrastive representation learning. Together, these properties make a contrastive router an efficient, supervision‑light and highly adaptable choice for directing traffic in modern multi‑LLM systems.

### 3.3 Background: The Classic InfoNCE Loss.

Let a minibatch contain BB queries {𝐱i}i\=1B\\{\\mathbf{x}\_{i}\\}\_{i=1}^{B} and a memory bank of MM keys {𝐞m}m\=1M\\{\\mathbf{e}\_{m}\\}\_{m=1}^{M}. A query encoder fθ:𝒳→ℝdf\_{\\theta}\\!:\\!\\mathcal{X}\\!\\rightarrow\\!\\mathbb{R}^{d} produces representations 𝐪i\=fθ​(𝐱i)/‖fθ​(𝐱i)‖2\\mathbf{q}\_{i}=f\_{\\theta}(\\mathbf{x}\_{i})/\\hskip 2.0pt\\!\\|\\!f\_{\\theta}(\\mathbf{x}\_{i})\\!\\|\_{2}, and the keys are ℓ2\\ell\_{2}‑normalized in advance, 𝐞m\=𝐄m/‖𝐄m‖2\\mathbf{e}\_{m}=\\mathbf{E}\_{m}/\\|\\mathbf{E}\_{m}\\|\_{2} (here, EmE\_{m} is the expert descriptor from Equation ([2](https://arxiv.org/html/2508.12491v2#S3.E2 "In 3.1.1 Logit–Footprint Descriptors for Transparent LLMs ‣ 3.1 Model Fingerprints ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")) or  ([4](https://arxiv.org/html/2508.12491v2#S3.E4 "In 3.1.2 Perplexity Fingerprints for Black‑Box or API‑Only LLMs ‣ 3.1 Model Fingerprints ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")), i.e. Em\=d​(hm)E\_{m}=d(h\_{m})). For each query ii let 𝒫​(i)⊂\[M\]\\mathcal{P}(i)\\subset\[M\] denote the positives (e.g. correct experts) and 𝒩​(i)\=\[M\]∖𝒫​(i)\\mathcal{N}(i)=\[M\]\\setminus\\mathcal{P}(i) the negatives. The vanilla InfoNCE objective \[[87](https://arxiv.org/html/2508.12491v2#bib.bib87)\] maximizes a log‑softmax over cosine similarities

ℒInfoNCE\=−1B​∑i\=1Blog⁡∑m∈𝒫​(i)exp⁡(𝐪i⊤​𝐞mτ)∑m′\=1Mexp⁡(𝐪i⊤​𝐞m′τ),\\mathcal{L}\_{\\text{InfoNCE}}\\;=\\;-\\frac{1}{B}\\sum\_{i=1}^{B}\\log\\frac{\\displaystyle\\sum\_{m\\in\\mathcal{P}(i)}\\!\\exp\\!\\bigl(\\tfrac{\\mathbf{q}\_{i}^{\\top}\\mathbf{e}\_{m}}{\\tau}\\bigr)}{\\displaystyle\\sum\_{m^{\\prime}=1}^{M}\\exp\\!\\bigl(\\tfrac{\\mathbf{q}\_{i}^{\\top}\\mathbf{e}\_{m^{\\prime}}}{\\tau}\\bigr)},

(6)

where τ\>0\\tau\\!>\\!0 is a temperature. It encourages queries to be close to any positive but far from all negatives, thereby learning a metric embedding.

### 3.4 Cost‑Spectrum InfoNCE.

##### Why Incorporate Cost:

Routing must balance two competing axes: _quality_ (the expert is correct) and _inference cost_ cmc\_{m} (e.g. dollars or latency). The classical InfoNCE loss ignores cmc\_{m}, so the encoder can satisfy the objective by clustering any correct experts—most often the cheapest ones, since there are usually more of them in the pool—around the query embedding. Compounding this, easier prompts tend to occur more frequently, so training examples are skewed toward cases where cheap experts suffice. Once those low-cost positives are nearby, there is no training signal to learn where the slightly more expensive but markedly more accurate models live. Empirically, this drives the router to over-use the bargain-bin checkpoints, hurting accuracy and leaving significant headroom untapped, even though paying a few extra cents would buy a large quality jump (see Table [5](https://arxiv.org/html/2508.12491v2#S4.T5 "Table 5 ‣ 4.3.2 Cost-Aware Training ‣ 4.3 Ablative Studies ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs")).

We therefore introduce a cost‑aware spectrum version that:

1.  1.
    
    Selects all positive per cost band, preventing domination by extremely cheap or extremely costly experts
    
2.  2.
    
    Assigns band‑specific temperatures so that harder (costlier) positives yield smoother gradients
    
3.  3.
    
    Penalizes negatives proportionally to their cost, pushing the encoder to prefer cheaper mistakes if it must err.
    

Formally we first normalize costs cm∈\[0,1\]c\_{m}\\!\\in\\!\[0,1\] and partition them into KK disjoint percentile bands ℬk\={m:cm∈\[βk,βk+1)}\\mathcal{B}\_{k}\\!=\\!\\{m:\\,c\_{m}\\!\\in\\!\[\\beta\_{k},\\beta\_{k+1})\\} with quantiles β0\=0<⋯<βK\=1\\beta\_{0}\\!=\\!0<\\!\\cdots\\!<\\!\\beta\_{K}\\!=\\!1. For each query ii and band kk, let 𝒫i​k\=𝒫​(i)∩ℬk\\mathcal{P}\_{ik}=\\mathcal{P}(i)\\cap\\mathcal{B}\_{k} denote the set of correct experts in that band. All experts in 𝒫i​k\\mathcal{P}\_{ik} are treated as positives, and weighted by a softmax over similarities scaled by a band-specific temperature

τk\=τmin+α⋅c¯k,\\tau\_{k}=\\tau\_{\\min}+\\alpha\\cdot\\bar{c}\_{k},

(7)

where c¯k\\bar{c}\_{k} is the mean cost of experts in ℬk\\mathcal{B}\_{k}.

With Φ​(𝐱)∈ℝD\\Phi(\\mathbf{x})\\!\\in\\!\\mathbb{R}^{D} being our frozen query encoder with a lightweight trainable MLP head gθ(.)g\_{\\theta}(.), the loss for a query 𝐪i\=gθ​(Φ​(𝐱i))/‖gθ​(Φ​(𝐱i))‖2\\mathbf{q}\_{i}=g\_{\\theta}(\\Phi(\\mathbf{x}\_{i}))/\\hskip 2.0pt\\!\\|\\!\\hskip 1.0ptg\_{\\theta}(\\Phi(\\mathbf{x}\_{i}))\\hskip 2.0pt\\!\\|\_{2} is then an average over all non-empty bands:

ℓiCS\\displaystyle\\ell\_{i}^{\\text{CS}}

\=−1|𝒦​i|​∑k∈𝒦​ilog⁡∑m∈𝒫i​kexp⁡(𝐪i⊤​𝐞mτk)∑m′\=1Mexp⁡(𝐪i⊤​𝐞m′−γ​cm′τk),\\displaystyle=-\\frac{1}{|\\mathcal{K}i|}\\sum\_{k\\in\\mathcal{K}i}\\log\\frac{\\sum\_{m\\in\\mathcal{P}\_{ik}}\\exp\\left(\\frac{\\mathbf{q}\_{i}^{\\top}\\mathbf{e}\_{m}}{\\tau\_{k}}\\right)}{\\sum\_{m^{\\prime}=1}^{M}\\exp\\left(\\frac{\\mathbf{q}\_{i}^{\\top}\\mathbf{e}\_{m^{\\prime}}-\\gamma c\_{m^{\\prime}}}{\\tau\_{k}}\\right)},

(8)

where 𝒦i\\mathcal{K}\_{i} is the set of cost bands that contain at least one positive and γ≥0\\gamma\\!\\geq\\!0 controls the negative cost penalty. Averaging over the minibatch yields ℒCS\=1B​∑i\=1BℓiCS\\mathcal{L}\_{\\text{CS}}=\\tfrac{1}{B}\\sum\_{i=1}^{B}\\ell\_{i}^{\\text{CS}}.

##### Banded positives.

By retaining all positives within each cost band, we ensure that high-cost, correct experts still receive gradient signal, even when low-cost models also answer correctly. This prevents cost-collapse, the failure mode discussed in [3.4](https://arxiv.org/html/2508.12491v2#S3.SS4.SSS0.Px1 "Why Incorporate Cost: ‣ 3.4 Cost‑Spectrum InfoNCE. ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs"), where training signal concentrates on cheap experts due to prompt and model imbalances.

##### Cost‑dependent temperature.

Higher bands (larger c¯k\\bar{c}\_{k}) get larger τk\\tau\_{k}, flattening their softmax and avoiding vanishing gradients when few difficult positives exist. In contrast, cheap bands keep a low temperature, sharpening the push towards inexpensive correct experts.

##### Negative cost penalty.

Subtracting λ​cm\\lambda c\_{m} in the denominator (not the numerator) means that expensive wrong experts contribute more to the partition function, hence increase the loss; the encoder is thus encouraged to separate from them first.

Thus the contrastive potential aligns three signals in the same metric space: _(i)_ semantic proximity via the query encoder, _(ii)_ expert capability via fingerprints, and _(iii)_ user preference via cost scaling. Previous cost‑aware objectives for retrieval weight the final scoring function at inference time (e.g. \[[48](https://arxiv.org/html/2508.12491v2#bib.bib48), [40](https://arxiv.org/html/2508.12491v2#bib.bib40)\]); our formulation also integrates cost during representation learning, inducing a feature geometry that naturally interpolates accuracy and cost. Equation ([8](https://arxiv.org/html/2508.12491v2#S3.E8 "In Why Incorporate Cost: ‣ 3.4 Cost‑Spectrum InfoNCE. ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")) collapses to standard InfoNCE when K\=1K\\!=\\!1 and λ\=0\\lambda\\!=\\!0. See Appendices  [C.3](https://arxiv.org/html/2508.12491v2#A3.SS3 "C.3 Band-Specific Temperatures and Smoother Gradients ‣ Appendix C Method ‣ Cost-Aware Contrastive Routing for LLMs") and [C.4](https://arxiv.org/html/2508.12491v2#A3.SS4 "C.4 Dense Human Annotations ‣ Appendix C Method ‣ Cost-Aware Contrastive Routing for LLMs") for a more detailed discussion.

### 3.5 Inference Router

Given a test prompt 𝐱\\mathbf{x} we retrieve

r^​(𝐱)\=arg​minh∈Topk​(𝐱)\[cos⁡⟨gθ​(Φ​(𝐱)),𝐝h⟩+λ​c​(h)\],\\hat{r}(\\mathbf{x})=\\mathop{\\rm arg\\,min}\_{h\\in\\text{Top}\_{k}(\\mathbf{x})}\\bigl\[\\cos\\langle g\_{\\theta}(\\Phi(\\mathbf{x})),\\mathbf{d}\_{h}\\rangle+\\lambda\\,c(h)\\bigr\],

(9)

λ\\lambda is the cost weight and Topk​(𝐱)\\text{Top}\_{k}(\\mathbf{x}) retrieves the kk most similar experts to the prompt from the FAISS index. We use k\=4k=4 by default. During training, a similar composite score appears in the cost-spectrum InfoNCE objective (Equation ([8](https://arxiv.org/html/2508.12491v2#S3.E8 "In Why Incorporate Cost: ‣ 3.4 Cost‑Spectrum InfoNCE. ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs"))), allowing the encoder to rank candidate models by similarity to expert descriptors plus the cost term γ​c​(h)\\gamma c(h), just as in the inference rule of Equation [9](https://arxiv.org/html/2508.12491v2#S3.E9 "In 3.5 Inference Router ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs"). We show in the next section that this alignment between training and inference is highly effective.

4 Experiments
-------------

### 4.1 Experimental Settings

##### Baselines

We compare our proposed method against a comprehensive set of baselines. Specifically, we include UMR \[[40](https://arxiv.org/html/2508.12491v2#bib.bib40)\], a recent technique that clusters prompt embeddings to route queries to LLM pools efficiently; Thompson Sampling \[[48](https://arxiv.org/html/2508.12491v2#bib.bib48), [2](https://arxiv.org/html/2508.12491v2#bib.bib2)\], which frames routing as a bandit exploration-exploitation problem to balance cost and accuracy dynamically; Pareto-optimal routing \[[34](https://arxiv.org/html/2508.12491v2#bib.bib34)\], a strategy that selects models by explicitly considering the cost-accuracy Pareto frontier; and two extreme baselines: Random, which selects models uniformly at random to represent naive routing without intelligent selection, and Oracle \[[40](https://arxiv.org/html/2508.12491v2#bib.bib40)\], which always selects the most accurate model at the lowest possible cost and thus represents a theoretical performance ceiling. Additionally, we evaluate against parametric-softmax gating methods inspired by mixture-of-experts architectures \[[62](https://arxiv.org/html/2508.12491v2#bib.bib62)\] and SoftMoE \[[64](https://arxiv.org/html/2508.12491v2#bib.bib64)\], which models router decisions via differentiable soft gating functions.

##### Datasets & Benchmarks

We train our router and evaluate it on three datasets: EmbedLLM \[[105](https://arxiv.org/html/2508.12491v2#bib.bib105)\], MixInstruct \[[39](https://arxiv.org/html/2508.12491v2#bib.bib39)\], and RouterBench \[[34](https://arxiv.org/html/2508.12491v2#bib.bib34)\]. For EmbedLLM and MixInstruct, we sample 192 probes from their respective validation sets. Each probe is processed to extract logit-based descriptors by capturing the top K\=256K=256 tokens over a horizon of T\=10T=10 tokens (Equation ([2](https://arxiv.org/html/2508.12491v2#S3.E2 "In 3.1.1 Logit–Footprint Descriptors for Transparent LLMs ‣ 3.1 Model Fingerprints ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs"))). For RouterBench, we sample 192 probes from its training set. We compute perplexity-based descriptors on RouterBench and use GPT-2 \[[66](https://arxiv.org/html/2508.12491v2#bib.bib66)\]. On both EmbedLLM and RouterBench, we use binary accuracy as the per-sample evaluation metric. For MixInstruct, we employ exponentiated BARTScore \[[97](https://arxiv.org/html/2508.12491v2#bib.bib97)\] as the evaluation metric, following the approach in prior work \[[40](https://arxiv.org/html/2508.12491v2#bib.bib40), [39](https://arxiv.org/html/2508.12491v2#bib.bib39)\].

##### Training

We use a frozen sentence-transformers/all-MiniLM-L6-v2\[[70](https://arxiv.org/html/2508.12491v2#bib.bib70)\] model as the embedding backbone across all experiments. Our trainable router component is a two-layer MLP which projects prompt embeddings into the expert descriptor space. We train our contrastive router on the training splits of each dataset. For the cost spectrum loss (Equation ([8](https://arxiv.org/html/2508.12491v2#S3.E8 "In Why Incorporate Cost: ‣ 3.4 Cost‑Spectrum InfoNCE. ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs"))), we set the number of cost bands K\=5K=5 and the negative cost penalty γ\=0.2\\gamma=0.2. The hyperparameters for band-specific temperatures (Equation ([7](https://arxiv.org/html/2508.12491v2#S3.E7 "In Why Incorporate Cost: ‣ 3.4 Cost‑Spectrum InfoNCE. ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs"))) are set as α\=0.25\\alpha=0.25 and τmin\=0.05\\tau\_{\\min}=0.05. See [D.1](https://arxiv.org/html/2508.12491v2#A4.SS1 "D.1 Experimental Settings ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs") for full details.

##### Evaluation

We evaluate each routing strategy using a deferral curve \[[40](https://arxiv.org/html/2508.12491v2#bib.bib40)\] which plots the average response quality against the total inference cost. Sweeping the routing penalty parameter λ\\lambda over the interval λ∈\[0,λmax\]\\lambda\\!\\in\\!\[0,\\lambda\_{\\max}\] (Equation ([9](https://arxiv.org/html/2508.12491v2#S3.E9 "In 3.5 Inference Router ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs"))) traces the deferral curve. For the EmbedLLM and MixInstruct datasets, we define the cost of processing a prompt as the number of parameters in the LLM, a proxy for computational resources and latency. In the case of RouterBench, we utilize the actual API call costs in USD, as provided in the dataset. Following \[[40](https://arxiv.org/html/2508.12491v2#bib.bib40)\] we employ evaluation metrics including Area Under the Deferral Curve (AUDC), peak accuracy and Query-Normalized Cost (QNC), the minimum relative cost required to match the performance of the most accurate tested LLM.

### 4.2 Results

EmbedLLM

Mix‑Instruct

RouterBench

Router

AUDC↑~\\uparrow

QNC↓~\\downarrow

Peak ↑\\uparrow

AUDC↑~\\uparrow

QNC↓~\\downarrow

Peak ↑\\uparrow

AUDC↑~\\uparrow

QNC↓~\\downarrow

Peak ↑\\uparrow

Oracle (upper bound)

0.960

2.87

0.979

0.079

10.17

0.081

0.891

0.290

0.910

UMR \[[40](https://arxiv.org/html/2508.12491v2#bib.bib40)\]

0.515

58.61

0.562

0.049

10.35

0.050

0.568

0.487

0.615

Thompson \[[2](https://arxiv.org/html/2508.12491v2#bib.bib2)\]

0.472

48.81

0.514

0.044

10.09

0.045

0.622

1.634

0.787

Soft‑MoE \[[64](https://arxiv.org/html/2508.12491v2#bib.bib64)\]

0.404

70.00

0.577

0.030

10.09

0.045

0.599

1.659

0.794

Parametric \[[62](https://arxiv.org/html/2508.12491v2#bib.bib62)\]

0.506

70.00

0.555

0.039

12.00

0.052

0.691

1.658

0.794

Pareto‑Random \[[34](https://arxiv.org/html/2508.12491v2#bib.bib34)\]

0.369

16.03

0.393

0.036

6.16

0.043

0.5172

0.589

0.545

Random

0.379

18.09

0.392

0.037

6.16

0.043

0.5147

0.585

0.542

CSCR (Ours)

0.541

43.28

0.578

0.051

9.32

0.052

0.7110

1.660

0.794

Table 1: Deferral-curve metrics across three benchmarks. Our Cost-Spectrum Contrastive Router achieves the highest area under the deferral curve (AUDC), competitive or superior peak accuracy and lower quality-neutral cost (QNC) compared to key baselines. The Oracle router serves as an upper bound, retrospectively selecting the lowest-cost LLM that yields the correct answer.

Table [1](https://arxiv.org/html/2508.12491v2#S4.T1 "Table 1 ‣ 4.2 Results ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs") presents deferral-curve metrics across the benchmarks. Our CSCR consistently outperforms all relevant baselines, achieving the highest AUDC and demonstrating competitive or superior peak accuracy. Notably, it attains lower QNC, indicating more cost-effective routing decisions. These results underscore the effectiveness of our cost-aware router learning approach in balancing performance and inference cost. The Oracle router, which retrospectively selects the optimal LLM for each query, establishes an upper bound for performance. The benchmarks are dominated by lower cost experts, hence the lower QNC for random baselines. See Appendix [D.6](https://arxiv.org/html/2508.12491v2#A4.SS6 "D.6 Statistical Significance ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs") for results on statistical significance.

#### 4.2.1 Generalization to New LLMs

![Refer to caption](figures/deferral_embedllm_subset.png)  

Figure 3: Deferral curves of test on new LLMs.

![Refer to caption](figures/deferral_embedllm_ood.png)  

Figure 4: Deferral curve of OOD prompts.

Router

EmbedLLM

AUDC ↑\\uparrow

QNC ↓\\downarrow

Peak ↑\\uparrow

Oracle (upper bound)

0.9111

4.118

0.951

UMR

0.4766

64.300

0.518

Thompson

0.4478

68.904

0.514

Soft‑MoE

0.4109

18.282

0.485

Pareto‑Random

0.3812

15.662

0.407

Random

0.3829

15.372

0.403

CSCR (Ours)

0.4848

70.000

0.565

  

Table 2: Deferral‑curve metrics on new LLMs. CSCR shows better robustness to new LLMs.

Router

EmbedLLM

AUDC ↑\\uparrow

QNC ↓\\downarrow

Peak ↑\\uparrow

Oracle (upper bound)

0.9511

2.872

0.979

UMR

0.4249

47.910

0.459

Thompson

0.4863

69.983

0.574

Soft‑MoE

0.4207

70.017

0.555

Pareto‑Random

0.3676

17.417

0.396

Random

0.3717

17.499

0.397

CSCR (Ours)

0.5146

42.338

0.568

  

Table 3: Deferral‑curve metrics on OOD prompts. CSCR shows superior robustness.

We also evaluate the robustness of our method and baselines in scenarios where new LLMs are introduced during testing. Specifically, we select two-thirds of the EmbedLLM training models to train our router and the baselines following \[[40](https://arxiv.org/html/2508.12491v2#bib.bib40)\], using only responses from these selected models. Table [2](https://arxiv.org/html/2508.12491v2#S4.T2 "Table 2 ‣ 4.2.1 Generalization to New LLMs ‣ 4.2 Results ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs") summarizes the performance metrics, while Figure [3](https://arxiv.org/html/2508.12491v2#S4.F3 "Figure 3 ‣ 4.2.1 Generalization to New LLMs ‣ 4.2 Results ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs") illustrates the corresponding deferral curves. Testing is conducted exclusively on the unseen LLM pool. Our results indicate that our approach exhibits superior robustness under these conditions.

#### 4.2.2 Generalization to Out-of-Distribution Prompts

We evaluate our approach on an _out-of-distribution (OOD) prompt at test time_ scenario. We split the EmbedLLM dataset into two subsets: one focusing on STEM-related prompts (e.g., science and technology) and the other comprising all remaining categories (see Appendix [D.2.1](https://arxiv.org/html/2508.12491v2#A4.SS2.SSS1 "D.2.1 Out-of-Distribution Prompt Experiments ‣ D.2 Results ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs") for detailed experimental settings). As illustrated in Figure [4](https://arxiv.org/html/2508.12491v2#S4.F4 "Figure 4 ‣ 4.2.1 Generalization to New LLMs ‣ 4.2 Results ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs") and quantified in Table [3](https://arxiv.org/html/2508.12491v2#S4.T3 "Table 3 ‣ 4.2.1 Generalization to New LLMs ‣ 4.2 Results ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs"), CSCR significantly outperforms all baselines across all key metrics. Specifically, our router achieves an AUDC of 0.5146 compared to the next-best baseline, Thompson, at 0.4863, highlighting its superior robustness and accuracy when handling diverse OOD prompts. This performance advantage demonstrates that our method generalizes exceptionally well, maintaining reliable decision-making capability across varied and harsh distributional shifts.

### 4.3 Ablative Studies

#### 4.3.1 Descriptor Choice

Table [4](https://arxiv.org/html/2508.12491v2#S4.T4 "Table 4 ‣ 4.3.2 Cost-Aware Training ‣ 4.3 Ablative Studies ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs") compares two descriptors on Mix‑Instruct, the only benchmark where we can compute both. Perplexity descriptors (obtained by running every candidate answer through an auxiliary language model) lift the AUDC from 0.461 to 0.467. The absolute peak accuracy, however, changes less than 0.1%. Because the perplexity pipeline requires (i) generating text with the model in the pool and (ii) a second forward pass through a public LM, it is often at least 2×\\times slower. Logit descriptors, in contrast, need only a single pass on open‑weights models and still deliver competitive AUDC. We therefore adopt logit‑based descriptors for all open LLMs and fall back to perplexity descriptors only when logits are inaccessible. The “mixed” row in  [4](https://arxiv.org/html/2508.12491v2#S4.T4 "Table 4 ‣ 4.3.2 Cost-Aware Training ‣ 4.3 Ablative Studies ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs") represents the results obtained by using logit descriptors for 6 randomly selected LLMs and perplexity descriptors for the remaining 5. This shows that both descriptors can be combined within the same pool without negatively affecting the results. In fact, performance slightly improves compared to using only a single descriptor type. This observation further supports our discussion of the “unified metric” in Section [3.1.2](https://arxiv.org/html/2508.12491v2#S3.SS1.SSS2.Px2 "Unified metric space. ‣ 3.1.2 Perplexity Fingerprints for Black‑Box or API‑Only LLMs ‣ 3.1 Model Fingerprints ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs").

#### 4.3.2 Cost-Aware Training

![[Uncaptioned image]](figures/routing_sample.png)

Figure 5: Real sample of routing decisions made by UMR, vanilla contrastive router, and CSCR. CSCR chooses a more expensive but accurate expert than the vanilla router, while also selecting a cheaper option than UMR, achieving better accuracy-cost trade-offs on both ends.

Replacing the vanilla InfoNCE loss with our cost‑spectrum variant significantly increases the AUDC (0.342 →\\rightarrow 0.495) and raises the peak attainable accuracy from 36% to 54% as shown in Table [5](https://arxiv.org/html/2508.12491v2#S4.T5 "Table 5 ‣ 4.3.2 Cost-Aware Training ‣ 4.3 Ablative Studies ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs"). The trade‑off is a higher Quality‑Neutral Cost QNC, meaning the router now leans more on expensive but accurate models; however, the large AUDC gain shows that, for any realistic cost budget, users receive better accuracy‑per‑dollar overall. This confirms our discussion in Section [3.4](https://arxiv.org/html/2508.12491v2#S3.SS4.SSS0.Px1 "Why Incorporate Cost: ‣ 3.4 Cost‑Spectrum InfoNCE. ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs"), that explicitly teaching the encoder to respect the cost hierarchy of experts is crucial.

Router

MixInstruct

AUDC ↑\\uparrow

QNC ↓\\downarrow

Peak ↑\\uparrow

Logit Desc.

0.0461

6.426

0.046

Perp. Desc.

0.0467

8.967

0.047

Mixed

0.0473

6.233

0.047

  

Table 4: Effect of Descriptor Type. Perplexity descriptors slightly improve AUDC but require an extra pass compared to the faster logit descriptors. Mixing descriptors has no impact on results.

Router

EmbedLLM

AUDC ↑\\uparrow

QNC ↓\\downarrow

Peak ↑\\uparrow

Vanilla

0.3421

6.382

0.362

Cost-Aware

0.4951

11.065

0.540

  

Table 5: Effect of cost‑aware training: injecting cost awareness into the contrastive loss prevents the router from concentrating on cheap experts, and boosts the AUDC and peak accuracy of the router. The trade‑off is a higher QNC.

Furthermore, Figure [1](https://arxiv.org/html/2508.12491v2#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Cost-Aware Contrastive Routing for LLMs") and Table [1](https://arxiv.org/html/2508.12491v2#S4.T1 "Table 1 ‣ 4.2 Results ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs") show that while cost-aware contrastive training incurs more cost than the vanilla variant, it remains far more efficient than the baselines, achieving lower QNC by learning to distinguish good cheap experts from both bad cheap and bad expensive ones. See Figure [5](https://arxiv.org/html/2508.12491v2#S4.F5 "Figure 5 ‣ 4.3.2 Cost-Aware Training ‣ 4.3 Ablative Studies ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs") for an example.

#### 4.3.3 Cost-Spectrum Granularity

![Refer to caption](figures/band_ablation.png)

Figure 6: Deferral‑curves across different numbers of cost bands.

Figure [6](https://arxiv.org/html/2508.12491v2#S4.F6 "Figure 6 ‣ 4.3.3 Cost-Spectrum Granularity ‣ 4.3 Ablative Studies ‣ 4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs") shows that performance varies with the number of cost bands. Using 5 bands yields the best results, with the highest AUDC and Peak accuracy, suggesting a good balance between flexibility and generalization. Fewer bands (e.g., 2) limit routing precision, while too many bands (e.g., 15 or 20) degrade performance, likely due to over-fragmentation and increased decision noise. This highlights the importance of tuning the number of bands per dataset to avoid both under- and overfitting. See Appendix [D.2](https://arxiv.org/html/2508.12491v2#A4.SS2 "D.2 Results ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs") for more ablations and detailed discussion.

5 Theoretical Analysis
----------------------

##### Set‑up.

Let 𝒳\\mathcal{X} be the space of prompts and ℋ\={h1,…,hM}\\mathcal{H}=\\{h\_{1},\\dots,h\_{M}\\} a fixed pool of experts with per‑call cost c​(h)∈ℝ\>0c(h)\\in\\mathbb{R}\_{>0}. For a query x∈𝒳x\\in\\mathcal{X} and ground‑truth yy we write ℓ​(x,y,h)\=𝟙​\[h​(x)≠y\]\\ell(x,y,h)=\\mathbbm{1}\\!\[h(x)\\neq y\] for the 0–11 loss and γ​(x,h)\=ℙy∣x​\[ℓ​(x,y,h)\=1\]\\gamma(x,h)=\\mathbb{P}\_{y\\mid x}\[\\ell(x,y,h)=1\] for the Bayes error. The _cost–adjusted risk_ of a router rr is

ℛλ​(r)\=𝔼(x,y)∼𝒟​\[ℓ​(x,y,hr​(x))+λ​c​(hr​(x))\],\\mathcal{R}\_{\\lambda}(r)\\;=\\;\\mathbb{E}\_{(x,y)\\sim\\mathcal{D}}\\Bigl\[\\,\\ell\\!\\bigl(x,y,h\_{r(x)}\\bigr)+\\lambda\\,c\\!\\bigl(h\_{r(x)}\\bigr)\\Bigr\],

where λ∈ℝ≥0\\lambda\\!\\in\\!\\mathbb{R}\_{\\geq 0} trades accuracy for cost \[[23](https://arxiv.org/html/2508.12491v2#bib.bib23)\].

Our router embeds _queries_ via Φq:𝒳→ℝd\\Phi\_{q}:\\mathcal{X}\\!\\to\\!\\mathbb{R}^{d} and _experts_ via E\=\[e1;…;eM\]∈ℝM×dE=\\bigl\[e\_{1};\\dots;e\_{M}\\bigr\]\\!\\in\\!\\mathbb{R}^{M\\times d}, using either (i) logits fingerprints (EmbedLLM, Mix‑Instruct) or (ii) perplexity fingerprints (RouterBench). Given a query xx, the kk‑NN rule selects

r^k​(x;λ)\=arg​minm∈\[M\]\[1k​∑j∈𝒩k​(x)γ​(xj,hm)⏟local error+λ​c​(hm)\].\\hat{r}\_{k}(x;\\lambda)=\\mathop{\\rm arg\\,min}\_{m\\in\[M\]}\\Bigl\[\\underbrace{\\tfrac{1}{k}\\sum\_{j\\in\\mathcal{N}\_{k}(x)}\\gamma\\bigl(x\_{j},h\_{m}\\bigr)}\_{\\text{local error}}+\\lambda\\,c(h\_{m})\\Bigr\].

where 𝒩k​(x)\\mathcal{N}\_{k}(x) are the kk nearest training prompts to xx in ‖Φq​(⋅)‖2\\|\\Phi\_{q}(\\cdot)\\|\_{2}.

### 5.1 Excess‑risk of cost–spectrum k‑NN

###### Assumption 5.1 (Lipschitz Bayes error).

There exists L\>0L>0 s.t. for all x,x′∈𝒳x,x^{\\prime}\\!\\in\\!\\mathcal{X} and h∈ℋh\\!\\in\\!\\mathcal{H}, |γ​(x,h)−γ​(x′,h)|≤L​‖Φq​(x)−Φq​(x′)‖2.|\\gamma(x,h)-\\gamma(x^{\\prime},h)|\\leq L\\,\\|\\Phi\_{q}(x)-\\Phi\_{q}(x^{\\prime})\\|\_{2}.

###### Theorem 5.2 (Excess risk).

Let r^k\\hat{r}\_{k} be trained on nn i.i.d. prompt embeddings. Under Assumption [5.1](https://arxiv.org/html/2508.12491v2#S5.Thmtheorem1 "Assumption 5.1 (Lipschitz Bayes error). ‣ 5.1 Excess‑risk of cost–spectrum k‑NN ‣ 5 Theoretical Analysis ‣ Cost-Aware Contrastive Routing for LLMs"), for any λ≥0\\lambda\\!\\geq\\!0 and any k≤nk\\!\\leq\\!n,

𝔼​\[ℛλ​(r^k)\]−ℛλ​(r⋆)≤C​(kn+k−1/d),\\mathbb{E}\\bigl\[\\mathcal{R}\_{\\lambda}(\\hat{r}\_{k})\\bigr\]-\\mathcal{R}\_{\\lambda}(r^{\\star})\\;\\;\\leq\\;\\;C\\,\\Bigl(\\sqrt{\\tfrac{k}{n}}+k^{-1/d}\\Bigr),

where CC depends only on LL and the diameter of Φq​(𝒳)\\Phi\_{q}(\\mathcal{X}), and r⋆r^{\\star} is the Bayes‑optimal rule r⋆​(x)\=arg​minm\[γ​(x,hm)+λ​c​(hm)\]r^{\\star}(x)=\\mathop{\\rm arg\\,min}\_{m}\\bigl\[\\gamma(x,h\_{m})+\\lambda c(h\_{m})\\bigr\].

The proof follows the classical kk‑NN bound of \[[53](https://arxiv.org/html/2508.12491v2#bib.bib53), [5](https://arxiv.org/html/2508.12491v2#bib.bib5)\] with an extra λ​c​(h)\\lambda c(h) term that is constant w.r.t. xx and therefore preserves the rate.

### 5.2 Consistency of Cost–Spectrum InfoNCE

Fingerprints live on 𝕊d−1\\mathbb{S}^{d-1}, so dot products are scaled similarities Si​m\=qi⊤​emτkS\_{im}=\\tfrac{q\_{i}^{\\top}e\_{m}}{\\tau\_{k}} with _band‑dependent temperature_ τk\\tau\_{k} (Equation ([7](https://arxiv.org/html/2508.12491v2#S3.E7 "In Why Incorporate Cost: ‣ 3.4 Cost‑Spectrum InfoNCE. ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs"))). Let β0\=0<β1<⋯<βK\=1\\beta\_{0}\\!=\\!0<\\!\\beta\_{1}<\\!\\dots<\\!\\beta\_{K}\\!=\\!1 partition costs into bands ℬk\={m:c​(hm)∈\[βk,βk+1)}\\mathcal{B}\_{k}=\\{m:\\;c(h\_{m})\\!\\in\\!\[\\beta\_{k},\\beta\_{k+1})\\}, and denote 𝒫i​k\=𝒫​(i)∩ℬk\\mathcal{P}\_{ik}=\\mathcal{P}(i)\\cap\\mathcal{B}\_{k} the correct experts for query ii that fall in band kk. The _Cost‑Spectrum _InfoNCE__ loss for a single query ii is

ℓiCS\=−1|𝒦i|​∑k∈𝒦ilog⁡∑m∈𝒫i​kexp⁡(Si​m)∑m′\=1Mexp⁡(Si​m′−γ​cm′),\\ell\_{i}^{\\mathrm{CS}}=-\\frac{1}{|\\mathcal{K}\_{i}|}\\sum\_{k\\in\\mathcal{K}\_{i}}\\log\\frac{\\displaystyle\\sum\_{m\\in\\mathcal{P}\_{ik}}\\exp\\bigl(S\_{im}\\bigr)}{\\displaystyle\\sum\_{m^{\\prime}=1}^{M}\\exp\\!\\bigl(S\_{im^{\\prime}}-\\gamma\\,c\_{m^{\\prime}}\\bigr)},

where 𝒦i\={k:𝒫i​k≠∅}\\mathcal{K}\_{i}=\\{k:\\mathcal{P}\_{ik}\\!\\neq\\!\\varnothing\\} and γ≥0\\gamma\\!\\geq\\!0 is the negative cost penalty.

###### Lemma 5.3 (Directional alignment with cost bands).

At any stationary point of ℒCS\=1B​∑iℓiCS\\mathcal{L}\_{\\mathrm{CS}}=\\tfrac{1}{B}\\sum\_{i}\\ell\_{i}^{\\mathrm{CS}}, for every query ii and any m+∈𝒫i​km^{+}\\!\\in\\!\\mathcal{P}\_{ik}, m−∈𝒩​(i)m^{-}\\!\\in\\!\\mathcal{N}(i) with cm+≤cm−c\_{m^{+}}\\!\\leq\\!c\_{m^{-}}, qi⊤​em+\>qi⊤​em−.q\_{i}^{\\top}e\_{m^{+}}>q\_{i}^{\\top}e\_{m^{-}}.

Lemma [5.3](https://arxiv.org/html/2508.12491v2#S5.Thmtheorem3 "Lemma 5.3 (Directional alignment with cost bands). ‣ 5.2 Consistency of Cost–Spectrum InfoNCE ‣ 5 Theoretical Analysis ‣ Cost-Aware Contrastive Routing for LLMs") shows the optimum ranks cheaper correct experts ahead of expensive or wrong ones, explaining the empirical benefit of the cost term.

### 5.3 Discussion

Theorem [5.2](https://arxiv.org/html/2508.12491v2#S5.Thmtheorem2 "Theorem 5.2 (Excess risk). ‣ 5.1 Excess‑risk of cost–spectrum k‑NN ‣ 5 Theoretical Analysis ‣ Cost-Aware Contrastive Routing for LLMs") guarantees that _if_ query–expert descriptors are Lipschitz, CSCR converges to the Bayes‑optimal router at the usual kk‑NN rate. Lemma [5.3](https://arxiv.org/html/2508.12491v2#S5.Thmtheorem3 "Lemma 5.3 (Directional alignment with cost bands). ‣ 5.2 Consistency of Cost–Spectrum InfoNCE ‣ 5 Theoretical Analysis ‣ Cost-Aware Contrastive Routing for LLMs") justifies the specific form of our InfoNCE objective. Sec. [4](https://arxiv.org/html/2508.12491v2#S4 "4 Experiments ‣ Cost-Aware Contrastive Routing for LLMs") verifies these claims on three benchmarks. See proofs in Appendix [B](https://arxiv.org/html/2508.12491v2#A2 "Appendix B Proofs of Theoretical Results ‣ Cost-Aware Contrastive Routing for LLMs").

6 Conclusion
------------

We presented CSCR, a simple and efficient framework for cost-aware routing across a pool of LLMs. Our method uses two lightweight expert descriptors and trains a contrastive encoder to select the cheapest accurate expert within adaptive cost bands. Despite its simplicity, CSCR outperforms more complex routing baselines and demonstrates strong generalization to unseen models and out-of-distribution prompts. Our findings highlight the importance of embedding cost-awareness directly into the training objective of routers, rather than deferring it to test time. As model pools grow in size and diversity, activating the right-sized expert per query is critical for minimizing latency and cost. We see CSCR as a step toward more sustainable and adaptive LLM deployments, and believe this line of research is essential to avoid defaulting to unnecessarily large models for simple tasks.



Limitations and Broader Impact
------------------------------

This paper proposes a routing framework that improves inference efficiency in large language models (LLMs). By directing simple queries to smaller models, it cuts computation and memory overhead, lowering costs and environmental impact. Although the framework is built for large‑scale deployments, we could not test very large LLM pools due to resource limits; still, our experiments validate the concept. Future work will examine routing across specialized subnetworks and conditional computation within a single LLMN. Moreover, future work can answer: How big must a model be to recognize a problem’s difficulty even if it can’t solve the problem itself? Future work can study whether size and fine-tuning helps; Would RL on LLMs makes them able to purely as routers, at whether there is a model size when difficulty awareness kicks in.

Appendix A Related Work
-----------------------

### A.1 Enhancing and Optimizing LLMs

Large language models (LLMs) have demonstrated remarkable capabilities across diverse NLP tasks \[[66](https://arxiv.org/html/2508.12491v2#bib.bib66), [6](https://arxiv.org/html/2508.12491v2#bib.bib6)\]. To further improve their performance and efficiency, numerous strategies have been proposed.

##### Single-LLM Techniques.

Enhancement approaches targeting individual LLMs include fine-tuning \[[67](https://arxiv.org/html/2508.12491v2#bib.bib67)\], prompting strategies like Chain-of-Thought (CoT) \[[102](https://arxiv.org/html/2508.12491v2#bib.bib102), [90](https://arxiv.org/html/2508.12491v2#bib.bib90)\], Tree-of-Thoughts \[[96](https://arxiv.org/html/2508.12491v2#bib.bib96)\], and inference-acceleration techniques such as early exiting \[[84](https://arxiv.org/html/2508.12491v2#bib.bib84), [103](https://arxiv.org/html/2508.12491v2#bib.bib103), [72](https://arxiv.org/html/2508.12491v2#bib.bib72)\] and speculative decoding \[[77](https://arxiv.org/html/2508.12491v2#bib.bib77), [83](https://arxiv.org/html/2508.12491v2#bib.bib83), [9](https://arxiv.org/html/2508.12491v2#bib.bib9), [45](https://arxiv.org/html/2508.12491v2#bib.bib45), [8](https://arxiv.org/html/2508.12491v2#bib.bib8)\]. Additionally, Mixture-of-Experts (MoE) architectures \[[36](https://arxiv.org/html/2508.12491v2#bib.bib36), [41](https://arxiv.org/html/2508.12491v2#bib.bib41), [75](https://arxiv.org/html/2508.12491v2#bib.bib75), [24](https://arxiv.org/html/2508.12491v2#bib.bib24), [104](https://arxiv.org/html/2508.12491v2#bib.bib104), [38](https://arxiv.org/html/2508.12491v2#bib.bib38)\] route inputs through sparse sub-models or "experts," reducing cost while retaining performance. However, these methods typically operate within a single LLM’s structure and may not generalize to multi-model scenarios.

##### Model Fusion and Merging.

Fusion strategies synthesize outputs from multiple LLMs to improve output quality \[[68](https://arxiv.org/html/2508.12491v2#bib.bib68), [39](https://arxiv.org/html/2508.12491v2#bib.bib39), [29](https://arxiv.org/html/2508.12491v2#bib.bib29), [89](https://arxiv.org/html/2508.12491v2#bib.bib89), [52](https://arxiv.org/html/2508.12491v2#bib.bib52)\]. Fusion approaches often rely on unsupervised metrics \[[99](https://arxiv.org/html/2508.12491v2#bib.bib99), [73](https://arxiv.org/html/2508.12491v2#bib.bib73), [97](https://arxiv.org/html/2508.12491v2#bib.bib97)\] or ensemble voting to determine the final output \[[44](https://arxiv.org/html/2508.12491v2#bib.bib44)\]. A related but distinct technique is model merging \[[50](https://arxiv.org/html/2508.12491v2#bib.bib50)\], where weights from multiple pre-trained or fine-tuned models are combined—either directly via methods like weight averaging \[[92](https://arxiv.org/html/2508.12491v2#bib.bib92)\], Task Arithmetic \[[35](https://arxiv.org/html/2508.12491v2#bib.bib35)\], or Fisher merging \[[56](https://arxiv.org/html/2508.12491v2#bib.bib56)\]—or using more sophisticated techniques like TIES \[[94](https://arxiv.org/html/2508.12491v2#bib.bib94)\], AdaMerging \[[95](https://arxiv.org/html/2508.12491v2#bib.bib95)\], and ZipIt \[[78](https://arxiv.org/html/2508.12491v2#bib.bib78)\].

##### Cascading.

In contrast, cascading invokes models sequentially—often ordered by computational cost—and halts once a satisfactory output is generated \[[10](https://arxiv.org/html/2508.12491v2#bib.bib10), [98](https://arxiv.org/html/2508.12491v2#bib.bib98), [10](https://arxiv.org/html/2508.12491v2#bib.bib10), [30](https://arxiv.org/html/2508.12491v2#bib.bib30)\]. Such approaches strike a balance between quality and efficiency, making them particularly attractive in production settings.

### A.2 LLM Routing

Routing methods dynamically select the most appropriate LLM from a pool for each input, aiming to optimize performance and cost without querying all models. Two primary strategies dominate: non-predictive and predictive routing.

##### Non-Predictive Routing.

Non-predictive methods generate outputs from one or more models before making a selection. FrugalGPT \[[10](https://arxiv.org/html/2508.12491v2#bib.bib10)\] exemplifies this category, using a sequential strategy and a response quality threshold to minimize cost. Other works adopt layered inference architectures to escalate hard queries to more powerful models \[[91](https://arxiv.org/html/2508.12491v2#bib.bib91)\], or leverage cascades with self-verification \[[54](https://arxiv.org/html/2508.12491v2#bib.bib54), [98](https://arxiv.org/html/2508.12491v2#bib.bib98), [43](https://arxiv.org/html/2508.12491v2#bib.bib43), [51](https://arxiv.org/html/2508.12491v2#bib.bib51)\].

##### Predictive Routing.

In contrast, predictive routing aims to select the best model \*before\* any inference is performed. Strategies include supervised learning \[[76](https://arxiv.org/html/2508.12491v2#bib.bib76)\], reward-model-based routing \[[32](https://arxiv.org/html/2508.12491v2#bib.bib32), [51](https://arxiv.org/html/2508.12491v2#bib.bib51)\], and meta-models trained to predict LLM performance given an input \[[71](https://arxiv.org/html/2508.12491v2#bib.bib71)\]. Router models vary widely in implementation, including neural networks \[[19](https://arxiv.org/html/2508.12491v2#bib.bib19), [88](https://arxiv.org/html/2508.12491v2#bib.bib88), [11](https://arxiv.org/html/2508.12491v2#bib.bib11), [1](https://arxiv.org/html/2508.12491v2#bib.bib1)\], kk\-nearest neighbors \[[34](https://arxiv.org/html/2508.12491v2#bib.bib34), [76](https://arxiv.org/html/2508.12491v2#bib.bib76), [80](https://arxiv.org/html/2508.12491v2#bib.bib80), [43](https://arxiv.org/html/2508.12491v2#bib.bib43)\], matrix factorization \[[62](https://arxiv.org/html/2508.12491v2#bib.bib62), [105](https://arxiv.org/html/2508.12491v2#bib.bib105), [48](https://arxiv.org/html/2508.12491v2#bib.bib48)\], and graph neural networks \[[25](https://arxiv.org/html/2508.12491v2#bib.bib25)\]. Others incorporate model-specific tokens or train across multiple domains \[[18](https://arxiv.org/html/2508.12491v2#bib.bib18), [7](https://arxiv.org/html/2508.12491v2#bib.bib7)\].

##### Theoretical Foundations and Robustness.

Routing and cascading are grounded in broader literature, including selective classification \[[27](https://arxiv.org/html/2508.12491v2#bib.bib27), [60](https://arxiv.org/html/2508.12491v2#bib.bib60)\], learning to defer \[[55](https://arxiv.org/html/2508.12491v2#bib.bib55)\], and learning to reject \[[12](https://arxiv.org/html/2508.12491v2#bib.bib12), [4](https://arxiv.org/html/2508.12491v2#bib.bib4), [14](https://arxiv.org/html/2508.12491v2#bib.bib14)\]. Several works explore supervision levels \[[51](https://arxiv.org/html/2508.12491v2#bib.bib51), [101](https://arxiv.org/html/2508.12491v2#bib.bib101)\], robustness \[[16](https://arxiv.org/html/2508.12491v2#bib.bib16), [59](https://arxiv.org/html/2508.12491v2#bib.bib59), [74](https://arxiv.org/html/2508.12491v2#bib.bib74)\], and evaluation frameworks for routers \[[33](https://arxiv.org/html/2508.12491v2#bib.bib33), [32](https://arxiv.org/html/2508.12491v2#bib.bib32)\].

Despite rapid innovation, a standardized evaluation framework for LLM routers has been lacking. Our work addresses this gap by proposing a comprehensive benchmark for routing strategies.

### A.3 Routing as Recommendation

Routing can also be framed as a recommendation problem, wherein the input query plays the role of a "user," the pool of LLMs corresponds to "items," and past performance metrics form the implicit interaction history \[[100](https://arxiv.org/html/2508.12491v2#bib.bib100), [93](https://arxiv.org/html/2508.12491v2#bib.bib93)\]. However, unlike conventional recommender systems, routing has limited "user" features (i.e., input metadata), making label collection and generalization especially challenging \[[61](https://arxiv.org/html/2508.12491v2#bib.bib61), [49](https://arxiv.org/html/2508.12491v2#bib.bib49)\].

Matrix factorization, attention-based models, and graph neural networks are used in both recommenders and routers \[[62](https://arxiv.org/html/2508.12491v2#bib.bib62), [105](https://arxiv.org/html/2508.12491v2#bib.bib105), [25](https://arxiv.org/html/2508.12491v2#bib.bib25)\], reinforcing the close link between the two domains.

### A.4 Scaling Laws and Architecture Trends

Scaling laws \[[42](https://arxiv.org/html/2508.12491v2#bib.bib42), [47](https://arxiv.org/html/2508.12491v2#bib.bib47)\] describe predictable trends between model size, data, and performance, guiding the development of efficient LLM architectures. These insights have been extended to MoEs \[[24](https://arxiv.org/html/2508.12491v2#bib.bib24), [13](https://arxiv.org/html/2508.12491v2#bib.bib13)\], sparse models \[[26](https://arxiv.org/html/2508.12491v2#bib.bib26)\], and hybrid systems \[[28](https://arxiv.org/html/2508.12491v2#bib.bib28), [63](https://arxiv.org/html/2508.12491v2#bib.bib63)\], offering context for when routing or merging approaches might be most beneficial.

### A.5 Routing within MoE and Hybrid Architectures

Routing is a central mechanism in MoE models \[[36](https://arxiv.org/html/2508.12491v2#bib.bib36), [41](https://arxiv.org/html/2508.12491v2#bib.bib41), [75](https://arxiv.org/html/2508.12491v2#bib.bib75)\], where expert modules are dynamically activated based on input. While classical MoEs involved equally-sized sub-models, modern approaches like Switch Transformer \[[24](https://arxiv.org/html/2508.12491v2#bib.bib24)\] and Mixtral \[[38](https://arxiv.org/html/2508.12491v2#bib.bib38)\] employ sparse activation to minimize cost.

Routing LLMs can be viewed as a coarse-grained MoE, where each expert is a full LLM. Approaches like UltraFuser \[[20](https://arxiv.org/html/2508.12491v2#bib.bib20)\], Branch-Train-MiX \[[81](https://arxiv.org/html/2508.12491v2#bib.bib81)\], and token-level fusion highlight recent advances in combining model specialization and flexibility.

Appendix B Proofs of Theoretical Results
----------------------------------------

### B.1 Notation and Preliminaries

Let the Bayes‑optimal router be r⋆​(x)\=arg​minm∈\[M\]\[γ​(x,hm)+λ​cm\]r^{\\star}(x)=\\mathop{\\rm arg\\,min}\_{m\\in\[M\]}\\bigl\[\\gamma(x,h\_{m})+\\lambda c\_{m}\\bigr\] and define the _excess cost–error gap_

Δm​(x)\=γ​(x,hm)+λ​cm−(γ​(x,hr⋆​(x))+λ​cr⋆​(x)).\\Delta\_{m}(x)=\\gamma(x,h\_{m})+\\lambda c\_{m}-\\bigl(\\gamma(x,h\_{r^{\\star}(x)})+\\lambda c\_{r^{\\star}(x)}\\bigr).

Hence Δr⋆​(x)​(x)\=0\\Delta\_{r^{\\star}(x)}(x)=0 and Δm​(x)≥0\\Delta\_{m}(x)\\geq 0. For a query xx let rk​(x)r\_{k}(x) be the radius of the ball B​(x,r)⊂𝕊d−1B(x,r)\\subset\\mathbb{S}^{d-1} (in the cosine metric) that contains the kk\-th nearest training neighbor. If the marginal on q​(𝒳)q(\\mathcal{X}) has a density, 𝔼\[rk​(x)d\]≤C1​k/n\\operatorname\*{\\mathbb{E}}\\!\\bigl\[r\_{k}(x)^{d}\\bigr\]\\leq C\_{1}k/n \[[15](https://arxiv.org/html/2508.12491v2#bib.bib15), [79](https://arxiv.org/html/2508.12491v2#bib.bib79)\].

### B.2 Proof of Theorem [5.2](https://arxiv.org/html/2508.12491v2#S5.Thmtheorem2 "Theorem 5.2 (Excess risk). ‣ 5.1 Excess‑risk of cost–spectrum k‑NN ‣ 5 Theoretical Analysis ‣ Cost-Aware Contrastive Routing for LLMs")

Writing qi\=q​(xi)q\_{i}=q(x\_{i}) to lighten notation, decompose

𝔼\[ℛλ​(r^k)\]−ℛλ​(r⋆)\=𝔼x\[Δr^k​(x)​(x)−Δr^k​(x)​(xj∈𝒩k​(x))⏟(A)+Δr^k​(x)​(xj∈𝒩k​(x))−Δr⋆​(x)​(xj∈𝒩k​(x))⏟(B)\].\\operatorname\*{\\mathbb{E}}\\bigl\[\\mathcal{R}\_{\\lambda}(\\hat{r}\_{k})\\bigr\]-\\mathcal{R}\_{\\lambda}(r^{\\star})=\\operatorname\*{\\mathbb{E}}\_{x}\\!\\left\[\\underbrace{\\!\\Delta\_{\\hat{r}\_{k}(x)}(x)-\\Delta\_{\\hat{r}\_{k}(x)}(x\_{j\\in\\mathcal{N}\_{k}(x)})}\_{(A)}+\\underbrace{\\!\\Delta\_{\\hat{r}\_{k}(x)}(x\_{j\\in\\mathcal{N}\_{k}(x)})-\\Delta\_{r^{\\star}(x)}(x\_{j\\in\\mathcal{N}\_{k}(x)})}\_{(B)}\\right\].

##### Term (A): Lipschitz bias.

Assumption [5.1](https://arxiv.org/html/2508.12491v2#S5.Thmtheorem1 "Assumption 5.1 (Lipschitz Bayes error). ‣ 5.1 Excess‑risk of cost–spectrum k‑NN ‣ 5 Theoretical Analysis ‣ Cost-Aware Contrastive Routing for LLMs") gives |γ​(x,h)−γ​(x′,h)|≤L​‖q​(x)−q​(x′)‖2|\\gamma(x,h)-\\gamma(x^{\\prime},h)|\\leq L\\|q(x)-q(x^{\\prime})\\|\_{2}, so |(A)|≤L​rk​(x)|(A)|\\leq L\\,r\_{k}(x). Taking expectations and using 𝔼\[rk​(x)\]≤(C1​k/n)1/d\\operatorname\*{\\mathbb{E}}\[r\_{k}(x)\]\\leq(C\_{1}k/n)^{1/d} yields 𝔼\[(A)\]≤C2​k−1/d\\operatorname\*{\\mathbb{E}}\[(A)\]\\leq C\_{2}k^{-1/d}.

##### Term (B): Estimation variance.

Let the empirical cost‑adjusted risk be

Δ^m​(x)\=1k​∑j∈𝒩k​(x)\[γ​(xj,hm)+λ​cm\].\\widehat{\\Delta}\_{m}(x)=\\frac{1}{k}\\!\\sum\_{j\\in\\mathcal{N}\_{k}(x)}\\bigl\[\\gamma(x\_{j},h\_{m})+\\lambda c\_{m}\\bigr\].

Hoeffding’s inequality bounds

P​(|Δ^m​(x)−𝔼\[Δ^m​(x)∣x\]|≥t)≤2​e−2​k​t2,P\\!\\bigl(|\\widehat{\\Delta}\_{m}(x)-\\operatorname\*{\\mathbb{E}}\[\\widehat{\\Delta}\_{m}(x)\\mid x\]|\\geq t\\bigr)\\leq 2e^{-2kt^{2}},

and a union bound over m≤Mm\\leq M plus integration gives

𝔼\[maxm⁡|Δ^m​(x)−𝔼\[Δ^m​(x)∣x\]|\]≤C3​log⁡Mk.\\operatorname\*{\\mathbb{E}}\[\\max\_{m}|\\widehat{\\Delta}\_{m}(x)-\\operatorname\*{\\mathbb{E}}\[\\widehat{\\Delta}\_{m}(x)\\mid x\]|\]\\leq C\_{3}\\sqrt{\\tfrac{\\log M}{k}}.

Because r^k​(x)\\hat{r}\_{k}(x) minimizes Δ^m​(x)\\widehat{\\Delta}\_{m}(x), (B)≤2​maxm⁡|Δ^m​(x)−𝔼\[Δ^m​(x)∣x\]|(B)\\leq 2\\max\_{m}|\\widehat{\\Delta}\_{m}(x)-\\operatorname\*{\\mathbb{E}}\[\\widehat{\\Delta}\_{m}(x)\\mid x\]|. Combining with (A) implies

𝔼\[ℛλ​(r^k)\]−ℛλ​(r⋆)≤C​(k/n+k−1/d).\\operatorname\*{\\mathbb{E}}\[\\mathcal{R}\_{\\lambda}(\\hat{r}\_{k})\]-\\mathcal{R}\_{\\lambda}(r^{\\star})\\leq C\\bigl(\\sqrt{k/n}+k^{-1/d}\\bigr).

### B.3 Proof of Lemma [5.3](https://arxiv.org/html/2508.12491v2#S5.Thmtheorem3 "Lemma 5.3 (Directional alignment with cost bands). ‣ 5.2 Consistency of Cost–Spectrum InfoNCE ‣ 5 Theoretical Analysis ‣ Cost-Aware Contrastive Routing for LLMs")

For convenience write Si​m\=qi⊤​em/τkS\_{im}=q\_{i}^{\\top}e\_{m}/\\tau\_{k} when m∈ℬkm\\in\\mathcal{B}\_{k}. The single‑query loss (Equation ([8](https://arxiv.org/html/2508.12491v2#S3.E8 "In Why Incorporate Cost: ‣ 3.4 Cost‑Spectrum InfoNCE. ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs"))) is

ℓiCS\=−1|𝒦i|​∑k∈𝒦ilog⁡∑m∈𝒫i​kexp⁡(Si​m)∑m′\=1Mexp⁡(Si​m′−γ​cm′).\\ell\_{i}^{\\mathrm{CS}}=-\\frac{1}{|\\mathcal{K}\_{i}|}\\sum\_{k\\in\\mathcal{K}\_{i}}\\log\\frac{\\sum\_{m\\in\\mathcal{P}\_{ik}}\\exp(S\_{im})}{\\sum\_{m^{\\prime}=1}^{M}\\exp\\bigl(S\_{im^{\\prime}}-\\gamma c\_{m^{\\prime}}\\bigr)}.

Let

pi​m\=exp⁡(Si​m)/∑j∈𝒫i​kexp⁡(Si​j)p\_{im}=\\exp(S\_{im})/\\sum\_{j\\in\\mathcal{P}\_{ik}}\\exp(S\_{ij})

when m∈𝒫i​km\\in\\mathcal{P}\_{ik} and

qi​m\=exp⁡(Si​m−γ​cm)/∑j\=1Mexp⁡(Si​j−γ​cj)q\_{im}=\\exp(S\_{im}-\\gamma c\_{m})/\\sum\_{j=1}^{M}\\exp(S\_{ij}-\\gamma c\_{j})

Then

ℓiCS\=−1|𝒦i|​∑klog​∑m∈𝒫i​kpi​m/qi​m\\ell\_{i}^{\\mathrm{CS}}=-\\tfrac{1}{|\\mathcal{K}\_{i}|}\\sum\_{k}\\log\\sum\_{m\\in\\mathcal{P}\_{ik}}p\_{im}/q\_{im}

Taking the gradient w.r.t. qiq\_{i} and setting it to zero gives ∑m(pi​m−qi​m)​em\=0\\sum\_{m}(p\_{im}-q\_{im})e\_{m}=0. Project onto qiq\_{i}: ∑m(pi​m−qi​m)​Si​m\=0.\\sum\_{m}(p\_{im}-q\_{im})S\_{im}=0. Fix m+∈𝒫i​km^{+}\\in\\mathcal{P}\_{ik}, m−∈𝒩​(i)m^{-}\\in\\mathcal{N}(i) with cm+≤cm−c\_{m^{+}}\\leq c\_{m^{-}}. Because pi​m−\=0p\_{im^{-}}=0 while pi​m+\>0p\_{im^{+}}>0, the equality forces qi​m+\>qi​m−q\_{im^{+}}>q\_{im^{-}}, hence Si​m+−γ​cm+\>Si​m−−γ​cm−S\_{im^{+}}-\\gamma c\_{m^{+}}>S\_{im^{-}}-\\gamma c\_{m^{-}}. Re‑arranging yields qi⊤​em+\>qi⊤​em−q\_{i}^{\\top}e\_{m^{+}}>q\_{i}^{\\top}e\_{m^{-}}, establishing directional alignment.

□\\Box

Appendix C Method
-----------------

### C.1 Logit-Footprint Descriptors

##### Why take the most frequent tokens?

We use the most frequent tokens so the basis is shared and stable: they appear in all models, give low-noise estimates with few probes, and make calibration comparable across experts. They give less noisy estimates because they get non-negligible probability across many contexts, so their averaged log-probs vary less than rare or Out-of-Vocabulary tokens. In all experiments, we set K\=256K=256 and T\=10T=10 ([D.1](https://arxiv.org/html/2508.12491v2#A4.SS1 "D.1 Experimental Settings ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs")), which is large enough that the basis isn’t dominated by a few function words.

##### Are frequent tokens trivial?

These tokens aren’t used for their meaning. They’re probes of each model’s output behavior. Even common words get scored differently across models (temperature, punctuation/number handling, style). By averaging over many prompts and steps, the descriptor captures overall model behavior, not any single word’s semantics. Also, a recent work \[[82](https://arxiv.org/html/2508.12491v2#bib.bib82)\] shows that LLMs exhibit stable, word-level idiosyncrasies (as the authors call them) that enable near-perfect model attribution using only the first few generated tokens (even after paraphrasing or translation), implying that common tokens still provide discriminative signals about a model’s predictive calibration.

##### Shared token set vs. per-model selection.

A shared basis gives all descriptors a common coordinate system. If each model used a different token set, cosine distances would mix basis changes with true behavior, hurting comparability. It would also require computing many more probes to align per-model bases that are different across models.

##### Beyond raw frequency.

We deliberately kept the descriptors simple to isolate and quantify the contrastive router’s contribution. Nonetheless, frequency is a pragmatic, not necessarily optimal, choice. Two variants that we considered and could be explored are:

*   •
    
    TF–IDF-weighted selection over the probe corpus.
    
*   •
    
    Picking tokens with the largest across-model log-prob variance.
    

These can be dropped into Equation [2](https://arxiv.org/html/2508.12491v2#S3.E2 "In 3.1.1 Logit–Footprint Descriptors for Transparent LLMs ‣ 3.1 Model Fingerprints ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs") without changing downstream training or inference.

### C.2 The Step from Equation [3](https://arxiv.org/html/2508.12491v2#S3.E3 "In 3.1.2 Perplexity Fingerprints for Black‑Box or API‑Only LLMs ‣ 3.1 Model Fingerprints ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs") to Equation [5](https://arxiv.org/html/2508.12491v2#S3.E5 "In 3.1.2 Perplexity Fingerprints for Black‑Box or API‑Only LLMs ‣ 3.1 Model Fingerprints ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")

Equation (3) defines a per-prompt token NLL that requires access to an expert’s next-token distribution ph(⋅∣⋅)p\_{h}(\\cdot\\mid\\cdot). For API-only (black-box) experts, logits/probabilities are not exposed, so Equation (3) is not computable. Our remedy is to (a) let the API expert hh produce a deterministic continuation y^h​(x)\\hat{y}\_{h}(x) for prompt xx (greedy decoding), and (b) evaluate that sequence under a single shared, public scorer pSp\_{S} (kept fixed across all experts). This yields Equation (5), a _pseudo-perplexity_:

ℓ~h(x)\=−1|y^h​(x)|∑t\=1|y^h​(x)|logpS(y^h,t|y^h,<t),\\tilde{\\ell}\_{h}(x)\\;=\\;-\\frac{1}{|\\hat{y}\_{h}(x)|}\\sum\_{t=1}^{|\\hat{y}\_{h}(x)|}\\log p\_{S}\\!\\left(\\hat{y}\_{h,t}\\,\\middle|\\,\\hat{y}\_{h,<t}\\right),

which we then normalize (similar to Equation (4)) and use as the fingerprint coordinate(s) for black-box experts.

If y^h\\hat{y}\_{h} is a typical (high-probability) output of hh (i.e., y^h∼ph\\hat{y}\_{h}\\sim p\_{h}) then averaging the pseudo-perplexity ℓ~h​(x)\=−1|y^h​(x)|​∑tlog⁡pS​(y^h,t∣y^h,<t)\\tilde{\\ell}\_{h}(x)=-\\tfrac{1}{|\\hat{y}\_{h}(x)|}\\sum\_{t}\\log p\_{S}(\\hat{y}\_{h,t}\\mid\\hat{y}\_{h,<t}) over many prompts/tokens is equivalent to taking an expectation over y∼phy\\!\\sim\\!p\_{h}:

𝔼y∼ph​\[−log⁡pS​(y)\]\=H​(ph,pS)\=H​(ph)+KL​(ph∥pS).\\mathbb{E}\_{y\\sim p\_{h}}\\big\[-\\log p\_{S}(y)\\big\]\\;=\\;H(p\_{h},p\_{S})\\;=\\;H(p\_{h})\\;+\\;\\mathrm{KL}\\!\\big(p\_{h}\\;\\|\\;p\_{S}\\big).

Here H​(ph,pS)H(p\_{h},p\_{S}) is the cross-entropy of php\_{h} with respect to pSp\_{S}, which decomposes into the entropy of hh’s own distribution H​(ph)H(p\_{h}) and its divergence from the scorer KL​(ph∥pS)\\mathrm{KL}(p\_{h}\\|p\_{S}).

Because pSp\_{S} is fixed for all experts, H​(ph,pS)H(p\_{h},p\_{S}) is a stable, model-specific quantity that makes descriptors comparable across experts (“same yardstick”). It is not the true NLL under php\_{h}, but it preserves differences between experts via H​(ph)H(p\_{h}) and their mismatch to pSp\_{S} via KL​(ph∥pS)\\mathrm{KL}(p\_{h}\\|p\_{S}). In practice we use deterministic (greedy) decoding to reduce variance; averaging over many prompts/tokens makes the empirical ℓ~h​(x)\\tilde{\\ell}\_{h}(x) closely track the expectation above. Figure [2](https://arxiv.org/html/2508.12491v2#S3.F2 "Figure 2 ‣ Why perplexity? ‣ 3.1.2 Perplexity Fingerprints for Black‑Box or API‑Only LLMs ‣ 3.1 Model Fingerprints ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs") empirically validates this argument.

### C.3 Band-Specific Temperatures and Smoother Gradients

Most prompts in everyday interactions (and in our datasets) can be handled by cheaper models; plus there are usually fewer very expensive experts overall. These expensive experts are only needed for a small fraction of hard prompts, so within those high-cost bands there are fewer suitable positives per query. With few positives, the similarity distribution becomes very peaky.

A larger τk\\tau\_{k} in Equation [7](https://arxiv.org/html/2508.12491v2#S3.E7 "In Why Incorporate Cost: ‣ 3.4 Cost‑Spectrum InfoNCE. ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs") flattens the per-band softmax, reducing gradient variance and preventing the update from collapsing onto a single rare positive. Formally, for band kk the per-query gradient w.r.t. the query embedding is

∇qℒk\=−∑m∈Pi​kpm(+)​emτk+∑m′\=1Mpm′(−)​em′τk,\\nabla\_{q}\\mathcal{L}\_{k}\\;=\\;-\\sum\_{m\\in P\_{ik}}p^{(+)}\_{m}\\,\\frac{e\_{m}}{\\tau\_{k}}\\;+\\;\\sum\_{m^{\\prime}=1}^{M}p^{(-)}\_{m^{\\prime}}\\,\\frac{e\_{m^{\\prime}}}{\\tau\_{k}},

(10)

where p(+)p^{(+)} and p(−)p^{(-)} are the band-restricted softmaxes over positives and all experts (with the negative cost penalty in the denominator). As τk\\tau\_{k} increases, both softmaxes become less concentrated, so (i) the gradient magnitude scales like 1/τk1/\\tau\_{k} and (ii) its direction is averaged over more positives, lowering variance across minibatches.

This is the sense in which band-specific temperatures yield smoother gradients; it is especially helpful in high-cost bands that otherwise have few positives and highly variable similarities. Without band-specific temperatures, the router can exhibit oscillatory updates on hard prompts (rare positives dominate, then vanish), slowing convergence and encouraging over-use of cheap experts. Empirically, we observed that bands and band-specific temperatures are important (Table [7](https://arxiv.org/html/2508.12491v2#A4.T7 "Table 7 ‣ D.3 Ablation of Cost Bands ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs"))

### C.4 Dense Human Annotations

We considered using human annotations but intentionally avoided them: model pools change quickly, so adding/replacing experts would require fresh labels that are costly and often unavailable. Instead, we train with sparse correctness and cost signals, which remain portable across experts. If dense feedback is available, it could help in several ways:

*   •
    
    Positive sets P​(i)P(i) with preference structure. Replace binary "correct expert" labels with pairwise preferences (cheap‐and‐good ≻\\succ expensive‐and‐similar ≻\\succ clearly wrong), yielding band-aware positives and margin constraints. This can be implemented by expanding P​(i)P(i) and adding a lightweight pairwise ranking (DPO-style) regularizer within each cost band.
    
*   •
    
    Difficulty-aware reweighting. Use human "hardness" scores to upweight rare/hard prompts when computing the contrastive loss, especially in higher cost bands. this could balance the effective sample sizes across easy vs. hard (and cheap vs. expensive band) cases so the gradient isn’t dominated by the abundant, easy examples.
    
*   •
    
    Band calibration. We can ask users how much quality they’re willing to trade for a lower cost, then use that to set the cost bands and the penalty for picking more expensive models, so the router’s choices match what users actually prefer.
    

Appendix D Experiments
----------------------

### D.1 Experimental Settings

##### Baselines

We compare our proposed method against a comprehensive set of baselines designed to capture key routing strategies and their trade-offs. Specifically, we include UMR \[[40](https://arxiv.org/html/2508.12491v2#bib.bib40)\], a recent state-of-the-art technique that clusters prompt embeddings to route queries to LLM pools efficiently; Thompson Sampling \[[48](https://arxiv.org/html/2508.12491v2#bib.bib48), [2](https://arxiv.org/html/2508.12491v2#bib.bib2)\], which frames routing as a bandit exploration-exploitation problem to balance cost and accuracy dynamically; Pareto-optimal routing \[[34](https://arxiv.org/html/2508.12491v2#bib.bib34)\], a strategy that selects models by explicitly considering the cost-accuracy Pareto frontier; and two extreme baselines—Random, which selects models uniformly at random to represent naive routing without intelligent selection, and Oracle (Clairvoyant Upper-Bound \[[40](https://arxiv.org/html/2508.12491v2#bib.bib40)\]), which always selects the most accurate model at the lowest possible cost and thus represents a theoretical performance ceiling. Additionally, we evaluate against parametric gating methods (Parametric Softmax Router) inspired by classical mixture-of-experts architectures \[[62](https://arxiv.org/html/2508.12491v2#bib.bib62)\] and SoftMoE, which models router decisions via differentiable soft gating functions \[[64](https://arxiv.org/html/2508.12491v2#bib.bib64)\]. Collectively, these baselines enable us to rigorously assess whether our contrastive routing approach delivers meaningful improvements in performance, cost-efficiency, and generalization capabilities relative to existing strategies.

##### Datasets, Benchmarks, and Evaluation

We train our router and evaluate it on three datasets: EmbedLLM \[[105](https://arxiv.org/html/2508.12491v2#bib.bib105)\], MixInstruct \[[39](https://arxiv.org/html/2508.12491v2#bib.bib39)\], and RouterBench \[[34](https://arxiv.org/html/2508.12491v2#bib.bib34)\]. For EmbedLLM and MixInstruct, we sample 192 probes from their respective validation sets. Each probe is processed to extract logit-based descriptors by capturing the top K\=256K=256 tokens over a horizon of T\=10T=10 tokens (Equation ([2](https://arxiv.org/html/2508.12491v2#S3.E2 "In 3.1.1 Logit–Footprint Descriptors for Transparent LLMs ‣ 3.1 Model Fingerprints ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs"))), resulting in a 256-dimensional vector per model. For RouterBench, we sample 192 probes from its training set, ensuring these probes are excluded from the training data used for the contrastive router. We compute perplexity-based descriptors on RouterBench and use GPT-2 \[[66](https://arxiv.org/html/2508.12491v2#bib.bib66)\]. On both EmbedLLM and RouterBench, we use binary accuracy as the per-sample evaluation metric, meaning an LLM response is classified strictly as correct or incorrect. For MixInstruct, we employ exponentiated BARTScore \[[97](https://arxiv.org/html/2508.12491v2#bib.bib97)\] as the evaluation metric, following the approach in prior work \[[40](https://arxiv.org/html/2508.12491v2#bib.bib40), [39](https://arxiv.org/html/2508.12491v2#bib.bib39)\].

We evaluate each routing strategy using a deferral curve \[[40](https://arxiv.org/html/2508.12491v2#bib.bib40)\] which plots the average response quality against the total inference cost. Sweeping the routing penalty parameter λ\\lambda over the interval λ∈\[0,λmax\]\\lambda\\!\\in\\!\[0,\\lambda\_{\\max}\] (refer to Equation ([9](https://arxiv.org/html/2508.12491v2#S3.E9 "In 3.5 Inference Router ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs"))) traces the deferral curve. For the EmbedLLM and MixInstruct datasets, we define the cost of processing a prompt as the number of parameters in the LLM, serving as a proxy for computational resources and latency. In the case of RouterBench, we utilize the actual API call costs in USD, as provided in the dataset. Following \[[40](https://arxiv.org/html/2508.12491v2#bib.bib40)\] we employ evaluation metrics including Area Under the Deferral Curve (AUDC), Query-Normalized Cost (QNC), and peak accuracy. QNC is the minimum relative cost required to match the performance of the most accurate tested LLM.

##### Training

We use a frozen sentence-transformers/all-MiniLM-L6-v2\[[70](https://arxiv.org/html/2508.12491v2#bib.bib70)\] model as the embedding backbone (Φ​(x)\\Phi(x) in Section[3](https://arxiv.org/html/2508.12491v2#S3 "3 Method ‣ Cost-Aware Contrastive Routing for LLMs")) across all experiments. Our trainable router component is a two-layer MLP, denoted as gθ(.)g\_{\\theta}(.), which projects prompt embeddings into the expert descriptor space. We train our contrastive router on the training splits of each dataset, excluding the probe examples from RouterBench. Training is performed for 10 epochs using the AdamW optimizer with a batch size of 512 and a learning rate of 5×10−45\\times 10^{-4}. For the cost spectrum loss (Equation ([8](https://arxiv.org/html/2508.12491v2#S3.E8 "In Why Incorporate Cost: ‣ 3.4 Cost‑Spectrum InfoNCE. ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs"))), we set the number of cost bands to K\=5K=5 and the negative cost penalty to λ\=0.1\\lambda=0.1. The hyperparameters for the linear schedule of band-specific temperatures (Equation ([7](https://arxiv.org/html/2508.12491v2#S3.E7 "In Why Incorporate Cost: ‣ 3.4 Cost‑Spectrum InfoNCE. ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs"))) are set as α\=0.25\\alpha=0.25 and τmin\=0.05\\tau\_{\\min}=0.05. All training and descriptor extraction are done on RTX6000Ada GPUs with ∼\\sim48GB GPU memory.

### D.2 Results

#### D.2.1 Out-of-Distribution Prompt Experiments

In the out-of-distribution (OOD) experiments, we divided the prompts in the EmbedLLM dataset into two challenging sets based on their categories: STEM-related (Science, Technology, Engineering, and Mathematics) and Non-STEM-related (covering Social sciences, Humanities, Arts, etc.). A detailed summary of the train and test categories for the OOD experiments is provided in Table [6](https://arxiv.org/html/2508.12491v2#A4.T6 "Table 6 ‣ D.2.1 Out-of-Distribution Prompt Experiments ‣ D.2 Results ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs"). Our splitting yielded 18,193 out of 36,054 total training questions and 1,060 distinct test prompts. Training and testing is done on all available LLMs across both splits.

Table 6: Full breakdown of training and testing categories used in OOD experiments.

Set

Categories

Train (STEM)

asdiv, gsm8k, medmcqa, mathqa, piqa, logiqa, gpqa\_main\_cot\_n\_shot, gpqa\_main\_cot\_zeroshot, gpqa\_main\_n\_shot,

gpqa\_main\_zeroshot, gpqa\_diamond\_cot\_n\_shot, gpqa\_diamond\_cot\_zeroshot, gpqa\_diamond\_n\_shot, gpqa\_diamond\_zeroshot,

gpqa\_extended\_cot\_n\_shot, gpqa\_extended\_cot\_zeroshot, gpqa\_extended\_n\_shot, gpqa\_extended\_zeroshot, mmlu\_college\_medicine,

mmlu\_astronomy, mmlu\_conceptual\_physics, mmlu\_college\_computer\_science, mmlu\_college\_biology, mmlu\_electrical\_engineering,

mmlu\_medical\_genetics, mmlu\_college\_physics, mmlu\_high\_school\_chemistry, mmlu\_computer\_security, mmlu\_clinical\_knowledge,

mmlu\_virology, mmlu\_machine\_learning, mmlu\_college\_mathematics, mmlu\_elementary\_mathematics, mmlu\_professional\_medicine,

mmlu\_college\_chemistry, mmlu\_high\_school\_biology, mmlu\_anatomy, mmlu\_high\_school\_statistics, mmlu\_high\_school\_physics,

mmlu\_high\_school\_computer\_science, mmlu\_high\_school\_mathematics

Test (Non-STEM)

social\_iqa, truthfulqa\_mc1, mmlu\_high\_school\_european\_history, mmlu\_us\_foreign\_policy,

mmlu\_high\_school\_microeconomics, mmlu\_business\_ethics, mmlu\_public\_relations, mmlu\_jurisprudence, mmlu\_nutrition,

mmlu\_high\_school\_world\_history, mmlu\_miscellaneous, mmlu\_formal\_logic, mmlu\_management, mmlu\_high\_school\_psychology,

mmlu\_high\_school\_government\_and\_politics, mmlu\_high\_school\_geography, mmlu\_world\_religions, mmlu\_international\_law,

mmlu\_human\_aging, mmlu\_sociology, mmlu\_professional\_accounting, mmlu\_prehistory, mmlu\_logical\_fallacies, mmlu\_moral\_disputes,

mmlu\_human\_sexuality, mmlu\_professional\_psychology, mmlu\_high\_school\_us\_history, mmlu\_high\_school\_macroeconomics,

mmlu\_abstract\_algebra, mmlu\_global\_facts, mmlu\_security\_studies, mmlu\_philosophy, mmlu\_professional\_law, mmlu\_moral\_scenarios,

mmlu\_marketing

### D.3 Ablation of Cost Bands

We measured AUCD and peak accuracy with and without bands. Table [7](https://arxiv.org/html/2508.12491v2#A4.T7 "Table 7 ‣ D.3 Ablation of Cost Bands ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs") presents these results, showing having cost bands is indeed effective.

Setting

AUCD

Peak Accuracy

Without bands

0.4574

0.482

With bands

0.4951

0.540

Table 7: Effect of banded cost temperatures on model performance

#### D.3.1 Ablation of Number of Cost bands

Table [8](https://arxiv.org/html/2508.12491v2#A4.T8 "Table 8 ‣ D.3.1 Ablation of Number of Cost bands ‣ D.3 Ablation of Cost Bands ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs") and Figure [7](https://arxiv.org/html/2508.12491v2#A4.F7 "Figure 7 ‣ D.3.1 Ablation of Number of Cost bands ‣ D.3 Ablation of Cost Bands ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs") show that performance varies with the number of cost bands. Using 5 bands yields the best results, with the highest AUDC (0.5480) and Peak accuracy (0.590), suggesting a good balance between flexibility and generalization. Fewer bands (e.g., 2) limit routing precision, while too many bands (e.g., 15 or 20) degrade performance, likely due to over-fragmentation and increased decision noise. This highlights the importance of tuning the number of bands to avoid both under- and overfitting.

![Refer to caption](figures/band_ablation.png)  

Figure 7: Deferral‑curve metrics across different numbers of cost bands.

Router (Bands)

AUDC ↑\\uparrow

QNC ↓\\downarrow

Peak ↑\\uparrow

bands:2

0.5357

47.693

0.563

bands:5

0.5480

66.057

0.590

bands:11

0.5173

30.412

0.547

bands:15

0.4460

13.474

0.496

bands:20

0.4649

11.541

0.477

  

Table 8: Deferral‑curve metrics across different numbers of cost bands.

#### D.3.2 Ablation of the Number of Neighbors

Increasing kk (the number of ANN neighbors selected before cost-aware scoring) generally provides modest improvements before reaching a plateau. (The baseline UMR \[[40](https://arxiv.org/html/2508.12491v2#bib.bib40)\] also included an ablation on kk in a K-NN router.) Beyond a small KK, the gains in accuracy or AUDC become minimal, while both latency and the likelihood of choosing unnecessarily expensive experts increase. We selected a default of K\=4K=4 and it worked reasonably well so we did not do further hyperparameter tuning. We performed an ablation on a subset of embedllm prompts. The results are shown in Table [9](https://arxiv.org/html/2508.12491v2#A4.T9 "Table 9 ‣ D.3.2 Ablation of the Number of Neighbors ‣ D.3 Ablation of Cost Bands ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs") which are consistent with the trend observed in UMR.

KK

AUDC

cost@max\_acc

1

0.5186

7.665

2

0.5294

7.759

4

0.5344

7.831

8

0.5378

7.940

16

0.5402

8.034

Table 9: Effect of Number of Nearest Neighbors (KK) on AUDC and Average Cost

#### D.3.3 Ablation of Negative Cost Penalty

γ\\gamma is a soft deterrent against assigning probability mass to costly, wrong experts during training (it appears in the denominator of the band-softmax in the loss in Equation [8](https://arxiv.org/html/2508.12491v2#S3.E8 "In Why Incorporate Cost: ‣ 3.4 Cost‑Spectrum InfoNCE. ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs"). We ablate this hyperparameter in Table [10](https://arxiv.org/html/2508.12491v2#A4.T10 "Table 10 ‣ D.3.3 Ablation of Negative Cost Penalty ‣ D.3 Ablation of Cost Bands ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs"). If too small, the router learns to over-consider expensive hub experts (cost creep). If too large, it over-penalizes cost and undertrains on valid high-cost positives, hurting hard prompts. We picked 0.20.2 as it’s a light regularizer: enough to push apart costly negatives first, but not so strong that it drowns the similarity signal for truly necessary expensive experts.

Router

AUDC

max\_acc

cost@max\_acc

γ\=0\\gamma{=}0

0.5518

0.5980

55.543

γ\=0.1\\gamma{=}0.1

0.5566

0.5850

44.499

γ\=0.2\\gamma{=}0.2

0.5526

0.5720

43.083

γ\=0.3\\gamma{=}0.3

0.5272

0.5307

14.901

γ\=0.5\\gamma{=}0.5

0.5129

0.5147

8.836

Table 10: Gamma ablation

#### D.3.4 Ablation of Band-Specific Temperature Slope

In the band-specific temperature schedule τk\=τmin+α​c¯k\\tau\_{k}=\\tau\_{\\min}+\\alpha\\,\\bar{c}\_{k} (Equation [7](https://arxiv.org/html/2508.12491v2#S3.E7 "In Why Incorporate Cost: ‣ 3.4 Cost‑Spectrum InfoNCE. ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs")), the slope α\\alpha sets how much flatter the softmax is in higher-cost bands. Increasing α\\alpha raises τk\\tau\_{k} for expensive bands, which flattens their per-band softmax over experts. This reduces gradient variance in those bands (where each query has fewer suitable positive) mitigating collapse onto a single rare positive and preventing cost-creep during training.

α\\alpha

AUDC

max\_acc

cost@max\_acc

0

0.5567

0.5930

64.292

0.1

0.5704

0.6033

55.680

0.25

0.5701

0.6003

48.360

0.4

0.5557

0.5657

32.854

0.5

0.5448

0.5517

29.175

Table 11: Ablation over the band slope α\\alpha in τk\=τmin+α​c¯k\\tau\_{k}=\\tau\_{\\min}+\\alpha\\,\\bar{c}\_{k}. Moderate α\\alpha (0.1–0.25) maximizes AUDC and lowers the cost required to reach peak accuracy; large α\\alpha oversmooths high-cost bands and reduces peak accuracy.

See Table [11](https://arxiv.org/html/2508.12491v2#A4.T11 "Table 11 ‣ D.3.4 Ablation of Band-Specific Temperature Slope ‣ D.3 Ablation of Cost Bands ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs") for an ablation of this hyperparameter. As α\\alpha increases from 0 to a moderate value (0.10.1–0.250.25), _AUDC_ improves and the _cost@max\_acc_ drops, indicating we reach peak quality at lower cost, while _max\_acc_ remains comparable. For larger α\\alpha (≥0.4\\geq 0.4), the high-cost bands become oversmoothed, weakening discrimination among expensive experts; _max\_acc_ and _AUDC_ decline despite further cost reductions. This validates our choice of adopting a moderate setting (default α\=0.25\\alpha{=}0.25, with 0.10.1 performing slightly better).

#### D.3.5 Ablation of Cheapest Cost Band

τmin\\tau\_{\\min} is the softmax temperature for the cheapest cost band in Equation [7](https://arxiv.org/html/2508.12491v2#S3.E7 "In Why Incorporate Cost: ‣ 3.4 Cost‑Spectrum InfoNCE. ‣ 3 Method ‣ Cost-Aware Contrastive Routing for LLMs"); all other bands inherit τk\=τmin+α​c¯k\\tau\_{k}=\\tau\_{\\min}+\\alpha\\,\\bar{c}\_{k}. A very small τmin\\tau\_{\\min} makes the cheap-band softmax sharp (highly discriminative but prone to noisy, peaky gradients) while a larger τmin\\tau\_{\\min} smooths the distribution, lowering variance but also blurring differences among cheap experts.

τmin\\tau\_{\\min}

AUDC

max\_acc

cost@max\_acc

0

0.5484

0.5877

51.232

0.02

0.5602

0.6037

51.704

0.05

0.5581

0.5883

44.633

0.08

0.5515

0.5707

38.395

Table 12: Ablation over the base temperature τmin\\tau\_{\\min} (with α\=0.25\\alpha=0.25). Moderate values improve AUDC and lower the cost required for peak or near-peak accuracy; very small or large values underperform.

Table [12](https://arxiv.org/html/2508.12491v2#A4.T12 "Table 12 ‣ D.3.5 Ablation of Cheapest Cost Band ‣ D.3 Ablation of Cost Bands ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs") ablates τmin\\tau\_{\\text{min}}. Raising τmin\\tau\_{\\min} from 0 to 0.020.02 increases AUDC and peak accuracy, showing that a touch of smoothing stabilizes learning without hurting discrimination. At τmin\=0.05\\tau\_{\\min}=0.05 we keep nearly the same AUDC while cutting the cost needed to achieve peak accuracy by ≈14%\\approx 14\\% (from 51.7 to 44.6). Pushing to τmin\=0.08\\tau\_{\\min}=0.08 oversmoothes the cheap band: accuracy at low cost rises slightly, but _max\_acc_ and AUDC both fall. Thus a moderate setting (τmin≈0.02\\tau\_{\\min}\\!\\approx\\!0.02–0.050.05) offers the best efficiency–stability trade-off.

### D.4 Larger Encoder Size

We kept the router is deliberately small: a frozen sentence-transformer plus a 2-layer MLP. We performed an ablation where we increased the dimension of the middle layer in Table [13](https://arxiv.org/html/2508.12491v2#A4.T13 "Table 13 ‣ D.4 Larger Encoder Size ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs"). As the router size increases, AUDC improves, but latency also increases.

Router

AUDC

MLP-x1

0.5189

MLP-x4

0.5384

Table 13: Effect of Router Size on AUDC

Broadly, future work can answer: How big must a model be to recognize a problem’s difficulty even if it can’t solve the problem itself? Future work can also study whether size and fine-tuning helps and whether RL on LLMs makes them able to purely as routers? Future work can also study different model sizes to pinpoint when difficulty awareness kicks in.

### D.5 Qualitative Insights and Interpretability of Routing

We performed an analysis on embedllm, which features a large pool of models with diverse costs. Note that these observed trends are specific to the dataset and may not generalize to other datasets or prompt pools with more challenging examples.

#### D.5.1 Selection Profiles

Qwen/Qwen1.5-0.5B-Chat (selected 91 times), google/gemma-2b-it (233), and microsoft/phi-2 (292) are smaller experts that are selected frequently. Some expensive experts (e.g., Qwen/Qwen-72B, ibivibiv/alpaca-dragon-72b-v1) are rarely chosen, since a less costly correct expert typically exists in the dataset.

We also report per-expert selection rates by cost band in Table [14](https://arxiv.org/html/2508.12491v2#A4.T14 "Table 14 ‣ D.5.1 Selection Profiles ‣ D.5 Qualitative Insights and Interpretability of Routing ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs").

Band Index

Count

0

533

1

837

2

805

3

694

4

131

Table 14: Prompt Counts by Band Index

#### D.5.2 Routing Error Breakdown

We present a confusion-style breakdown of routing errors in Table [15](https://arxiv.org/html/2508.12491v2#A4.T15 "Table 15 ‣ D.5.2 Routing Error Breakdown ‣ D.5 Qualitative Insights and Interpretability of Routing ‣ Appendix D Experiments ‣ Cost-Aware Contrastive Routing for LLMs")

*   •
    
    Too cheap—the router selects a cheap but incorrect expert
    
*   •
    
    Too expensive—the router selects an unnecessarily costly expert, though correct
    
*   •
    
    Optimal—the router selects a correct and minimally costly expert
    
*   •
    
    No correct—no available expert produces a correct answer
    

Outcome

Count

Too cheap

235

Too expensive

586

Optimal

773

No correct

63

Table 15: Routing Outcome Breakdown

### D.6 Statistical Significance

We perform full evaluation on Embedllm \[[105](https://arxiv.org/html/2508.12491v2#bib.bib105)\] using paired, prompt-level significance tests to concretely assess statistical significance. Specifically, we computed a paired bootstrap \[[22](https://arxiv.org/html/2508.12491v2#bib.bib22)\] (sampling 3,000 prompts with replacement, 5,000 times) to obtain a 95% confidence interval for ΔAUDC\=AUDCCSCR−AUDCUMR\\Delta\_{\\mathrm{AUDC}}=\\mathrm{AUDC}\_{\\text{CSCR}}-\\mathrm{AUDC}\_{\\text{UMR}} (UMR is the best baseline overall). We also report a one-sided p-value for the hypothesis Δ\>0\\Delta>0. Additionally, we ran McNemar’s \[[57](https://arxiv.org/html/2508.12491v2#bib.bib57)\] test at a matched budget (using the median of the combined cost grids) to compare per-prompt wins and losses at equivalent operating cost. These tests quantify uncertainty over the test prompts.

Dataset

ΔAUDC\\Delta\_{\\textbf{AUDC}}

95% CI

𝒑\\boldsymbol{p} (bootstrap)

c⋆c^{\\star}

n10/n01n\_{10}/n\_{01}

𝒑\\boldsymbol{p} (McNemar)

Embedllm

+0.053

\[0.037, 0.069\]

2.0×10−42.0\\times 10^{-4}

6.08

560/366

9.76×10−119.76\\times 10^{-11}

Table 16: Paired significance versus the strongest baseline. ΔAUDC\=AUDCCSCR−AUDCUMR\\Delta\_{\\mathrm{AUDC}}=\\mathrm{AUDC}\_{\\text{CSCR}}-\\mathrm{AUDC}\_{\\text{UMR}} (area under the deferral curve; higher is better). “95% CI” and the one-sided pp come from a paired bootstrap over prompts (N\=3000N=3000, B\=5000B=5000; H1:Δ\>0H\_{1}\\!:\\Delta>0). c⋆c^{\\star} is the matched budget used for McNemar. n10/n01n\_{10}/n\_{01} are discordant counts (CSCR correct / baseline correct), and “pp (McNemar)” is the one-sided exact binomial pp for CSCR \>\> baseline at c⋆c^{\\star}. CIs that exclude 0 and small pp\-values indicate a statistically significant improvement of CSCR.

Δ​AUDC\\Delta\\mathrm{AUDC} is positive with CIs that exclude zero, and McNemar shows win rates above 0.5 with very small p-values at the matched budget. In other words: CSCR’s deferral curve encloses more area (higher accuracy at the same or lower cost on average), and at a fixed budget it wins on more prompts than it loses. This complements the Pareto-frontier plots: the gains are not an artifact of a single operating point or random variation, but hold paired, prompt by prompt. The paired test establishes statistical significance.
