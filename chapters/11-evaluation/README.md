# Chapter 11: Evaluation

This chapter covers how to measure whether an LLM is good, and how to detect and reduce one of its most important failures: hallucination.

## Sections

| # | Section | Summary |
|---|---|---|
| 1 | [Why Evaluation Is Hard](01-why-evaluation-is-hard.md) | Why training loss is not usefulness, how offline and online metrics differ, and how Goodhart's law and benchmark saturation distort measurement. |
| 2 | [Intrinsic Metrics](02-intrinsic-metrics.md) | Cross-entropy, perplexity, and bits per byte: what they measure and miss; data contamination and how to detect and prevent it. |
| 3 | [Benchmarks and Task Suites](03-benchmarks-and-task-suites.md) | Knowledge, reasoning, math, code, and chat benchmarks (MMLU to Humanity's Last Exam, pass@k, SWE-bench, Arena Elo and Bradley-Terry), and how to read a leaderboard critically. |
| 4 | [Hallucination](04-hallucination.md) | What hallucination is, its types and causes, how to measure it (SimpleQA-style grading, FActScore, self-consistency, calibration), and how to reduce it. |
| 5 | [Human Evaluation](05-human-evaluation.md) | Rubrics and pairwise comparisons, inter-annotator agreement with Cohen's kappa, and the cost, bias, and expertise limits of human judgment. |
| 6 | [LLM-as-a-Judge](06-llm-as-a-judge.md) | Reference-based and reference-free judging, position, verbosity, and self-preference biases, validating judges against humans, and when humans are still needed. |
| 7 | [Safety Evaluation](07-safety-evaluation.md) | Toxicity and harmful-content benchmarks, jailbreak robustness and red teaming, over-refusal, and frontier dangerous-capability evaluations. |
| 8 | [Reporting Results Honestly](08-reporting-results-honestly.md) | Confidence intervals, paired comparisons and the bootstrap, seeds and other variance, avoiding cherry-picking, and an evaluation checklist for a new model or fine-tune. |

## Suggested code labs

1. Compute perplexity on a short held-out text with a small open model.
2. Build a tiny factual QA set, run a model on it, and measure its hallucination rate.
3. Use self-consistency (sample five answers) to flag likely hallucinations.
4. Score a handful of answers with a rubric and with an LLM judge, and compare how often they agree.

## Key takeaways

- No single number summarizes an LLM; pick the metric for the job.
- Hallucination comes from how LLMs are trained, so it must be measured, not assumed away.
- Grounding, abstention, and verification reduce hallucination; none eliminates it.
- Treat benchmarks as evidence, not proof.

## Further reading

Biderman, Stella, et al. "Lessons from the Trenches on Reproducible Evaluation of Language Models." arXiv preprint arXiv:2405.14782, 2024. https://arxiv.org/abs/2405.14782.

Bouthillier, Xavier, et al. "Accounting for Variance in Machine Learning Benchmarks." arXiv preprint arXiv:2103.03098, 2021. https://arxiv.org/abs/2103.03098.

Brown, Tom B., et al. "Language Models Are Few-Shot Learners." In *Advances in Neural Information Processing Systems 33*, 2020. https://arxiv.org/abs/2005.14165.

Card, Dallas, et al. "With Little Power Comes Great Responsibility." arXiv preprint arXiv:2010.06595, 2020. https://arxiv.org/abs/2010.06595.

Chao, Patrick, et al. "JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models." arXiv preprint arXiv:2404.01318, 2024. https://arxiv.org/abs/2404.01318.

Chen, Mark, et al. "Evaluating Large Language Models Trained on Code." arXiv preprint arXiv:2107.03374, 2021. https://arxiv.org/abs/2107.03374.

Chiang, Wei-Lin, et al. "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference." In *Proceedings of the 41st International Conference on Machine Learning*, 2024. https://arxiv.org/abs/2403.04132.

Clark, Elizabeth, et al. "All That's 'Human' Is Not Gold: Evaluating Human Evaluation of Generated Text." In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing*, 2021. https://arxiv.org/abs/2107.00061.

Cohen, Jacob. "A Coefficient of Agreement for Nominal Scales." *Educational and Psychological Measurement* 20, no. 1 (1960): 37–46.

Dodge, Jesse, et al. "Show Your Work: Improved Reporting of Experimental Results." arXiv preprint arXiv:1909.03004, 2019. https://arxiv.org/abs/1909.03004.

Dubois, Yann, et al. "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators." arXiv preprint arXiv:2404.04475, 2024. https://arxiv.org/abs/2404.04475.

Ganguli, Deep, et al. "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned." arXiv preprint arXiv:2209.07858, 2022. https://arxiv.org/abs/2209.07858.

Gao, Leo, et al. "Scaling Laws for Reward Model Overoptimization." arXiv preprint arXiv:2210.10760, 2022. https://arxiv.org/abs/2210.10760.

Gehman, Samuel, et al. "RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models." In *Findings of the Association for Computational Linguistics: EMNLP 2020*, 2020. https://arxiv.org/abs/2009.11462.

Gekhman, Zorik, et al. "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?" arXiv preprint arXiv:2405.05904, 2024. https://arxiv.org/abs/2405.05904.

Golchin, Shahriar, et al. "Time Travel in LLMs: Tracing Data Contamination in Large Language Models." arXiv preprint arXiv:2308.08493, 2023. https://arxiv.org/abs/2308.08493.

Gu, Jiawei, et al. "A Survey on LLM-as-a-Judge." arXiv preprint arXiv:2411.15594, 2024. https://arxiv.org/abs/2411.15594.

Hendrycks, Dan, et al. "Measuring Massive Multitask Language Understanding." In *International Conference on Learning Representations*, 2021. https://arxiv.org/abs/2009.03300.

Hosking, Tom, et al. "Human Feedback Is Not Gold Standard." arXiv preprint arXiv:2309.16349, 2023. https://arxiv.org/abs/2309.16349.

Huang, Lei, et al. "A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions." arXiv preprint arXiv:2311.05232, 2023. https://arxiv.org/abs/2311.05232.

Jain, Naman, et al. "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code." arXiv preprint arXiv:2403.07974, 2024. https://arxiv.org/abs/2403.07974.

Jimenez, Carlos E., et al. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" In *International Conference on Learning Representations*, 2024. https://arxiv.org/abs/2310.06770.

Kalai, Adam Tauman, et al. "Why Language Models Hallucinate." arXiv preprint arXiv:2509.04664, 2025. https://arxiv.org/abs/2509.04664.

Kaplan, Jared, et al. "Scaling Laws for Neural Language Models." arXiv preprint arXiv:2001.08361, 2020. https://arxiv.org/abs/2001.08361.

Karpinska, Marzena, et al. "The Perils of Using Mechanical Turk to Evaluate Open-Ended Text Generation." arXiv preprint arXiv:2109.06835, 2021. https://arxiv.org/abs/2109.06835.

Kim, Seungone, et al. "Prometheus: Inducing Fine-Grained Evaluation Capability in Language Models." arXiv preprint arXiv:2310.08491, 2023. https://arxiv.org/abs/2310.08491.

Landis, J. Richard, et al. "The Measurement of Observer Agreement for Categorical Data." *Biometrics* 33, no. 1 (1977): 159–174.

Lewis, Patrick, et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." In *Advances in Neural Information Processing Systems 33*, 2020. https://arxiv.org/abs/2005.11401.

Liang, Percy, et al. "Holistic Evaluation of Language Models." arXiv preprint arXiv:2211.09110, 2022. https://arxiv.org/abs/2211.09110.

Lin, Stephanie, et al. "TruthfulQA: Measuring How Models Mimic Human Falsehoods." In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics*, 2022. https://arxiv.org/abs/2109.07958.

Liu, Yang, et al. "G-Eval: NLG Evaluation Using GPT-4 with Better Human Alignment." arXiv preprint arXiv:2303.16634, 2023. https://arxiv.org/abs/2303.16634.

Madaan, Lovish, et al. "Quantifying Variance in Evaluation Benchmarks." arXiv preprint arXiv:2406.10229, 2024. https://arxiv.org/abs/2406.10229.

Magnusson, Ian, et al. "Paloma: A Benchmark for Evaluating Language Model Fit." arXiv preprint arXiv:2312.10523, 2023. https://arxiv.org/abs/2312.10523.

Manakul, Potsawee, et al. "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models." In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 2023. https://arxiv.org/abs/2303.08896.

Manheim, David, et al. "Categorizing Variants of Goodhart's Law." arXiv preprint arXiv:1803.04585, 2018. https://arxiv.org/abs/1803.04585.

Mazeika, Mantas, et al. "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal." arXiv preprint arXiv:2402.04249, 2024. https://arxiv.org/abs/2402.04249.

Miller, Evan. "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations." arXiv preprint arXiv:2411.00640, 2024. https://arxiv.org/abs/2411.00640.

Min, Sewon, et al. "FActScore: Fine-Grained Atomic Evaluation of Factual Precision in Long Form Text Generation." In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 2023. https://arxiv.org/abs/2305.14251.

Mitchell, Margaret, et al. "Model Cards for Model Reporting." In *Proceedings of the Conference on Fairness, Accountability, and Transparency*, 2019. https://arxiv.org/abs/1810.03993.

OpenAI. "GPT-4 Technical Report." arXiv preprint arXiv:2303.08774, 2023. https://arxiv.org/abs/2303.08774.

OpenAI. "Why SWE-bench Verified No Longer Measures Frontier Coding Capabilities." February 23, 2026. https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/.

Oren, Yonatan, et al. "Proving Test Set Contamination in Black Box Language Models." arXiv preprint arXiv:2310.17623, 2023. https://arxiv.org/abs/2310.17623.

Panickssery, Arjun, et al. "LLM Evaluators Recognize and Favor Their Own Generations." arXiv preprint arXiv:2404.13076, 2024. https://arxiv.org/abs/2404.13076.

Phan, Long, et al. "Humanity's Last Exam." arXiv preprint arXiv:2501.14249, 2025. https://arxiv.org/abs/2501.14249.

Raji, Inioluwa Deborah, et al. "AI and the Everything in the Whole Wide World Benchmark." arXiv preprint arXiv:2111.15366, 2021. https://arxiv.org/abs/2111.15366.

Röttger, Paul, et al. "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models." arXiv preprint arXiv:2308.01263, 2023. https://arxiv.org/abs/2308.01263.

Sainz, Oscar, et al. "NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for Each Benchmark." arXiv preprint arXiv:2310.18018, 2023. https://arxiv.org/abs/2310.18018.

Sclar, Melanie, et al. "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I Learned to Start Worrying about Prompt Formatting." arXiv preprint arXiv:2310.11324, 2023. https://arxiv.org/abs/2310.11324.

Shevlane, Toby, et al. "Model Evaluation for Extreme Risks." arXiv preprint arXiv:2305.15324, 2023. https://arxiv.org/abs/2305.15324.

Shi, Weijia, et al. "Detecting Pretraining Data from Large Language Models." arXiv preprint arXiv:2310.16789, 2023. https://arxiv.org/abs/2310.16789.

Singh, Shivalika, et al. "The Leaderboard Illusion." arXiv preprint arXiv:2504.20879, 2025. https://arxiv.org/abs/2504.20879.

Souly, Alexandra, et al. "A StrongREJECT for Empty Jailbreaks." arXiv preprint arXiv:2402.10260, 2024. https://arxiv.org/abs/2402.10260.

Stiennon, Nisan, et al. "Learning to Summarize from Human Feedback." In *Advances in Neural Information Processing Systems 33*, 2020. https://arxiv.org/abs/2009.01325.

Thakur, Aman Singh, et al. "Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges." arXiv preprint arXiv:2406.12624, 2024. https://arxiv.org/abs/2406.12624.

Veselovsky, Veniamin, et al. "Artificial Artificial Artificial Intelligence: Crowd Workers Widely Use Large Language Models for Text Production Tasks." arXiv preprint arXiv:2306.07899, 2023. https://arxiv.org/abs/2306.07899.

Wang, Peiyi, et al. "Large Language Models Are Not Fair Evaluators." arXiv preprint arXiv:2305.17926, 2023. https://arxiv.org/abs/2305.17926.

Wang, Yubo, et al. "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark." arXiv preprint arXiv:2406.01574, 2024. https://arxiv.org/abs/2406.01574.

Wei, Alexander, et al. "Jailbroken: How Does LLM Safety Training Fail?" arXiv preprint arXiv:2307.02483, 2023. https://arxiv.org/abs/2307.02483.

Wei, Jason, et al. "Measuring Short-Form Factuality in Large Language Models." arXiv preprint arXiv:2411.04368, 2024. https://arxiv.org/abs/2411.04368.

Zhang, Hugh, et al. "A Careful Examination of Large Language Model Performance on Grade School Arithmetic." arXiv preprint arXiv:2405.00332, 2024. https://arxiv.org/abs/2405.00332.

Zheng, Lianmin, et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." In *Advances in Neural Information Processing Systems 36*, 2023. https://arxiv.org/abs/2306.05685.
