This analysis compares performance across two sample types:
   * Medium: Prompt and context with more content than the standard short prompts used for benchmarking against user study. 
   * Long: Enriched with extensive (and potentially distracting) context information.


  ┌───────────────────┬─────────────┬─────────────┬───────────────┬─────────────────┐
  │ Model             │ Sample Type │ Accuracy    │ Avg. Time (s) │ Avg. Confidence │
  ├───────────────────┼─────────────┼─────────────┼───────────────┼─────────────────┤
  │ Claude Haiku      │ Medium      │ 55.6% (5/9) │ 7.58s         │ 0.52            │
  │                   │ Long        │ 33.3% (3/9) │ 7.16s         │ 0.40            │
  │ Claude Sonnet     │ Medium      │ 55.6% (5/9) │ 11.36s        │ 0.50            │
  │                   │ Long        │ 22.2% (2/9) │ 12.72s        │ 0.37            │
  │ DeepSeek Reasoner │ Medium      │ 55.6% (5/9) │ 221.96s       │ 0.43            │
  │                   │ Long        │ 44.4% (4/9) │ 450.65s       │ 0.20            │
  └───────────────────┴─────────────┴─────────────┴───────────────┴─────────────────┘


   * The short prompt variation has a comprehensive analysis based on 1,620 total task evaluations.
   * While the sample runs contains only a small, recent subset of these same evaluations (9 tasks per model).

  A small sample size is not statistically representative of the full dataset. Therefore, while the sample runs for the provide a snapshot of recent performance, they do not have the same statistical power that is reflected in the overall performance documented by the short prompt variation's results.  But they are indicative of the performance degradation expected with the increase in signal in the words context space that cause LLMs to hallucinate those signals with noise in the data. 