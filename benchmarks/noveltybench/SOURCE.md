Source: NoveltyBench benchmark assets.

Upstream commit:
- `984876e92a5eefe6945f273b8c8543a4e6cff070`

Copied files:
- `data/curated.jsonl` -> `benchmarks/noveltybench/curated.jsonl`
- `data/wildchat-1k.jsonl` -> `benchmarks/noveltybench/wildchat_1k.jsonl`

Notes:
- These assets are vendored so the evaluation does not depend on an external checkout at runtime.
- The runtime implementation preserves upstream split names and metric semantics, but it does not preserve the upstream off-by-one `distinct` bug.
