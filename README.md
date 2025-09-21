# reranker-test

Cross-encoder document reranking with Transformers.js. This repo provides a small, production-ready API for semantic reranking and a comprehensive test/benchmark suite to validate quality and performance across models.

- Library: @xenova/transformers (ONNX/WASM, CPU-friendly)
- Default model: mixedbread-ai/mxbai-rerank-xsmall-v1 (confident, smaller batches)
- Alternative: Xenova/ms-marco-MiniLM-L-6-v2 (fast, general purpose)


## What’s inside

- `reranker.js` — Production API with batching, multi-model cache, logging
- `test-suite.js` — End-to-end tests, metrics (NDCG, MRR, Precision@K, Recall@K), and benchmark scenarios
- `models/` — Optional local cache of models for offline/fast startup
  - `mixedbread-ai/mxbai-rerank-xsmall-v1/`
  - `Xenova/ms-marco-MiniLM-L-6-v2/`


## Requirements

- Node.js 18+ (ES modules enabled via `"type": "module"`)
- Internet access on first run unless models are pre-cached in `./models`
- Windows, macOS, or Linux. No GPU required (runs via ONNX/WASM).


## Install

In the project root:

```powershell
npm install
```

This installs `@xenova/transformers` (see `package.json`).


## Quick start (programmatic API)

The API centers on `NativeEmbeddingReranker`, which loads a cross-encoder model and ranks documents against a query.

```js
import { NativeEmbeddingReranker } from './reranker.js';

const reranker = new NativeEmbeddingReranker({
  // Optional overrides shown with defaults
  model: 'mixedbread-ai/mxbai-rerank-xsmall-v1',
  cacheDir: null,         // null → uses "./models" under the current working directory
  logLevel: 'info',       // 'debug' | 'info' | 'warn' | 'error' | 'silent'
});

await reranker.initialize();

const query = 'How does AI work?';
const documents = [
  'Machine learning uses data to train models.',
  'Gardening tips for spring flowers.',
  'Neural networks learn hierarchical features.',
];

const results = await reranker.rerank(query, documents, { topK: 2, batchSize: 128 });
console.log(results);
// [
//   { _rerank_corpus_id, _rerank_score, text, ...originalProps },
//   ...
// ]

reranker.dispose(); // optional: free model from memory in long-running apps
```

Input formats for `documents`:
- Array of strings: `["doc text", ...]`
- Array of objects: `[{ text: "doc text", ...customFields }]`
- Mixed arrays are supported. Any extra fields are preserved in the output.

Output fields added by the reranker:
- `_rerank_corpus_id`: original index in the input array
- `_rerank_score`: relevance score (higher is more relevant)


## Switching models

```js
// Fast, general-purpose
const fast = new NativeEmbeddingReranker({ model: 'Xenova/ms-marco-MiniLM-L-6-v2' });
await fast.initialize();

// Higher confidence, often slower
const precise = new NativeEmbeddingReranker({ model: 'mixedbread-ai/mxbai-rerank-xsmall-v1' });
await precise.initialize();
```

Under the hood, the reranker:
- Uses sigmoid for single-logit models (e.g., many MS-MARCO-style heads)
- Uses softmax and the positive class probability for multi-logit models (auto-detected using `config.id2label`)


## Caching and offline use

- By default, model files are cached under `./models` (relative to the process working directory).
- If the model isn’t found locally, `@xenova/transformers` will download it on first use.
- To run fully offline, place the model files in the expected folder structure:

```
models/
  mixedbread-ai/
    mxbai-rerank-xsmall-v1/
      config.json
      tokenizer.json
      tokenizer_config.json
      onnx/
        model_quantized.onnx
  Xenova/
    ms-marco-MiniLM-L-6-v2/
      config.json
      tokenizer.json
      tokenizer_config.json
      onnx/
        model_quantized.onnx
```

Tips:
- You can change the cache location via the `cacheDir` constructor option.
- Multiple `NativeEmbeddingReranker` instances that use the same `model` share one in-memory cache (via `ModelLoader`).


## Extended API

```js
import { NativeEmbeddingReranker, ModelLoader } from './reranker.js';

const reranker = new NativeEmbeddingReranker({ logLevel: 'debug' });
await reranker.preload();        // alias of initialize() with service-style logging
const info = reranker.getModelInfo();
console.log(info);
// { currentModel, isLoaded, cacheDirectory, totalCachedModels, cachedModels, logLevel }

reranker.dispose();              // unloads the current model from memory

// Static cache utilities
ModelLoader.getCachedModels();   // [ 'mixedbread-ai/mxbai-rerank-xsmall-v1', ... ]
ModelLoader.clearCache();        // clear all cached models
ModelLoader.clearCache('Xenova/ms-marco-MiniLM-L-6-v2'); // clear one
```


## Benchmark and test suite

`test-suite.js` runs scenario-based benchmarks with accuracy metrics. Use Node directly:

```powershell
# Standard suite (4 scenarios)
node .\test-suite.js

# Quick validation (1 scenario)
node .\test-suite.js --quick

# Full comprehensive suite (8 scenarios)
node .\test-suite.js --full
```

What you’ll see per scenario:
- Candidate distribution (high/medium/low relevance)
- Throughput and total time
- NDCG@k, MRR@k, Precision@k, Recall@k
- Top-K results with scores
- Summary across all scenarios (averages and throughput)

Note: `package.json` currently maps some scripts to `reranker.js`; use the commands above to run the test suite directly. If you prefer npm scripts, you can add:

```json
{
  "scripts": {
    "bench": "node test-suite.js",
    "bench:quick": "node test-suite.js --quick",
    "bench:full": "node test-suite.js --full"
  }
}
```


## Performance notes

- Batch processing is enabled; tune `batchSize` to balance memory vs. speed.
- Mixedbread model tends to produce higher-confidence scores but may process fewer docs/sec than MS-MARCO.
- First run of a model includes download and warmup; subsequent runs are faster due to cache.


## Troubleshooting

- Model download is slow or blocked
  - Pre-download the model files and place them under `./models` as shown above.
  - Or set a custom `cacheDir` to a location you control.

- ESM import errors (e.g., "Cannot use import statement outside a module")
  - Ensure Node 18+ and that `package.json` contains `{ "type": "module" }`.

- No results or empty output
  - `rerank()` returns `[]` when `documents` is empty; verify your inputs.

- Memory usage is high in long-running apps
  - Call `reranker.dispose()` when switching models or shutting down to free memory.

- Windows path tips
  - Use PowerShell-friendly paths like `node .\test-suite.js` when running commands from the project root.


## License

ISC (see `package.json`).


## Acknowledgements

- Transformers.js by Xenova
- Mixedbread AI and MS MARCO reranking models