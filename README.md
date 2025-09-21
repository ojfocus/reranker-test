# Transformers.js Multi-Model Reranker Benchmark

A comprehensive, single-file benchmarking suite for cross-encoder reranking models using Transformers.js in Node.js. Now supports multiple models including MS-MARCO and MixedBread AI models.

## Features

- **Multi-Model Support**: Switch between different reranking models seamlessly
- **Single Test Runner**: One comprehensive test file handles everything
- **Multiple Test Modes**: Quick validation, standard suite, or full benchmark  
- **Model Loading**: Automatic loading and caching of cross-encoder models
- **Concurrent Model Support**: Load and use multiple models simultaneously
- **Reranking**: Process query-passage pairs and rank passages by relevance
- **Comprehensive Metrics**: Calculate precision, improvement over random, throughput
- **Performance Timing**: Measure latency and processing speed
- **Detailed Reporting**: Generate comprehensive reports with visual results
- **Centralized Test Data**: Reusable, categorized document pools for consistent testing

## Supported Models

- **Xenova/ms-marco-MiniLM-L-6-v2** (default): Well-established MS-MARCO trained model
- **mixedbread-ai/mxbai-rerank-xsmall-v1**: High-quality compact reranking model
- Any compatible cross-encoder model available in Transformers.js

## Installation

1. Clone or set up the project:
   ```bash
   npm install
   ```

2. The project uses ES modules, so make sure `"type": "module"` is in your `package.json`.

## Project Structure

```
reranker-test/
├── reranker.js          # Complete multi-model reranker implementation
├── example-usage.js     # Multi-model usage examples
├── models/              # Model cache directory (created automatically)  
├── package.json
└── README.md
```

## Usage

### Quick Start

1. **Quick validation test** (~30 seconds):
   ```bash
## Usage

### Running Tests

1. **Quick validation test** (~30 seconds):
   ```bash
   node reranker.js --quick
   ```

2. **Standard test suite** (~2-3 minutes):
   ```bash
   node reranker.js
   ```

3. **Full comprehensive benchmark** (~5-10 minutes):
   ```bash
   node reranker.js --full
   ```

### Using as a Module

#### Default Model (MS-MARCO)
```javascript
import { NativeEmbeddingReranker } from './reranker.js';

const reranker = new NativeEmbeddingReranker();
await reranker.initialize();

const query = "How does machine learning work?";
const documents = [
  "Machine learning trains algorithms on data to recognize patterns.",
  "Photosynthesis is how plants create energy from sunlight.",
  "Cloud platforms provide scalable AI infrastructure."
];

const results = await reranker.rerank(query, documents, { topK: 3 });
console.log(results);
```

#### Specifying a Model
```javascript
// Using MixedBread AI model
const reranker = new NativeEmbeddingReranker({
  model: 'mixedbread-ai/mxbai-rerank-xsmall-v1'
});
await reranker.initialize();

// Using MS-MARCO model explicitly
const msMarcoReranker = new NativeEmbeddingReranker({
  model: 'Xenova/ms-marco-MiniLM-L-6-v2'
});
await msMarcoReranker.initialize();
```

#### Multiple Models Simultaneously
```javascript
// Load different models for different use cases
const generalReranker = new NativeEmbeddingReranker({
  model: 'Xenova/ms-marco-MiniLM-L-6-v2'
});

const compactReranker = new NativeEmbeddingReranker({
  model: 'mixedbread-ai/mxbai-rerank-xsmall-v1'
});

await Promise.all([
  generalReranker.initialize(),
  compactReranker.initialize()
]);

// Use both models independently
const generalResults = await generalReranker.rerank(query, documents);
const compactResults = await compactReranker.rerank(query, documents);
```

#### Custom Cache Directory
```javascript
const reranker = new NativeEmbeddingReranker({
  model: 'mixedbread-ai/mxbai-rerank-xsmall-v1',
  cacheDir: './custom-model-cache'
});
```

### Running the Example
```bash
node example-usage.js
```

## Test Data Module

The project includes a centralized test data module for consistent testing:

### Available Test Queries

- `ai_ml_fundamentals` - "How does machine learning and artificial intelligence work?"
- `photosynthesis` - "What is photosynthesis and how do plants create energy?"
- `software_development` - "How does artificial intelligence work in software development?"
- `climate_change` - "Climate change effects on global weather patterns"

### Document Categories

- `core_ai_ml` - High-quality AI/ML concept explanations
- `tech_applications` - AI applications in technology
- `data_science_business` - Data science and business analytics
- `hard_science` - Biology, chemistry, physics topics
- `humanities` - History, literature, arts
- `health_lifestyle` - Health, fitness, wellness topics

### Usage Example

```javascript
import { TEST_QUERIES, generateTestCandidates } from './reranker.js';

// Generate 20 test candidates for AI/ML query
const candidates = generateTestCandidates('ai_ml_fundamentals', 20, {
  highRelevanceRatio: 0.4,   // 40% high relevance
  mediumRelevanceRatio: 0.3, // 30% medium relevance  
  lowRelevanceRatio: 0.3     // 30% low relevance
});
```
- `npm run quick-test` - Run benchmark with small dataset (2 queries)
- `npm start` - Run full benchmark with complete dataset (5 queries)
- `npm run benchmark` - Alias for `npm start`

### Custom Usage

You can also import and use the benchmark programmatically:

```javascript
import { Benchmark } from './src/benchmark.js';

const benchmark = new Benchmark();
const results = await benchmark.runBenchmark();
```

Or use individual components:

```javascript
import { ModelLoader } from './src/model.js';
import { Reranker } from './src/reranker.js';

// Initialize model
const modelLoader = new ModelLoader();
await modelLoader.initialize();

// Rank passages
const reranker = new Reranker(modelLoader);
const query = "How many people live in Berlin?";
const passages = [
  "Berlin has a population of 3,520,031 registered inhabitants.",
  "New York City is famous for the Metropolitan Museum of Art."
];

const result = await reranker.rankPassages(query, passages);
console.log('Ranked passages:', result.rankedPassages);
```

## Dataset Format

The dataset should be a JSON array with the following structure:

```json
[
  {
    "query": "What is the capital of Germany?",
    "positive_passages": [
      "Berlin is the capital of Germany...",
      "Germany's capital city is Berlin..."
    ],
    "negative_passages": [
      "Paris is the capital of France...",
      "London is the capital city of England..."
    ]
  }
]
```

## Evaluation Metrics

The benchmark calculates several metrics at different cutoff values (k=1,3,5,10):

- **MRR@k**: Mean Reciprocal Rank - measures the rank of the first relevant document
- **NDCG@k**: Normalized Discounted Cumulative Gain - measures ranking quality with position discounting
- **Precision@k**: Fraction of retrieved documents that are relevant
- **Recall@k**: Fraction of relevant documents that are retrieved

## Output

The benchmark generates:

1. **Console Output**: Real-time progress and final results
2. **JSON Results** (`results/benchmark_results.json`): Detailed results in JSON format
3. **Text Report** (`results/benchmark_report.txt`): Human-readable report

## Sample Output

```
=== EVALUATION METRICS ===

MRR:
  MRR@1: 85.00%
  MRR@3: 85.00%
  MRR@5: 85.00%
  MRR@10: 85.00%

NDCG:
  NDCG@1: 85.00%
  NDCG@3: 92.14%
  NDCG@5: 94.23%
  NDCG@10: 95.67%

=== TIMING REPORT ===
Average Latency per Query: 1234.56ms
Queries Processed: 5
Average Throughput: 0.81 queries/second
```

## Model Information

- **Model**: Xenova/ms-marco-MiniLM-L-6-v2
- **Base Model**: microsoft/MiniLM-L12-H384-uncased
- **Original**: cross-encoder/ms-marco-MiniLM-L6-v2 (converted to ONNX for web compatibility)
- **Type**: Cross-encoder for passage reranking
- **Framework**: Transformers.js (@xenova/transformers)
- **Task**: Text classification for relevance scoring
- **Format**: ONNX weights optimized for JavaScript runtime
- **Input**: Query-passage pairs as [query, passage] arrays
- **Output**: Relevance scores (higher = more relevant)

### Model Details
This model is specifically optimized for MS MARCO passage reranking. It takes a query and a passage as input and outputs a relevance score. The model was trained on the MS MARCO dataset to distinguish between relevant and non-relevant query-passage pairs.

**Important Implementation Notes:**
- Uses the lower-level `AutoModelForSequenceClassification` API to access raw logits
- Returns continuous relevance scores (not just binary classification)  
- Positive scores indicate relevance, negative scores indicate irrelevance
- Higher absolute values indicate stronger confidence in the prediction
- Batch processing is supported for efficient inference

## Performance Considerations

- The model downloads and caches automatically on first run (~45MB ONNX model)
- Subsequent runs will be faster as the model is cached locally
- Processing time depends on the number and length of passages
- Batch processing is used for efficiency
- Expected output: Higher scores indicate higher relevance (typically ranging from negative to positive values)
- Model handles input sequences up to 512 tokens (query + passage combined)

### Expected Performance
Based on the model's design:
- **Accuracy**: Optimized for MS MARCO passage ranking task
- **Speed**: Lightweight 6-layer architecture for faster inference
- **Input Limits**: Query + passage should not exceed 512 tokens combined
- **Score Range**: Relevance scores can be positive (relevant) or negative (not relevant)
- **Score Interpretation**: Higher positive scores = more relevant, lower negative scores = less relevant
- **Typical Range**: Scores often range from -15 to +15, but can vary based on query complexity

## Customization

### Custom Dataset

Replace `data/dataset.json` with your own dataset following the required format.

### Custom Metrics

Modify the `kValues` parameter in the benchmark to evaluate different cutoff points:

```javascript
const results = await benchmark.runBenchmark([1, 5, 10, 20]);
```

### Adding New Metrics

Extend the `EvaluationMetrics` class in `src/metrics.js` to add new evaluation metrics.

## Dependencies

- `@xenova/transformers`: For running the cross-encoder model
- Node.js built-in modules: `fs`, `path`, `url`

## License

ISC