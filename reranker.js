import { AutoTokenizer, AutoModelForSequenceClassification } from '@xenova/transformers';
import path from 'path';
import fs from 'fs';

/**
 * ====================================================================
 * RERANKER MODULE - Production reranking API
 * ====================================================================
 * 
 * This module provides a clean, production-ready document reranking API
 * using cross-encoder models with Transformers.js.
 * 
 * Core functionality:
 * - Cross-encoder model loading and inference
 * - Multi-model support with caching
 * - Semantic document reranking
 * 
 * Usage:
 *   import { NativeEmbeddingReranker } from './reranker.js';
 *   const reranker = new NativeEmbeddingReranker();
 *   await reranker.initialize();
 *   const results = await reranker.rerank(query, documents);
 */

// ====================================================================
// MODEL LOADER CLASS
// ====================================================================

/**
 * Model loader class for the cross-encoder reranking model
 */
class ModelLoader {
  // Use Maps to store multiple models and tokenizers, keyed by model name
  static #models = new Map();
  static #tokenizers = new Map();
  static #initializationPromises = new Map();

  constructor({ modelName, cacheDir = null }) {
    if (!modelName) throw new Error("A model name must be provided.");
    this.modelName = modelName;
    
    // Define the cache directory for models, with a fallback
    this.cacheDir = cacheDir
      ? path.resolve(cacheDir)
      : path.resolve(process.cwd(), 'models'); // Default to a local 'models' folder

    // Ensure the cache directory exists
    if (!fs.existsSync(this.cacheDir)) {
      fs.mkdirSync(this.cacheDir, { recursive: true });
    }

    // More robust check for model existence - look for actual model files
    // Transformers.js creates nested directories like: models/mixedbread-ai/mxbai-rerank-xsmall-v1/
    const modelPath = path.resolve(this.cacheDir, this.modelName);
    this.modelDownloaded = this.checkModelExists(modelPath);
    console.log(`[ModelLoader] Initialized for model "${this.modelName}" with cache: ${this.cacheDir}`);
  }

  /**
   * Check if model files exist in the cache directory
   * @param {string} modelPath - Path to the model directory
   * @returns {boolean} - True if model appears to be downloaded
   */
  checkModelExists(modelPath) {
    // Check for key files that indicate a complete model download
    const requiredFiles = ['config.json', 'tokenizer.json'];
    const optionalFiles = ['tokenizer_config.json'];
    
    // First check if the model directory exists
    if (!fs.existsSync(modelPath)) {
      return false;
    }

    // Check for required files
    const hasRequiredFiles = requiredFiles.every(file => 
      fs.existsSync(path.join(modelPath, file))
    );

    if (!hasRequiredFiles) {
      return false;
    }

    // Check for model files (could be .bin, .safetensors, or in onnx/ subdirectory)
    const modelFiles = fs.readdirSync(modelPath);
    const hasModelWeights = modelFiles.some(file => 
      file.endsWith('.bin') || 
      file.endsWith('.safetensors') ||
      file === 'onnx' // ONNX models are in a subdirectory
    );

    // If there's an onnx directory, check for model files inside it
    if (!hasModelWeights && modelFiles.includes('onnx')) {
      const onnxPath = path.join(modelPath, 'onnx');
      try {
        const onnxFiles = fs.readdirSync(onnxPath);
        return onnxFiles.some(file => file.endsWith('.onnx'));
      } catch (error) {
        return false;
      }
    }

    return hasModelWeights;
  }

  /**
   * Initialize the cross-encoder model and tokenizer with caching
   * @returns {Promise<void>}
   */
  async initialize() {
    // Check if this specific model is already loaded
    if (ModelLoader.#models.has(this.modelName)) {
      console.log(`[ModelLoader] Model "${this.modelName}" already initialized.`);
      return;
    }

    // Check if this specific model is currently being initialized
    if (ModelLoader.#initializationPromises.has(this.modelName)) {
      console.log(`[ModelLoader] Waiting for "${this.modelName}" initialization to complete...`);
      await ModelLoader.#initializationPromises.get(this.modelName);
      return;
    }

    const promise = (async () => {
      try {
        console.log(`[ModelLoader] Loading model: ${this.modelName}...`);
        const start = performance.now();

        // Load the model and tokenizer
        const model = await AutoModelForSequenceClassification.from_pretrained(this.modelName, {
          cache_dir: this.cacheDir,
        });
        
        const tokenizer = await AutoTokenizer.from_pretrained(this.modelName, {
          cache_dir: this.cacheDir,
        });

        // Store the model and tokenizer
        ModelLoader.#models.set(this.modelName, model);
        ModelLoader.#tokenizers.set(this.modelName, tokenizer);

        const duration = performance.now() - start;
        console.log(`[ModelLoader] Model "${this.modelName}" loaded in ${duration.toFixed(2)}ms`);
        console.log(`Model cached at: ${this.cacheDir}`);

      } catch (error) {
        console.error(`[ModelLoader] Failed to load model "${this.modelName}":`, error);
        throw error;
      } finally {
        // Clean up the initialization promise
        ModelLoader.#initializationPromises.delete(this.modelName);
      }
    })();

    // Store the promise so other concurrent calls can wait
    ModelLoader.#initializationPromises.set(this.modelName, promise);
    await promise;
  }

  /**
   * Get relevance scores for query-passage pairs using tensor operations
   * @param {Array<Array<string>>} queryPassagePairs - Array of [query, passage] pairs
   * @returns {Promise<Array<{score: number, label: string}>>}
   */
  async getScores(queryPassagePairs) {
    // Retrieve the correct model and tokenizer from the maps
    const model = ModelLoader.#models.get(this.modelName);
    const tokenizer = ModelLoader.#tokenizers.get(this.modelName);

    if (!model || !tokenizer) {
      throw new Error(`Model "${this.modelName}" not initialized. Call initialize() first.`);
    }

    if (!Array.isArray(queryPassagePairs) || queryPassagePairs.length === 0) {
      return [];
    }

    try {
      // Extract queries and passages
      const queries = queryPassagePairs.map(pair => pair[0]);
      const passages = queryPassagePairs.map(pair => pair[1]);

      // Tokenize using the text_pair format
      const features = tokenizer(queries, {
        text_pair: passages,
        padding: true,
        truncation: true,
      });

      // Run model inference to get raw logits
      const output = await model(features);
      
      // Use tensor operations for efficiency
      const sigmoidScores = output.logits.sigmoid().tolist();
      
      // Convert to expected format
      const results = sigmoidScores.map(([score]) => ({
        score: score,
        label: score > 0.5 ? 'RELEVANT' : 'NOT_RELEVANT'
      }));
      
      return results;
      
    } catch (error) {
      console.error('Error getting scores:', error);
      throw error;
    }
  }

  /**
   * Get model and tokenizer for direct access
   * @returns {{model: any, tokenizer: any}}
   */
  getModelAndTokenizer() {
    return {
      model: ModelLoader.#models.get(this.modelName),
      tokenizer: ModelLoader.#tokenizers.get(this.modelName)
    };
  }

  /**
   * Check if model is loaded
   * @returns {boolean}
   */
  isLoaded() {
    return ModelLoader.#models.has(this.modelName) && ModelLoader.#tokenizers.has(this.modelName);
  }

  /**
   * Get model name
   * @returns {string}
   */
  getModelName() {
    return this.modelName;
  }
}

// ====================================================================
// MAIN RERANKER CLASS
// ====================================================================

/**
 * Cross-encoder document reranker using Transformers.js
 * 
 * OVERVIEW:
 * This class provides semantic document reranking using cross-encoder models.
 * It takes a query and a list of documents, then ranks them by semantic relevance.
 * 
 * SUPPORTED MODELS:
 * - 'mixedbread-ai/mxbai-rerank-xsmall-v1' (default): Higher confidence scores, slower
 * - 'Xenova/ms-marco-MiniLM-L-6-v2': Fast, general-purpose reranking
 * - Model switching requires only constructor parameter change
 * 
 * TYPICAL WORKFLOW:
 * 1. Constructor: new NativeEmbeddingReranker({ model: 'model-name' })
 * 2. Initialize: await reranker.initialize()
 * 3. Rerank: await reranker.rerank(query, documents, { topK: N })
 * 
 * INTEGRATION PATTERNS:
 * - Vector Search + Reranking: Use after initial vector/BM25 retrieval
 * - Multi-stage Ranking: Combine multiple reranking models
 * - A/B Testing: Switch models without changing application logic
 * 
 * PERFORMANCE CHARACTERISTICS:
 * - Batch processing optimized for 10-1000+ documents
 * - Model loading cached across instances (same model name)
 * - First initialization downloads model files locally
 * - Subsequent runs use cached models for faster startup
 */
class NativeEmbeddingReranker {
  /**
   * Creates a new document reranker instance
   * 
   * @param {Object} [options={}] - Configuration options
   * @param {string} [options.model='mixedbread-ai/mxbai-rerank-xsmall-v1'] - Model identifier to use:
   *   - 'mixedbread-ai/mxbai-rerank-xsmall-v1': Higher confidence, slower (default)
   *   - 'Xenova/ms-marco-MiniLM-L-6-v2': Fast, general-purpose
   * @param {string|null} [options.cacheDir=null] - Custom model cache directory
   *   - null: Uses './models' in current working directory
   *   - string: Custom path for model storage
   * 
   * @example
   * // Default MixedBread model
   * const reranker = new NativeEmbeddingReranker();
   * 
   * // Specific model selection
   * const reranker = new NativeEmbeddingReranker({
   *   model: 'Xenova/ms-marco-MiniLM-L-6-v2'
   * });
   * 
   * // Custom cache directory
   * const reranker = new NativeEmbeddingReranker({
   *   model: 'mixedbread-ai/mxbai-rerank-xsmall-v1',
   *   cacheDir: '/custom/path/to/models'
   * });
   */
  constructor(options = {}) {
    // Destructure modelName and cacheDir from options
    const { 
      cacheDir = null, 
      model = 'mixedbread-ai/mxbai-rerank-xsmall-v1' // Default model
    } = options;

    this.model = model; // Store the model name
    this.cacheDir = cacheDir;

    // Pass the model name to the ModelLoader
    this.modelLoader = new ModelLoader({ 
      modelName: this.model, 
      cacheDir: this.cacheDir 
    });
  }

  /**
   * Initialize the reranker model (must be called before rerank())
   * 
   * Downloads model files on first run and loads them into memory.
   * Subsequent calls with the same model name use cached instances.
   * 
   * @returns {Promise<void>}
   * @throws {Error} If model loading fails
   * 
   * @example
   * const reranker = new NativeEmbeddingReranker();
   * await reranker.initialize(); // Required before rerank()
   */
  async initialize() {
    console.log('Initializing NativeEmbeddingReranker...');
    await this.modelLoader.initialize();
    console.log('Reranker ready');
  }

  /**
   * Reranks a list of documents based on semantic relevance to the query using cross-encoder models.
   * 
   * INPUT SIGNATURE:
   * @param {string} query - The search query or question to rerank documents against
   * @param {(string|Object)[]} documents - Array of documents to rerank. Can be:
   *   - Array of strings: ["doc1 text", "doc2 text", ...]
   *   - Array of objects: [{text: "doc1"}, {text: "doc2", metadata: "..."}, ...]
   *   - Mixed format supported
   * @param {Object} [options] - Optional configuration object
   * @param {number} [options.topK=4] - Maximum number of top-ranked documents to return
   * 
   * OUTPUT FORMAT:
   * @returns {Promise<Array<{
   *   rerank_corpus_id: number,     // Original index position in input array
   *   rerank_score: number,         // Relevance score (higher = more relevant)
   *   text: string,                 // Document text content
   *   ...originalProperties         // Any additional properties from input objects
   * }>>} Array of reranked documents, sorted by relevance score (descending)
   * 
   * EXAMPLE USAGE:
   * ```javascript
   * // String array input
   * const results = await reranker.rerank(
   *   "How does AI work?",
   *   ["AI uses algorithms", "Cooking is fun", "Machine learning trains models"],
   *   { topK: 2 }
   * );
   * // Returns: [
   * //   { rerank_corpus_id: 0, rerank_score: 0.95, text: "AI uses algorithms" },
   * //   { rerank_corpus_id: 2, rerank_score: 0.87, text: "Machine learning trains models" }
   * // ]
   * 
   * // Object array input with metadata preservation
   * const results = await reranker.rerank(
   *   "Python programming",
   *   [
   *     { text: "Python is a programming language", id: "doc1", category: "tech" },
   *     { text: "Snakes are reptiles", id: "doc2", category: "animals" }
   *   ]
   * );
   * // Returns: [
   * //   { rerank_corpus_id: 0, rerank_score: 0.92, text: "Python is...", id: "doc1", category: "tech" }
   * // ]
   * ```
   * 
   * PERFORMANCE NOTES:
   * - MixedBread model (default): ~25-50 documents/second, higher confidence scores
   * - MS-MARCO model: ~90-150 documents/second, score range 0.0-1.0
   * - Batch processing is optimized for arrays up to 1000+ documents
   * - Empty input returns empty array immediately
   */
  async rerank(query, documents, options = {}) {
    if (!documents || documents.length === 0) {
      return [];
    }
    
    const { topK = 4 } = options;
    const start = performance.now();
    
    console.log(`Reranking ${documents.length} documents for query: "${query}"`);

    try {
      // Create query-document pairs for getScores
      const queryDocPairs = documents.map(doc => {
        const text = typeof doc === 'string' ? doc : doc.text;
        return [query, text];
      });

      // Delegate core inference to ModelLoader
      const results = await this.modelLoader.getScores(queryDocPairs);

      // Process scores and format the output
      const reranked = results
        .map((result, i) => ({
          rerank_corpus_id: i,
          rerank_score: result.score,
          ...(typeof documents[i] === 'string' ? { text: documents[i] } : documents[i]),
        }))
        .sort((a, b) => b.rerank_score - a.rerank_score)
        .slice(0, topK);

      const duration = performance.now() - start;
      console.log(`Reranking ${documents.length} documents to top ${topK} took ${duration.toFixed(2)}ms`);
      console.log(`  Top result score: ${reranked[0]?.rerank_score.toFixed(4) || 'N/A'}`);
      
      // Return the final, sorted array of documents
      return reranked;

    } catch (error) {
      console.error('Reranking failed:', error);
      throw error;
    }
  }

  /**
   * Preload the reranker model for faster subsequent calls
   * 
   * Alias for initialize() with service-oriented logging.
   * Useful for server applications that want to warm up models at startup
   * rather than on first request.
   * 
   * @returns {Promise<void>}
   * 
   * @example
   * // Server startup
   * const reranker = new NativeEmbeddingReranker();
   * await reranker.preload(); // Warm up model
   * // Now first rerank() call will be faster
   * 
   * // Equivalent to:
   * await reranker.initialize();
   */
  async preload() {
    try {
      console.log('Preloading reranker suite...');
      await this.initialize();
      console.log('Preloaded reranker suite. Reranking is available as a service now.');
    } catch (e) {
      console.error('Failed to preload reranker suite:', e);
      console.log('Reranking will be available on the first rerank call.');
    }
  }
}

// ====================================================================
// MODULE EXPORTS
// ====================================================================

// Export the production classes
export { 
  NativeEmbeddingReranker, 
  ModelLoader
};