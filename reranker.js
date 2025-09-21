import { AutoTokenizer, AutoModelForSequenceClassification } from '@xenova/transformers';
import path from 'path';
import fs from 'fs';

// --- New: Safe high-resolution timer with a fallback ---
const now = (typeof performance !== 'undefined' && performance.now) 
  ? () => performance.now() 
  : () => Date.now();

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

  constructor({ modelName, cacheDir = null, logger = null }) {
    if (!modelName) throw new Error("A model name must be provided.");
    this.modelName = modelName;
    
    // Define the cache directory for models, with a fallback
    this.cacheDir = cacheDir
      ? path.resolve(cacheDir)
      : path.resolve(process.cwd(), 'models'); // Default to a local 'models' folder

    // Set up logger (use default if none provided)
    this.logger = logger || {
      debug: (msg) => console.debug(`[DEBUG] ${msg}`),
      info: (msg) => console.info(`[INFO] ${msg}`),
      warn: (msg) => console.warn(`[WARN] ${msg}`),
      error: (msg) => console.error(`[ERROR] ${msg}`)
    };

    // Ensure the cache directory exists
    if (!fs.existsSync(this.cacheDir)) {
      fs.mkdirSync(this.cacheDir, { recursive: true });
    }

    // The 'from_pretrained' function will handle checking the cache automatically.
    // No need for a manual check.
    this.logger.info(`[ModelLoader] Initialized for model "${this.modelName}" with cache: ${this.cacheDir}`);
  }



  /**
   * Initialize the cross-encoder model and tokenizer with caching
   * @returns {Promise<void>}
   */
  async initialize() {
    // Check if this specific model is already loaded
    if (ModelLoader.#models.has(this.modelName)) {
      this.logger.debug(`Model "${this.modelName}" already initialized.`);
      return;
    }

    // Check if this specific model is currently being initialized
    if (ModelLoader.#initializationPromises.has(this.modelName)) {
      this.logger.debug(`Waiting for "${this.modelName}" initialization to complete...`);
      await ModelLoader.#initializationPromises.get(this.modelName);
      return;
    }

    const promise = (async () => {
      try {
        this.logger.info(`Loading model: ${this.modelName}...`);
        const start = now();

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

        const duration = now() - start;
        this.logger.info(`Model "${this.modelName}" loaded in ${duration.toFixed(2)}ms`);
        this.logger.debug(`Model cached at: ${this.cacheDir}`);

      } catch (error) {
        this.logger.error(`Failed to load model "${this.modelName}": ${error.message}`);
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

  // --- New: Pure JavaScript math functions for robustness ---
  #sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  #softmax(arr) {
    const exps = arr.map(x => Math.exp(x));
    const sumExps = exps.reduce((a, b) => a + b);
    return exps.map(e => e / sumExps);
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
      // --- New: Get the model's configuration to find the positive class ---
      const config = model.config || {};
      const id2label = config.id2label || {};
      const positiveClassIndex = Number(
        Object.keys(id2label).find(key => 
          /relevant|positive/i.test(id2label[key]) // Look for "relevant" or "positive"
        ) ?? 1 // Fallback to 1 if not found
      );
      this.logger.debug(`Determined positive class index: ${positiveClassIndex}`);

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
      const logits = output.logits;
      const shape = logits.dims || logits.shape || []; // Get tensor shape

      let scores;
      const logitsData = logits.tolist(); // Get the raw nested array of numbers

      // The key change: Check the number of output columns (logits)
      if (shape.length > 1 && shape[1] === 1) {
        // --- Case 1: Handle single-logit models (like ms-marco) ---
        // The model outputs one score per document, so we use sigmoid.
        this.logger.debug('Applying SIGMOID for single-logit model (JS fallback).');
        // --- Updated: Use the JS helper ---
        scores = logitsData.map(([logit]) => this.#sigmoid(logit));

      } else {
        // --- Case 2: Handle multi-logit models (like many modern rerankers) ---
        // The model outputs a score for each class (e.g., 'not relevant', 'relevant').
        // We use softmax to get a probability distribution.
        this.logger.debug('Applying SOFTMAX for multi-logit model (JS fallback).');
        // --- Updated: Use the JS helper ---
        const probabilities = logitsData.map(row => this.#softmax(row));
        scores = probabilities.map(row => row[positiveClassIndex]);
      }
      
      // Convert to expected format
      const results = scores.map(score => ({
        score: score,
        label: score > 0.5 ? 'RELEVANT' : 'NOT_RELEVANT'
      }));
      
      return results;
      
    } catch (error) {
      this.logger.error(`Error getting scores: ${error.message}`);
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

  /**
   * Clear model cache for memory management
   * @param {string|null} modelNameToClear - Specific model to clear, or null to clear all
   */
  static clearCache(modelNameToClear = null) {
    if (modelNameToClear) {
      // Clear specific model
      const wasPresent = ModelLoader.#models.has(modelNameToClear) || ModelLoader.#tokenizers.has(modelNameToClear);
      ModelLoader.#models.delete(modelNameToClear);
      ModelLoader.#tokenizers.delete(modelNameToClear);
      ModelLoader.#initializationPromises.delete(modelNameToClear); // Also clear any pending promises
      
      if (wasPresent) {
        console.log(`[INFO] Cleared cache for model "${modelNameToClear}"`);
      } else {
        console.log(`[WARN] Model "${modelNameToClear}" was not found in cache`);
      }
    } else {
      // Clear all models
      const totalModels = ModelLoader.#models.size;
      ModelLoader.#models.clear();
      ModelLoader.#tokenizers.clear();
      ModelLoader.#initializationPromises.clear();
      
      console.log(`[INFO] Cleared cache for all models (${totalModels} models removed)`);
    }
  }

  /**
   * Get information about currently cached models
   * @returns {Array<string>} Array of cached model names
   */
  static getCachedModels() {
    return Array.from(ModelLoader.#models.keys());
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
   * @param {Object|null} [options.logger=null] - Custom logger object with debug, info, warn, error methods
   *   - null: Uses default console-based logger
   *   - object: Custom logger (e.g., Winston, Pino, etc.)
   * @param {string} [options.logLevel='info'] - Log level when using default logger
   *   - 'debug': Verbose logging for troubleshooting
   *   - 'info': Standard operational messages (default)
   *   - 'warn': Warning messages only
   *   - 'error': Error messages only
   *   - 'silent': No logging
   * 
   * @example
   * // Default MixedBread model with info logging
   * const reranker = new NativeEmbeddingReranker();
   * 
   * // Specific model with custom log level
   * const reranker = new NativeEmbeddingReranker({
   *   model: 'Xenova/ms-marco-MiniLM-L-6-v2',
   *   logLevel: 'debug'
   * });
   * 
   * // Production setup with custom logger and cache directory
   * const reranker = new NativeEmbeddingReranker({
   *   model: 'mixedbread-ai/mxbai-rerank-xsmall-v1',
   *   cacheDir: '/custom/path/to/models',
   *   logger: myWinstonLogger, // Custom logger
   *   logLevel: 'warn' // Only warnings and errors
   * });
   * 
   * // Silent mode for minimal logging
   * const reranker = new NativeEmbeddingReranker({
   *   logLevel: 'silent'
   * });
   */
  constructor(options = {}) {
    // Destructure options with new logging parameters
    const { 
      cacheDir = null, 
      model = 'mixedbread-ai/mxbai-rerank-xsmall-v1', // Default model
      logger = null, // Allow custom logger
      logLevel = 'info' // Default log level
    } = options;

    this.model = model; // Store the model name
    this.cacheDir = cacheDir;
    this.logLevel = logLevel; // <-- Add this line

    // --- Professional Logger Logic ---
    // If a custom logger isn't provided, create a simple default one
    this.logger = logger || this.createDefaultLogger(logLevel);

    // Pass the model name and logger to the ModelLoader
    this.modelLoader = new ModelLoader({ 
      modelName: this.model, 
      cacheDir: this.cacheDir,
      logger: this.logger // Pass logger down to ModelLoader
    });
  }

  /**
   * Create a default logger with configurable log levels
   * @param {string} level - Log level: 'debug', 'info', 'warn', 'error', 'silent'
   * @returns {Object} Logger object with debug, info, warn, error methods
   */
  createDefaultLogger(level) {
    const levels = { 'debug': 1, 'info': 2, 'warn': 3, 'error': 4, 'silent': 5 };
    const currentLevel = levels[level] || 2;
    
    return {
      debug: (message) => currentLevel <= 1 && console.debug(`[DEBUG] ${message}`),
      info:  (message) => currentLevel <= 2 && console.info(`[INFO] ${message}`),
      warn:  (message) => currentLevel <= 3 && console.warn(`[WARN] ${message}`),
      error: (message) => currentLevel <= 4 && console.error(`[ERROR] ${message}`)
    };
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
    this.logger.info('Initializing NativeEmbeddingReranker...');
    await this.modelLoader.initialize();
    this.logger.info('Reranker ready');
  }

  /**
   * Reranks a list of documents based on semantic relevance to the query using cross-encoder models.
   * Now with batching support for scalability.
   * 
   * INPUT SIGNATURE:
   * @param {string} query - The search query or question to rerank documents against
   * @param {(string|Object)[]} documents - Array of documents to rerank. Can be:
   *   - Array of strings: ["doc1 text", "doc2 text", ...]
   *   - Array of objects: [{text: "doc1"}, {text: "doc2", metadata: "..."}, ...]
   *   - Mixed format supported
   * @param {Object} [options] - Optional configuration object
   * @param {number} [options.topK=4] - Maximum number of top-ranked documents to return
   * @param {number} [options.batchSize=128] - Number of documents to process in a single batch
   * 
   * OUTPUT FORMAT:
   * @returns {Promise<Array<{
   *   _rerank_corpus_id: number,    // Original index position in input array  
   *   _rerank_score: number,        // Relevance score (higher = more relevant)
   *   text: string,                 // Document text content
   *   ...originalProperties         // Any additional properties from input objects
   * }>>} Array of reranked documents, sorted by relevance score (descending)
   * 
   * EXAMPLE USAGE:
   * ```javascript
   * // String array input with custom batch size
   * const results = await reranker.rerank(
   *   "How does AI work?",
   *   ["AI uses algorithms", "Cooking is fun", "Machine learning trains models"],
   *   { topK: 2, batchSize: 64 }
   * );
   * // Returns: [
   * //   { _rerank_corpus_id: 0, _rerank_score: 0.95, text: "AI uses algorithms" },
   * //   { _rerank_corpus_id: 2, _rerank_score: 0.87, text: "Machine learning trains models" }
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
   * //   { _rerank_corpus_id: 0, _rerank_score: 0.92, text: "Python is...", id: "doc1", category: "tech" }
   * // ]
   * ```
   * 
   * PERFORMANCE NOTES:
   * - MixedBread model (default): ~25-50 documents/second, higher confidence scores
   * - MS-MARCO model: ~90-150 documents/second, score range 0.0-1.0
   * - Batch processing prevents memory issues when handling thousands of documents
   * - Default batch size is 128 documents per batch for optimal memory usage
   * - Empty input returns empty array immediately
   */
  async rerank(query, documents, options = {}) {
    if (!documents || documents.length === 0) {
      return [];
    }
    
    // Destructure options with sensible defaults
    const { topK = 4, batchSize = 128 } = options;
    const start = now();
    
    this.logger.info(`Reranking ${documents.length} documents for query: "${query}" with batch size ${batchSize}`);

    try {
      // --- Core Batching Logic ---
      
      // 1. Create an array to hold results from all batches
      let allResults = [];

      // 2. Loop through the documents in chunks of batchSize
      for (let i = 0; i < documents.length; i += batchSize) {
        const chunkStart = i;
        const chunkEnd = Math.min(i + batchSize, documents.length);
        
        // Get the current chunk of documents
        const chunk = documents.slice(chunkStart, chunkEnd);
        this.logger.debug(`Processing batch ${Math.floor(i / batchSize) + 1}: documents ${chunkStart + 1}-${chunkEnd}`);

        // Create query-document pairs for the current chunk
        const queryDocPairs = chunk.map(doc => {
          const text = typeof doc === 'string' ? doc : doc.text;
          return [query, text];
        });

        // Get scores for the current chunk
        const chunkScores = await this.modelLoader.getScores(queryDocPairs);

        // Process and store results for this chunk, making sure to use the original index
        const chunkResults = chunkScores.map((result, j) => {
          const originalIndex = chunkStart + j;
          return {
            // --- API Namespacing: Use underscores to prevent field collisions ---
            _rerank_corpus_id: originalIndex,
            _rerank_score: result.score,
            ...(typeof documents[originalIndex] === 'string' ? { text: documents[originalIndex] } : documents[originalIndex]),
          };
        });

        allResults.push(...chunkResults);
      }
      
      // --- Final Processing ---
      
      // 3. Now, sort all the collected results together
      const reranked = allResults.sort((a, b) => b._rerank_score - a._rerank_score);
      
      // 4. Finally, slice to get the topK results
      const finalTopK = reranked.slice(0, topK);

      const duration = now() - start;
      this.logger.info(`Reranking ${documents.length} documents to top ${topK} took ${duration.toFixed(2)}ms`);
      this.logger.debug(`Top result score: ${finalTopK[0]?._rerank_score.toFixed(4) || 'N/A'}`);
      
      return finalTopK;

    } catch (error) {
      this.logger.error(`Reranking failed: ${error.message}`);
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
      this.logger.info('Preloading reranker suite...');
      await this.initialize();
      this.logger.info('Preloaded reranker suite. Reranking is available as a service now.');
    } catch (e) {
      this.logger.error(`Failed to preload reranker suite: ${e.message}`);
      this.logger.warn('Reranking will be available on the first rerank call.');
    }
  }

  /**
   * Unload the associated model from memory and clear it from the cache.
   * Useful in long-running applications to manage memory usage.
   * 
   * @example
   * const reranker = new NativeEmbeddingReranker();
   * await reranker.initialize();
   * // ... use the reranker ...
   * 
   * // When shutting down or switching models:
   * reranker.dispose(); // Frees up memory
   */
  dispose() {
    this.logger.info(`Disposing model: ${this.model}`);
    ModelLoader.clearCache(this.model);
    this.logger.debug('Model resources have been released from memory');
  }

  /**
   * Get information about the current model and cache status
   * @returns {Object} Model information and cache statistics
   */
  getModelInfo() {
    const cachedModels = ModelLoader.getCachedModels();
    const isLoaded = this.modelLoader.isLoaded();
    
    return {
      currentModel: this.model,
      isLoaded: isLoaded,
      cacheDirectory: this.cacheDir,
      totalCachedModels: cachedModels.length,
      cachedModels: cachedModels,
      logLevel: this.logLevel // <-- Change this line
    };
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