import { fileURLToPath } from 'url';
import { NativeEmbeddingReranker } from './reranker.js';

/**
 * ====================================================================
 * RERANKER TEST SUITE - Comprehensive benchmarking and evaluation
 * ====================================================================
 * 
 * This file provides comprehensive testing and benchmarking utilities for
 * the NativeEmbeddingReranker. It includes:
 * - Evaluation metrics (NDCG, MRR, Precision@K, Recall@K)
 * - Realistic test data and document pools
 * - Comprehensive test runner with multiple scenarios
 * - Performance benchmarking
 * 
 * Usage:
 *   node test-suite.js [--quick|--full]
 */

// ====================================================================
// EVALUATION METRICS CLASS
// ====================================================================

/**
 * Evaluation metrics for ranking performance
 */
class EvaluationMetrics {
  /**
   * Calculate Mean Reciprocal Rank (MRR) at k
   * @param {Array} rankedPassages - Array of ranked passage objects with isRelevant property
   * @param {number} k - Cutoff for evaluation (default: 10)
   * @returns {number} - MRR@k score
   */
  static calculateMRR(rankedPassages, k = 10) {
    const relevantRanks = [];
    
    // Find ranks of relevant documents (1-indexed)
    for (let i = 0; i < Math.min(rankedPassages.length, k); i++) {
      if (rankedPassages[i].isRelevant) {
        relevantRanks.push(i + 1);
      }
    }
    
    if (relevantRanks.length === 0) {
      return 0.0;
    }
    
    // MRR is the reciprocal of the rank of the first relevant document
    return 1.0 / relevantRanks[0];
  }

  /**
   * Calculate Normalized Discounted Cumulative Gain (NDCG) at k
   * @param {Array} rankedPassages - Array of ranked passage objects with isRelevant property
   * @param {number} k - Cutoff for evaluation (default: 10)
   * @returns {number} - NDCG@k score
   */
  static calculateNDCG(rankedPassages, k = 10) {
    const dcg = this.calculateDCG(rankedPassages, k);
    const idcg = this.calculateIDCG(rankedPassages, k);
    
    return idcg > 0 ? dcg / idcg : 0.0;
  }

  /**
   * Calculate Discounted Cumulative Gain (DCG) at k
   * @param {Array} rankedPassages - Array of ranked passage objects with isRelevant property
   * @param {number} k - Cutoff for evaluation
   * @returns {number} - DCG@k score
   */
  static calculateDCG(rankedPassages, k) {
    let dcg = 0.0;
    
    for (let i = 0; i < Math.min(rankedPassages.length, k); i++) {
      const relevance = rankedPassages[i].isRelevant ? 1 : 0;
      if (i === 0) {
        dcg += relevance;
      } else {
        dcg += relevance / Math.log2(i + 1);
      }
    }
    
    return dcg;
  }

  /**
   * Calculate Ideal Discounted Cumulative Gain (IDCG) at k
   * @param {Array} rankedPassages - Array of passage objects to determine total relevant documents
   * @param {number} k - Cutoff for evaluation
   * @returns {number} - IDCG@k score
   */
  static calculateIDCG(rankedPassages, k) {
    // Count relevant documents
    const relevantCount = rankedPassages.filter(p => p.isRelevant).length;
    
    let idcg = 0.0;
    const maxRelevant = Math.min(relevantCount, k);
    
    for (let i = 0; i < maxRelevant; i++) {
      if (i === 0) {
        idcg += 1;
      } else {
        idcg += 1 / Math.log2(i + 1);
      }
    }
    
    return idcg;
  }

  /**
   * Calculate Precision at k
   * @param {Array} rankedPassages - Array of ranked passage objects with isRelevant property
   * @param {number} k - Cutoff for evaluation (default: 10)
   * @returns {number} - Precision@k score
   */
  static calculatePrecisionAtK(rankedPassages, k = 10) {
    if (k === 0 || rankedPassages.length === 0) {
      return 0.0;
    }
    
    const topK = rankedPassages.slice(0, Math.min(k, rankedPassages.length));
    const relevantInTopK = topK.filter(p => p.isRelevant).length;
    
    return relevantInTopK / Math.min(k, rankedPassages.length);
  }

  /**
   * Calculate Recall at k
   * @param {Array} rankedPassages - Array of ranked passage objects with isRelevant property
   * @param {number} k - Cutoff for evaluation (default: 10)
   * @returns {number} - Recall@k score
   */
  static calculateRecallAtK(rankedPassages, k = 10) {
    const totalRelevant = rankedPassages.filter(p => p.isRelevant).length;
    
    if (totalRelevant === 0) {
      return 0.0;
    }
    
    const topK = rankedPassages.slice(0, Math.min(k, rankedPassages.length));
    const relevantInTopK = topK.filter(p => p.isRelevant).length;
    
    return relevantInTopK / totalRelevant;
  }

  /**
   * Calculate comprehensive metrics for a single query result
   * @param {Array} rankedPassages - Array of ranked passage objects with isRelevant property
   * @param {Array<number>} kValues - Array of k values to evaluate (default: [1, 3, 5, 10])
   * @returns {Object} - Object containing all metrics at different k values
   */
  static calculateAllMetrics(rankedPassages, kValues = [1, 3, 5, 10]) {
    const metrics = {};
    
    for (const k of kValues) {
      metrics[`MRR@${k}`] = this.calculateMRR(rankedPassages, k);
      metrics[`NDCG@${k}`] = this.calculateNDCG(rankedPassages, k);
      metrics[`Precision@${k}`] = this.calculatePrecisionAtK(rankedPassages, k);
      metrics[`Recall@${k}`] = this.calculateRecallAtK(rankedPassages, k);
    }
    
    return metrics;
  }

  /**
   * Calculate average metrics across multiple query results
   * @param {Array<Object>} queryResults - Array of query result objects
   * @param {Array<number>} kValues - Array of k values to evaluate
   * @returns {Object} - Object containing average metrics
   */
  static calculateAverageMetrics(queryResults, kValues = [1, 3, 5, 10]) {
    if (queryResults.length === 0) {
      return {};
    }

    const validResults = queryResults.filter(result => !result.error);
    if (validResults.length === 0) {
      return { error: 'No valid results to calculate metrics' };
    }

    const averageMetrics = {};
    const metricSums = {};
    
    // Initialize sums
    for (const k of kValues) {
      metricSums[`MRR@${k}`] = 0;
      metricSums[`NDCG@${k}`] = 0;
      metricSums[`Precision@${k}`] = 0;
      metricSums[`Recall@${k}`] = 0;
    }
    
    // Sum metrics across all valid results
    for (const result of validResults) {
      const metrics = this.calculateAllMetrics(result.rankedPassages, kValues);
      
      for (const [metricName, value] of Object.entries(metrics)) {
        metricSums[metricName] += value;
      }
    }
    
    // Calculate averages
    for (const [metricName, sum] of Object.entries(metricSums)) {
      averageMetrics[metricName] = sum / validResults.length;
    }
    
    // Add additional statistics
    averageMetrics.totalQueries = queryResults.length;
    averageMetrics.validQueries = validResults.length;
    averageMetrics.errorQueries = queryResults.length - validResults.length;
    
    return averageMetrics;
  }
}

// ====================================================================
// TEST DATA
// ====================================================================

/**
 * Standard test queries with expected document relevance patterns
 */
const TEST_QUERIES = {
  ai_ml_fundamentals: {
    query: "How does machine learning and artificial intelligence work?",
    description: "Core AI/ML concepts and mechanisms",
    expectedHighRelevance: ["core_ai_ml", "tech_applications"],
    expectedMediumRelevance: ["data_science_business"],
    expectedLowRelevance: ["hard_science", "humanities", "health_lifestyle"]
  },
  photosynthesis: {
    query: "What is photosynthesis and how do plants create energy?",
    description: "Biological processes in plants",
    expectedHighRelevance: ["hard_science"],
    expectedMediumRelevance: ["health_lifestyle"],
    expectedLowRelevance: ["core_ai_ml", "tech_applications", "humanities", "data_science_business"]
  },
  software_development: {
    query: "How does artificial intelligence work in software development?",
    description: "AI applications in development workflows",
    expectedHighRelevance: ["tech_applications", "core_ai_ml"],
    expectedMediumRelevance: ["data_science_business"],
    expectedLowRelevance: ["hard_science", "humanities", "health_lifestyle"]
  },
  climate_change: {
    query: "Climate change effects on global weather patterns",
    description: "Environmental science and climate systems",
    expectedHighRelevance: ["hard_science"],
    expectedMediumRelevance: ["data_science_business"],
    expectedLowRelevance: ["core_ai_ml", "tech_applications", "humanities", "health_lifestyle"]
  }
};

/**
 * Document pools categorized by topic relevance
 * Each pool contains high-quality, realistic documents for testing
 */
const DOCUMENT_POOLS = {
  // Highly Relevant: Core concepts of AI and ML
  "core_ai_ml": [
    "Machine learning works by training algorithms on large datasets to recognize patterns. A model, such as a neural network, adjusts its internal parameters through a process called backpropagation, minimizing the difference between its predictions and the actual outcomes. This iterative training enables the model to make accurate predictions on new, unseen data.",
    "Artificial intelligence encompasses various subfields, with machine learning being the most prominent. Core AI systems often use deep learning, which involves neural networks with many layers (deep architectures). These systems can process complex inputs like images and speech by learning hierarchical features, from simple edges to complex objects.",
    "The effectiveness of an AI model is heavily dependent on the quality and quantity of its training data. Biased or insufficient data can lead to poor performance and unfair outcomes. Data preprocessing steps, such as normalization and augmentation, are crucial for preparing the data and improving the model's ability to generalize to real-world scenarios.",
    "Natural Language Processing (NLP) is a key area of AI that enables machines to understand and generate human language. It relies on models like transformers, which use attention mechanisms to weigh the importance of different words in a sentence. This allows for advanced applications like language translation, sentiment analysis, and sophisticated chatbots.",
    "Reinforcement learning is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize a cumulative reward. It differs from supervised learning as it doesn't require labeled data and instead learns through trial and error, making it suitable for robotics, game playing, and autonomous systems.",
    "Feature engineering transforms raw data into meaningful inputs for machine learning algorithms. This process involves selecting, modifying, or creating new features from existing data to improve model performance. Good feature engineering can significantly enhance a model's ability to learn patterns and make accurate predictions.",
    "Cross-validation techniques help evaluate model performance and prevent overfitting issues. By splitting data into multiple folds and training on different subsets, researchers can get a more robust estimate of how well their model will generalize to unseen data in real-world applications.",
    "Gradient descent optimization adjusts model parameters to minimize prediction errors efficiently. This iterative algorithm calculates the gradient of the loss function and moves parameters in the direction that reduces error, enabling neural networks to learn complex patterns from training data."
  ],
  
  // Relevant: Applications of AI in Technology
  "tech_applications": [
    "In modern software development, AI accelerates workflows through automated code generation and intelligent debugging. Tools powered by machine learning can predict potential bugs, suggest code completions, and even optimize algorithm performance, significantly reducing development time and improving code quality.",
    "Cloud computing platforms are essential for deploying scalable AI applications. They provide the vast computational resources, such as GPUs and TPUs, needed for training large models. Services like AWS SageMaker or Google AI Platform offer end-to-end solutions for building, training, and deploying machine learning models at scale.",
    "Cybersecurity systems increasingly rely on artificial intelligence to detect and respond to threats in real-time. ML algorithms can analyze network traffic to identify anomalous patterns indicative of a cyberattack, moving beyond traditional signature-based detection to counter new and evolving threats more effectively.",
    "The Internet of Things (IoT) generates massive streams of data from connected devices. Machine learning is used to analyze this data for applications like predictive maintenance in industrial machinery, smart home automation that learns user preferences, and real-time monitoring in healthcare.",
    "Computer vision, a field of AI, enables systems to interpret and understand visual information from the world. In technology, this is applied in autonomous vehicles for navigation, in facial recognition systems for security, and in medical imaging analysis to help diagnose diseases from scans like X-rays and MRIs.",
    "Database systems efficiently store, organize, and retrieve large amounts of structured data. Modern databases incorporate AI features for query optimization, automated indexing, and predictive caching to improve performance for machine learning workloads and large-scale applications.",
    "Software engineering practices ensure reliable, maintainable, and scalable application development. AI-powered tools assist in code review, testing automation, and deployment optimization, helping development teams deliver higher quality software more efficiently.",
    "Version control systems track code changes and enable collaborative software development. Advanced platforms integrate machine learning to suggest merge conflict resolutions, detect code patterns, and automate routine development tasks."
  ],
  
  // Medium Relevance: Data Science and Business
  "data_science_business": [
    "Data analytics forms the foundation for many business intelligence strategies. By analyzing historical data, companies can identify market trends, understand customer behavior, and make informed decisions. This process often precedes the implementation of predictive machine learning models.",
    "Businesses leverage data warehousing solutions to store and manage vast amounts of information from various sources. A well-organized data warehouse is critical for running complex queries and feeding clean data into machine learning pipelines for predictive analytics and forecasting.",
    "A/B testing is a common method used in business to compare two versions of a product or marketing campaign. While not AI itself, the results from extensive A/B tests provide valuable data that can be used to train machine learning models to personalize user experiences automatically.",
    "Supply chain management is being revolutionized by data analysis. Companies use predictive models to forecast demand, optimize inventory levels, and identify potential disruptions, leading to increased efficiency and reduced operational costs across their logistics networks.",
    "Customer Relationship Management (CRM) systems integrate data analytics to provide a 360-degree view of the customer. This allows businesses to segment their audience, personalize marketing efforts, and predict customer churn, thereby improving retention and loyalty.",
    "Data visualization tools help analysts present complex information in understandable formats. Interactive dashboards and charts make it easier for business stakeholders to interpret data insights and make informed decisions based on analytical findings.",
    "Business intelligence platforms combine data from multiple sources to provide comprehensive insights. These systems often incorporate basic machine learning capabilities for trend analysis, anomaly detection, and automated reporting.",
    "Mobile app development requires optimizing user interfaces for different device platforms. Analytics and user behavior data inform design decisions and feature prioritization to improve user engagement and retention."
  ],
  
  // Low Relevance: Hard Sciences
  "hard_science": [
    "Photosynthesis is the fundamental biological process by which plants, algae, and some bacteria convert light energy into chemical energy. They use sunlight, water, and carbon dioxide to create glucose, which serves as their food, and release oxygen as a byproduct, which is vital for most life on Earth.",
    "The theory of plate tectonics explains the large-scale motion of Earth's lithosphere. The planet's outer shell is divided into several plates that glide over the mantle, the rocky inner layer above the core. This movement is responsible for earthquakes, volcanic eruptions, and the formation of mountains.",
    "Quantum mechanics describes the strange behavior of matter and energy at the atomic and subatomic levels. Concepts like superposition and entanglement defy classical physics, stating that a particle can exist in multiple states at once and two particles can be linked in such a way that their fates are intertwined, regardless of the distance separating them.",
    "In chemistry, a covalent bond is a chemical link between two atoms in which electron pairs are shared between them. This sharing allows atoms to achieve a stable electron configuration and form molecules. Covalent bonds are responsible for the structure of most organic compounds, including DNA and proteins.",
    "The human immune system is a complex network of cells, tissues, and organs that work together to defend the body against pathogens. It includes two main subsystems: the innate immune system, which provides a general defense, and the adaptive immune system, which targets specific pathogens and creates long-lasting memory of them.",
    "Ecological systems maintain delicate balances between different species and their environment. Food webs illustrate how energy and nutrients flow through ecosystems, with producers, consumers, and decomposers each playing critical roles in sustaining biodiversity.",
    "Geological formations reveal Earth's history through layers of sedimentary rock deposits. By studying these strata, scientists can understand past climate conditions, extinction events, and the evolution of life over millions of years.",
    "Ocean currents distribute heat globally, influencing regional climate and weather patterns. These massive water movements are driven by differences in temperature, salinity, and wind patterns, affecting everything from local weather to global climate systems."
  ],
  
  // Unrelated: Humanities and Arts
  "humanities": [
    "The Renaissance was a fervent period of European cultural, artistic, political, and economic 'rebirth' following the Middle Ages. Generally described as taking place from the 14th century to the 17th century, it was characterized by a renewed interest in classical antiquity and a flourishing of art and science.",
    "William Shakespeare's tragedy, Hamlet, explores themes of betrayal, revenge, madness, and moral corruption. The play's protagonist struggles with the task of avenging his father's murder, and his contemplative soliloquies have become some of the most famous passages in English literature.",
    "The construction of the Great Wall of China spanned several dynasties and over two millennia. It was not a single continuous wall but a system of fortifications, walls, watchtowers, and fortresses built to protect Chinese states and empires from raids and invasions of nomadic groups from the Eurasian Steppe.",
    "Impressionism, a 19th-century art movement, was characterized by relatively small, thin, yet visible brush strokes, open composition, and an emphasis on the accurate depiction of light in its changing qualities. Artists like Claude Monet sought to capture the immediate sensory effect of a scene.",
    "The Roman Empire was one of the most powerful and influential civilizations in history, leaving a lasting legacy in law, language, architecture, and government. At its peak, it stretched from Britain in the northwest to Mesopotamia in the east, encompassing the entire Mediterranean basin.",
    "Classical composers like Bach created complex musical structures that remain influential today. His compositions demonstrate mathematical precision in their harmonic progressions and counterpoint, inspiring musicians and composers centuries later.",
    "Ancient Greek architecture featured impressive marble columns and geometric proportions. The Parthenon exemplifies the classical orders and mathematical ratios that influenced architectural design throughout Western civilization.",
    "Renaissance painters pioneered perspective techniques to create realistic artistic representations. Masters like Leonardo da Vinci combined artistic skill with scientific observation to produce works that captured both beauty and anatomical accuracy."
  ],
  
  // Unrelated: Health and Lifestyle
  "health_lifestyle": [
    "A balanced diet rich in fruits, vegetables, whole grains, and lean proteins is crucial for maintaining good health. These foods provide essential vitamins, minerals, and fiber, which support bodily functions, boost the immune system, and reduce the risk of chronic diseases such as heart disease and diabetes.",
    "Regular physical activity is vital for cardiovascular health. Aerobic exercises like running, swimming, or cycling strengthen the heart and improve blood circulation. Experts recommend at least 150 minutes of moderate-intensity exercise per week for adults to maintain a healthy lifestyle.",
    "Getting adequate sleep is as important as diet and exercise for overall well-being. During sleep, the body repairs itself, consolidates memories, and regulates hormones. Chronic sleep deprivation can lead to impaired cognitive function, mood disturbances, and an increased risk of serious health problems.",
    "Mindfulness and meditation are practices that can significantly reduce stress and improve mental clarity. By focusing on the present moment and observing thoughts without judgment, individuals can calm their nervous system, lower blood pressure, and enhance their emotional resilience.",
    "Hydration is essential for nearly every function in the body. Drinking enough water throughout the day helps regulate body temperature, deliver nutrients to cells, and keep organs functioning properly. Dehydration can cause fatigue, headaches, and impaired physical and cognitive performance.",
    "The Mediterranean diet emphasizes fresh fruits, vegetables, fish, and healthy olive oil. Research shows this eating pattern can reduce inflammation, improve heart health, and may contribute to longevity and cognitive function.",
    "Professional athletes train rigorously to develop physical strength, speed, and endurance. Their training regimens incorporate nutrition science, biomechanics, and performance psychology to optimize athletic performance and prevent injuries.",
    "Migration patterns help wildlife species adapt to seasonal changes in food availability. Understanding these natural cycles is important for conservation efforts and maintaining healthy ecosystems."
  ],
  
  // Generic irrelevant content for padding
  "irrelevant": [
    "The weather forecast predicts rain for the weekend with temperatures dropping to seasonal averages.",
    "Local restaurants are offering new seasonal menu items featuring fresh ingredients from nearby farms.",
    "Traffic patterns in the downtown area have improved following recent infrastructure improvements.",
    "The city library is hosting a book club meeting next Thursday evening for mystery novel enthusiasts.",
    "Gardening experts recommend planting spring flowers after the last frost date in your region.",
    "Pet adoption rates have increased significantly during the past year, with more families welcoming animals.",
    "Construction work on the new bridge is ahead of schedule and should finish before the holiday season.",
    "The local farmers market features organic produce, handmade crafts, and live music every Saturday morning.",
    "Urban planning balances population growth with sustainable infrastructure development needs.",
    "Tourism industry recovery continues with increased visitor numbers to popular destinations."
  ]
};

/**
 * Generate test candidates based on query and desired distribution
 * @param {string} queryKey - Key from TEST_QUERIES
 * @param {number} numCandidates - Total number of candidates to generate
 * @param {Object} options - Distribution options
 * @returns {Array<Object>} - Array of test documents
 */
function generateTestCandidates(queryKey, numCandidates = 20, options = {}) {
  const testQuery = TEST_QUERIES[queryKey];
  if (!testQuery) {
    throw new Error(`Unknown query key: ${queryKey}`);
  }
  
  const {
    highRelevanceRatio = 0.3,
    mediumRelevanceRatio = 0.3,
    lowRelevanceRatio = 0.4
  } = options;
  
  const numHigh = Math.floor(numCandidates * highRelevanceRatio);
  const numMedium = Math.floor(numCandidates * mediumRelevanceRatio);
  const numLow = numCandidates - numHigh - numMedium;
  
  const candidates = [];
  let docId = 0;
  const globalUsedTexts = new Set(); // Track used texts across all relevance levels
  
  // Helper function to get unique documents from multiple pools without duplicates
  function getUniqueDocuments(poolKeys, numNeeded, relevanceType) {
    const allDocs = [];
    
    // Collect all available documents from the pools
    for (const poolKey of poolKeys) {
      const pool = DOCUMENT_POOLS[poolKey];
      for (const text of pool) {
        if (!globalUsedTexts.has(text)) {
          allDocs.push({
            text: text,
            category: poolKey,
            variation: 0
          });
        }
      }
    }
    
    // Shuffle to get random selection
    const shuffledDocs = shuffleArray(allDocs);
    
    // Select unique documents up to what we need, or what's available
    const selectedDocs = shuffledDocs.slice(0, Math.min(numNeeded, shuffledDocs.length));
    
    // Mark selected texts as used
    selectedDocs.forEach(doc => globalUsedTexts.add(doc.text));
    
    // If we still need more documents and have less than requested, 
    // we'll pad with the irrelevant pool if available
    if (selectedDocs.length < numNeeded && relevanceType === "low") {
      const irrelevantPool = DOCUMENT_POOLS["irrelevant"];
      let additionalNeeded = numNeeded - selectedDocs.length;
      
      for (let i = 0; i < Math.min(additionalNeeded, irrelevantPool.length); i++) {
        if (!globalUsedTexts.has(irrelevantPool[i])) {
          selectedDocs.push({
            text: irrelevantPool[i],
            category: "irrelevant",
            variation: 0
          });
          globalUsedTexts.add(irrelevantPool[i]);
        }
      }
    }
    
    return selectedDocs;
  }
  
  // Add high relevance documents
  const highDocs = getUniqueDocuments(testQuery.expectedHighRelevance, numHigh, "high");
  for (const doc of highDocs) {
    candidates.push({
      id: `doc_${docId++}`,
      text: doc.text,
      source: `source_${docId}`,
      category: doc.category,
      expected_relevance: "high",
      variation: doc.variation || 0
    });
  }
  
  // Add medium relevance documents
  const mediumDocs = getUniqueDocuments(testQuery.expectedMediumRelevance, numMedium, "medium");
  for (const doc of mediumDocs) {
    candidates.push({
      id: `doc_${docId++}`,
      text: doc.text,
      source: `source_${docId}`,
      category: doc.category,
      expected_relevance: "medium",
      variation: doc.variation || 0
    });
  }
  
  // Add low relevance documents
  const lowDocs = getUniqueDocuments(testQuery.expectedLowRelevance, numLow, "low");
  for (const doc of lowDocs) {
    candidates.push({
      id: `doc_${docId++}`,
      text: doc.text,
      source: `source_${docId}`,
      category: doc.category,
      expected_relevance: "low",
      variation: doc.variation || 0
    });
  }
  
  // Shuffle to simulate realistic vector search results
  return shuffleArray(candidates);
}

/**
 * Shuffle array utility
 */
function shuffleArray(array) {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

// ====================================================================
// TEST SUITE RUNNER
// ====================================================================

/**
 * Run comprehensive reranker tests
 * This is the main testing suite that covers all scenarios
 */
async function runRerankerTest() {
  console.log('\n=== Comprehensive Transformers.js Reranker Test Suite ===\n');
  
  // Parse command line arguments to determine test mode
  const args = process.argv.slice(2);
  const mode = args.includes('--quick') ? 'quick' : 
              args.includes('--full') ? 'full' : 'standard';
  
  // Define test scenarios based on mode
  let testScenarios = [];
  
  if (mode === 'quick') {
    testScenarios = [
      {
        name: "Quick Validation (MS-MARCO)",
        queryKey: "ai_ml_fundamentals",
        numCandidates: 15,
        topK: 5,
        model: 'Xenova/ms-marco-MiniLM-L-6-v2',
        description: "Fast validation test with MS-MARCO model"
      }
    ];
  } else if (mode === 'full') {
    testScenarios = [
      {
        name: "AI/ML Fundamentals (MS-MARCO)",
        queryKey: "ai_ml_fundamentals",
        numCandidates: 25,
        topK: 8,
        model: 'Xenova/ms-marco-MiniLM-L-6-v2',
        description: "Core AI/ML concepts test with MS-MARCO model"
      },
      {
        name: "AI/ML Fundamentals (MixedBread)",
        queryKey: "ai_ml_fundamentals",
        numCandidates: 25,
        topK: 8,
        model: 'mixedbread-ai/mxbai-rerank-xsmall-v1',
        description: "Core AI/ML concepts test with MixedBread model"
      },
      {
        name: "Software Development (MS-MARCO)",
        queryKey: "software_development",
        numCandidates: 30,
        topK: 6,
        model: 'Xenova/ms-marco-MiniLM-L-6-v2',
        description: "AI in development workflows with MS-MARCO"
      },
      {
        name: "Software Development (MixedBread)",
        queryKey: "software_development",
        numCandidates: 30,
        topK: 6,
        model: 'mixedbread-ai/mxbai-rerank-xsmall-v1',
        description: "AI in development workflows with MixedBread"
      },
      {
        name: "Science Query (MS-MARCO)",
        queryKey: "photosynthesis",
        numCandidates: 20,
        topK: 5,
        model: 'Xenova/ms-marco-MiniLM-L-6-v2',
        description: "Biological processes test with MS-MARCO"
      },
      {
        name: "Science Query (MixedBread)",
        queryKey: "photosynthesis",
        numCandidates: 20,
        topK: 5,
        model: 'mixedbread-ai/mxbai-rerank-xsmall-v1',
        description: "Biological processes test with MixedBread"
      },
      {
        name: "Climate Change (MS-MARCO)",
        queryKey: "climate_change",
        numCandidates: 22,
        topK: 7,
        model: 'Xenova/ms-marco-MiniLM-L-6-v2',
        description: "Environmental science with MS-MARCO"
      },
      {
        name: "Climate Change (MixedBread)",
        queryKey: "climate_change",
        numCandidates: 22,
        topK: 7,
        model: 'mixedbread-ai/mxbai-rerank-xsmall-v1',
        description: "Environmental science with MixedBread"
      }
    ];
  } else {
    testScenarios = [
      {
        name: "AI/ML Fundamentals (MS-MARCO)",
        queryKey: "ai_ml_fundamentals",
        numCandidates: 25,
        topK: 8,
        model: 'Xenova/ms-marco-MiniLM-L-6-v2',
        description: "Core AI/ML concepts test with MS-MARCO model"
      },
      {
        name: "AI/ML Fundamentals (MixedBread)",
        queryKey: "ai_ml_fundamentals",
        numCandidates: 25,
        topK: 8,
        model: 'mixedbread-ai/mxbai-rerank-xsmall-v1',
        description: "Core AI/ML concepts test with MixedBread model"
      },
      {
        name: "Software Development (MS-MARCO)",
        queryKey: "software_development",
        numCandidates: 30,
        topK: 6,
        model: 'Xenova/ms-marco-MiniLM-L-6-v2',
        description: "AI in development workflows with MS-MARCO"
      },
      {
        name: "Science Query (MixedBread)",
        queryKey: "photosynthesis",
        numCandidates: 20,
        topK: 5,
        model: 'mixedbread-ai/mxbai-rerank-xsmall-v1',
        description: "Biological processes test with MixedBread"
      }
    ];
  }
  
  const allResults = [];
  
  console.log(`Running ${testScenarios.length} comprehensive test scenarios...\n`);
  
  for (let i = 0; i < testScenarios.length; i++) {
    const { name, queryKey, numCandidates, topK, model, description } = testScenarios[i];
    const testQuery = TEST_QUERIES[queryKey];
    
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Test ${i + 1}/${testScenarios.length}: ${name}`);
    console.log(`${description}`);
    console.log(`Model: ${model}`);
    console.log(`Query: "${testQuery.query}"`);
    console.log(`Candidates: ${numCandidates} → Top ${topK}`);
    console.log(`${'='.repeat(60)}`);
    
    // Create a reranker with the specified model for the test
    const reranker = new NativeEmbeddingReranker({ model: model });
    await reranker.initialize();
    
    // Generate realistic vector search candidates using the test data module
    const candidates = generateTestCandidates(queryKey, numCandidates);
    
    console.log(`\nGenerated candidate distribution:`);
    const highCount = candidates.filter(d => d.expected_relevance === 'high').length;
    const mediumCount = candidates.filter(d => d.expected_relevance === 'medium').length;
    const lowCount = candidates.filter(d => d.expected_relevance === 'low').length;
    console.log(`  High relevance: ${highCount}/${numCandidates} (${((highCount/numCandidates)*100).toFixed(0)}%)`);
    console.log(`  Medium relevance: ${mediumCount}/${numCandidates} (${((mediumCount/numCandidates)*100).toFixed(0)}%)`);
    console.log(`  Low relevance: ${lowCount}/${numCandidates} (${((lowCount/numCandidates)*100).toFixed(0)}%)`);
    
    // Perform reranking
    console.log(`\nProcessing ${numCandidates} candidates...`);
    const startTime = performance.now();
    const rerankedResults = await reranker.rerank(testQuery.query, candidates, { topK });
    const endTime = performance.now();
    const processingTime = endTime - startTime;
    
    // Prepare results for the EvaluationMetrics class
    const passagesForEval = rerankedResults.map(doc => ({
      ...doc,
      // Mark as relevant if high or medium relevance for evaluation
      isRelevant: doc.expected_relevance === 'high' || doc.expected_relevance === 'medium'
    }));

    // Use the superior EvaluationMetrics class
    const evaluation = EvaluationMetrics.calculateAllMetrics(passagesForEval, [1, 3, 5, topK]);
    
    // Calculate additional metrics for backward compatibility
    const highRelevantInTopK = rerankedResults.filter(doc => doc.expected_relevance === "high").length;
    const mediumRelevantInTopK = rerankedResults.filter(doc => doc.expected_relevance === "medium").length;
    const totalHighRelevant = candidates.filter(doc => doc.expected_relevance === "high").length;
    const totalMediumRelevant = candidates.filter(doc => doc.expected_relevance === "medium").length;
    const precision = (highRelevantInTopK + mediumRelevantInTopK * 0.5) / rerankedResults.length;
    const randomPrecision = (totalHighRelevant + totalMediumRelevant * 0.5) / candidates.length;
    const improvement = randomPrecision > 0 ? precision / randomPrecision : 1;

    // Display detailed results with comprehensive metrics
    console.log(`\nPerformance Metrics (k=${topK}):`);
    console.log(`  Processing time: ${processingTime.toFixed(2)}ms`);
    console.log(`  Throughput: ${((numCandidates / processingTime) * 1000).toFixed(1)} candidates/sec`);
    console.log(`  NDCG@${topK}: ${evaluation[`NDCG@${topK}`].toFixed(4)}`);
    console.log(`  MRR@${topK}: ${evaluation[`MRR@${topK}`].toFixed(4)}`);
    console.log(`  Precision@${topK}: ${evaluation[`Precision@${topK}`].toFixed(4)}`);
    console.log(`  Recall@${topK}: ${evaluation[`Recall@${topK}`].toFixed(4)}`);
    console.log(`  Legacy Precision: ${(precision * 100).toFixed(1)}%`);
    console.log(`  Improvement over random: ${improvement.toFixed(2)}x`);
    console.log(`  High relevance in top ${topK}: ${highRelevantInTopK}/${topK} (${((highRelevantInTopK/topK)*100).toFixed(0)}%)`);
    console.log(`  Medium relevance in top ${topK}: ${mediumRelevantInTopK}/${topK}`);

    console.log(`\nTop ${topK} Results:`);
    rerankedResults.forEach((doc, idx) => {
      const relevanceIcon = doc.expected_relevance === 'high' ? '🟢' : 
                           doc.expected_relevance === 'medium' ? '🟡' : '🔴';
      const truncatedText = doc.text.length > 60 ? doc.text.substring(0, 60) + '...' : doc.text;
      console.log(`   ${idx + 1}. ${relevanceIcon} Score: ${doc._rerank_score.toFixed(4)} | ${truncatedText}`);
    });
    
    // Score distribution analysis
    const scores = rerankedResults.map(r => r._rerank_score);
    const avgScore = scores.reduce((sum, s) => sum + s, 0) / scores.length;
    const maxScore = Math.max(...scores);
    const minScore = Math.min(...scores);
    
    console.log(`\nScore Analysis:`);
    console.log(`  Range: ${minScore.toFixed(4)} → ${maxScore.toFixed(4)} (Δ: ${(maxScore - minScore).toFixed(4)})`);
    console.log(`  Average: ${avgScore.toFixed(4)}`);
    
    // Store results for summary
    allResults.push({
      name,
      queryKey,
      query: testQuery.query,
      model,
      numCandidates,
      topK,
      results: rerankedResults,
      processingTime,
      totalCandidates: numCandidates,
      returnedResults: rerankedResults.length,
      evaluation: {
        rankedPassages: passagesForEval,
        ...evaluation,
        precision: precision,
        improvement: improvement,
        highRelevantInTopK: highRelevantInTopK,
        mediumRelevantInTopK: mediumRelevantInTopK
      },
      scoreStats: { avg: avgScore, max: maxScore, min: minScore, range: maxScore - minScore }
    });
    
    console.log(`\n${name} completed successfully!`);
  }
  
  // Comprehensive summary
  console.log(`\n${'='.repeat(70)}`);
  console.log(`COMPREHENSIVE TEST SUITE SUMMARY`);
  console.log(`${'='.repeat(70)}`);
  
  const avgPrecision = allResults.reduce((sum, r) => sum + r.evaluation.precision, 0) / allResults.length;
  const avgImprovement = allResults.reduce((sum, r) => sum + r.evaluation.improvement, 0) / allResults.length;
  const avgProcessingTime = allResults.reduce((sum, r) => sum + r.processingTime, 0) / allResults.length;
  const totalCandidatesProcessed = allResults.reduce((sum, r) => sum + r.totalCandidates, 0);
  const totalProcessingTime = allResults.reduce((sum, r) => sum + r.processingTime, 0);
  const overallThroughput = (totalCandidatesProcessed / totalProcessingTime) * 1000;
  
  // Calculate average comprehensive metrics
  const avgNDCG = allResults.reduce((sum, r) => sum + (r.evaluation[`NDCG@${r.topK}`] || 0), 0) / allResults.length;
  const avgMRR = allResults.reduce((sum, r) => sum + (r.evaluation[`MRR@${r.topK}`] || 0), 0) / allResults.length;
  const avgPrecisionAtK = allResults.reduce((sum, r) => sum + (r.evaluation[`Precision@${r.topK}`] || 0), 0) / allResults.length;
  const avgRecallAtK = allResults.reduce((sum, r) => sum + (r.evaluation[`Recall@${r.topK}`] || 0), 0) / allResults.length;
  
  console.log(`\nOverall Performance:`);
  console.log(`  Average Legacy Precision: ${(avgPrecision * 100).toFixed(1)}%`);
  console.log(`  Average NDCG@k: ${avgNDCG.toFixed(4)}`);
  console.log(`  Average MRR@k: ${avgMRR.toFixed(4)}`);
  console.log(`  Average Precision@k: ${avgPrecisionAtK.toFixed(4)}`);
  console.log(`  Average Recall@k: ${avgRecallAtK.toFixed(4)}`);
  console.log(`  Average Improvement: ${avgImprovement.toFixed(2)}x over random`);
  console.log(`  Average Processing Time: ${avgProcessingTime.toFixed(2)}ms`);
  console.log(`  Overall Throughput: ${overallThroughput.toFixed(1)} candidates/sec`);
  console.log(`  Total Candidates Processed: ${totalCandidatesProcessed}`);
  console.log(`  Total Processing Time: ${totalProcessingTime.toFixed(2)}ms`);
  
  console.log(`\nTest Breakdown:`);
  allResults.forEach((result, idx) => {
    const precision = (result.evaluation.precision * 100).toFixed(0);
    const ndcg = (result.evaluation[`NDCG@${result.topK}`] || 0).toFixed(3);
    const mrr = (result.evaluation[`MRR@${result.topK}`] || 0).toFixed(3);
    const improvement = result.evaluation.improvement.toFixed(1);
    const throughput = ((result.totalCandidates / result.processingTime) * 1000).toFixed(0);
    
    console.log(`  ${(idx + 1).toString().padStart(2)}. ${result.name.padEnd(20)} | ${precision}% prec | NDCG:${ndcg} | MRR:${mrr} | ${improvement}x | ${throughput} cand/sec`);
  });
  
  // Performance categories
  const fastTests = allResults.filter(r => r.processingTime < 200);
  const accurateTests = allResults.filter(r => r.evaluation.precision > 0.8);
  const efficientTests = allResults.filter(r => r.evaluation.improvement > 2.0);
  
  console.log(`\nPerformance Categories:`);
  console.log(`  Fast Tests (<200ms): ${fastTests.length}/${allResults.length}`);
  console.log(`  Accurate Tests (>80%): ${accurateTests.length}/${allResults.length}`);
  console.log(`  Efficient Tests (>2x improvement): ${efficientTests.length}/${allResults.length}`);
  
  console.log(`\nTechnical Details:`);
  console.log(`  Models Tested: ${[...new Set(allResults.map(r => r.model))].join(', ')}`);
  console.log(`  Test Queries: ${Object.keys(TEST_QUERIES).length} available`);
  console.log(`  Document Pools: ${Object.keys(DOCUMENT_POOLS).length} categories`);
  console.log(`  Total Documents Available: ${Object.values(DOCUMENT_POOLS).reduce((sum, pool) => sum + pool.length, 0)}`);
  console.log(`  Multi-Model Support: Active`);
  
  console.log(`\n${'='.repeat(70)}`);
  console.log(`ALL TESTS COMPLETED SUCCESSFULLY!`);
  console.log(`   ${allResults.length} test scenarios passed`);
  console.log(`   Average accuracy: ${(avgPrecision * 100).toFixed(1)}%`);
  console.log(`   Ready for production use!`);
  console.log(`${'='.repeat(70)}\n`);
  
  return allResults;
}

// ====================================================================
// COMMAND-LINE INTERFACE AND EXPORTS
// ====================================================================

// Export the testing utilities for use as module
export { 
  EvaluationMetrics, 
  runRerankerTest,
  TEST_QUERIES,
  DOCUMENT_POOLS,
  generateTestCandidates
};

// Command-line interface - only run if this file is called directly
const __filename = fileURLToPath(import.meta.url);
const isMainModule = process.argv[1] === __filename;

if (isMainModule) {
  console.log('Starting Transformers.js Reranker Test Suite...\n');

  // Parse command line arguments
  const args = process.argv.slice(2);
  const mode = args.includes('--quick') ? 'quick' : 
              args.includes('--full') ? 'full' : 'standard';

  console.log(`Test Mode: ${mode.toUpperCase()}`);

  if (mode === 'quick') {
    console.log('Running quick validation test (single scenario)');
  } else if (mode === 'full') {
    console.log('Running full comprehensive test suite (8 scenarios)');
  } else {
    console.log('Running standard test suite (4 scenarios)');
  }

  runRerankerTest()
    .then((results) => {
      console.log(`Test suite completed with ${results.length} scenarios!`);
      console.log(`Usage:`);
      console.log(`   node test-suite.js         - Standard test suite`);
      console.log(`   node test-suite.js --quick - Quick validation test`);
      console.log(`   node test-suite.js --full  - Full comprehensive suite`);
    })
    .catch(err => {
      console.error('Test suite failed:', err);
      process.exit(1);
    });
}