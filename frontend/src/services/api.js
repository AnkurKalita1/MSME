const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3000';

/**
 * Generic API fetch function with error handling
 */
const apiCall = async (endpoint, options = {}) => {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || `HTTP error! status: ${response.status}`);
    }

    return data;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

/**
 * BOQ API Services
 */
export const boqAPI = {
  /**
   * Predict rates for BOQ items
   * @param {Array} items - Array of items with material field
   */
  predictRates: async (items) => {
    return apiCall('/api/boq/predict', {
      method: 'POST',
      body: JSON.stringify({ items }),
    });
  },

  /**
   * Extract BOQ from text and predict rates
   * @param {string} text - Text to extract BOQ from
   */
  extractAndPredict: async (text) => {
    return apiCall('/api/boq/extract', {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
  },

  /**
   * Health check
   */
  healthCheck: async () => {
    return apiCall('/api/boq/health');
  },
};

/**
 * WBS API Services
 */
export const wbsAPI = {
  /**
   * Generate WBS from single BOQ item
   * @param {Object} payload - WBS generation payload
   */
  generate: async (payload) => {
    return apiCall('/api/wbs/generate', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  /**
   * Generate WBS for multiple BOQ items (batch)
   * @param {Array} boqs - Array of BOQ items
   */
  generateBatch: async (boqs) => {
    return apiCall('/api/wbs/batch', {
      method: 'POST',
      body: JSON.stringify({ boqs }),
    });
  },

  /**
   * Health check
   */
  healthCheck: async () => {
    return apiCall('/api/wbs/health');
  },
};

/**
 * BOM API Services
 */
export const bomAPI = {
  /**
   * Generate BOM from BOQ
   * @param {Object} payload - BOM generation payload with boq object
   */
  generate: async (payload) => {
    return apiCall('/api/bom/generate', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  /**
   * Generate BOM for multiple BOQs (batch)
   * @param {Array} boqs - Array of BOQ items
   */
  generateBatch: async (boqs) => {
    return apiCall('/api/bom/generate-batch', {
      method: 'POST',
      body: JSON.stringify({ boqs }),
    });
  },

  /**
   * Get material substitutions
   * @param {string} materialName - Material name
   */
  getSubstitutions: async (materialName) => {
    return apiCall(`/api/bom/substitutions/${encodeURIComponent(materialName)}`);
  },

  /**
   * Predict wastage for a material
   * @param {Object} materialData - Material data
   */
  predictWastage: async (materialData) => {
    return apiCall('/api/bom/predict-wastage', {
      method: 'POST',
      body: JSON.stringify(materialData),
    });
  },
};

