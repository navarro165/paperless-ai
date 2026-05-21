// services/ragService.js
const axios = require('axios');
const config = require('../config/config');
const AIServiceFactory = require('./aiServiceFactory');
const paperlessService = require('./paperlessService');

class RagService {
  constructor() {
    this.baseUrl = process.env.RAG_SERVICE_URL || 'http://localhost:8000';
  }

  /**
   * Check if the RAG service is available and ready
   * @returns {Promise<{status: string, index_ready: boolean, data_loaded: boolean}>}
   */
  async checkStatus() {
    try {
      const response = await axios.get(`${this.baseUrl}/status`);
      //make test call to the LLM service to check if it is available
      return response.data;
    } catch (error) {
      console.error('Error checking RAG service status:', error.message);
      return {
        server_up: false,
        data_loaded: false,
        index_ready: false,
        error: error.message
      };
    }
  }

  /**
   * Search for documents matching a query
   * @param {string} query - The search query
   * @param {Object} filters - Optional filters for search
   * @returns {Promise<Array>} - Array of search results
   */
  async search(query, filters = {}) {
    try {
      const response = await axios.post(`${this.baseUrl}/search`, {
        query,
        ...filters
      });
      return response.data;
    } catch (error) {
      console.error('Error searching documents:', error);
      throw error;
    }
  }

  /**
   * Ask a question about documents and get an AI-generated answer in the same language as the question
   * @param {string} question - The question to ask
   * @returns {Promise<{answer: string, sources: Array}>} - AI response and source documents
   */
  async askQuestion(question) {
    try {
      const aiService = AIServiceFactory.getService();

      // 1. Expand the query into retrieval-optimized keywords so the embedding
      //    model can match the right documents even for vague personal questions
      let retrievalQuery = question;
      try {
        const expansionPrompt = `You are a document retrieval assistant. Convert the user's question into 5-10 specific keywords that would appear verbatim in the document containing the answer. Output ONLY the keywords comma-separated, no explanation.\n\nQuestion: "${question}"\n\nKeywords:`;
        const expanded = await aiService.generateText(expansionPrompt);
        if (expanded && expanded.trim()) retrievalQuery = expanded.trim();
      } catch (err) {
        console.error('Query expansion failed, using original:', err.message);
      }

      // 2. Get context from the RAG service using the expanded query
      const response = await axios.post(`${this.baseUrl}/context`, {
        question: retrievalQuery,
        max_sources: 20
      }, { timeout: 30000 });

      const { context, sources } = response.data;

      // 3. Fetch full document objects (content + metadata) for each source
      let enhancedContext = context;

      if (sources && sources.length > 0) {
        const fullDocContents = await Promise.all(
          sources.map(async (source) => {
            if (source.doc_id) {
              try {
                const doc = await paperlessService.getDocument(source.doc_id);
                const content = doc.content || '';
                const truncated = content.length > 8000 ? content.slice(0, 8000) : content;
                // Prepend metadata so Gemini can distinguish which document belongs to whom
                const fileName = doc.archived_file_name || '';
                const created = doc.created_date || '';
                const header = [
                  `Document: ${doc.title || source.title}`,
                  fileName ? `File: ${fileName}` : '',
                  created ? `Date: ${created}` : '',
                ].filter(Boolean).join(' | ');
                return `${header}\n${truncated}`;
              } catch (error) {
                console.error(`Error fetching content for document ${source.doc_id}:`, error.message);
                return '';
              }
            }
            return '';
          })
        );

        enhancedContext = context + '\n\n' + fullDocContents.filter(content => content).join('\n\n---\n\n');
      }

      // 5. Answer using the ORIGINAL question (not the expanded retrieval query)
      const prompt = `
        You are a helpful assistant that answers questions about documents.

        Answer the following question precisely, based on the provided documents:

        Question: ${question}

        Context from relevant documents:
        ${enhancedContext}

        Important instructions:
        - Use ONLY information from the provided documents
        - If the answer is not contained in the documents, respond: "This information is not contained in the documents." (in the same language as the question)
        - Avoid assumptions or speculation beyond the given context
        - Answer in the same language as the question was asked
        - Do not mention document numbers or source references, answer as if it were a natural conversation
        `;

      let answer;
      try {
        answer = await aiService.generateText(prompt);
      } catch (error) {
        console.error('Error generating answer with AI service:', error);
        answer = "An error occurred while generating an answer. Please try again later.";
      }
      
      return {
        answer,
        sources
      };
    } catch (error) {
      console.error('Error in askQuestion:', error);
      throw new Error("An error occurred while processing your question. Please try again later.");
    }
  }

  /**
   * Start indexing documents in the RAG service
   * @param {boolean} force - Whether to force refresh from source
   * @returns {Promise<Object>} - Indexing status
   */
  async indexDocuments(force = false) {
    try {
      const response = await axios.post(`${this.baseUrl}/indexing/start`, { 
        force, 
        background: true 
      });
      return response.data;
    } catch (error) {
      console.error('Error indexing documents:', error);
      throw error;
    }
  }

  /**
   * Check if the RAG service needs document updates
   * @returns {Promise<{needs_update: boolean, message: string}>}
   */
  async checkForUpdates() {
    try {
      const response = await axios.post(`${this.baseUrl}/indexing/check`);
      return response.data;
    } catch (error) {
      console.error('Error checking for updates:', error);
      throw error;
    }
  }

  /**
   * Get current indexing status
   * @returns {Promise<Object>} - Current indexing status
   */
  async getIndexingStatus() {
    try {
      const response = await axios.get(`${this.baseUrl}/indexing/status`);
      return response.data;
    } catch (error) {
      console.error('Error getting indexing status:', error);
      throw error;
    }
  }

  /**
   * Initialize the RAG service
   * @param {boolean} force - Whether to force initialization
   * @returns {Promise<Object>} - Initialization status
   */
  async initialize(force = false) {
    try {
      const response = await axios.post(`${this.baseUrl}/initialize`, { force });
      return response.data;
    } catch (error) {
      console.error('Error initializing RAG service:', error);
      throw error;
    }
  }

  /**
   * Get AI status
   * @returns {Promise<{status: string}>}
   */
  async getAIStatus() {
    try {
      const aiService = AIServiceFactory.getService();
      const status = await aiService.checkStatus();
      return status;
    } catch (error) {
      console.error('Error checking AI service status:', error);
      throw error;
    }
  }
}


module.exports = new RagService();
