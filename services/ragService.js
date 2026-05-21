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

      // 1. Expand the query into retrieval-optimized keywords.
      //    The expansion prompt tells Gemini this is a personal document archive
      //    so it interprets abbreviations in that context (SSN = Social Security Number,
      //    not a stock ticker; W2 = tax form, not something else).
      //    Optional: set RAG_OWNER_NAME in .env to anchor first-person pronouns
      //    (my/I/me) to the archive owner's real name for better personal queries.
      const ownerName = process.env.RAG_OWNER_NAME || '';
      const ownerHint = ownerName
        ? ` The archive owner is ${ownerName}. When the query uses "my", "I", or "me", include "${ownerName}" in the search terms.`
        : '';
      let retrievalQuery = question;
      try {
        const expansionPrompt = `You are a search assistant for a personal document archive containing tax forms, government documents, financial records, insurance documents, and legal papers.${ownerHint} Expand the user's query into 5-10 specific search terms that would appear verbatim in the document containing the answer. Interpret all abbreviations in the context of personal finance and government documents (e.g. SSN = Social Security Number, EIN = Employer Identification Number, W-2 = Wage and Tax Statement, 1040 = tax return). Output ONLY the search terms comma-separated, no explanation.\n\nQuery: "${question}"\n\nSearch terms:`;
        const expanded = await aiService.generateText(expansionPrompt);
        if (expanded && expanded.trim()) retrievalQuery = expanded.trim();
      } catch (err) {
        console.error('Query expansion failed, using original:', err.message);
      }

      // 2. Retrieve context using the expanded query
      const response = await axios.post(`${this.baseUrl}/context`, {
        question: retrievalQuery,
        max_sources: 20
      }, { timeout: 30000 });

      const { context, sources } = response.data;

      // 3. Fetch full document content and prepend rich metadata so Gemini can
      //    distinguish documents by correspondent, date, and tags.
      //    Correspondent and date are already in the source object from the RAG
      //    service (zero extra API calls). Tags come from the full document fetch.
      let enhancedContext = context;

      if (sources && sources.length > 0) {
        const fullDocContents = await Promise.all(
          sources.map(async (source) => {
            if (!source.doc_id) return '';
            try {
              const doc = await paperlessService.getDocument(source.doc_id);
              const content = doc.content || '';
              const truncated = content.length > 8000 ? content.slice(0, 8000) : content;

              // Build a rich metadata header. Use source.correspondent (already resolved
              // to a name by the RAG Python service) and source.date rather than IDs.
              const correspondent = source.correspondent || '';
              const date = source.date || doc.created_date || '';

              // Resolve tag names via a single bulk request (id__in= filter)
              let tagNames = '';
              if (doc.tags && doc.tags.length > 0) {
                try {
                  const tagIds = doc.tags.slice(0, 5).join(',');
                  const tagResp = await paperlessService.client.get(`/tags/?id__in=${tagIds}&page_size=5`);
                  tagNames = (tagResp.data.results || []).map(t => t.name).join(', ');
                } catch (e) {
                  // tags are enhancement only; don't fail the whole document
                }
              }

              const header = [
                `Document: ${doc.title || source.title}`,
                correspondent ? `Correspondent: ${correspondent}` : '',
                date ? `Date: ${date}` : '',
                tagNames ? `Tags: ${tagNames}` : '',
              ].filter(Boolean).join(' | ');

              return `${header}\n${truncated}`;
            } catch (error) {
              console.error(`Error fetching content for document ${source.doc_id}:`, error.message);
              return '';
            }
          })
        );

        enhancedContext = context + '\n\n' + fullDocContents.filter(c => c).join('\n\n---\n\n');
      }

      // 4. Answer using the ORIGINAL question against the enriched context
      const ownerInstruction = ownerName
        ? `When the question uses "my", "I", "me", or "mine", it refers to ${ownerName} and not to any other person whose documents may appear in the context.\n`
        : '';
      const prompt = `
        You are a helpful assistant that searches a personal document archive and surfaces relevant findings.
        ${ownerInstruction}
        The user is asking: ${question}

        Documents retrieved from the archive:
        ${enhancedContext}

        Instructions:
        - Search ALL provided documents for anything relevant to the question.
        - If you find relevant information in multiple documents, list ALL of them with the correspondent/source so the user can identify which one applies to them.
        - Do NOT pick just one answer and discard the others. Surface every potential match.
        - Do NOT say "this information is not contained in the documents" if you found anything even partially relevant — show what you found and where.
        - Only say information is not available if you genuinely found nothing relevant after scanning all documents.
        - Answer in the same language as the question.
        - Format clearly: if multiple matches, list each with its source (correspondent name and date from the document header).
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
