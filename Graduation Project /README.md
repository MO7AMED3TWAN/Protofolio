# Speak with Your Book 📚💬

An intelligent chatbot system that enables users to have meaningful conversations with any uploaded book or document using Retrieval-Augmented Generation (RAG) and Large Language Models.

## 🎯 Project Overview

**Speak with Your Book** is an advanced RAG-based system that transforms static documents into interactive conversational partners. Users can upload books or documents and engage in natural language conversations, receiving accurate, context-aware answers grounded in the specific content of their uploaded materials.

### Key Features

- 📖 **Interactive Document Conversations**: Chat naturally with any book or document
- 🔍 **Semantic Search**: Intelligent retrieval of relevant content segments
- 🤖 **Fine-tuned LLM**: Custom Qwen model optimized for document-based conversations
- 💾 **Vector Database**: ChromaDB for efficient similarity search and retrieval
- 🌐 **User-Friendly Interface**: Clean UI for seamless interaction
- 🔮 **Future Arabic Support**: Planned multilingual capabilities

## 🏗️ System Architecture

### Workflow Overview
   ## Part (1)
1. **Document Preprocessing**
   - Extract and clean text from uploaded documents
   - Segment content into meaningful units (chapters, paragraphs)
   - Generate vector embeddings for each segment

2. **Vector Storage**
   - Store embeddings in ChromaDB for fast semantic search
   - Index content for efficient retrieval

   ## Part (2)
3. **Query Processing**
   - Convert user questions into vector embeddings
   - Retrieve most relevant document segments
   - Feed context to fine-tuned LLM

4. **Response Generation**
   - Generate contextual responses using retrieved information
   - Ensure accuracy and coherence based on source material


## Workflow Digram
```
[Document Upload] → [Preprocessing] → [Embedding Generation] → [ChromaDB Storage]
                                                                       ↓
[User Query] → [Query Embedding] → [Similarity Search] → [Context Retrieval] → [LLM Response]
```

## 📁 Project Structure

```
speak-with-your-book/
├── development/                          # Development Folder
│   ├── 01_qwen_fine_tuning.ipynb         # Fine-tune Qwen LLM
│   ├── 02_data_preparation_rag.ipynb     # RAG data prep & DB creation
│   ├── inputs/
│       └── data/
│           ├── pdf/                      # Original pdf documents
│           └── otherwise/                # JSON and other supported document formats
│   └── outputs/
│         └── data/
│               ├── processed/            # Cleaned and segmented data
│               └── vector_database/      # Generated vector embeddings
│         └── Model/                      # Fine-tuned Model
├── 03_rag_system.ipynb                   # Main RAG system implementation
├── ui/streamlit_app.py                   # User interface
├── requirements.txt                      # Python dependencies 
│   
└── README.md                     
```
## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for model training)
- Minimum 16GB RAM (32GB recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/speak-with-your-book.git
   cd speak-with-your-book
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Step 1: Fine-tune the LLM
Run the fine-tuning notebook to customize Qwen for your specific use case:
```bash
jupyter notebook notebooks/01_qwen_fine_tuning.ipynb
```

#### Step 2: Prepare RAG Data
Process your documents and create the vector database:
```bash
jupyter notebook notebooks/02_data_preparation_rag.ipynb
```

#### Step 3: Run the RAG System
Launch the main RAG system:
```bash
jupyter notebook notebooks/03_rag_system.ipynb
```

#### Step 4: Use the Interface (Optional)
Launch the user interface for easier interaction:
```bash
streamlit run ui/streamlit_app.py
```

## 📚 System Components

### Main RAG System (`main_rag_system.ipynb`) - **🎯 Primary Interface**
- **Purpose**: Complete RAG system built with LangChain
- **Features**:
  - Load pre-built fine-tuned Qwen model
  - Connect to pre-built ChromaDB vector database
  - Interactive chat interface for book conversations
  - Query processing with context retrieval
  - Performance evaluation and metrics
  - **Ready to run immediately!**

### Development Notebooks (Reference Only)

#### Database Creation (`development/database_creation.ipynb`) - ✅ **Completed**
- **Purpose**: Built vector database from JSON files with RAG-optimized structure
- **Process**:
  - JSON file parsing and processing
  - Text chunking with optimal RAG structure
  - Embedding generation for semantic search
  - ChromaDB database creation and population
  - **Output**: Vector database ready for RAG system

#### Model Fine-tuning (`development/model_fine_tuning.ipynb`) - ✅ **Completed** 
- **Purpose**: Fine-tuned Qwen model specifically for RAG conversations
- **Process**:
  - Model configuration for document-based chat
  - Training data preparation and optimization
  - Fine-tuning with conversation-specific objectives
  - Model evaluation and validation
  - **Output**: Optimized model for RAG system

## 🛠️ Technical Stack

- **Language Models**: Qwen (fine-tuned)
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers / OpenAI Embeddings
- **Framework**: Python, Transformers, LangChain
- **Interface**: Jupyter Notebooks, Streamlit (planned)
- **Deployment**: Local development, cloud deployment ready

## 🔮 Future Enhancements

### Planned Features
- **Multi-format Support**: PDF, EPUB, DOCX, and more document types
- **Advanced Interactions**: Follow-up questions, conversation memory
- **Arabic Language Support**: Multilingual capabilities for Arabic documents
- **Web Interface**: Enhanced UI with document management
- **Batch Processing**: Handle multiple documents simultaneously
- **Export Options**: Save conversations and insights

### Technical Improvements
- **Hybrid Search**: Combine semantic and keyword search
- **Dynamic Chunking**: Adaptive text segmentation based on content
- **Model Optimization**: Quantization and efficiency improvements
- **Caching System**: Faster response times for repeated queries

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the documentation in the notebooks
- Review the troubleshooting section below

## 📈 Results and Examples

### Sample Interactions

**User**: "ما الشكل الديموجرافي للمجتمع المصري"

**System**:  العدد والكثافة: أكثر من 106 مليون نسمة (الأكبر عربيًا)، يعيش 95% منهم على 4.5% فقط من المساحة (وادي النيل والدلتا)، مما يخلق كثافة سكانية عالية جدًا.

مجتمع فتي: مجتمع شاب بشكل لافت، حيث أكثر من 60% من السكان تحت سن 30 عامًا. هذه سمة تمثل طاقة بشرية هائلة إذا تم استثمارها، ولكنها تشكل ضغطًا كبيرًا على الخدمات وسوق العمل.

معدل النمو: مرتفع (حوالي 1.7% سنويًا)، مما يعني زيادة سريعة ومستمرة في عدد السكان تلتهم مكاسب التنمية.

التوزيع الجغرافي: تركيز شديد حول النيل واتجاه متزايد للعمرنة (الانتقال إلى المدن)، حيث يعيش نحو 43% من السكان في مناطق حضرية.

التكوين الثقافي: متجانس إلى حد كبير، حيث يشكل المسلمون الغالبية العظمى (90-95%)، ويشكل الأقباط الأرثوذكس معظم النسبة المتبقية.

التحدي الرئيسي: تحويل الطاقة البشرية (الشباب) من عبء إلى عائد ديموغرافي من خلال توفير التعليم الجيد، الرعاية الصحية، وفرص العمل المنتج.