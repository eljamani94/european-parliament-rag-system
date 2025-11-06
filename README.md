# RAG System for European Parliament Documents

A production-ready Retrieval Augmented Generation (RAG) system for querying European Parliament plenary session transcripts using semantic search and LLM-powered question answering.

## Overview

This project implements a RAG pipeline that enables natural language queries over European Parliament transcripts. Instead of manually searching through hundreds of pages of parliamentary proceedings, users can ask specific questions and receive contextually relevant, source-grounded answers.

### Key Features

- **Semantic Search**: Embedding-based retrieval that understands context beyond keyword matching
- **Persistent Storage**: Chroma vector database for efficient, scalable document storage
- **Temporal Filtering**: Query specific dates, months, or years of parliamentary sessions
- **Modular Architecture**: Reusable functions for easy corpus expansion
- **Cost Optimized**: Embeddings computed once and reused across sessions

## Technical Stack

- **LLM**: Google Gemini 2.0 Flash
- **Embeddings**: Google text-embedding-004 (768 dimensions)
- **Vector Store**: Chroma (local persistent storage)
- **Framework**: LangChain
- **Document Processing**: PyPDF, RecursiveCharacterTextSplitter

## Project Structure
```
.
├── rag_notebook.ipynb          # Main notebook with RAG implementation
├── data/                        # European Parliament PDF transcripts
│   ├── CRE-10-2025-05-05_EN.pdf
│   └── CRE-10-2025-05-06_EN.pdf
├── chroma_ep_follower/          # Persistent vector store (generated)
├── .env                         # API keys (not committed)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Prerequisites

- Python 3.12+
- Google AI API key (for Gemini and embeddings)

## Installation

1. **Clone the repository**
```bash
   git clone https://github.com/eljamani94/european-parliament-rag-system.git
   cd ep-rag-system
```

2. **Create and activate virtual environment**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
```
   GOOGLE_API_KEY=your_google_api_key_here
```

   Get your API key from: https://aistudio.google.com/app/apikey

5. **Create data directory**
```bash
   mkdir data
```

## Usage

### Quick Start

Open and run the Jupyter notebook:
```bash
jupyter notebook rag_notebook.ipynb
```

The notebook walks through the complete pipeline:
1. Document loading and preprocessing
2. Text chunking with overlap
3. Embedding generation and vector storage
4. Semantic retrieval
5. LLM-powered answer generation

### Core Functions

#### Adding Documents to the System
```python
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize components
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = Chroma(
    collection_name="ep_plenary",
    embedding_function=embeddings,
    persist_directory="./chroma_ep_follower"
)

# Add a new transcript
document_ids = embed_and_store_fancy(
    file_path="data/CRE-10-2025-05-05_EN.pdf",
    vector_store=vector_store,
    session_date="2025-05-05"
)
```

#### Querying the System
```python
from langchain.chat_models import init_chat_model

# Initialize LLM
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# Ask a question
query = "What was discussed about agricultural policy?"
answer = answer_fancy(
    query=query,
    vector_store=vector_store,
    llm=llm,
    session_date="2025-05-05"  # Optional: filter by date
)

print(answer)
```

#### Temporal Filtering

Query specific time periods:
```python
# Query specific date
answer_fancy(query, vector_store, llm, session_date="2025-05-05")

# Query specific month
answer_fancy(query, vector_store, llm, session_month="2025-05")

# Query specific year
answer_fancy(query, vector_store, llm, session_year="2025")

# Query all documents
answer_fancy(query, vector_store, llm)
```

## How It Works

### 1. Document Processing
- PDFs are loaded and split into 2,000-character chunks with 400-character overlap
- Overlap preserves context across chunk boundaries
- Metadata (date, year, month) is added to each chunk

### 2. Embedding & Storage
- Each chunk is converted to a 768-dimensional vector using Google's embedding model
- Vectors and metadata are stored in Chroma's persistent database
- Embeddings are computed once and reused

### 3. Retrieval
- User queries are embedded using the same model
- Vector similarity search retrieves the top-k most relevant chunks
- Optional metadata filtering narrows results by date

### 4. Generation
- Retrieved chunks are concatenated into context
- A structured prompt template guides the LLM
- The LLM generates an answer grounded in the provided context

## Example Queries
```python
# Policy discussions
"Summarize the discussion on agricultural policy."
"What was said about international trade?"

# Specific topics
"Who spoke about climate change and what were their main points?"
"What amendments were proposed regarding digital regulation?"

# Procedural questions
"What votes were taken during this session?"
"Which committees presented reports?"
```

## Configuration

### Chunk Size & Overlap

Adjust in the text splitter configuration:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2_000,      # Adjust based on your needs
    chunk_overlap=400,     # 20% overlap recommended
    add_start_index=True,
)
```

### Retrieval Parameters

Modify number of chunks retrieved:
```python
retrieved_docs = vector_store.similarity_search(query, k=6)  # Adjust k
```

### LLM Model

Switch to different Gemini models:
```python
llm = init_chat_model("gemini-1.5-pro", model_provider="google_genai")
```

## Data Sources

European Parliament plenary session transcripts are available at:
https://www.europarl.europa.eu/plenary/en/debates-video.html

Download verbatim reports in PDF format and place them in the `data/` directory.

## Performance Considerations

- **Embedding Cost**: First-time embedding of documents incurs API costs. Subsequent queries use cached embeddings.
- **Retrieval Speed**: Chroma provides fast similarity search even with large corpora.
- **Context Window**: The system retrieves 6 chunks (~12,000 characters) by default, well within Gemini's context limits.

## Limitations

- Currently supports English transcripts only
- Retrieval quality depends on chunk size and overlap settings
- Answers are limited to information in the indexed documents
- No cross-document reasoning (answers are based on retrieved chunks)

## Future Enhancements

- [ ] Batch processing for historical sessions
- [ ] Hybrid search (semantic + keyword)
- [ ] Citation tracking with source page numbers
- [ ] Web interface for non-technical users
- [ ] Multi-language support
- [ ] Document summarization for long contexts
- [ ] Query result caching

## Troubleshooting

### LangSmith Warnings
Warnings about missing LangSmith API keys can be safely ignored. Suppress with:
```python
os.environ["LANGCHAIN_TRACING_V2"] = "false"
```

### Chroma Telemetry Warnings
Telemetry errors don't affect functionality. Disable with:
```python
os.environ["ANONYMIZED_TELEMETRY"] = "False"
```

### API Rate Limits
If you hit rate limits, add delays between embedding operations or use batch processing.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a clear description

## License

MIT License - see LICENSE file for details

## Acknowledgments

- European Parliament for providing open access to transcripts
- LangChain for the RAG framework
- Google for Gemini and embedding models
- Chroma for the vector database

## Contact

For questions or issues, please open a GitHub issue or contact [eljamani.aej@gmail.com]

---

**Note**: This is a research/educational project. For production use with sensitive data, implement appropriate security measures and comply with data protection regulations.
