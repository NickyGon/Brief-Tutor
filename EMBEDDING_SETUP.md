# Embedding Model Configuration Guide

## Current Setup

The system now uses **OpenAI embeddings** consistently for both ingestion and retrieval. This ensures that:
- Documents are embedded using the same model they'll be searched with
- Search quality is optimal
- No embedding mismatches occur

## Embedding Models Available

### Option 1: `text-embedding-3-large` (Default - Recommended for Better Quality)
- **Dimensions**: 3,072
- **Token Limit**: 8,192 tokens per input
- **Cost**: $0.13 per million tokens
- **Best for**: Highest quality embeddings, complex PDFs with technical content, better semantic understanding
- **Status**: ✅ Currently set as default

### Option 2: `text-embedding-3-small` (Alternative for Cost Efficiency)
- **Dimensions**: 1,536
- **Token Limit**: 8,192 tokens per input
- **Cost**: $0.02 per million tokens (6.5x cheaper)
- **Best for**: Cost-conscious deployments, simpler documents, high-volume use cases

## Configuration

The system defaults to `text-embedding-3-large` for optimal quality. You can override this via environment variable:

```bash
# In your .env file or environment (optional - defaults to text-embedding-3-large):
EMBEDDING_MODEL=text-embedding-3-large  # Default
# or
EMBEDDING_MODEL=text-embedding-3-small  # For cost savings
```

The vector size will automatically adjust:
- `text-embedding-3-large` → 3072 dimensions (default)
- `text-embedding-3-small` → 1536 dimensions

You can also override the vector size manually:
```bash
RAG_VECTOR_SIZE=1536  # or 3072
```

## Chunk Size for PDFs

Current chunk size: **1000 characters** (approximately 250-400 tokens)

This is well within the 8,192 token limit, so you have plenty of room to increase it if needed.

### When to Increase Chunk Size

If your PDFs contain:
- Long paragraphs that shouldn't be split
- Technical documentation with code blocks
- Tables or structured data
- Complex multi-sentence explanations

You can increase chunk size:

```bash
# In your .env file:
RAG_CHUNK_SIZE_CHARS=2000  # Double the current size (still well under token limit)
RAG_CHUNK_OVERLAP_CHARS=400  # Increase overlap proportionally
```

### Token Estimation

- 1,000 characters ≈ 250-400 tokens (depending on content)
- 2,000 characters ≈ 500-800 tokens
- 4,000 characters ≈ 1,000-1,600 tokens

All well within the 8,192 token limit!

## PDF Processing

PDFs are extracted using `pypdf`, which handles:
- Text extraction from all pages
- Basic formatting preservation
- Multi-page documents

The extracted text is then:
1. Chunked into manageable pieces (default 1000 chars)
2. Embedded using the selected embedding model
3. Stored in Qdrant for retrieval

## Recommendations

### Default Configuration (Optimal Quality):
```bash
# No configuration needed - uses text-embedding-3-large by default
RAG_CHUNK_SIZE_CHARS=1000
RAG_CHUNK_OVERLAP_CHARS=200
```

### For Complex PDFs with Technical Content:
```bash
# Uses text-embedding-3-large by default, just increase chunk size
RAG_CHUNK_SIZE_CHARS=2000
RAG_CHUNK_OVERLAP_CHARS=400
```

### For Cost-Conscious Deployments:
```bash
EMBEDDING_MODEL=text-embedding-3-small  # Override default
RAG_CHUNK_SIZE_CHARS=1500  # Slightly larger chunks, fewer total chunks
RAG_CHUNK_OVERLAP_CHARS=300
```

## Migration Notes

If you have existing data in Qdrant:
- **You'll need to re-index** your documents after changing the embedding model
- The vector dimensions must match between ingestion and retrieval
- Old embeddings won't work with a new model

To re-index:
1. Delete the old collection in Qdrant (or use a new collection name)
2. Run the ingestion script again with the new embedding model
3. The new embeddings will be stored with the correct dimensions

## Troubleshooting

### "Dimension mismatch" errors
- Ensure `EMBEDDING_MODEL` is the same for ingestion and retrieval
- Check that `RAG_VECTOR_SIZE` matches the model's output dimensions
- Re-index if you changed models

### Poor search results
- Try `text-embedding-3-large` for better quality
- Increase chunk size to preserve more context
- Check that PDFs are being extracted correctly

### High costs
- Use `text-embedding-3-small` instead of `-large`
- Reduce chunk size (fewer chunks = fewer API calls)
- Consider caching embeddings for unchanged documents
