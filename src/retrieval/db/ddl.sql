-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create video embeddings table
CREATE TABLE IF NOT EXISTS video_embeddings (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(255) NOT NULL UNIQUE,
    embedding vector(768) NOT NULL,
    summary_ground_truth TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS video_embeddings_embedding_idx
ON video_embeddings USING ivfflat (embedding vector_cosine_ops);
