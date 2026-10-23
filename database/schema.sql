-- ==============================================================================
-- Schema Database: PixMe (Supabase PostgreSQL + pgvector)
-- Stack: 100% Free-Tier Compatible
-- ==============================================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Events Table (Marathon, Cycling, Graduation, etc.)
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    slug VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    location TEXT NOT NULL,
    event_date DATE NOT NULL,
    cover_image_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Photos Table (Published Photos in Event)
CREATE TABLE IF NOT EXISTS photos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_url TEXT NOT NULL,          -- Full resolution original
    thumbnail_url TEXT NOT NULL,     -- Compressed thumbnail
    phash VARCHAR(64),               -- 64-bit perceptual hash (anti-duplicate)
    width INT,
    height INT,
    size_bytes BIGINT,
    faces_detected INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Face Embeddings Table (Vector 128-d from SFace ONNX)
CREATE TABLE IF NOT EXISTS face_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    photo_id UUID NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    event_id UUID NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    embedding VECTOR(128) NOT NULL,
    bbox_x INT NOT NULL,
    bbox_y INT NOT NULL,
    bbox_w INT NOT NULL,
    bbox_h INT NOT NULL,
    confidence FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 5. Fast Indexing for Events & Photo Lookups
CREATE INDEX IF NOT EXISTS idx_photos_event_id ON photos(event_id);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_event_id ON face_embeddings(event_id);
CREATE INDEX IF NOT EXISTS idx_photos_phash ON photos(phash);

-- 6. HNSW Cosine Vector Index for Sub-Millisecond Face Search
CREATE INDEX IF NOT EXISTS idx_face_embeddings_vector 
ON face_embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 7. SQL Helper Function: Match Faces by Event
CREATE OR REPLACE FUNCTION match_event_faces(
    query_embedding VECTOR(128),
    target_event_id UUID,
    match_threshold FLOAT DEFAULT 0.40,
    match_count INT DEFAULT 50
)
RETURNS TABLE (
    photo_id UUID,
    file_url TEXT,
    thumbnail_url TEXT,
    similarity FLOAT
)
LANGUAGE sql STABLE AS $$
    SELECT 
        p.id AS photo_id,
        p.file_url,
        p.thumbnail_url,
        (1 - (fe.embedding <=> query_embedding)) AS similarity
    FROM face_embeddings fe
    JOIN photos p ON fe.photo_id = p.id
    WHERE fe.event_id = target_event_id
      AND (fe.embedding <=> query_embedding) < match_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
$$;
