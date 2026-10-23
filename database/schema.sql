-- ==============================================================================
-- Schema Database: PicMe (Supabase PostgreSQL + pgvector + Auth)
-- Stack: 100% Free-Tier Compatible
-- ==============================================================================

-- 1. Enable pgvector & pgcrypto extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 2. Profiles Table (Linked to Supabase Auth & Role Management)
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    username VARCHAR(64) UNIQUE NOT NULL,
    full_name VARCHAR(128),
    avatar_url TEXT,
    role VARCHAR(32) DEFAULT 'user' CHECK (role IN ('user', 'creator', 'superadmin')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on Profiles
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public profiles are viewable by everyone" 
ON public.profiles FOR SELECT USING (true);

CREATE POLICY "Users can update their own profile" 
ON public.profiles FOR UPDATE USING (auth.uid() = id);

-- 3. Automatic Profile Creation Trigger on Sign-Up (Google / Email)
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, username, full_name, avatar_url, role)
    VALUES (
        new.id,
        COALESCE(new.raw_user_meta_data->>'username', split_part(new.email, '@', 1)),
        COALESCE(new.raw_user_meta_data->>'full_name', new.raw_user_meta_data->>'name', split_part(new.email, '@', 1)),
        new.raw_user_meta_data->>'avatar_url',
        CASE 
            WHEN new.email = 'superadmin@picme.id' OR new.raw_user_meta_data->>'username' = 'superadmin' THEN 'superadmin'
            ELSE 'user'
        END
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- 4. Events Table (Music Festival, Concerts, Marathon, etc.)
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
