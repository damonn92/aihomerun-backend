-- Supabase SQL Migration: analyses + analysis_jobs tables
-- Run in Supabase Dashboard > SQL Editor

-- ── Analyses table ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    action_type TEXT NOT NULL CHECK (action_type IN ('swing', 'pitch')),
    overall_score INTEGER NOT NULL,
    technique_score INTEGER NOT NULL,
    power_score INTEGER NOT NULL,
    balance_score INTEGER NOT NULL,
    peak_wrist_speed REAL,
    hip_shoulder_separation REAL,
    balance_metric REAL,
    follow_through BOOLEAN DEFAULT FALSE,
    plain_summary TEXT,
    video_id TEXT,
    video_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_analyses_user_action ON analyses(user_id, action_type);
CREATE INDEX IF NOT EXISTS idx_analyses_user_score ON analyses(user_id, overall_score DESC);

ALTER TABLE analyses ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users read own analyses"
    ON analyses FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Service role can insert analyses"
    ON analyses FOR INSERT
    WITH CHECK (TRUE);

CREATE POLICY "Users delete own analyses"
    ON analyses FOR DELETE
    USING (auth.uid() = user_id);


-- ── Analysis jobs table (for async processing) ─────────────────────────────

CREATE TABLE IF NOT EXISTS analysis_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    action_type TEXT NOT NULL,
    age INTEGER NOT NULL,
    video_storage_key TEXT,
    analysis_id UUID REFERENCES analyses(id),
    error_message TEXT,
    progress INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_jobs_user ON analysis_jobs(user_id, status);
CREATE INDEX IF NOT EXISTS idx_jobs_pending ON analysis_jobs(status)
    WHERE status IN ('pending', 'processing');

ALTER TABLE analysis_jobs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users read own jobs"
    ON analysis_jobs FOR SELECT
    USING (auth.uid() = user_id);


-- ── Leaderboard RPC function ────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION leaderboard_top20()
RETURNS TABLE (user_id UUID, score INTEGER, display_name TEXT)
LANGUAGE sql STABLE
AS $$
    SELECT
        a.user_id,
        MAX(a.overall_score)::INTEGER AS score,
        COALESCE(p.full_name, '') AS display_name
    FROM analyses a
    LEFT JOIN profiles p ON p.id = a.user_id
    GROUP BY a.user_id, p.full_name
    ORDER BY score DESC
    LIMIT 20;
$$;
