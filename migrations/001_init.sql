-- 001_init.sql
-- Core job table. Per CLAUDE.md the job row IS the audit trail: it must carry
-- content hash + model_version_id + aggregation method/params for every result.
--
-- retention_hold lands in this same migration as the delete path (df/retention.py)
-- deliberately: the flag column and the check that reads it ship together, before
-- anything is able to set it true.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS jobs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    media_type          TEXT        NOT NULL
                        CHECK (media_type IN ('video', 'image', 'audio')),
    status              TEXT        NOT NULL DEFAULT 'awaiting_upload'
                        CHECK (status IN ('awaiting_upload', 'queued', 'preprocessing',
                                          'inference', 'aggregating', 'complete',
                                          'failed', 'dead_letter')),

    -- ---- audit trail (Tier 1) ----
    content_hash        TEXT,           -- sha256 of the raw upload, computed server-side
    model_version_id    TEXT,           -- which weights actually produced the scores
    aggregation_method  TEXT,           -- e.g. 'weighted_trimmed_mean'
    aggregation_params  JSONB,          -- exact params used for this row's score

    -- ---- result ----
    result_class        TEXT
                        CHECK (result_class IN ('authentic', 'manipulated',
                                                'uncertain', 'undetermined')),
    band                TEXT,           -- routing band label, see df/bands.py
    aggregate_score     DOUBLE PRECISION,   -- 0-100, aggregated (never a raw item score)
    item_count          INTEGER,        -- frames / chunks scored
    face_count          INTEGER,        -- faces detected (0 => undetermined)

    -- ---- storage ----
    raw_object_key      TEXT,           -- the uploaded original
    derived_prefix      TEXT,           -- face crops / spectrograms live under here
    raw_deleted_at      TIMESTAMPTZ,
    derived_deleted_at  TIMESTAMPTZ,

    -- ---- retention ----
    -- Hold flag. Every delete path checks this column first (df/retention.py):
    -- the completion-triggered delete, the crash-recovery sweeper, AND the
    -- cold-storage expiry. No exceptions.
    retention_hold      BOOLEAN     NOT NULL DEFAULT FALSE,
    hold_set_at         TIMESTAMPTZ,
    hold_reason         TEXT,
    -- Tier 2 "extended retention window": a FIXED 30-day timer that auto-expires.
    -- This is NOT a legal hold and must never be described as one.
    extended_retention_until TIMESTAMPTZ,
    -- Cold storage holding the face crops / chunks that DROVE a flagged score.
    -- The window protects this media, not merely the job row -- otherwise the
    -- >80 branch would preserve nothing a dispute could use.
    cold_prefix         TEXT,
    cold_deleted_at     TIMESTAMPTZ,

    -- ---- queue bookkeeping ----
    attempts            INTEGER     NOT NULL DEFAULT 0,
    error               TEXT,

    submitted_by        TEXT,           -- API key id / principal, for rate-limit forensics
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS jobs_status_idx  ON jobs (status);
CREATE INDEX IF NOT EXISTS jobs_created_idx ON jobs (created_at DESC);
CREATE INDEX IF NOT EXISTS jobs_hash_idx    ON jobs (content_hash);
-- Sweeper looks for rows still holding bytes past their retention decision.
CREATE INDEX IF NOT EXISTS jobs_undeleted_idx
    ON jobs (completed_at)
    WHERE raw_deleted_at IS NULL AND retention_hold = FALSE;


-- Per-frame / per-chunk / per-face scores. Kept after raw media is deleted:
-- these are numbers, not biometric media.
CREATE TABLE IF NOT EXISTS job_items (
    id              BIGSERIAL PRIMARY KEY,
    job_id          UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    item_index      INTEGER NOT NULL,       -- frame index / chunk index
    item_kind       TEXT    NOT NULL CHECK (item_kind IN ('frame', 'chunk', 'image')),
    face_index      INTEGER,                -- NULL for audio chunks
    score           DOUBLE PRECISION NOT NULL,     -- 0-100, raw model output (calibrated)
    confidence      DOUBLE PRECISION NOT NULL,     -- detection/alignment confidence -> weight
    -- Where this item's crop/spectrogram lives under derived/. Recorded so the
    -- router can preserve exactly the objects that drove a flagged score
    -- instead of reconstructing key names by convention.
    object_key      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS job_items_job_idx ON job_items (job_id, item_index);

-- Sweeper looks for windows whose fixed timer has run out. A timer nothing
-- enforces quietly becomes "retained forever", which is its own liability.
CREATE INDEX IF NOT EXISTS jobs_window_expiry_idx
    ON jobs (extended_retention_until)
    WHERE cold_prefix IS NOT NULL AND cold_deleted_at IS NULL AND retention_hold = FALSE;


-- Append-only event log. Retention/deletion decisions land here so a deleted
-- job can still prove *that* it was deleted and why.
CREATE TABLE IF NOT EXISTS job_events (
    id          BIGSERIAL PRIMARY KEY,
    job_id      UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    event       TEXT NOT NULL,
    detail      JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS job_events_job_idx ON job_events (job_id, created_at);


-- Tier 3 substitute for the human-in-the-loop dashboard: a DB flag + an alert.
-- Rows here are what the Slack/email notifier reads.
CREATE TABLE IF NOT EXISTS review_flags (
    id           BIGSERIAL PRIMARY KEY,
    job_id       UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    reason       TEXT NOT NULL,
    -- 'low' is the 60-80 band: recorded and alerted so it is never a silent
    -- pass-through into normal deletion, without paging anyone.
    urgency      TEXT NOT NULL DEFAULT 'normal' CHECK (urgency IN ('low', 'normal')),
    notified_at  TIMESTAMPTZ,
    resolved_at  TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS review_flags_open_idx
    ON review_flags (created_at) WHERE resolved_at IS NULL;
