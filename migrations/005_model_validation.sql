-- 005_model_validation.sql
--
-- Record how far the weights behind a score may be trusted.
--
-- Until now the caveat on a result was produced by matching the substring
-- "stub" against model_version_id in the gateway. That check fails OPEN: load
-- any real checkpoint and the id becomes something like
-- face-efficientnet_b4-<hash>, the substring is gone, and every advisory
-- disappears from every result silently -- at exactly the moment scores start
-- looking plausible enough to be believed. A caveat that vanishes when the
-- thing it warns about gets more convincing is worse than no caveat.
--
-- It also conflated two different questions. is_real_detector asks whether a
-- model ran at all. Validation asks whether its output means anything here. A
-- public research checkpoint is a real detector AND is not validated: trained
-- on someone else's distribution, thresholded for someone else's task, never
-- measured against this pipeline's bands. One boolean cannot say that.
--
-- Stored on job_items as well as jobs, and derived by the router from the rows
-- that actually produced the score -- the same rule migration 003 established
-- for model_version_id. The message is hearsay; the rows are the evidence.
--
-- Nullable on purpose. Rows written before this migration have no observed
-- validation level, and defaulting them to any value would assert something
-- nobody measured. The gateway treats NULL as the STRONGEST caveat, not the
-- absence of one: unknown provenance is a reason to warn, not a reason to stay
-- quiet.

ALTER TABLE job_items ADD COLUMN IF NOT EXISTS model_validation TEXT;
ALTER TABLE jobs      ADD COLUMN IF NOT EXISTS model_validation TEXT;

COMMENT ON COLUMN jobs.model_validation IS
    'placeholder | research-checkpoint | production-validated. '
    'NULL = never recorded; callers must treat NULL as untrusted.';
