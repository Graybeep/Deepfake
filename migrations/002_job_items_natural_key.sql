-- 002_job_items_natural_key.sql
--
-- Stop one item being scored into two rows.
--
-- Streams reclaim hands a taken-but-unacked message to another consumer once it
-- has been idle past DF_QUEUE_RECLAIM_MS. Idle is not the same as dead: a slow
-- handler is still working when its message is claimed. If both consumers then
-- finish, each writes its own job_items rows for the same frame.
--
-- Measured, not assumed: today that does NOT move the score. rollup.py groups
-- by item_index and rolls up with worst_case (max), which is idempotent under
-- exact duplication -- 17 rows become 34 and the aggregate is unchanged. That
-- is luck, and it holds only while two conditions both happen to be true:
--
--   1. the rollup stays idempotent. Change worst_case to a mean over rows, or
--      let a per-face average in, and every duplicate starts pulling the score.
--   2. both consumers produce the SAME score. They will not during a rolling
--      deploy, where two workers run different model_version_ids -- and then
--      max() takes the higher of the two, silently raising severity toward the
--      >80 band that opens extended retention. Nondeterministic inference does
--      the same thing.
--
-- Independent of the score, duplicate rows corrupt the audit trail: the job row
-- names one model_version_id and one aggregation method, while the item rows
-- show a frame scored twice, possibly by different weights. CLAUDE.md calls
-- that row the whole audit trail, so it has to mean one thing.
--
-- This is not the rare crash window that sweep_stalled_jobs covers. It happens
-- to any handler that is occasionally slower than the reclaim threshold, which
-- is a normal operating condition, not a fault.
--
-- The natural key is (job_id, item_index, face_index): one score per face per
-- frame/chunk. face_index is NULL for audio chunks, and NULL compares distinct
-- from NULL in a unique index by default -- so without NULLS NOT DISTINCT every
-- audio chunk would slip past the constraint while video was protected, which
-- is the worst kind of half-fix. Requires PG15+; compose runs 16.

-- Any duplicate written before this constraint existed keeps its first row.
-- IS NOT DISTINCT FROM so audio rows (face_index NULL) are compared too.
DELETE FROM job_items a
      USING job_items b
      WHERE a.id > b.id
        AND a.job_id = b.job_id
        AND a.item_index = b.item_index
        AND a.face_index IS NOT DISTINCT FROM b.face_index;

CREATE UNIQUE INDEX IF NOT EXISTS job_items_natural_key
    ON job_items (job_id, item_index, face_index) NULLS NOT DISTINCT;
