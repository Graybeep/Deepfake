-- 003_job_items_model_version.sql
--
-- Record which model actually scored each item.
--
-- Migration 002 stopped one item becoming two rows, but ON CONFLICT DO NOTHING
-- only decides that ONE row survives -- not which. The winner is whichever
-- write reached Postgres first, which during a rolling deploy is arbitrary with
-- respect to model version.
--
-- The router made that reachable in a second way: it computes the score from
-- the surviving rows (db.get_items) but takes model_version_id from the queue
-- message, so the two came from different places. With duplicate delivery there
-- are two aggregate messages, write_result is an UPDATE, and the last one wins.
-- Consumer A (v1) could win the rows while consumer B (v2) won the job row --
-- a score computed from v1's numbers, stored as if v2 produced it.
--
-- CLAUDE.md calls the job row the whole audit trail, and a score attributed to
-- weights that did not produce it is exactly the unauditable result that row
-- exists to prevent. Nothing recorded the producer per row, so the mismatch was
-- also undetectable after the fact.
--
-- Nullable: rows written before this migration genuinely have no recorded
-- producer, and backfilling them from the job row would invent provenance that
-- was never observed -- the same lie in a different column.

ALTER TABLE job_items ADD COLUMN IF NOT EXISTS model_version_id TEXT;

-- The router reads the distinct set per job to detect a mixed-version job.
CREATE INDEX IF NOT EXISTS job_items_model_idx
    ON job_items (job_id, model_version_id);
