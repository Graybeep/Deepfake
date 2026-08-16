-- 004_jobs_items_unattributed.sql
--
-- Put the attribution caveat on the audit row itself.
--
-- Migration 003 recorded which model scored each item, and the router flags a
-- job whose rows are only partly attributed. But the flag lands in
-- review_flags, which is operational: it exists to get someone's attention now.
-- The permanent record is the jobs row, and that row still asserted
-- model_version_id with no qualifier at all.
--
-- Those are different audiences with different lifetimes. A dispute review
-- months later pulls the job row; it does not replay an alert log or trawl a
-- side table for an open flag, and a flag nobody watched does not change what
-- the permanent record claims. CLAUDE.md is explicit that the job row IS the
-- audit trail -- so a job whose evidence is partly unattributed has to say so
-- on that row, not merely somewhere adjacent to it.
--
-- Lifetime matters too, and the earlier framing of this as "confined to the
-- migration-003 deploy window" was wrong. The window bounds when NEW straddling
-- jobs stop being created. It says nothing about how long the ones already
-- written stay under-evidenced: job rows persist indefinitely as the audit
-- record, which is the entire point of Tier 1, so every straddling job from
-- that window would keep overstating its own attribution for the life of the
-- row.
--
-- Nullable, and deliberately not defaulted to 0: rows written before this
-- migration were never measured, and 0 would assert "all items attributed",
-- which is precisely the unverified claim this column exists to stop. NULL
-- means not measured; 0 means measured and complete.

ALTER TABLE jobs ADD COLUMN IF NOT EXISTS items_unattributed INTEGER;

COMMENT ON COLUMN jobs.items_unattributed IS
    'Item rows with no recorded model_version_id at decision time. '
    'NULL = never measured (pre-004 row). 0 = measured, fully attributed.';
