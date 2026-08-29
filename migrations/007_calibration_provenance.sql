-- 007_calibration_provenance.sql
--
-- Record which calibration produced a score.
--
-- CLAUDE.md: the job row is the whole audit trail, and "anything qualifying the
-- result belongs on this row". How a raw logit became the 0-100 number a reader
-- is looking at qualifies it about as directly as anything can -- and it was
-- recorded nowhere at all. `ModelVersion.calibration` existed, was set on both
-- backends, and was never written to a column, never returned by the API, and
-- never read by anything. The one honest field, `Temperature.fitted_on`, was
-- discarded entirely.
--
-- Worse, the value being carried was wrong. It was the constant
-- "temperature.v1:launch-snapshot", stamped onto every torch-backend result
-- while T was 1.0 and `fitted_on` said "NOT YET FITTED". So the field a reader
-- would consult to find out whether a calibration snapshot had been taken
-- asserted that one had. It is now derived from whether a fit actually
-- happened.
--
--
-- WHY THIS IS PER-ITEM AND NOT JUST A JOB COLUMN
--
-- Same rule migration 003 established for model_version_id, for the same
-- reason: the rows that produced the score are the evidence, the queue message
-- is hearsay. The router derives the job-level value from the item rows and
-- refuses a job whose rows disagree.
--
-- It matters more here than it looks. `model_version_id` is derived from the
-- weights sha256, so it does NOT change when only the temperature changes:
-- refit T, ship it, and every new job carries the same model_version_id as
-- every old one while computing a materially different score from the same
-- logit. Without a separate calibration column, two rows that are identical in
-- every recorded field would be incomparable and nothing would say so. This is
-- the column that makes a re-calibration visible after the fact.
--
-- Nullable, and deliberately not defaulted: rows written before this migration
-- were produced under a calibration nobody recorded. NULL means exactly that.
-- Backfilling them with today's scheme string would invent provenance -- and
-- would be inventing it for precisely the rows written while the scheme string
-- was lying.

ALTER TABLE job_items ADD COLUMN IF NOT EXISTS calibration TEXT;
ALTER TABLE jobs      ADD COLUMN IF NOT EXISTS calibration TEXT;

COMMENT ON COLUMN job_items.calibration IS
    'Calibration scheme that produced this item score, e.g. '
    '"temperature.v1:unfitted" or "temperature.v1:launch-snapshot". '
    'NULL = not recorded (pre-007 row).';

COMMENT ON COLUMN jobs.calibration IS
    'Calibration rolled up from the item rows that produced the score, by the '
    'same rule as model_version_id. NULL = never measured (pre-007 row). '
    'Note that model_version_id is keyed on the weights hash alone, so it does '
    'not move when only the temperature is refitted -- this column is what '
    'distinguishes those results.';

-- The router reads the distinct set per job to detect a job whose rows were
-- scored under two different calibrations, exactly as it does for model
-- version. Same shape of index, same access pattern.
CREATE INDEX IF NOT EXISTS job_items_calibration_idx
    ON job_items (job_id, calibration);
