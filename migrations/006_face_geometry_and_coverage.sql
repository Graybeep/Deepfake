-- 006_face_geometry_and_coverage.sql
--
-- Record the two things the verdict was reduced from, so the reduction stops
-- being the only artifact.
--
-- Two separate gaps, same shape: a decision was baked down to a scalar and the
-- inputs to it were thrown away, which makes the decision impossible to revisit
-- without re-running every job.
--
--
-- 1. job_items.face_w / face_h -- face geometry
--
-- The multi-face rule rolls N faces up to worst-case severity, so one
-- high-scoring background face sets the label for a whole crowd scene. The
-- obvious refinement is a per-face bar that varies with face size and quality
-- rather than a global constant, fitted so the false-positive rate is flat
-- across size buckets.
--
-- That cannot be fitted here, and the reason is not only the missing labels.
-- `FaceCrop.bbox` has existed since the first commit and is populated by
-- OpenCVFaceExtractor -- and it was dropped on the floor in the CPU worker,
-- never entering the manifest, never reaching job_items. So face size was not
-- merely un-thresholded, it was unmeasurable after the fact: across every job
-- this system has ever run there is no record of how big any face was.
--
-- Storing it does not fit a threshold and does not pretend to. It stops the
-- next 500 jobs from being as unanalysable as the last 500, which is the part
-- that gets more expensive the longer it waits.
--
-- Absolute pixels, not relative-to-frame. Relative area is the better bucketing
-- feature and it is NOT available: source frame dimensions are not recorded
-- anywhere either. Flagged rather than approximated -- deriving relative size
-- from an assumed frame size would be a fabricated feature, and a fitted
-- threshold is only as honest as the feature under it.
--
-- NULL means not recorded: a pre-006 row, or an item with no face at all
-- (audio chunks, and the undetermined path where extraction returned nothing).
-- Not defaulted to 0, which would assert a zero-pixel face was observed.

ALTER TABLE job_items ADD COLUMN IF NOT EXISTS face_w INTEGER;
ALTER TABLE job_items ADD COLUMN IF NOT EXISTS face_h INTEGER;

COMMENT ON COLUMN job_items.face_w IS
    'Detected face width in pixels of the source frame. NULL = not recorded '
    '(pre-006 row, or a non-face item). Absolute, not relative to frame: '
    'frame dimensions are not stored, so relative area cannot be derived.';

--
-- 2. jobs.items_total -- coverage
--
-- The job row records item_count, which is items USED after low-confidence
-- drops and trimming. It does not record how many there were to begin with, so
-- a verdict off 1 usable frame of 50 and a verdict off 50 of 50 are the same
-- row. The consumer cannot tell a well-covered result from a thin one, which
-- is what forces the minimum-items gate to be load-bearing: with no way to
-- express "scored, but barely", the only way to protect the reader is to
-- refuse to answer.
--
-- With coverage on the row the gate stops being the only defence and a
-- downstream consumer can set its own bar -- which is the point, because this
-- codebase is in no position to set a defensible one (see below).
--
-- NULL = never measured (pre-006 row). Distinct from 0, which cannot occur:
-- a job with zero items takes the undetermined path and stores no score.

ALTER TABLE jobs ADD COLUMN IF NOT EXISTS items_total INTEGER;

COMMENT ON COLUMN jobs.items_total IS
    'Items entering aggregation, before confidence drops and trimming. '
    'item_count is what survived. coverage = item_count / items_total. '
    'NULL = never measured (pre-006 row).';

-- Bucketing face size is the intended use of face_w/face_h. The index is not
-- created here: there is no query yet, the fitting work is blocked on labelled
-- validation data that does not exist, and an index built for a query nobody
-- writes is a cost with no reader. Add it with the query.
