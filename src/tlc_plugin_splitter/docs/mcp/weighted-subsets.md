# Weighted training subsets

Use `splitter_weighted_subset` for a smaller training selection that keeps revision history.
Read `table_weight_summary` on the run's exact training input first. A table with 29,537 rows
and 5,000 positive weights is a 5k selection. Keeping 3,000 zeroes 2,000 additional weights,
not 26,537. Existing zero weights stay zero and retained positive weights are unchanged.
The job writes an EditedTable beside its source with a unique name and source lineage;
it does not copy images, delete rows, overwrite the source, or follow latest().
The default is uniform random selection without replacement with seed 42. It is not
class stratified. Ask for a selection criterion only if the task depends on one.
Use its returned table_url for training, keep the validation revision and recorded trainer
settings, and ensure zero-weight exclusion is enabled. No GPU is needed to prepare the subset.
