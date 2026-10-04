# DNS ML Learning Lifecycle, Inputs, Outputs and Configuration

## Single feature-engineering rule

Feature engineering is performed once in the Zeek DNS Log Manager workflow.

Shared feature output:
`/data/dns-ml/features/dns_training_features.csv`

The Autoencoder and Isolation Forest trainers consume this CSV directly. They do
not reload merged DNS JSON and do not regenerate the 36-feature matrix.

The GUI displays only the first 20 rows of this feature dataset under Zeek DNS Log Manager.

## Lifecycle

1. Upload `.json`, `.jsonl`, or a JSON-formatted Zeek `dns.log`.
2. Merge and verify all uploaded files.
3. Analyze and extract features once.
4. Before ML starts, `Clear training data` resets the prepared cycle.
5. As soon as either ML model starts, Zeek DNS Log Manager becomes read-only/frozen.
6. It remains frozen after training completes.
7. `Purge All` under either ML section purges both ML families and unlocks Zeek.

## Autoencoder

Input:
`/data/dns-ml/features/dns_training_features.csv`

Trainer/config source:
`/opt/dns-ml/train_dns_autoencoder.py`

Parameters:
- DNS_TRAINING_EPOCHS (50)
- DNS_BATCH_SIZE (256)
- DNS_THRESHOLD_QUANTILE (0.99)
- DNS_MIN_WINDOWS_REQUIRED (50)
- DNS_EXPECTED_FEATURE_COUNT (36)
- DNS_RANDOM_STATE (42)
- DNS_KERAS_VERBOSE (2)

Outputs:
- /data/dns-ml/models/dns_autoencoder.keras
- /data/dns-ml/models/dns_scaler.joblib
- /data/dns-ml/models/dns_autoencoder_config.json
- /data/dns-ml/runs/latest_training_status.json
- /data/dns-ml/runs/latest_training.log
- /data/dns-ml/runs/latest_training_history.csv
- /data/dns-ml/runs/latest_reconstruction_errors.csv
- /data/dns-ml/runs/latest_training_result.json

## Isolation Forest

Input:
`/data/dns-ml/features/dns_training_features.csv`

Trainer/config source:
`/opt/dns-ml/train_dns_isolation_forest.py`

Parameters:
- IF_N_ESTIMATORS (300)
- IF_MAX_SAMPLES (auto)
- IF_CONTAMINATION (0.01)
- IF_RANDOM_STATE (42)
- IF_N_JOBS (-1)
- IF_MIN_WINDOWS (50)
- IF_SHAP_SAMPLE_SIZE (500)
- IF_SILHOUETTE_SAMPLE_SIZE (5000)
- IF_PCA_FIT_SAMPLE_SIZE (50000)
- IF_CONTOUR_GRID_SIZE (70)

Outputs:
- /data/dns-ml/models/dns_isolation_forest.joblib
- /data/dns-ml/models/dns_isolation_forest_scaler.joblib
- /data/dns-ml/models/dns_isolation_forest_pca.joblib
- /data/dns-ml/models/dns_isolation_forest_config.json
- /data/dns-ml/runs/latest_isolation_forest_status.json
- /data/dns-ml/runs/latest_isolation_forest.log
- /data/dns-ml/runs/latest_isolation_forest_result.json
- /data/dns-ml/runs/latest_isolation_forest_scores.csv

## Stop Learning

Stop Learning is enabled while a process is active or its state still says
starting/running/busy. It terminates the process group when alive, clears partial
run/status/chart/table artifacts for that model, and preserves the shared feature
CSV so Start Learning can be used again.

## Purge All

`Purge All` is global even though it appears under both ML sections. It removes:
- both ML model families,
- both ML learning result sets,
- uploaded/prepared training data,
- merged Zeek training artifacts,
- extracted training features.

It does not remove:
- /opt/zeek/logs
- alerts
- scoring cursors/buffers
- scoring configuration
- detector cron configuration
