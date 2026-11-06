# NMFs-CSL pose extraction

This tool reuses `data_prep/wlasl/extract_mmpose.py` inference and person-selection
functions unchanged. It does not change training, validation or test membership.
It preserves all source frames and class IDs. No frame sampling or batching is
introduced. Always retain the frame-repair and readability reports.

## Output and compatibility

Output at `NMFs-CSL/masa_ready/NMFs-CSL`:

```text
Joint_2D_mmpose/<original class>/<original clip>.pkl
jpg_video -> ../../extracted/jpg_video
traintestlist -> ../../downloads/NMFs-CSL
extraction_protocol.json
attempts/
failures-0.jsonl                   # append-only history, not current failure count
summary-0-pilot.json
summary-0-full.json
pose_audit.json                    # produced by final audit
overlays/
```

The actual relative link targets are calculated, not hard-coded. Pickles contain
float32 `[T,133,3]` original-image x/y/confidence and T original image names.
WLASL's extra trailing filename sentinel is removed **only from the name list**:
the existing NMFs loader indexes every name, while WLASL removes the sentinel.
Keypoint values are unchanged. Provenance fields are additional dictionary keys.

This is pose preparation, not certification of the legacy NMFs loader for a new
training study. That loader defaults to a placeholder cache path; use_cache must
be disabled or configured during training integration. It also hard-codes image
dimensions by class group; confirm those against actual images before training.
No train/validation allocation is made here. Validation will be drawn from the
official training clips later, without changing official test membership.

## 1. Plan once on university

Use the pinned pose environment, not the training environment. Before starting,
confirm the three sessions occupy different GPUs/hosts and no other compute job
is using each selected GPU. One process per GPU, with its allocated CPU limits.

```bash
cd /shared/g00107537/jupyter/masa-thesis
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/shared/g00107537/jupyter/environments/mmpose/bin/python
BASE=/shared/g00107537/jupyter/datasets/NMFs-CSL
PLAN="$BASE/pose_plan_3gpu.json"

"$PY" -u data_prep/nmfs_csl/extract_poses.py plan \
  --frames-root "$BASE/extracted/jpg_video" \
  --lists-root "$BASE/downloads/NMFs-CSL" \
  --out-root "$BASE/masa_ready/NMFs-CSL" \
  --frame-report "$BASE/frame_readability.json" \
  --repair-record "$BASE/frame_repair_h_250_3.json" \
  --pose-config /shared/g00107537/jupyter/tools/mmpose_configs/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py \
  --pose-weights /shared/g00107537/jupyter/checkpoints/pose/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
  --shards 3 --manifest "$PLAN"
```

Planning reads 32,010 clip directories, checks consecutive frame names and the
existing full-readability report, and prints progress every 500 clips. It does
not redo 2.64 million image decodes. Allocation uses largest-first greedy frame
balancing; class folders can be split across jobs, but no clip appears twice.
Planning requires the official counts and original class range 0..1066.

Input/config/checkpoint/helper/source/report SHA256s are frozen in the manifest.
Frame names/counts are rechecked per clip; the individual JPEG bytes are **not**
hashed by this tool. Keep frames immutable after the completed readability scan.
Do not edit extraction source or manifest during jobs or between pilot and full
run. Source changes require deliberate new provenance/output, not bypass flags.

## 2. Visual gate: do not launch full extraction yet

```bash
"$PY" -u data_prep/nmfs_csl/extract_poses.py overlay \
  --manifest "$PLAN" --clip 637/P21_02_07_2
"$PY" -u data_prep/nmfs_csl/extract_poses.py overlay \
  --manifest "$PLAN" --clip 250/h_250_3
```

Inspect the saved beginning/middle/end JPEG overlays in both class groups. Dots:
body/feet green, face yellow, left hand cyan, right hand red (confidence >= 0.3
for visualization only; extraction values are not thresholded). Confirm correct
person and meaningful hand, face and body alignment. This gate requires human
inspection, not merely array shapes or positive confidence counts.

The loaded pose and detector tensor hashes, resolved configs, versions, detector
categories and inference behavior are recorded. Three workers must match that
protocol. Automatic detector selection remains the same as WLASL's None/None
arguments; resolved detector details are now recorded, not silently assumed.

## 3. Three-GPU pilot, after overlays are accepted

On each separate one-GPU session repeat the cd/export/PY/BASE/PLAN variables from
step 1. Set **a different** SHARD (0, 1 or 2); do not run all on the same GPU.

```bash
SHARD=0  # second GPU: 1; third GPU: 2
mkdir -p /shared/g00107537/jupyter/logs
nohup "$PY" -u data_prep/nmfs_csl/extract_poses.py run \
  --manifest "$PLAN" --shard "$SHARD" --limit 10 \
  > "/shared/g00107537/jupyter/logs/nmfs_pose_pilot_${SHARD}.log" 2>&1 < /dev/null &
echo "Pilot PID: $!"
```

Pilot results are real outputs and will be reused in the full run. Review each
`summary-<shard>-pilot.json`: zero failures, frame throughput and zero-detection
frames. Inspect warnings/zero detections, and more overlays if needed. Runtime
includes disk reads and writes but excludes model initialization. No promised
speedup; shared-storage contention can limit parallelism. Allow adequate session
wall time. Closing a laptop is safe for nohup, but allocation expiry stops work.

## 4. Full extraction, only after pilot review

On the same three separate GPUs, preserve their 0/1/2 assignment. Drop only the
pilot limit. No changes to config, plan or output root are needed:

```bash
nohup "$PY" -u data_prep/nmfs_csl/extract_poses.py run \
  --manifest "$PLAN" --shard "$SHARD" \
  > "/shared/g00107537/jupyter/logs/nmfs_pose_full_${SHARD}.log" 2>&1 < /dev/null &
echo "Extraction PID: $!"
```

The identical command resumes interrupted work. Existing pickles are checked
for shape, dtype, finiteness, exact frame alignment and extraction provenance
before skipping. A malformed or foreign existing pickle is reported as a failure
and not overwritten: inspect/quarantine it, then rerun that shard. Missing outputs
are inferred again. Process death can leave `.tmp` files; they are never used as
completion indicators. Only complete pickles are atomically renamed into place.
Each shard has an advisory lock; do not run duplicate copies of one shard.
Filesystem lock/rename behavior still requires verification on the university
shared filesystem during the pilot. No claim of bitwise CUDA reproducibility is
made. Inference choices, packages, weights and frame order are preserved.

## 5. Final audit, after all three workers stop

```bash
"$PY" -u data_prep/nmfs_csl/extract_poses.py audit --manifest "$PLAN"
```

Requires all 32,010 outputs and validates them against the plan and resolved
protocol. Reports missing/invalid outputs and zero-detection frames. A structural
pass with some zero-detection frames still needs quality review, not automatic
acceptance for training. Failed-clip logs are append-only historical records;
current summaries and the final audit determine unresolved issues.
