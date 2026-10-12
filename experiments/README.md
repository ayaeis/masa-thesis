# Compression Study

## Scope

Datasets: WLASL100, WLASL300, MSASL100, MSASL200, MSASL1000.

| Result | Post-training | Architecture-first |
| --- | --- | --- |
| Baseline | Pretraining -> dense task fine-tuning, 60 epochs | Shared reference and KD teacher |
| Baseline + quantization | Quantize the baseline | Shared reference |
| Ghost | Baseline -> Ghost, all kernel sizes, ratio 2 -> 60 epochs | Pretraining -> Ghost -> two 60-epoch stages |
| Ghost + quantization | Quantize FP32 Ghost before KD | Quantize FP32 Ghost before KD |
| Ghost + KD | FP32 Ghost -> 60 KD epochs | FP32 Ghost after stage 2 -> 60 KD epochs |
| Ghost + KD + quantization | Quantize KD-Ghost | Quantize KD-Ghost |
| Low-rank | Baseline -> all eligible target layers, r=0.125 -> 60 epochs | Pretraining -> low-rank -> two 60-epoch stages |
| Low-rank + quantization | Quantize low-rank | Quantize low-rank |

Both KD paths use the same dataset's fine-tuned dense baseline as teacher.
KD starts from FP32 Ghost, independently of the quantization branch.
There is no low-rank KD in the final study. Low-rank targets are
`transformer,conv,project_head`, with `low_rank_min_features=64`.

Architecture-first stage 2 restores the stage-1 validation-best checkpoint and
starts a fresh optimizer and 60-epoch learning-rate schedule. It restores the
existing low-rank factors rather than factorizing them again. This matches the
earlier MSASL100 architecture-first protocol.

The budgets are 60+60 scheduled epochs before KD and another 60 for Ghost KD.
Because each stage selects its validation-best checkpoint, the final model does
not necessarily descend from the last epoch of every earlier stage. Report stage
budgets and selected epochs separately. Architecture-first KD additionally needs
the separately trained dense teacher; record that shared training cost.

## Training and Evaluation Protocol

The settings agreed for each 60-epoch stage are:

| Setting | Value |
| --- | --- |
| Training batch size | 64 |
| Workers | 8; must be reviewed against university CPU allocation before launch |
| Optimizer | SGD |
| Momentum | 0.9 |
| Initial learning rate | 0.01 |
| Weight decay | 0.0 |
| Scheduler | MultiStepLR, milestones 20 and 40, gamma 0.1 |
| Target frames | 32 |
| Temporal sampling | index |
| Freeze / warmup epochs | 0 / 0 |
| Head learning-rate multiplier | 1.0 |
| Training temporal crop minimum | 1.0 |
| Patience / min delta | 999 / 0.0 |
| Seed | 123 |
| KD alpha / temperature | 0.5 / 4.0 |
| Checkpoint selection | Validation accuracy; evaluate the selected checkpoint on test |

Metrics: test accuracy, parameter count, latency, model size, and FLOPs.
Accuracy differences use percentage points; other differences use relative
percentages. Keep batch size, input shape, precision, and measurement units with
each report. Earlier evaluators use batch size 32 and five warmup steps.
Quantization reports describe supported INT8 layers with FP16 fallback; do not
label the entire model INT8.

Use the same benchmark procedure on an otherwise idle GPU for all latency
comparisons. Concurrent training jobs should use separate GPU allocations and
must not overlap latency measurements on the same GPU.

## University Workspace

```text
/shared/g00107537/jupyter/
    masa-thesis/
    datasets/
        WLASL/latest_full_wlasl/
            official_wlasl100/masa_ready/WLASL/
            official_wlasl300/masa_ready/WLASL/
            raw_2000/
        MSASL/native_masa_ready/
            MSASL/
            MSASL1000/
    checkpoints/pretrained_model.pth.tar
    environments/
    results/
    logs/
```

WLASL2000 is outside the five-dataset study. Its raw videos are retained for
future preprocessing. MSASL100 and MSASL200 use class subsets of the same prepared
`MSASL` tree; MSASL1000 uses its own archive loader and recorded exclusions.

Preserve relative links between the prepared trees. Absolute symlinks referring
to `/workspace` need repair after relocation. A matching rsync listing does not
prove that symlinks resolve at their destination.

## Code Layout

| Directory | Purpose |
| --- | --- |
| `training/wlasl/` | WLASL supervised, Ghost KD, and low-rank trainers |
| `training/msasl/` | MSASL trainers, including low-rank continuation |
| `moco/` | MASA architecture and compression modules |
| `feeder/` | Base data loading and preprocessing |
| `data_prep/` | Dataset conversion, archive loaders, and validation |
| `analysis/` | Evaluation, quantization, audits, and result export |
| `presentations/` | Google Slides generators for earlier results |
| `experiments/legacy/` | Earlier RunPod runners and preflights |

The legacy scripts retain their old experiment scope and `/workspace` paths.
They include extra ratios and low-rank KD ablations that are excluded from the
final study. Their matching preflights now reference runners in this directory.
Moving scripts does not change the trainers or existing result artifacts.

## Added Study Support

`training/wlasl/finetune_lowrank_architecture_first.py` provides the same
`pretrained_dense` and `resume_low_rank` stages as the MSASL counterpart, with
WLASL's dataset adapter and `--subset-num`. Optimizer, scheduler, epoch loop, and
checkpoint selection match the MSASL implementation.

MSASL trainers and the checkpoint/quantization evaluators now use
`data_prep/msasl/archive_loader.py`: class count 1000 selects
`MSASL1000Archive`; existing 100/200 runs retain `MSASLArchive`. The underlying
loader filtering rules and prepared dataset files are unchanged.

Run the focused CPU checks from the repository root using the study environment:

```bash
python -m unittest discover -s tests -v
```

These check loader routing and exclusions, exact factor restoration without SVD,
rejection of incompatible checkpoints, new-stage optimizer/schedule reset, and
training-logic parity. They use small synthetic fixtures rather than the full
datasets, so a university model/data smoke test is still required.

## Before Launching the New Study

The new runners are available. Before a full university launch:

1. Finish transfer verification and repair relocated symlinks.
2. Verify an isolated university environment and one short loader/model test.
3. Smoke-test the added WLASL continuation and MSASL1000 loader routing with
   the real university data and checkpoints.
4. Review each runner's `--dry-run` plan and CPU allocation.

Retain old results as comparisons. Do not mix old and fresh runs in one output
directory or skip fresh work merely because an old checkpoint exists.

## Launch and Resume

The execution code is organized into five files:

| File | Responsibility |
| --- | --- |
| `experiments/run_wlasl_study.sh` | WLASL entry point |
| `experiments/run_msasl_study.sh` | MSASL entry point |
| `experiments/run_study.py` | Protocol, preflight, GPU checks, execution, provenance, export, and link repair |
| `analysis/benchmark_study.py` | Independent standardized latency/resource measurement |
| `training/study_state.py` | Optional epoch-boundary restart state |

Dataset-specific trainers and loaders are reused. The orchestration module keeps
PyTorch imports inside its GPU/data preflight, so ordinary orchestration does not
create a CUDA context. Preflight and export run as internal subprocess modes of
the same module; there are no separate helper scripts to launch.

From the repository root on university:

```bash
bash experiments/run_wlasl_study.sh 100 --dry-run
bash experiments/run_msasl_study.sh 100 --dry-run
```

Dry runs only print the full plan; they do not train or write outputs. Launch one
dataset per GPU allocation, choosing workers explicitly if the allocation has
fewer than the legacy eight CPUs:

```bash
bash experiments/run_wlasl_study.sh 100 --workers 4
bash experiments/run_msasl_study.sh 100 --workers 4
```

Use the first command in one GPU session, the second in another. Later use WLASL
300 and MSASL 200, then MSASL 1000. The default workers are eight to match legacy
arguments. An explicit worker adjustment is recorded; batch size and training
hyperparameters stay fixed. The runner refuses oversubscription when Slurm
reports fewer allocated CPUs than requested workers.

Defaults: workspace `/shared/g00107537/jupyter`, interpreter
`environments/masa/bin/python`, output root `results/compression_study_v1`.
Override workspace/interpreter with `STUDY_WORKSPACE` / `STUDY_PYTHON`, and output
with `--results-root`. Do not set these to old `fall_results` directories.

Every launch includes a preflight: sample/split checks, all listed clips' pose
and first-image presence, pretrained compatibility, and a disposable real-data
CUDA forward/backward. Dataset hashes cover effective split lists and metadata,
not all frame/pose contents. Existing intra-training duplicates are recorded;
exact cross-split filename overlap is rejected.

Each dataset completes Method 1, then Method 2, then benchmarks and exports all
14 results. It creates 25 tasks: nine training stages, seven FP32 evaluations,
seven quantizations, one benchmark, and one export. Individual evaluation
reports contain legacy timings, but final exports replace resource metrics with
the standardized benchmark values.

Monitor the main launch output and `DATASET/logs/TASK.log`. Task logs stream epoch
progress. Use Slurm batch submission for runs intended to survive browser/session
closure; these launchers do not create allocations or bypass their time limits.

After an allocation interruption, rerun the same command with `--resume`:

```bash
bash experiments/run_wlasl_study.sh 100 --workers 4 --resume
```

Completed tasks are skipped only after checking their configuration and artifact
hashes. Interrupted training restores `last.pth.tar`: model, optimizer, scheduler,
Python/NumPy/Torch/CUDA RNG, history, and matching best checkpoint. An incomplete
epoch is repeated from the last completed boundary. A planned Method 2 second
stage uses the earlier best weights and a new optimizer/schedule instead. No
last-state file is written in legacy launches unless `--save-last` is supplied.

Restart snapshots include the corresponding best checkpoint to prevent a
partially committed epoch from changing validation-best selection. These full
snapshots are large; they are kept as explicit restart artifacts. Quantization
keeps its actual `after_int8_fp16_state_dict.pth` for final benchmarking.
Redundant FP32 serialization copies from metric/quantization reports are removed
after successful evaluation; original checkpoints and quantized states stay.

`source.json`, `plan.json`, `preflight.json`, each stage's `launch.json`, and
`task_records/` preserve source/environment/configuration and file hashes.
Changing source, data manifests, packages, or task inputs blocks accidental
mixing on resume. A clean rerun should use a fresh results root.

The final files are `comparisons/results.json` and `comparisons/results.csv`.
All comparisons use the shared FP32 baseline; KD additionally compares with its
pre-KD FP32 Ghost. Accuracy changes are percentage points; resource changes are
relative percentages with negative values denoting reductions.

## Standardized Benchmark

The last phase requires an otherwise idle GPU. GPU locks prevent the study
runners from sharing a GPU, and NVIDIA process checks reject other compute jobs
before/after measurements. Other datasets may run on separate GPUs.

Defaults are a fixed real batch of 32 at target T=32, 30 warmups and 100 timed
forwards per round, three rounds. All models use the same saved input tensors;
measurements exclude loading/transfers/FLOP hooks, synchronize CUDA, and report
median, quartiles, p95, and raw samples. TF32 is disabled for this benchmark only.
The values are per batch; the per-sample equivalent is not batch-1 latency.

Rebenchmark without retraining if needed:

```bash
PYTHONPATH="$PWD" /shared/g00107537/jupyter/environments/masa/bin/python \
  analysis/benchmark_study.py \
  --manifest /shared/g00107537/jupyter/results/compression_study_v1/wlasl100/models.json \
  --out /shared/g00107537/jupyter/results/compression_study_v1/wlasl100/comparisons/rebenchmark.json
```

The quantization algorithm is the existing one: INT8 stored weights dequantized
to the input dtype during forward, with FP16 fallback. It is not a native INT8
GPU-kernel benchmark. Final FLOPs no longer use the partial legacy module hooks.
A separate JIT trace counts convolutions, linear layers, attention matrix
multiplications, GCN einsum contractions, normalization and elementwise arithmetic,
including explicit INT8 weight dequantization arithmetic. The exported
`parameter_count` counts learned weight/bias elements, including weights moved to
INT8 `qweight` buffers, but excludes quantization scales and running statistics.
The benchmark separately retains legacy `param_count_tensors` (all state tensors)
and `parameter_numel` (registered Parameters only) for traceability.

Final JSON/CSV report `gflops_per_sample`, calculated from audited forward
arithmetic for the fixed batch of 32 with 32 frames per sample, divided by 32 and
1e9. The benchmark also retains the actual `flops_per_batch` and per-operator
breakdown. This is an inference arithmetic estimate, not a hardware instruction
count. The exact convention is saved: MAC=2, explicit bias adds count separately;
scalar arithmetic/transcendentals count as one operation per element. ReLU/max
pool comparisons, integer shape calculations, indexing, casts and memory moves
are explicitly excluded. There is no claimed precision-based FLOP discount.

Unknown operators or unaudited data-dependent tracing warnings stop execution
instead of exporting a partial total. Preflight traces all six architecture/
precision layouts on the allocated GPU before training. Existing MASA warnings
from audited fixed-shape calculations are retained in the audit; their traces are
specific to the fixed input shape. Formula tests cover linear/grouped convolution,
low-rank, attention, GCN contractions and dequantization. Legacy standalone
evaluation reports remain unchanged and must not supply final-study FLOPs.

Every study evaluation saves per-clip target, predicted class, and correctness
from its existing forwards in a `*.predictions.json` sidecar. Export verifies the
same ordered clip identities across all 14 models and recomputes each reported
accuracy from those predictions. `results.json` contains pairwise prediction
disagreement counts: equal accuracies need not imply equal predictions.

Before timing, `comparisons/weight_audit.json` fingerprints all nine training-stage
best model states, ignoring optimizer/epoch metadata and checkpoint packaging.
Exact duplicates stop the benchmark for investigation; no training is rerun
automatically. Identical predictions alone are recorded, not treated as an error.
These safeguards do not change the training, evaluation sampling, or quantization
algorithms. Prediction recording is enabled only by the study runner.

If transfer relocation left absolute links, preview known RunPod-prefix repairs:

```bash
PYTHONPATH="$PWD" /shared/g00107537/jupyter/environments/masa/bin/python \
  experiments/run_study.py --repair-links
```

Add `--apply` to that command to perform validated repairs. It preserves relative
links and regular file contents. Run it once before concurrent study launches;
it may take time to scan all dataset entries.
