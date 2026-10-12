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

## Before Launching the New Study

This cleanup organizes existing code; it does not supply a new five-dataset
university launcher yet. Remaining implementation and environment work:

1. Finish transfer verification and repair relocated symlinks.
2. Verify an isolated university environment and one short loader/model test.
3. Route every MSASL1000 trainer and evaluator through its archive loader,
   including exclusions; test class count and split boundaries.
4. Extend architecture-first low-rank continuation to WLASL while preserving
   the agreed training behavior.
5. Create the reduced study launcher with fresh output directories, saved full
   arguments, source Git revision, environment versions, dataset manifest hashes,
   and initialization/teacher/student checkpoint hashes.
6. Support job continuation within Slurm time limits. Resuming an interrupted
   stage must preserve optimizer, scheduler, and random state; beginning the
   second planned stage intentionally resets optimizer and schedule.

Retain old results as comparisons. Do not mix old and fresh runs in one output
directory or skip fresh work merely because an old checkpoint exists.
