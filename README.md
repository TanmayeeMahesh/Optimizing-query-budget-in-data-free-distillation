# qbkd: query-budgeted data-free KD + mode connectivity

Pipeline for the project: distil a pretrained ResNet-34 into a ResNet-18 using a **fixed number of teacher queries**, with no CIFAR training images. Then connect two students with a low-loss curve (Garipov et al., 2018) **without spending any new queries**.

## The one rule: what a query is

One query is one teacher forward pass on one image. `query.budget` is the **total** number of queries. Augmentation and mixup happen **before** the query, so they are counted inside the budget, not on top of it:

```
budget B (e.g. 2,000,000)
 └─ per source s: n_s = weight_s · B queries
      └─ n_s / views_per_base base images  ──RandAugment/strong aug──▶  views_per_base views each
           └─ batch mixup (λ~U(0,1)), rounded to uint8  ──▶  teacher (once)  ──▶  store (image, logits)
```

For example, `views_per_base: 20` with B = 2M gives 100k noise images × 20 augmented views = 2M queries.

Training and curve fitting read only the stored logits and never load the teacher. Every teacher call is appended to `query_ledger.jsonl`, with the purpose recorded as `transfer`, `monitor`, or `probe`. The `report` command sums the ledger.

Priyanka's runs and Frank & Davis re-augment online every epoch. That costs epochs × 45–50k queries (4.5M for her 100-epoch runs, about 20M for F&D's 400 epochs). This pipeline freezes the queried set instead, so the budget is exact.

## Augmentation (from Priyanka's report, Sec. 6.4)

Her GitHub repo only contains Assignment 1, which uses crop and flip. The data-free notebooks are on her Drive. Her final `strong_mix` setting, which gave 88.90% with Leaves and 86.07% with pink2 at T=20, is reproduced as `augment.name: strong`:

RandomCrop(32, pad 4) → HFlip → RandAugment(4, 14) → Elastic (p 0.5) → Invert (p 0.5), then batch mixup with λ ~ U(0,1). The teacher labels the mixed image itself, and the targets are not blended.

Other presets are `none`, `basic`, `randaugment`, `trivialaugment`, and `strong_trivial` (TrivialAugmentWide in place of RandAugment).

## Teachers (downloaded, not trained)

| config | source | reported test acc |
|---|---|---|
| CIFAR-10 | `edadaltocg/resnet34_cifar10` (Hugging Face, MIT) | 95.4% |
| CIFAR-100 | `edadaltocg/resnet34_cifar100` (Hugging Face, MIT) | 79.78% |

These are timm ResNet-34s with a 3×3 stride-1 stem and no max-pool. They are loaded without the `detectors` package. Normalisation stats are read from the repo's `config.json`. **Always run `verify-teacher` first.** If the accuracy is far from the model card, the stats are wrong; set `teacher.mean` and `teacher.std` by hand.

To use another teacher, set `teacher.kind: checkpoint`, `arch:`, `path:`, `mean:`, and `std:`. To use another student, set `student.arch` to `resnet34_cifar`, `timm_cifar:<name>`, or `timm:<name>`. The curve code works with any architecture.

## Commands

```bash
pip install -r requirements.txt
C=configs/cifar10_r34_r18.yaml

python -m qbkd.cli verify-teacher --config $C
python -m qbkd.cli probe   --config $C --n 20000              # coverage per source (Phase 0)
python -m qbkd.cli monitor --config $C                         # 5k SVHN-test images, queried once
python -m qbkd.cli query   --config $C query.budget=500000    # the transfer store
python -m qbkd.cli train   --config $C query.budget=500000 --seed 0
python -m qbkd.cli train   --config $C query.budget=500000 --seed 1
python -m qbkd.cli connect --config $C query.budget=500000    # auto: top-2 by monitor agreement
python -m qbkd.cli connect --config $C query.budget=500000 connect.kind=linear
python -m qbkd.cli report  --config $C
```

Any key can be overridden as `dotted.key=value`, for example `train.temperature=4`, `query.sources.svhn.weight=0.25`, `query.augment.name=trivialaugment`, or `student.arch=resnet34_cifar`. The store folder name is a hash of everything that changes what the teacher is asked. A different budget, source mix, augmentation, seed, or teacher therefore gets its own store, and `train`/`connect` automatically pick the store matching the config you pass.

For CIFAR-100, swap the config to `configs/cifar100_r34_r18.yaml`. It inherits everything else.

## Disk

Images are stored as uint8 (3 KB each). Logits are stored as float16.

| budget | images | logits C-10 | logits C-100 |
|---|---|---|---|
| 100k | 0.3 GB | 2 MB | 20 MB |
| 500k | 1.5 GB | 10 MB | 100 MB |
| 2M | 6.1 GB | 40 MB | 400 MB |

On Colab, build stores on `/content` (local disk) and copy finished stores to Drive. A query run resumes from the last committed chunk if the session drops.

## Selection rules (carried over from Priyanka's discipline)

The CIFAR test set is never used to choose anything. Checkpoints (`best_monitor.pt`), curve endpoints (`auto`), and the best point on a curve are all chosen by **teacher–student agreement on held-out natural images** (the monitor store). Priyanka showed this ranks runs correctly, while agreement on held-out transfer data ranks them almost backwards. Test accuracy is computed once at the end, for reporting only.

## Mode connectivity: what is reported

`curves/<run>/summary.json` contains the following:

- Each endpoint with its own BatchNorm statistics, and with BN recomputed on the store.
- The best curve point (chosen by monitor agreement), plus the KD loss and accuracies along t (also plotted in `curve.png`).
- The **2-model endpoint ensemble**, which is the baseline a curve ensemble has to beat.
- The ensemble of all grid points on the curve, and of the interior points only.
- `extra_queries_for_curve: 0`.

Run `connect.kind=linear` as the no-training baseline. If the Bezier curve's loss barrier is no lower than the linear path's, the two students already sit in one basin.
