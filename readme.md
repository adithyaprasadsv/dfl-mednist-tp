# BloodMNIST – Federated Learning Experiments
**Lab Variant 8** | Medical Image Analysis with Federated Learning  
*Course: NTUU Lab: Distributed & Federated Learning*

---

## Overview

This project implements a federated learning (FL) pipeline for blood cell classification
using the BloodMNIST dataset from the MedMNIST benchmark collection. The assignment
investigates how FL strategies behave under data heterogeneity, model architecture
differences, varying image resolutions, and data poisoning attacks. These conditions
closely mirror real-world clinical deployment scenarios where patient data cannot leave
individual hospitals.

The full pipeline is implemented in a single Jupyter notebook
(`BloodMNIST_FL_Experiments.ipynb`) designed to run on Google Colab using the
Flower (`flwr`) simulation engine.

Why a notebook rather than .py files?
The lab is run on Google Colab, which is a notebook-native environment. Notebooks allow
inline result inspection, plots, printed metrics, and cell-by-cell execution, without
a separate script-runner or file transfer step. For an experimental workload where
hyperparameters are adjusted frequently and results are compared after each run,
the interactive format reduces iteration time. Modular .py files are the appropriate 
next step once the code is required to be packaged for reuse.

---

## Project Structure

```
dfl-mednist-tp/
├── BloodMNIST_FL_Experiments.ipynb              # main notebook with data loading, modelling & experiment
├── dfl-mednist-tp_report.pdf                    # Report with results from all experiments
├── environment.yml                              # dependencies
├── readme.md                                    # README
└── assets/
        └──{.png and .json}                      # Contains output by experiments
```

---

## Dataset

| Property        | Value                                                                 |
|-----------------|-----------------------------------------------------------------------|
| Name            | BloodMNIST (part of MedMnist)                                         |
| Source          | A large-scale dataset of each normal cells from peripheral blood      |
| Task            | Multi-class classification (8 classes)                                |
| num classes     | 8                                                                     |
| Channels        | 3 (RGB)                                                               |
| Resolutions     | 28×28, 64×64                                                          |
| Train           | 11,959                                                                |
| Val             | 1,712                                                                 |
| Test            | 3,421                                                                 |

---

## Environment Setup

### Requirements (environment.yml)

```
Python >= 3.10
PyTorch >= 2.0
flwr[simulation] >= 1.8
flwr-datasets
medmnist
scikit-learn
numpy
pandas
matplotlib
seaborn
```

### Installation

```bash
pip install "flwr[simulation]>=1.8" flwr-datasets medmnist scikit-learn
```

All other dependencies (`torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`,
`seaborn`) are pre-installed in Google Colab.

### Running on Google Colab

1. Upload `BloodMNIST_FL_Experiments.ipynb` to Google Colab or connect using extension in VS code to leverage GPU.
2. Set runtime to GPU.
3. Mount Google Drive when prompted — all outputs (plots and JSON files) are saved to
   `/content/drive/MyDrive/BloodMNIST_FL/` (change as per requirement).
4. Run all cells in order.

### Running Locally

Change the `OUTPUT_DIR` variable in the plotting cell to a local path:

```python
OUTPUT_DIR = "./outputs"   # replace the Google Drive path
```
Remove or comment out the `drive.mount(...)` call.

---

## Reproducibility

`SEED = 42`:

```python
random.seed('SEED')
numpy.random.seed('SEED')
torch.manual_seed('SEED')
torch.cuda.manual_seed_all('SEED')
```

Dirichlet partitioning also accepts the seed explicitly, so client splits are
identical across runs for the same `alpha` and `num_clients`. To reproduce any
specific experiment, set the config variables as hyperparameter in the run_fl_experiment call.

---

## Experimental Configuration

All experiments share the following default settings unless noted otherwise.

| Parameter           | Default Value | Notes                                      |
|---------------------|---------------|--------------------------------------------|
| `IMAGE_SIZE`        | 28            | 28 or 64                                   |
| `BATCH_SIZE`        | 64            |                                            |
| `NUM_CLIENTS`       | 5             | Minimum 3 required by rubric               |
| `NUM_ROUNDS`        | 20            | FL communication rounds                    |
| `LOCAL_EPOCHS`      | 1             | Local training steps per round             |
| `LR`                | 0.01          | SGD learning rate                          |
| `MOMENTUM`          | 0.9           | SGD momentum                               |
| `DIRICHLET_ALPHA`   | 100           | Non-IID concentration parameter            |
| `PROXIMAL_MU`       | 0.1           | FedProx penalty (used only with FedProx)   |
| `MALICIOUS_FRACTION`| 0.2           | Fraction of clients running poisoning attack|
| `POISON_STRATEGY`   | `'random'`    | `'random'` or `'target'`                   |

---

## Experiments

### Experiment 1 — IID vs Non-IID Heterogeneity
**Config:** FedAvg, CNN, 28×28, α ∈ {0.01, 100}, 20 rounds  
**Output:** `exp1_heterogeneity_fedavg_cnn_28.png / .json`

Compares near-uniform (α=100) against extreme non-IID (α=0.01) data distribution
across 5 clients. Establishes how much performance is lost due to statistical
heterogeneity, independent of strategy choice.

---

### Experiment 2 — CNN vs ResNet18
**Config:** FedAvg, IID (α=100), 28×28, 20 rounds  
**Output:** `exp2_model_fedavg_iid_28.png / .json`

Benchmarks the lightweight CNN against ResNet18 under identical FL conditions. 
Isolates the contribution of model capacity to convergence speed and final accuracy.

---

### Experiment 3 — Strategy Comparison (FedAvg, FedProx, FedMedian, Krum)
**Config:** CNN, IID (α=100), 28×28, 20 rounds  
**Output:** `exp3_strategy_iid_cnn_28.png / .json`

Runs all four aggregation strategies under clean IID conditions to establish a
fair baseline. Any performance difference here is attributable to the aggregation
algorithm itself rather than data distribution.

---

### Experiment 4 — Resolution: 28×28 vs 64×64
**Config:** FedAvg, CNN, Non-IID (α=0.01), 20 rounds  
**Output:** `exp4_resolution_fedavg_cnn_noniid.png / .json`

Tests whether higher-resolution inputs improve accuracy enough to justify the
additional per-client compute and storage cost in a federated setting.

---

### Experiment 5 — Data Poisoning Robustness
**Config:** CNN, 28×28, Non-IID (α=0.5), 20% malicious clients, random label flipping, 20 rounds  
**Strategies compared:** FedAvg (no defence), FedMedian, Krum, FedAvg clean (reference)  
**Output:** `exp5_poisoning_cnn_28.png / .json`

Introduces one malicious client (out of 5) running a random label-flipping attack.
Compares the degradation under FedAvg against the recovery achieved by the two
Byzantine-robust strategies.

---

### Experiment 6 — Client Drift: Local Epochs × Heterogeneity
**Config:** FedAvg, CNN, 28×28, local epochs ∈ {1, 5}, α ∈ {0.01, 100}, 20 rounds  
**Output:** `exp6_le5_noniid.png / .json`

Demonstrates client drift by comparing 1 vs 5 local epochs under both IID and
extreme non-IID splits. More local steps increase model divergence across clients,
particularly when the data is heterogeneous.

---

## Output Files

Each experiment produces two files saved to `OUTPUT_DIR`:

Files:
1. `expN_<name>.png` - 2×3 subplot: val loss, val accuracy, val F1 (solid) + test loss, test accuracy, test F1 (dashed) per round.
2. `expN_<name>.json` - Per-round metrics for all runs in the experiment plus a `summary` block with final-round values.

### JSON structure

```json
{
  "saved_at": "2025-06-10T14:00:00",
  "summary": {
    "FedMedian": {
      "final_val_loss": 0.42,
      "final_val_acc":  0.84,
      "final_val_f1":   0.83,
      "final_test_loss": 0.44,
      "final_test_acc":  0.83,
      "final_test_f1":   0.82
    }
  },
  "runs": {
    "FedMedian": [
      {"round": 1, "val_loss": 0.71, "accuracy": 0.74, "f1": 0.72,
       "test_loss": 0.73, "test_acc": 0.73, "test_f1": 0.71},
      ...
    ]
  }
}
```

---

## Key Decisions

### Hyperparameters

**Learning rate 0.01, momentum 0.9** — standard SGD settings for image classification
on small datasets. High enough to converge within 20 rounds; low enough to remain stable
under heterogeneous gradients.

**Batch size 64** — balances GPU utilisation and gradient noise. Larger batches risk overfitting on small per-client partitions; smaller batches slow GPU throughput.

---

### Why those alpha values (0.01, 0.5, 1, 100)?

Alpha controls the concentration of the Dirichlet distribution.

- **α = 0.01**: near-degenerate as each client receives samples almost exclusively from
  one or two classes. Represents the worst-case clinical scenario where one hospital
  only sees rare subtypes.
- **α = 0.5**: moderate heterogeneity and the standard benchmark value used in the
  original FedProx and FedAvg non-IID papers, making results comparable to the literature.
- **α = 100**: effectively IID and proportions are nearly uniform, used as the clean
  upper-bound reference.

---

### Why 20 rounds?

Twenty rounds provides enough communication steps to observe convergence or plateau
behaviour for all four strategies across both IID and non-IID conditions. Preliminary
runs showed that the CNN converges within 10–15 rounds under IID and begins to plateau
under non-IID by round 20, giving sufficient curve length to compare strategies. Fewer
rounds would truncate the convergence story and many more would exceed practical Colab
runtime limits for six experiments.

---

### Why 5 clients?

Five clients is the smallest number that makes FL behaviourally meaningful while keeping
Colab simulation time manageable. It satisfies the rubric minimum of three, produces
non-trivial aggregation behaviour (Krum selects from a real pool, FedMedian has a
meaningful median), and maps cleanly to the poisoning scenario: 20% malicious = exactly
one attacker, which is the minimum Byzantine scenario for Krum to demonstrate its defence.

---

### Why FedProx in this experimental context?

FedProx is the natural first extension to FedAvg under data heterogeneity. It adds a
proximal term to each client's local objective that penalises large deviation from the
global model, explicitly addressing client drift caused by non-IID data. In the context
of BloodMNIST, where different laboratories may hold entirely different cell type
distributions, the proximal term provides a theoretically grounded mechanism to
stabilise convergence. Including it allows the report to compare a naive baseline
(FedAvg), a regularised variant (FedProx), and robust aggregators (FedMedian, Krum)
within a single unified framework.

---

### Why local epoch = 1?

One local epoch is the standard FL setting used in the original FedAvg paper and most
subsequent benchmarks. It minimises client drift by ensuring each local update stays
close to the global optimum before aggregation. Experiment 6 explicitly studies the
effect of increasing local epochs to 5 to demonstrate drift under non-IID conditions,
making epoch=1 the clean baseline against which drift is measured.

---

### Why data poisoning, why 20%, why FedMedian and Krum?

**Why poisoning:** Byzantine robustness is a core requirement for real-world federated
healthcare systems. A malicious or compromised laboratory client could deliberately
corrupt the global model. Simulating this with label flipping is the standard threat
model in the FL robustness literature and directly tests whether the chosen aggregation
strategies provide meaningful protection.

**Why 20%:** 20% (1 in 5 clients) is the canonical Byzantine fraction used in the
FedMedian and Krum papers. Both algorithms offer theoretical guarantees when fewer than
half the clients are malicious, so 20% represents a realistic but non-trivial attack
that sits well within the provable tolerance range of both defences.

**Why FedMedian and Krum:** These are the two most widely studied Byzantine-robust
aggregation strategies. FedMedian replaces the mean with a coordinate-wise median,
naturally downweighting outliers without needing to know how many attackers are present.
Krum selects the single update closest to its neighbours, providing a stronger
geometric defence but requiring the number of malicious clients as a parameter. Their
contrasting designs make the comparison informative: FedMedian is parameter-free and
practical; Krum is stronger when attacker count is known.

---

## Metrics Tracked

All metrics are recorded at every communication round for both validation and test sets.

| Metric   | Split      | Description                                             |
|----------|------------|---------------------------------------------------------|
| Loss     | Val + Test | Cross-entropy, averaged over all samples                |
| Accuracy | Val + Test | Fraction of correctly classified samples                |
| Macro F1 | Val + Test | Per-class F1 averaged without class-frequency weighting |

**Why macro F1:** Under non-IID conditions, some clients hold very few samples of
certain classes. Accuracy is misleading in this case because a model ignoring a rare
class can still score highly. Macro F1 weights all eight classes equally, matching
the clinical requirement that all blood cell types are correctly identified.

**Val vs test:** Both metrics track closely under clean conditions, confirming that
BloodMNIST's official splits share the same data distribution. Under poisoning attacks
the gap widens for FedAvg (val is inflated, test reveals real damage) while
Byzantine-robust strategies maintain alignment — making the val/test gap a useful
diagnostic for attack severity.

---

## References

- Yang, J. et al. (2023). MedMNIST v2 — A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification. *Scientific Data*, 10(1), 1–10.
- McMahan, B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS*.
- Li, T. et al. (2020). Federated Optimization in Heterogeneous Networks (FedProx). *MLSys*.
- Blanchard, P. et al. (2017). Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent (Krum). *NeurIPS*.

---

*Author: Adithya Prasad SV | BloodMNIST FL Variant 8 | Framework: Flower ≥ 1.8 | Dataset: MedMNIST v2*