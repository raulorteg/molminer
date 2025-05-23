# MolMiner &nbsp;—&nbsp; Towards Controllable, 3D-Aware, Fragment-Based Molecular Design
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<p align="center">
  <img src="figures/logo.png" alt="MolMiner logo" width="22%">
</p>

MolMiner is a **multi-property-conditioned, geometry-aware transformer** that builds molecules *fragment-by-fragment* while 3-D aware.  
It supports:

* **Order-agnostic roll-outs** with symmetry-aware attachment handling  
* Up to **12 simultaneous property conditions** (logP, QED, …)  
* End-to-end scripts for vocab extraction, preprocessing, GMM, starter model and full MolMiner training

---

## 1. Installation

```bash
git clone https://github.com/raulorteg/molminer.git
cd molminer
pip install -r requirements.txt
```

## 2. Training

Below is the minimal happy path from raw CSV -> trained MolMiner model.
All scripts share the common option --help for full CLI details.

### 2.1 Build fragment vocabularies
```bash
python extract_vocabulary.py \
  --dataset ../data/test/example.csv
# -> vocab_anchors.csv, vocab_attachments.csv, vocab_fragments.csv, stats.json
```
### 2.2 Train / validation / test split
```bash
python dataset_split.py \
  --dataset ../data/test/example.csv
# -> train.csv, valid.csv, test.csv
```
### 2.3 Pre-process for the Fragment-Starter
```bash
python preprocess_starter.py \
  --data_dir ../data/test
# -> train_starter.pkl, valid_starter.pkl, test_starter.pkl
```
### 2.4 Pre-process for MolMiner
```bash
python preprocess_molminer.py \
  --data_dir ../data/test \
  --total_epochs 10 \
  --max_workers 2
# -> steps/test, steps/valid, steps/{epoch}, ...
```
### 2.5 Train auxiliary models
stage	command
| stage                | command                                                                                         |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| **GMM**     | `python train_gmm.py   --data_dir ../data/test   --model_out ../checkpoints/test_gmm_model.pkl` |
| **Fragment-Starter** | `python train_starter.py --data_dir ../data/test --ckpt_dir ../checkpoints`        |


### 2.6 Train MolMiner
```bash
python train_molminer.py \
  --data_dir ../data/test \
  --ckpt_dir ../checkpoints \
  --fixedrollout  \      # remove to use adaptive roll-outs
  --total_epochs 10
# -> ckpt_dir/best_molminer.pth, cpkt_dir/last_molminer/pth
```

## 3. Generating

### 3.1 Create calibration 

```bash
python postprocess_calibration.py --samples=10 --ckpt_molminer='../checkpoints/best_molminer.pth' --ckpt_starter='../checkpoints/best_starter.pth' --ckpt_gmm='../checkpoints/gmm_model.pkl' --stats_path='../data/zinc/stats.json' --vocab_fragments=
'../data/zinc/vocab_fragments.csv' --vocab_attachments='../data/zinc/vocab_attachments.csv' --vocab_anchors='../data/zinc/vocab_anchors.csv' --device=cpu
# -> data/calibration/{prop}_calibration.txt
```

### 3.2 Generate unconditionally 
```bash
python generate_random.py --samples=10 --ckpt_molminer='../checkpoints/best_molminer.pth' --ckpt_starter='../checkpoints/best_starter.pth' --ckpt_gmm='../checkpoints/gmm_model.pkl' --stats_path='../data/zinc/stats.json' --vocab_fragments='../data/zinc/vocab_fragments.csv' --vocab_attachments='../data/zinc/vocab_attachments.csv' --vocab_anchors='../data/zinc/vocab_anchors.csv' --device=cpu
# -> data/generated.txt
```

## 4. Analyze results

### 4.1 Compute statistics on molecules generated unconditionally
```bash
python postprocess_generated_statistics.py
```

### 4.2 Create the Calibration plot for each of the twelve properties
```bash
python postprocess_calibration_plot.py --calibration_dir='../data/calibration' --stats_path='../data/zinc/stats.json' --figure_savepath='../figures/calibration.png'
# -> figures/calibration.png
```

## 5. Citation

If you use MolMiner in academic work, please cite:

Note: To be updated.
```bibtex
@misc{ochoa2024molminertransformerarchitecturefragmentbased,
      title={MolMiner: Transformer architecture for fragment-based autoregressive generation of molecular stories}, 
      author={Raul Ortega Ochoa and Tejs Vegge and Jes Frellsen},
      year={2024},
      eprint={2411.06608},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.06608}, 
}
```

## 6. License

MolMiner is released under the Apache 2.0 License – see LICENSE for details.
Contributions are welcome via pull requests or issues!