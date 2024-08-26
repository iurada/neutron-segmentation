# DieHardNet for Semantic Segmentation
Base code for running the hardened segmentation experiments on simulated environments

## Getting Started

We implemented the base code starting with python 3.10 on CUDA 11.8. Once a clean virtual environment is properly setup, run from the shell
```bash
pip install -r requirements.txt
```

### Datasets
To download the Cityscapes dataset we've added a script in the `data/` folder. You can download it by running
```bash
python data/download_cityscapes.py
```

While the GTA5 dataset can be downloaded from: https://download.visinf.tu-darmstadt.de/data/from_games/ (eg. via `wget`)

### Running the Experiments
To run the experiments we've included the scripts to launch a dummy experiment in the `launch_scripts/` folder. You can start from there to launch the experiments or add your custom runs.
