# LASST(Language-guided Semantic Style Transfer of 3D Indoor Scenes)
**LASST** is a method for Language-guided Semantic Style Transfer of 3D Indoor Scenes.

## Getting Started
### Installation

```
conda env create --name LASST python=3.7
conda install --yes --file requirements.txt
```

System requirements
### System Requirements
- Python 3.7
- CUDA 11.0
- GPU w/ minimum 8 GB ram

###  Data Preparation
The dataset we used is ScanNetV2 dataset. See [HERE](https://github.com/ScanNet/ScanNet) for more details. Remember to fix the data path in `src/local.py` as your own datapath.


### Run examples
Call the below shell scripts to generate example styles. 
```bash
# wooden floor,steel refridgerator
./scripts/go.sh
# ...
```

The outputs will be saved to `results/`.

#### Outputs
<p float="center">
<img alt="example" height="228" src="examples/example.png" width="1132"/>
</p>

<p float="center">
<img alt="semantic mask" height="1100" src="examples/sem_mask.png" width="600"/>
</p>

<p float="center">
<img alt="sampling method" height="500" src="examples/sampling.png" width="600"/>
</p>

<p float="center">
<img alt="predicted label" height="2400" src="examples/gt_pred.png" width="600"/>
</p>

<p float="center">
<img alt="hsv loss" height="600" src="examples/hsv.png" width="600"/>
</p>

## Citation
```
@article{LASST,
    author = {Bu Jin
              and Beiwen Tian
              and Hao Zhao
              and Guyue Zhou
              },
    title = {Language-guided Semantic Style Transfer of 3D Indoor Scenes},
    journal = {fixmeee},
    year  = {2022}
}
```
