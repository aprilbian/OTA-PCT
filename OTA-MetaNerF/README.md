# OTA-MetaNeRF

Official implementation of [Over-the-Air Learning-based Geometry Point Cloud Transmission](https://arxiv.org/abs/2306.08730).

<img src="https://github.com/EmilienDupont/coinpp/raw/main/imgs/fig1.png" width="800">


## Data

Download the point cloud data from Semantic-KiTTi [dataset](), following [](), we use the sequence 08, spilt it to training and test dataset (the first 3k as training, the remaining as test dataset). 

## Training

To train a model, run

```python main.py @config.txt```.

See `config.txt` and `main.py` for setting various arguments. Note that if using wandb, you need to change the wandb entity and project name to your own.

A few example configs used to train the models in the paper can be found in the `configs` folder.


#### Evaluation

To evaluate the performance of a given modulation dataset (in terms of PSNR), run

```python evaluate.py --wandb_run_path <wandb_run_path> --modulation_dataset <modulation_dataset>```.




## Baselines

The authors adopt the G-PCC baseline used in this [repo](https://github.com/NJUVISION/PCGCv2). More precisely, the authors 
