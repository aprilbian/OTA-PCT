# OTA-MetaNeRF

Official implementation of [Over-the-Air Learning-based Geometry Point Cloud Transmission](https://arxiv.org/abs/2306.08730).

<img src="https://github.com/aprilbian/OTA-PCT/blob/main/OTA-MetaNerF/ota-metanerf.png" width="800">


## Data

Download the point cloud data from Semantic-KITTI [dataset](https://semantic-kitti.org/dataset.html), following [Deep Compression for Dense Point Cloud Maps](https://github.com/PRBonn/deep-point-map-compression), we use the sequence 08, consisting of 4k point clouds.

### Data prepropessing

Put the downloaded Semantic-KITTI dataset into the `KITTI/` folder, then run the `data_nerf.py` to generate `*.npy` files under the `./data` folder. More precisely, we spilt the dataset to training and test dataset (the first 3k as training, the remaining as test dataset). The `data_nerf.py` file normalizes and then voxelizes the input `*.bin` files with a default voxel resolution, $V = 6$.



## Training

To train a model, run

```python main.py --test False```.

See `main.py` file for more hyper parameters, e.g., **inner_step** determines the encoder latency, while **latent_dim** determines bandwidth usage. 

The model will be stored in the `wandb/` folder.




#### Evaluation

To evaluate the performance of a given setting

```python main.py --wandb_run_path <wandb_run_path> --test True```.

The readers can directly use the default parameter settings (testing mode) to achieve the same performances of the leftmost point of the green curves shown in Fig. 15 in the [paper](https://arxiv.org/abs/2306.08730). Download the checkpoint (`model.pt`) using [Google Drive](https://drive.google.com/file/d/1Y9SzK29sepB_76NrNrKVSnhyzxacLd3J/view?usp=drive_link), then place it to the folder indicated by **wandb_run_path** in `main.py` file. 

## File Organization

<pre> OTA-MetaNeRF/
  ├── data_nerf.py
  ├── main.py
  ├── wandb_utils.py
  ├── helpers.py
  ├── coinpp/ 
  │ ├── losses.py
  │ ├── metalearning.py
  │ ├── ...
  │ └── training.py
  
  ├── data/
  │ ├── train
        . `*.npy`
  │ ├── test
        . `*.npy`
  │ ├── kitti.py      # precess the data

  ├── KITTI/  
  │ ├── 08
        . `*.bin`

  ├── wandb/ # Pretrained weights (download separately) 
  │ ├── run-*
        ├── files
           ├── model.pt
           ├── ...

  
</pre>


## Baselines

The authors adopt the G-PCC baseline used in this [repo](https://github.com/NJUVISION/PCGCv2).
