import coinpp.conversion as conversion
import coinpp.losses as losses
import coinpp.metalearning as metalearning
import torch
import wandb
import open3d as o3d
import numpy as np

import time
from torch.utils.data import Subset

def nearest_neighbor_indices(points, voxels):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kd_tree = o3d.geometry.KDTreeFlann(pcd)
    indices = np.zeros(len(voxels), dtype = np.int64)
    for i, voxel in enumerate(voxels):
        indices[i] = kd_tree.search_knn_vector_3d(voxel, 1)[1][0]
    return indices


def estimate_normals(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamKNN(knn = 30))
    normals = np.asarray(pcd.normals).astype('float32')
    return normals



def distortion(points_A, points_B):
    
    device = points_A.device
    points_A_numpy = points_A.cpu().numpy()
    points_B_numpy = points_B.cpu().numpy()
    normals_A = estimate_normals(points_A_numpy)
    normals_B = estimate_normals(points_B_numpy)
    indices_AB = nearest_neighbor_indices(points_B_numpy, points_A_numpy)
    indices_BA = nearest_neighbor_indices(points_A_numpy, points_B_numpy)
    normals_A = torch.from_numpy(normals_A).to(device)
    normals_B = torch.from_numpy(normals_B).to(device)
    indices_AB = torch.from_numpy(indices_AB).to(device)
    indices_BA = torch.from_numpy(indices_BA).to(device)
    d1_AB = torch.sum((points_A - points_B[indices_AB]) ** 2, dim = -1)
    d1_BA = torch.sum((points_B - points_A[indices_BA]) ** 2, dim = -1)
    d1 = torch.maximum(torch.mean(d1_AB), torch.mean(d1_BA))
    d2_AB = torch.sum((points_A - points_B[indices_AB]) * normals_A, dim = -1) ** 2
    d2_BA = torch.sum((points_B - points_A[indices_BA]) * normals_B, dim = -1) ** 2
    d2 = torch.maximum(torch.mean(d2_AB), torch.mean(d2_BA))
    return d1, d2


class Trainer:
    def __init__(
        self,
        func_rep,
        converter,
        args,
        train_dataset,
        test_dataset,
        patcher=None,
        model_path="",
    ):
        """Module to handle meta-learning of COIN++ model.

        Args:
            func_rep (models.ModulatedSiren):
            converter (conversion.Converter):
            args: Training arguments (see main.py).
            train_dataset:
            test_dataset:
            patcher: If not None, patcher that is used to create random patches during
                training and to partition data into patches during validation.
            model_path: If not empty, wandb path where best (validation) model
                will be saved.
        """
        self.func_rep = func_rep
        self.converter = converter
        self.args = args
        self.patcher = patcher

        self.outer_optimizer = torch.optim.Adam(
            self.func_rep.parameters(), lr=args.outer_lr
        )

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # consider a subset 
        subset = Subset(test_dataset, [1, 265, 2815])
        self.test_dataset = subset

        self._process_datasets()

        self.model_path = model_path
        self.step = 0
        self.best_val_psnr = 0.0

    def _process_datasets(self):
        """Create dataloaders for datasets based on self.args."""
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.num_workers > 0,
        )

        # If we are using patching, require data loader to have a batch size of 1,
        # since we can potentially have different sized outputs which cannot be batched
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=1 if (self.patcher or self.args.test_dataset == 'kitti') else self.args.batch_size,
            num_workers=self.args.num_workers,
        )

    def train_epoch(self):
        """Train model for a single epoch."""
        for data in self.train_dataloader:
            data = data.to(self.args.device)
            coordinates, features = self.converter.to_coordinates_and_features(data)

            # Optionally subsample points
            if self.args.subsample_num_points != -1:
                # Coordinates have shape (batch_size, *, coordinate_dim)
                # Features have shape (batch_size, *, feature_dim)
                # Flatten both along spatial dimension and randomly select points
                coordinates = coordinates.reshape(
                    coordinates.shape[0], -1, coordinates.shape[-1]
                )
                features = features.reshape(features.shape[0], -1, features.shape[-1])
                # Compute random indices (no good pytorch function to do this,
                # so do it this slightly hacky way)
                permutation = torch.randperm(coordinates.shape[1])
                idx = permutation[: self.args.subsample_num_points]
                coordinates = coordinates[:, idx, :]
                features = features[:, idx, :]

            outputs = metalearning.outer_step(
                self.func_rep,
                coordinates,
                features,
                inner_steps=self.args.inner_steps,
                inner_lr=self.args.inner_lr,
                is_train=True,
                return_reconstructions=False,
                gradient_checkpointing=self.args.gradient_checkpointing,
            )

            # Update parameters of base network
            self.outer_optimizer.zero_grad()
            outputs["loss"].backward(create_graph=False)
            self.outer_optimizer.step()

            if self.step % self.args.validate_every == 0 and self.step != 0:
                self.validation()

            log_dict = {"loss": outputs["loss"].item(), "psnr": outputs["psnr"]}

            self.step += 1

            #print(f'Step {self.step}, Loss {log_dict["loss"]:.3f}, PSNR {log_dict["psnr"]:.3f}')

            if self.args.use_wandb:
                wandb.log(log_dict, step=self.step)

    def validation(self):
        """Run trained model on validation dataset."""
        print(f"\nValidation, Step {self.step}:")

        # If num_validation_points is -1, validate on entire validation dataset,
        # otherwise validate on a subsample of points
        full_validation = self.args.num_validation_points == -1
        num_validation_batches = self.args.num_validation_points // self.args.batch_size

        # Initialize validation logging dict
        log_dict = {}

        # Evaluate model for different numbers of inner loop steps
        for inner_steps in self.args.validation_inner_steps:
            log_dict[f"val_psnr_{inner_steps}_steps"] = 0.0
            log_dict[f"val_loss_{inner_steps}_steps"] = 0.0

            # Fit modulations for each validation datapoint
            for i, data in enumerate(self.test_dataloader):

                if self.args.train_dataset == 'kitti':
                    data, _, _, _, _ = data
                    data = data[0]

                data = data.to(self.args.device)
                if self.patcher:
                    # If using patching, test data will have a batch size of 1.
                    # Remove batch dimension and instead convert data into
                    # patches, with patch dimension acting as batch size
                    patches, spatial_shape = self.patcher.patch(data[0])
                    coordinates, features = self.converter.to_coordinates_and_features(
                        patches
                    )

                    # As num_patches may be much larger than args.batch_size,
                    # split the fitting of patches into batch_size chunks to
                    # reduce memory
                    outputs = metalearning.outer_step_chunked(
                        self.func_rep,
                        coordinates,
                        features,
                        inner_steps=inner_steps,
                        inner_lr=self.args.inner_lr,
                        chunk_size=self.args.batch_size,
                        gradient_checkpointing=self.args.gradient_checkpointing,
                    )

                    # Shape (num_patches, *patch_shape, feature_dim)
                    patch_features = outputs["reconstructions"]

                    # When using patches, we cannot directly use psnr and loss
                    # output by outer step, since these are calculated on the
                    # padded patches. Therefore we need to reconstruct the data
                    # in its original unpadded form and manually calculate mse
                    # and psnr
                    # Shape (num_patches, *patch_shape, feature_dim) ->
                    # (num_patches, feature_dim, *patch_shape)
                    patch_data = conversion.features2data(patch_features, batched=True)
                    # Shape (feature_dim, *spatial_shape)
                    data_recon = self.patcher.unpatch(patch_data, spatial_shape)
                    # Calculate MSE and PSNR values and log them
                    mse = losses.mse_fn(data_recon, data[0])
                    psnr = losses.mse2psnr(mse)
                    log_dict[f"val_psnr_{inner_steps}_steps"] += psnr.item()
                    log_dict[f"val_loss_{inner_steps}_steps"] += mse.item()
                else:
                    
                    
                    coordinates, features = self.converter.to_coordinates_and_features(
                        data
                    )

                    outputs = metalearning.outer_step(
                        self.func_rep,
                        coordinates,
                        features,
                        inner_steps=inner_steps,
                        inner_lr=self.args.inner_lr,
                        is_train=False,
                        return_reconstructions=True,
                        gradient_checkpointing=self.args.gradient_checkpointing,
                    )

                    log_dict[f"val_psnr_{inner_steps}_steps"] += outputs["psnr"]
                    log_dict[f"val_loss_{inner_steps}_steps"] += outputs["loss"].item()

                if not full_validation and i >= num_validation_batches - 1:
                    break

            # Calculate average PSNR and loss by dividing by number of batches
            log_dict[f"val_psnr_{inner_steps}_steps"] /= i + 1
            log_dict[f"val_loss_{inner_steps}_steps"] /= i + 1

            mean_psnr, mean_loss = (
                log_dict[f"val_psnr_{inner_steps}_steps"],
                log_dict[f"val_loss_{inner_steps}_steps"],
            )
            print(
                f"Inner steps {inner_steps}, Loss {mean_loss:.3f}, PSNR {mean_psnr:.3f}"
            )

            # Use first setting of inner steps for best validation PSNR
            if inner_steps == self.args.validation_inner_steps[0]:
                if mean_psnr > self.best_val_psnr:
                    self.best_val_psnr = mean_psnr
                    # Optionally save new best model
                    if self.args.use_wandb and self.model_path:
                        torch.save(
                            {
                                "args": self.args,
                                "state_dict": self.func_rep.state_dict(),
                            },
                            self.model_path,
                        )

            if self.args.use_wandb:
                # Store final batch of reconstructions to visually inspect model
                # Shape (batch_size, channels, *spatial_dims)
                reconstruction = self.converter.to_data(
                    None, outputs["reconstructions"]
                )
                if self.patcher:
                    # If using patches, unpatch the reconstruction
                    # Shape (channels, *spatial_dims)
                    reconstruction = self.patcher.unpatch(reconstruction, spatial_shape)
                if self.converter.data_type == "mri":
                    # To store an image, slice MRI data along a single dimension
                    # Shape (1, depth, height, width) -> (1, height, width)
                    reconstruction = reconstruction[:, reconstruction.shape[1] // 2]

                if self.converter.data_type == "audio":
                    # Currently only support audio saving when using patches
                    if self.patcher:
                        # Unnormalize data from [0, 1] to [-1, 1] as expected by wandb
                        if self.test_dataloader.dataset.normalize:
                            reconstruction = 2 * reconstruction - 1
                        # Saved audio sample needs shape (num_samples, num_channels),
                        # so transpose
                        log_dict[
                            f"val_reconstruction_{inner_steps}_steps"
                        ] = wandb.Audio(
                            reconstruction.T.cpu(),
                            sample_rate=self.test_dataloader.dataset.sample_rate,
                        )
                elif self.converter.data_type == "image":
                    log_dict[f"val_reconstruction_{inner_steps}_steps"] = wandb.Image(
                        reconstruction
                    )

                wandb.log(log_dict, step=self.step)

        print("\n")


    def construct_points(self, voxels, block_info, thresholds):

        # reconstruct points clouds through a list of thresholds
        device = voxels.device
        stride = 16
        reconstruction_list = []

        for thres in thresholds:

            full_voxels = torch.zeros((64, 64, 64)).to(device)
            for i in range(len(block_info)):
                x, y, z = block_info[i]
                # first perform thresholding
                occupancy_i = voxels[i, :, :, :, 0] > thres
                full_voxels[x*stride:(x+1)*stride, y*stride:(y+1)*stride, z*stride:(z+1)*stride] = occupancy_i
            
            # then calculate the non-zero elements
            reconstructed_points  = torch.nonzero(full_voxels)
            if len(reconstructed_points) != 0:
                reconstruction_list.append(reconstructed_points)
            else:
                break

        return reconstruction_list
    

    def calculate_d1d2(self, voxels, block_info, point_max, point_min, raw_points):

        # the point clouds from reconstruction

        thresholds = [i*0.01 for i in range(5, 30)]
        recon_list = self.construct_points(voxels, block_info, thresholds)

        # reconstruct the point clouds from the predictions
        d1_list, d2_list = [],  []
        for i in range(len(recon_list)):
            reconstructed_points = recon_list[i]
            #print(thresholds[i])

            point1 = reconstructed_points/(2**6 - 1)
            rescale_rec = point1*(point_max - point_min) + point_min #+ torch.from_numpy(meta_data[2]).to(device)
            
            d1, d2 = distortion(raw_points, rescale_rec)
            peak_square = 3 * (point_max[0]-point_min[0]) ** 2
            d1_psnr, d2_psnr = 10 * torch.log10(peak_square / d1), 10 * torch.log10(peak_square / d2)

            d1_list.append(d1_psnr.item())
            d2_list.append(d2_psnr.item())
        
        print(d1_list)
        print(d2_list)

        return np.max(d1_list), np.max(d2_list)



    def evaluate(self):
        """Run trained model on evaluation dataset."""
        print(f"\nEvaluation, Step {self.step}:")

        # If num_validation_points is -1, validate on entire validation dataset,
        # otherwise validate on a subsample of points
        full_validation = self.args.num_validation_points == -1
        num_validation_batches = self.args.num_validation_points // self.args.batch_size

        # Initialize validation logging dict
        log_dict = {}
        N_iter = 3

        # Evaluate model for different numbers of inner loop steps
        for inner_steps in self.args.validation_inner_steps:
            log_dict[f"d1_{inner_steps}_steps"] = 0.0
            log_dict[f"d2_{inner_steps}_steps"] = 0.0
            log_dict["num_blks"] = 0.0

            # Fit modulations for each validation datapoint
            for i, data in enumerate(self.test_dataloader):

                #if i in [1, 265, 2815]:

                    voxels, point_max, point_min, gt_points, block_info = data
                    voxels, point_max, point_min, gt_points, block_info = \
                                voxels[0].to(self.args.device), point_max[0].to(self.args.device), point_min[0].to(self.args.device), gt_points[0].to(self.args.device), block_info[0]

                    
                    coordinates, features = self.converter.to_coordinates_and_features(
                        voxels
                    )

                    d1_avg, d2_avg = 0, 0

                    for _ in range(N_iter):
                    
                        start_time = time.time()
                        outputs = metalearning.outer_step(
                            self.func_rep,
                            coordinates,
                            features,
                            inner_steps=inner_steps,
                            inner_lr=self.args.inner_lr,
                            is_train=False,
                            return_reconstructions=True,
                            gradient_checkpointing=self.args.gradient_checkpointing,
                        )
                        print(time.time() - start_time)

                        rec_voxels = outputs["reconstructions"]   # (num_blk, 16, 16, 16, 1)
                        d1_psnr, d2_psnr = self.calculate_d1d2(rec_voxels, block_info, point_max, point_min, gt_points)

                        print(d1_psnr)
                        print(d2_psnr)

                        d1_avg += d1_psnr
                        d2_avg += d2_psnr
                    
                    d1_psnr, d2_psnr = d1_avg/N_iter, d2_avg/N_iter
                    print(d1_psnr)
                    print(d2_psnr)

                    log_dict[f"d1_{inner_steps}_steps"] += d1_psnr.item()
                    log_dict[f"d2_{inner_steps}_steps"] += d2_psnr.item()
                    log_dict["num_blks"] += len(block_info)

                    if not full_validation and i >= num_validation_batches - 1:
                        break

            # Calculate average PSNR and loss by dividing by number of batches
            log_dict[f"d1_{inner_steps}_steps"] /= i + 1
            log_dict[f"d2_{inner_steps}_steps"] /= i + 1
            log_dict["num_blks"] /= i+1

            D1_mean, D2_mean = (
                log_dict[f"d1_{inner_steps}_steps"],
                log_dict[f"d2_{inner_steps}_steps"],
            )
            print("average blks:", log_dict["num_blks"])
            print(
                f"Inner steps {inner_steps}, D1 {D1_mean:.3f}, D2 {D2_mean:.3f}"
            )


        print("\n")