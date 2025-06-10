import argparse
import numpy as np

import os
import numpy as np


def prepare_dir(path, exist = False):
    if exist and (path == None or not os.path.exists(path)):
        raise FileNotFoundError(f'\'{path}\' does not exist.')
    if not exist and path != None and path != 'None' and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

def log(path, content, output = True):
    if output:
        print(content)
    if path != None and path != 'None':
        with open(path, 'a') as file:
            content += '\n'
            file.write(content)



def voxelize_cloud(points, resolution):    

    max_coord, min_coord = np.max(points, axis=0), np.min(points, axis = 0)
    points = points - min_coord
    points = points / (max_coord - min_coord)
    points = np.round(points * (2**resolution - 1))
    point_dict = {}
    for _, point in enumerate(points):
        point_tuple = tuple(point)
        if point_tuple not in point_dict:
            point_dict[point_tuple] = []
    points = []
    for point_tuple, _ in point_dict.items():
        points.append(np.array([point_tuple]))
    points = np.concatenate(points, axis = 0)
    return points, [max_coord, min_coord]

def partition_blocks(points, resolution, block_resolution):
    block_occupancy = np.zeros([2**block_resolution] * 3)
    block_width = 2**(resolution - block_resolution)
    local_occupancy = {}
    for point in points:
        block, local_voxel = divmod(point.astype('int64'), block_width)
        block_tuple = tuple(block)
        if block_occupancy[block_tuple] == 0:
            block_occupancy[block_tuple] = 1
            local_occupancy[block_tuple] = np.zeros([block_width] * 3)
        local_occupancy[block_tuple][tuple(local_voxel)] = 1

    blocks = np.transpose(np.nonzero(block_occupancy))
    empty_voxels = []
    for block_tuple in local_occupancy.keys():
        local_voxel = np.transpose(np.nonzero(1 - local_occupancy[block_tuple]))
        voxel = local_voxel + np.array(block_tuple) * block_width
        empty_voxels.append(voxel.astype(np.float32))
    empty_voxels = np.concatenate(empty_voxels, axis = 0)

    train_data = np.zeros((len(blocks), block_width, block_width, block_width))
    for i in range(len(blocks)):
        train_data[i] = local_occupancy[tuple(blocks[i])]

    return empty_voxels, blocks, train_data

def parse_args():
    parser = argparse.ArgumentParser('data.py')
    parser.add_argument('--cloud_path', default = './KITTI/dataset/sequences/08/velodyne/002816.bin', type = str)
    parser.add_argument('--log_path', default = './log', type = str)
    parser.add_argument('--resolution', default = 6, type = int)
    parser.add_argument('--block_resolution', default = 2, type = int)
    return parser.parse_args()

def main(args):
    prepare_dir(args.cloud_path, exist = True)
    prepare_dir(args.log_path)
    log(args.log_path, 'data_nerf.py')
    log(args.log_path, str(args))

    cloud_path_dir = './KITTI/dataset/sequences/08/velodyne/'

    for i in range(1000):
        idx_str = str(i+3001).zfill(6)
        name_i = cloud_path_dir + idx_str + '.bin'
        data_path = './data/kitti/test/' + str(i+1)  + '.npy'
        prepare_dir(data_path)

        points = np.fromfile(name_i, dtype=np.float32).reshape(-1, 4)
        raw_points = points[:, 0:3]
        points, meta_data = voxelize_cloud(points[:, 0:3], args.resolution)

        _, blocks, train_data = partition_blocks(points, args.resolution, args.block_resolution)


        # calculate the rescaled points
        point1 = points/(2**args.resolution - 1)
        rescale_points = point1*(meta_data[0]-meta_data[1]) + meta_data[1] #+ meta_data[2]
        data_package = {
            'blocks': blocks,
            'train_data': train_data,
            'raw_points': raw_points,
            'meta_data': meta_data
        }
        np.save(data_path, data_package, allow_pickle=True)

if __name__ == '__main__':
    args = parse_args()
    main(args)