# -*- coding = utf-8 -*-
# @File Name : loss_utils
# @Date : 2022/10/6 15:24
# @Author : dengzhiwei
# @E-mail : zhiweide@usc.edu


import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode


def preproc_output(output):
    """
    Pre-process the output of the network
    :param output: Output dictionary of the network
    :return: vessel: 2D image [B, H, W, 2], 3D image [B, H, W, D, 3]. Optimal direction
             radius: 2D image [B, H, W, 1], 3D image [B, H, W, D, 1].
    """
    vessel = output['vessel']  # this should have shape of [B, 3, H, W, D] / [B, 2, H, W]
    radius = output['radius']  # this should have shape of [B, 1, H, W, D] / [B, 1, H, W]
    vessel = F.normalize(vessel, dim=1)         # normalize the optimal dir
    # if 3D image such as CT and MRA
    if len(vessel.size()) == 5:
        vessel = vessel.permute(0, 2, 3, 4, 1)  # change to [B, H, W, D, 3]
        radius = radius.permute(0, 2, 3, 4, 1)  # change to [B, H, W, D, 1]
    else:
        # if 2D image such as OCT
        vessel = vessel.permute(0, 2, 3, 1)     # change to [B, H, W, 2]
        radius = radius.permute(0, 2, 3, 1)     # change to [B, H, W, 1]
    return vessel, radius


def get_orthogonal_basis(optimal_dir):
    """
    Get orthogonal vectors of other two directions
    :param optimal_dir: 3D image [B, H, W, D, 3] / 2D image [B, H, W, 2]
    :return: basis: 3D image [B, H, W, D, n(3), 3] / 2D image [B, H, W, n(2), 2]
    """
    # if 3D image such as CT and MRA
    if len(optimal_dir.size()) == 5:
        c = torch.randn_like(optimal_dir, device=optimal_dir.device)
        ortho_dir_1 = torch.cross(c, optimal_dir, dim=4)
        ortho_dir_1 = ortho_dir_1 / ortho_dir_1.norm(dim=4, keepdim=True) + 1e-10
        ortho_dir_2 = torch.cross(optimal_dir, ortho_dir_1, dim=4)
        ortho_dir_2 = ortho_dir_2 / ortho_dir_2.norm(dim=4, keepdim=True) + 1e-10
        basis = torch.stack((optimal_dir, ortho_dir_1, ortho_dir_2), dim=4)
    else:
        # if 2D image such as OCT
        index = torch.LongTensor([1, 0]).to(optimal_dir.device)
        ortho_dir_1 = torch.index_select(optimal_dir, -1, index)
        ortho_dir_1[:, :, :, 1] = -ortho_dir_1[:, :, :, 1]
        ortho_dir_1 = ortho_dir_1 / ortho_dir_1.norm(dim=3, keepdim=True) + 1e-10
        basis = torch.stack((optimal_dir, ortho_dir_1), dim=3)
    return basis


def get_sampling_vec(num_pts, estimated_r):
    """
    Get the sampling vectors, sphere or circle
    :param num_pts: sampling num points
    :param estimated_r: estimated radius, used to parse the device
    :return: sampling vectors
    """
    # if 3D image such as CT and MRA
    if len(estimated_r.size()) == 5:
        indices = torch.arange(0, num_pts, dtype=torch.float32)
        phi = torch.arccos(1 - 2 * indices / num_pts)
        theta = torch.pi * (1 + 5 ** 0.5) * indices
        x, y, z = torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)
        # flip coordinates according to the sample grid
        vectors = torch.vstack((z, y, x)).T.to(estimated_r.device)      # This is a sphere sampling
    else:
        # if 2D image such as OCT
        angle = 2.0 * torch.pi * torch.arange(0, num_pts) / num_pts
        x, y = 1.0 * torch.cos(angle), 1.0 * torch.sin(angle)
        vectors = torch.vstack((x, y)).T.to(estimated_r.device)         # This is a circle sampling
    return vectors


def get_gradients(image, dims=(2, 3, 4), channel_dim=1):
    """
    Get gradients of batch image
    :param image: 3D image [B, 1, H, W, D] / 2D image [B, 1, H, W]
    :param dims: 3D image (2, 3, 4) / 2D image (2, 3)
    :param channel_dim: channel dimension, default=1
    :return: gradients: 3D image [B, 3, H, W, D] / 2D image [B, 2, H, W]
    """
    gradients = torch.gradient(image, dim=dims)
    gradients = torch.cat(gradients, dim=channel_dim)
    gradients += torch.randn(gradients.size(), device=gradients.device) * 1e-10
    return gradients


def get_grid_base(gradients):
    """
    Get the image grid
    meshgrid method will switch x and z axis
    :param gradients: [B, 2, H, W] / [B, 3, H, W, D]
    :return: grid : [B, H, W, 2] / [B, H, W, D, 3]
    """
    shape = gradients.size()
    b, c, h, w = shape[0], shape[1], shape[2], shape[3]
    d = shape[4] if len(shape) == 5 else None
    dh = torch.linspace(-1.0, 1.0, h)
    dw = torch.linspace(-1.0, 1.0, w)
    if d:
        dd = torch.linspace(-1.0, 1.0, d)
        meshx, meshy, meshz = torch.meshgrid((dh, dw, dd), indexing='ij')
        # need to swap the order of xyz
        grid = torch.stack((meshz, meshy, meshx), dim=3).repeat((b, 1, 1, 1, 1))    # [B, H, W, D, 3]
    else:
        meshx, meshy = torch.meshgrid((dh, dw), indexing='ij')
        grid = torch.stack((meshy, meshx), dim=2).repeat((b, 1, 1, 1))           # [B, H, W, 2]
    return grid.to(gradients.device)


def sample_space_to_img_space(grid, h, w, d=None):
    """
    Convert the image space to sample space
    [[-1, 1], [-1, 1], [-1, 1]] -> [[0, H], [0, W], [0, D]]
    grid is of size [B, H, W, D, 3] or [B, H, W, 2]
    convert [-1, 1] scale to [0, H] scale
    :param grid: [B, H, W, D, 3] or [B, H, W, 2]
    :param h: image height, int
    :param w: image width, int
    :param d: image depth, int, only used if the grid is 3D
    :return: [B, H, W, D, 3] or [B, H, W, 2]
    """
    #
    grid = grid + 0
    grid = grid * 0.5 + 0.5
    grid[..., 0] = grid[..., 0] * h
    grid[..., 1] = grid[..., 1] * w
    if d:
        grid[..., 2] = grid[..., 2] * d
    return grid


def img_space_to_sample_space(grid, h, w, d=None):
    """
    Convert the image space to sample space
    [[0, H], [0, W], [0, D]] -> [[-1, 1], [-1, 1], [-1, 1]]
    grid is of size [B, H, W, D, 3] or [B, H, W, 2]
    convert [0, H] scale to [-1, 1] scale
    :param grid: [B, H, W, D, 3] or [B, H, W, 2]
    :param h: image height, int
    :param w: image width, int
    :param d: image depth, int, only used if the grid is 3D
    :return: [B, H, W, D, 3] or [B, H, W, 2]
    """
    grid = grid + 0
    grid[..., 0] = 2.0 * grid[..., 0] / h - 1
    grid[..., 1] = 2.0 * grid[..., 1] / w - 1
    if d:
        grid[..., 2] = 2.0 * grid[..., 2] / d - 1
    return grid


def grid_sample(image, sample_grid, permute):
    """
    Functional grid sample overload
    :param image: [B, C, H, W] / [B, C, H, W, D]
    :param sample_grid: [B, H, W, 2] / [B, H, W, D, 3]
    :param permute: bool, indicate to change the shape or not
    :return:  [B, H, W, 1, 2] / [B, H, W, D, 1, 3]
    """
    sampled = F.grid_sample(image, sample_grid, align_corners=True, padding_mode='border')
    if len(image.size()) == 5:
        if permute:
            sampled = sampled.permute(0, 2, 3, 4, 1).unsqueeze(4)
    else:
        if permute:
            sampled = sampled.permute(0, 2, 3, 1).unsqueeze(3)
    return sampled


def project(vectors, basis, proj=False):
    """
    Project gradients on to the basis
    :param vectors: 2D image [B, H, W, k(1/2), 2] / 3D image [B, H, W, D, k(1/3), 3]
    :param basis: 2D image [B, H, W, 2, 2] / 3D image [B, H, W, D, 3, 3]
    :param proj: bool, whether to project ob the base
    :return:
        proj = True
            2D image [B, H, W, 2, 2] / 3D image [B, H, W, D, 3, 3]
        proj = False
            2D image [B, H, W, 2] / 3D image [B, H, W, D, 3]
    """
    proj_vectors = torch.sum(torch.mul(vectors, basis), dim=-1)         # [B, H, W, 2] / [B, H, W, D, 3]
    if proj:
        proj_vectors = torch.mul(proj_vectors.unsqueeze(-1), basis)     # [B, H, W, 2, 2] / [B, H, W, D, 3, 3]
    return proj_vectors


def swap_order(direction, order, dim):
    """
    Swap the axis of the tensor
    :param direction: 2D image [B, H, W, 2] / 3D image [B, H, W, D, 3]
    :param order: list, the order of X, Y and Z axis
    :param dim: -1 or 3 or 4
    :return: 2D image [B, H, W, 2] / 3D image [B, H, W, D, 3]
    """
    index = torch.LongTensor(order).to(direction.device)
    direction = torch.index_select(direction, dim, index)
    return direction


def proc_sample_dir(sample_dir_scaled, estimated_r):
    """
    Process the sampled direction
    :param sample_dir_scaled: 2D image [B, H, W, 2] / 3D image [B, H, W, D, 3]
    :param estimated_r: 2D image [B, H, W, 1] / 3D image [B, H, W, D, 1]
    :return: 2D image [B, H, W, 2, 2] / 3D image [B, H, W, D, 3, 3]
    """
    if len(sample_dir_scaled.size()) == 5:
        sample_dir_scaled = swap_order(sample_dir_scaled, [2, 1, 0], dim=-1)
    else:
        sample_dir_scaled = swap_order(sample_dir_scaled, [1, 0], dim=-1)
    sample_dir_scaled = torch.div(sample_dir_scaled, estimated_r)   # [B, H, W, 2] / [B, H, W, D, 3]
    sample_dir_scaled = sample_dir_scaled.unsqueeze(-2)             # [B, H, W, 1, 2] / [B, H, W, D, 1, 3]
    repeat_size = torch.ones(len(sample_dir_scaled.size()), dtype=torch.int).tolist()
    repeat_size[-2] = sample_dir_scaled.size(-1)                    # [1, 1, 1, 2, 1] / [1, 1, 1, 1, 3, 1]
    sample_dir_scaled = sample_dir_scaled.repeat(repeat_size)       # [B, H, W, 2, 2] / [B, H, W, D, 3, 3]
    return sample_dir_scaled


def get_sample_grid(sample_dir, curr_radius, grid_base, b, h, w, d):
    """
    Compute the sampling grid based on the radius and sample direction
    :param sample_dir: sampling direction   2D or 3D vector
    :param curr_radius: radius corresponding to this direction
    :param grid_base: grid base for sampling of the original image [B, H, W, 2] or [B, H, W, D, 3]
    :param b: batch size, int
    :param h: image, height, int
    :param w: image width, int
    :param d: image depth, int or None
    :return: the sampling grid, same size as grid base
    """
    order = [2, 1, 0] if d else [1, 0]
    sample_dir = sample_dir.repeat((b, h, w, d, 1)) if d else sample_dir.repeat((b, h, w, 1))
    sample_dir_scaled = swap_order(torch.mul(sample_dir, curr_radius), order, dim=-1)
    sample_grid = img_space_to_sample_space(grid_base + sample_dir_scaled, h, w, d)     # convert to [-1, 1]
    return sample_grid, sample_dir_scaled


def calc_dir_response(sample_dir, curr_radius, gradients, basis, grid_base, b, h, w, d):
    """
    Compute the projected response for given direction and radius
    :param sample_dir: sampling direction 2D or 3D vector
    :param curr_radius: radius corresponding to this direction
    :param gradients: image gradients [B, 2, H, W] or [B, 3, H, W, D]
    :param basis: basis for vessel flow of each pixel or voxel [B, H, W, n(2), 2] or [B, H, W, D, n(3), 3]
    :param grid_base: grid base for sampling [B, H, W, 2] or [B, H, W, D, 3]
    :param b: batch size, int
    :param h: image, height, int
    :param w: image width, int
    :param d: image depth, int or None
    :return: projected responses [B, H, W, 2, 2] / [B, H, W, D, 3, 3]
    """
    # get sample grid [B, H, W, 2] or [B, H, W, D, 3]
    sample_grid, sample_dir_scaled = get_sample_grid(sample_dir, curr_radius, grid_base, b, h, w, d)
    # projected gradients has shape of [B, H, W, 1, 2] / [B, H, W, D, 1, 3]
    proj_gradients = project(grid_sample(gradients, sample_grid, True), basis, proj=True)
    # compute projected flux
    sample_dir_scaled = proc_sample_dir(sample_dir_scaled, curr_radius)     # [B, H, W, 2, 2] / [B, H, W, D, 3, 3]
    proj_response = project(proj_gradients, sample_dir_scaled)              # [B, H, W, 2, 2] / [B, H, W, D, 3, 3]
    return proj_response


def rotate_3d(image, angles, expand=False, fill=0.0):
    """
    rotate the 3d images with specified angles
    :param image: 3d volume images, [H, W, D]
    :param angles: angles of rotation in 3 different axis, (x, y, z), [,3]
    :param expand: ignored
    :param fill: if given a number, the value is used for all bands respectively.
    :return: image, rotated image or recovered image
    """
    image = rotate(image, interpolation=InterpolationMode.BILINEAR, angle=angles[0].item(), expand=expand, fill=fill)
    image = image.permute((1, 0, 2))
    image = rotate(image, interpolation=InterpolationMode.BILINEAR, angle=angles[1].item(), expand=expand, fill=fill)
    image = image.permute((1, 0, 2))
    image = image.permute((2, 1, 0))
    image = rotate(image, interpolation=InterpolationMode.BILINEAR, angle=-angles[2].item(), expand=expand, fill=fill)
    image = image.permute((2, 1, 0))
    return image.squeeze(0)


def rotate_3d_batch(batch_images, batch_angles):
    """
    rotate the 3d images with specified angles
    :param batch_images: 3d volume images, [B, C, H, W, D]
    :param batch_angles: angles of rotation in 3 different axis, (x, y, z), [B, 3]
    :return: batch_images_rotated, rotated image or recovered image
    """
    device = batch_images.device
    batch_num = batch_images.size(0)
    channel_num = batch_images.size(1)
    batch_images_rotated = torch.zeros(batch_images.size())
    for i in range(batch_num):
        for j in range(channel_num):
            batch_images_rotated[i][j] = rotate_3d(batch_images[i][j], batch_angles[i])
    return batch_images_rotated.to(device)


def get_rotation_matrix_3d(angles):
    """
    get the 3d rotation matrix
    :param angles: angles of rotation in 3 different axis, (x, y, z), [,3]
    :return: rotation matrix (3, 3)
    """
    alpha, beta, gamma = angles / 180 * torch.pi
    yaw = torch.tensor([[1, 0, 0],
                       [0, torch.cos(alpha), -torch.sin(alpha)],
                       [0, torch.sin(alpha), torch.cos(alpha)]])
    pitch = torch.tensor([[torch.cos(beta), 0, torch.sin(beta)],
                         [0, 1, 0],
                         [-torch.sin(beta), 0, torch.cos(beta)]])
    roll = torch.tensor([[torch.cos(gamma), -torch.sin(gamma), 0],
                        [torch.sin(gamma), torch.cos(gamma), 0],
                        [0, 0, 1]])
    r_matrix = torch.matmul(roll, torch.matmul(pitch, yaw))
    return r_matrix


def get_rotation_matrix_3d_batch(batch_angles):
    """
    get the 3d rotation matrix
    :param batch_angles: angles of rotation in 3 different axis, (x, y, z), [B, 3]
    :return: rotation matrix (3, 3)
    """
    device = batch_angles.device
    batch_num = batch_angles.size(0)
    batch_rotation_matrices = torch.zeros(batch_num, 3, 3)
    for i in range(batch_num):
        batch_rotation_matrices[i] = get_rotation_matrix_3d(batch_angles[i])
    return batch_rotation_matrices.to(device)
