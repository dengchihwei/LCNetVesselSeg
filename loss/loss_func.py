# -*- coding = utf-8 -*-
# @File Name : loss
# @Date : 2022/6/2 14:57
# @Author : dengzhiwei
# @E-mail : zhiweide@usc.edu
import torch

from loss.loss_utils import *


def recon_loss(image, output, sup=False):
    """
    Compute the reconstruction loss
    :param image: original image [B, 1, H, W] / [B, 1, H, W, D]
    :param output: reconstructed image [B, 1, H, W] / [B, 1, H, W, D]
    :param sup: whether supervision is used
    :return: reconstruction loss
    """
    recon = output['recon']
    if sup:
        recon = torch.sigmoid(recon)
    rec_loss = F.mse_loss(image, recon)
    return rec_loss


def flux_loss_symmetry(image, output, sample_num, grad_dims):
    """
    Compute the Symmetry Flux Loss of the output of the network
    :param image: original image [B, 1, H, W] / [B, 1, H, W, D]
    :param output: output dictionary 'vessel', 'radius', 'recon', 'attention'
    :param sample_num: num of sampling directions of a sphere / circle
    :param grad_dims: 2D image (2, 3) / 3D image (2, 3, 4)
    :return: flux response, mean flux loss
    """
    shape = image.size()                                            # 2D image [B, C, H, W] / 3D image [B, C, H, W, D]
    b, c, h, w = shape[0], shape[1], shape[2], shape[3]
    d = shape[4] if len(shape) == 5 else None
    # 2D image [B, H, W, 2], [B, H, W, 1] / 3D image [B, H, W, D, 3], [B, H, W, D, 1]
    optimal_dir, estimated_r = preproc_output(output)
    basis = get_orthogonal_basis(optimal_dir)                               # get the basis of the optimal directions
    sampling_vec = get_sampling_vec(sample_num, estimated_r)                # get sampling sphere / circle
    gradients = get_gradients(image, dims=grad_dims)                        # [B, 2, H, W] / [B, 3, H, W, D]
    grid_base = sample_space_to_img_space(get_grid_base(gradients), h, w, d)     # convert to [0, H]

    response = torch.zeros(optimal_dir.size(), device=optimal_dir.device)   # get responses of 2 / 3 directions
    for i in range(sample_num):
        sample_dir = sampling_vec[i]                                        # this is a 2d / 3d vector
        curr_radius = estimated_r[..., i:i+1] if estimated_r.size(-1) > 1 else estimated_r
        proj_response = calc_dir_response(sample_dir, curr_radius, gradients, basis, grid_base, b, h, w, d)
        response += proj_response / sample_num                              # [B, H, W, 2] / [B, H, W, D, 3]
    response = - torch.sum(response[..., 1:], dim=-1)                       # [B, H, W] / [B, H, W, D]
    response = torch.clip(response, min=0.0)
    if 'attention' in output.keys():
        response = torch.mul(response, 1.0 + output['attention'])
    max_flux_loss = - response.mean()
    return response, max_flux_loss


def flux_loss_asymmetry(image, output, sample_num, grad_dims):
    """
    Compute the None-Symmetry Flux Loss of the output of the network
    find the minimum magnitude of the direction and the opposite direction
    :param image: original image [B, 1, H, W] / [B, 1, H, W, D]
    :param output: output dictionary 'vessel', 'radius', 'recon', 'attention'
    :param sample_num: num of sampling directions of a sphere / circle
    :param grad_dims: 2D image (2, 3) / 3D image (2, 3, 4)
    :return: flux response, mean flux loss
    """
    shape = image.size()                                            # 2D image [B, C, H, W] / 3D image [B, C, H, W, D]
    b, c, h, w = shape[0], shape[1], shape[2], shape[3]
    d = shape[4] if len(shape) == 5 else None
    # 2D image [B, H, W, 2], [B, H, W, 1] / 3D image [B, H, W, D, 3], [B, H, W, D, 1]
    optimal_dir, estimated_r = preproc_output(output)
    basis = get_orthogonal_basis(optimal_dir)                               # get the basis of the optimal directions
    sampling_vec = get_sampling_vec(sample_num, estimated_r)                # get sampling sphere / circle
    gradients = get_gradients(image, dims=grad_dims)                        # [B, 2, H, W] / [B, 3, H, W, D]
    grid_base = sample_space_to_img_space(get_grid_base(gradients), h, w, d)     # convert to [0, H]

    shift = int(sample_num / 2)
    response = torch.zeros(optimal_dir.size(), device=optimal_dir.device)   # get responses of 2 / 3 directions
    for i in range(shift):
        # get the sampling direction and the opposite direction
        sample_dir1, sample_dir2 = sampling_vec[i], sampling_vec[i+shift]   # this is a 2d / 3d vector
        curr_rad1 = estimated_r[..., i:i+1] if estimated_r.size(-1) > 1 else estimated_r
        curr_rad2 = estimated_r[..., i+shift:i+shift+1] if estimated_r.size(-1) > 1 else estimated_r
        # compute the projected responses of the two directions [B, H, W, 2] / [B, H, W, D, 3]
        proj_response1 = calc_dir_response(sample_dir1, curr_rad1, gradients, basis, grid_base, b, h, w, d)
        proj_response2 = calc_dir_response(sample_dir2, curr_rad2, gradients, basis, grid_base, b, h, w, d)
        # find the minimum responses of the two directions
        response += torch.maximum(proj_response1, proj_response2) * 2 / sample_num
    response = - torch.sum(response[..., 1:], dim=-1)  # - response[..., 0]    # [B, H, W] / [B, H, W, D]
    response = torch.clip(response, min=0.0)
    if 'attentions' in output.keys():
        attention = output['attentions'][-1]
        response = torch.mul(response, 1.0 + attention)
    mean_flux_loss = - response.mean()
    return response, mean_flux_loss


def continuity_loss(image, output, flux_response, sample_num):
    """
    Compute the continuity loss
    :param image: original image [B, 1, H, W] / [B, 1, H, W, D]
    :param output: output dictionary 'vessel', 'radius', 'recon', 'attention'
    :param flux_response: flux response [B, H, W] / [B, H, W, D]
    :param sample_num: num of sampling directions of a sphere / circle
    :return: mean direction_loss and mean intensity loss
    """
    shape = image.size()                                            # 2D image [B, C, H, W] / 3D image [B, C, H, W, D]
    b, c, h, w = shape[0], shape[1], shape[2], shape[3]
    d = shape[4] if len(shape) == 5 else None
    # 2D image [B, H, W, 2], [B, H, W, 1] / 3D image [B, H, W, D, 3], [B, H, W, D, 1]
    optimal_dir, estimated_r = preproc_output(output)
    mean_rad = torch.mean(estimated_r, dim=-1).unsqueeze(-1)
    optimal_dir_scaled = torch.mul(optimal_dir, mean_rad)
    # get the sample grid on the optimal direction
    order = [2, 1, 0] if d else [1, 0]
    grid_base = sample_space_to_img_space(get_grid_base(image), h, w, d)              # convert to [0, H]
    sample_grid = grid_base + swap_order(optimal_dir_scaled, order, dim=-1)

    # compute the direction loss
    optimal_dir = output['vessel']  # original vessel direction, [B, 2, H, W] / [B, 3, H, W, D]
    sampled_optimal_dir = grid_sample(optimal_dir, sample_grid, permute=False)
    similarity = F.cosine_similarity(optimal_dir, sampled_optimal_dir)
    similarity_low = similarity * 0
    direction_loss = - torch.min(similarity, similarity_low).mean()

    # intensity continuity loss
    intensity_loss = 0.0
    flux_response = flux_response.unsqueeze(1)
    for scale in torch.linspace(-1.0, 1.0, sample_num):
        curr_grid = grid_base + optimal_dir_scaled * scale
        sampled_optimal_response = grid_sample(flux_response, curr_grid, permute=False)
        intensity_loss += F.mse_loss(flux_response, sampled_optimal_response) / sample_num
    return direction_loss, intensity_loss


def attention_loss(output, mean_val):
    att_loss = 0.0
    attentions = output['attentions']
    for i in range(len(attentions)):
        att_loss += torch.pow((attentions[i].mean() - mean_val), 2)
    return att_loss


def supervised_loss(ground_truth, output):
    recon = output['recon']
    recon = torch.sigmoid(recon)
    return F.mse_loss(recon, ground_truth) + F.l1_loss(recon, ground_truth)


def vessel_loss(image, output, loss_config):
    """
    Aggregate all the vessel loss
    :param image: original image [B, 1, H, W] / [B, 1, H, W, D]
    :param output: output dictionary 'vessel', 'radius', 'recon', 'attention'
    :param loss_config: dict, store the loss configurations
    :return: losses: dict, store the losses
    """
    grad_dims = loss_config['grad_dims']
    flux_sample_num = loss_config['flux_sample_num']
    intensity_sample_num = loss_config['intensity_sample_num']
    l_flux = loss_config['lambda_flux']
    l_direction = loss_config['lambda_direction']
    l_intensity = loss_config['lambda_intensity']
    l_recon = loss_config['lambda_recon']

    # get flux loss function configuration
    flux_loss_type = loss_config['flux_loss_type']
    if flux_loss_type == 'asymmetry':
        flux_loss_func = flux_loss_asymmetry
    else:
        flux_loss_func = flux_loss_symmetry

    # calculate losses
    flux_response, optimal_flux_loss = flux_loss_func(image, output, flux_sample_num, grad_dims)
    dir_loss, intense_loss = continuity_loss(image, output, flux_response, intensity_sample_num)
    rec_loss = recon_loss(image, output)

    # assign weights of losses
    dir_loss, intense_loss = dir_loss * l_direction, intense_loss * l_intensity
    optimal_flux_loss, rec_loss = optimal_flux_loss * l_flux, rec_loss * l_recon
    total_loss = optimal_flux_loss + rec_loss + dir_loss + intense_loss

    losses = {
        'flux_loss': optimal_flux_loss,
        'dirs_loss': dir_loss,
        'ints_loss': intense_loss,
        'rcon_loss': rec_loss,
        'total_loss': total_loss
    }
    if 'attentions' in output.keys():
        mean_exposure = loss_config['mean_exp']
        if mean_exposure != 0:
            l_att = loss_config['lambda_attention']
            att_loss = attention_loss(output, mean_exposure) * l_att
            total_loss += att_loss
            losses['attn_loss'] = att_loss
            losses['total_loss'] = total_loss
    return losses


def calc_local_contrast(image, estimated_r, sample_num, scale_steps):
    shape = image.size()                                            # 2D image [B, C, H, W] / 3D image [B, C, H, W, D]
    b, c, h, w = shape[0], shape[1], shape[2], shape[3]
    d = shape[4] if len(shape) == 5 else None
    sampling_vec = get_sampling_vec(sample_num, estimated_r)        # get sampling sphere / circle
    inside_scales = torch.linspace(0.1, 1.0, steps=scale_steps)     # multiple scales of inside
    outside_scales = torch.linspace(1.1, 2.0, steps=scale_steps)    # multiple scales of outside
    grid_base = sample_space_to_img_space(get_grid_base(image), h, w, d)     # Convert to [0, H]
    # distinguish the 2D and 3D image
    if d:
        estimated_r = estimated_r.permute(0, 2, 3, 4, 1)                    # [B, H, W, D, 1]
        img_contrast_i = torch.zeros(scale_steps, b, c, h, w, d).to(estimated_r.device)
        img_contrast_o = torch.zeros(scale_steps, b, c, h, w, d).to(estimated_r.device)
    else:
        estimated_r = estimated_r.permute(0, 2, 3, 1)                       # [B, H, W, 1]
        img_contrast_i = torch.zeros(scale_steps, b, c, h, w).to(estimated_r.device)
        img_contrast_o = torch.zeros(scale_steps, b, c, h, w).to(estimated_r.device)

    shift = int(sample_num / 2)
    for i in range(scale_steps):                                    # loop over the sampling scales
        scale_i, scale_o = inside_scales[i], outside_scales[i]
        for j in range(shift):                                      # loop over the sampling directions
            sample_dir1, sample_dir2 = sampling_vec[j], sampling_vec[j+shift]   # this is a 2d / 3d vector
            curr_rad1 = estimated_r[..., j:j+1] if estimated_r.size(-1) > 1 else estimated_r
            curr_rad2 = estimated_r[..., j+shift:j+shift+1] if estimated_r.size(-1) > 1 else estimated_r
            # get sample grids of the inside and outside for both directions
            sample_grid_pos_i, _ = get_sample_grid(sample_dir1, curr_rad1 * scale_i, grid_base, b, h, w, d)
            sample_grid_neg_i, _ = get_sample_grid(sample_dir2, curr_rad2 * scale_i, grid_base, b, h, w, d)
            sample_grid_pos_o, _ = get_sample_grid(sample_dir1, curr_rad1 * scale_o, grid_base, b, h, w, d)
            sample_grid_neg_o, _ = get_sample_grid(sample_dir2, curr_rad2 * scale_o, grid_base, b, h, w, d)
            # sampling intensities
            sampled_img_pos_i = torch.clip(image - grid_sample(image, sample_grid_pos_i, permute=False), min=0.0)
            sampled_img_neg_i = torch.clip(image - grid_sample(image, sample_grid_neg_i, permute=False), min=0.0)
            sampled_img_pos_o = torch.clip(image - grid_sample(image, sample_grid_pos_o, permute=False), min=0.0)
            sampled_img_neg_o = torch.clip(image - grid_sample(image, sample_grid_neg_o, permute=False), min=0.0)
            # adding the image local contrasts
            img_contrast_i[i] += torch.mul(sampled_img_pos_i, sampled_img_neg_i) / sample_num
            img_contrast_o[i] += torch.mul(sampled_img_pos_o, sampled_img_neg_o) / sample_num
    # compute the inside / outside ratio
    img_contrast_i = torch.mean(img_contrast_i, dim=0)
    img_contrast_o = torch.mean(img_contrast_o, dim=0)
    epsilon = 3e-2 if d else 1e-4
    img_local_contrast = torch.div(img_contrast_o, img_contrast_i + epsilon) - 1.0
    img_local_contrast = torch.sigmoid(img_local_contrast)
    return img_local_contrast


def direction_similarity_3d(original_dirs, rotated_dirs, rotate_angles, original_attentions):
    """
    Compute the direction similarities
    :param original_dirs: vessel directions of original images [B, 3, H, W, D]
    :param rotated_dirs: vessel directions of rotated images [B, 3, H, W, D]
    :param rotate_angles: angles of rotation in different axis [B, 3]:
    :param original_attentions: attention maps [B, H, W, D]
    :return: dir_sims, direction similarities
    """
    assert original_dirs.size() == rotated_dirs.size()
    batch_num = original_dirs.size(0)
    rotated_original_dirs = rotate_3d_batch(original_dirs, rotate_angles)
    rotated_original_atts = rotate_3d_batch(original_attentions.unsqueeze(1), rotate_angles)
    # change the shape of the directions to calculate matrix multiplication
    rotated_dirs = rotated_dirs.view(batch_num, 3, -1)
    rotated_original_dirs = rotated_original_dirs.view(batch_num, 3, -1)
    rotated_original_atts = rotated_original_atts.view(batch_num, 1, -1)
    # rotate the learnt vessel directions
    rotate_matrices = get_rotation_matrix_3d_batch(rotate_angles)
    rotated_original_dirs = torch.matmul(rotate_matrices, rotated_original_dirs)
    # normalization
    rotated_dirs = F.normalize(rotated_dirs, dim=1)
    rotated_original_dirs = F.normalize(rotated_original_dirs, dim=1)
    # exclude the empty areas
    indices1 = rotated_original_dirs[:, 0, :] == 0.0
    indices2 = rotated_original_dirs[:, 1, :] == 0.0
    indices3 = rotated_original_dirs[:, 2, :] == 0.0
    indices = torch.logical_and(torch.logical_and(indices1, indices2), indices3)
    # calculate the cosine similarities
    dirs_sims = F.cosine_similarity(rotated_original_dirs, rotated_dirs, dim=1)
    dirs_sims = torch.multiply(dirs_sims, rotated_original_atts.squeeze(1))
    dirs_sims = torch.abs(dirs_sims[torch.logical_not(indices)])
    return dirs_sims.mean()


def vessel_loss_ssl(image, original_output, rotated_output, loss_config, rotate_angles):
    """
    Aggregate all the vessel loss and SSL Loss
    :param image: original image [B, 1, H, W] / [B, 1, H, W, D]
    :param original_output: output dictionary 'vessel', 'radius', 'recon', 'attention'
    :param rotated_output: output dictionary 'vessel', 'radius', 'recon', 'attention'
    :param loss_config: dict, store the loss configurations
    :param rotate_angles: angles of rotation in different axis [B, 3]
    :return: losses: dict, store the losses
    """
    losses = vessel_loss(image, original_output, loss_config)
    l_augment = loss_config['lambda_augment']
    # output direction and rotated direction
    original_dirs = original_output['vessel']
    original_attentions = original_output['attentions'][-1]
    rotated_dirs = rotated_output['vessel']
    augment_loss = direction_similarity_3d(original_dirs, rotated_dirs, rotate_angles, original_attentions)
    augment_loss = -l_augment * augment_loss
    total_loss = losses['total_loss'] + augment_loss
    losses['total_loss'] = total_loss
    losses['augment_loss'] = augment_loss
    return losses


if __name__ == '__main__':
    test_orig_dirs = torch.rand(4, 3, 48, 48, 48) * 2 - 1
    test_rotate_angles = torch.rand(4, 3) * 360 - 180
    test_rotated_dirs = rotate_3d_batch(test_orig_dirs, test_rotate_angles)
    # change the shape of the directions to calculate matrix multiplication
    test_rotated_dirs = test_rotated_dirs.view(4, 3, -1)
    # rotate the learnt vessel directions
    test_rotate_matrices = get_rotation_matrix_3d_batch(test_rotate_angles)
    test_rotated_dirs = torch.matmul(test_rotate_matrices, test_rotated_dirs)
    test_rotated_dirs = test_rotated_dirs.view(4, 3, 48, 48, 48)
    print(direction_similarity_3d(test_orig_dirs, test_rotated_dirs, test_rotate_angles))
