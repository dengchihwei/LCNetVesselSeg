# -*- coding = utf-8 -*-
# @File Name : visual_test
# @Date : 2022/10/6 15:36
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


from loss.loss_func import *
from loss.loss_utils import *
from matplotlib import pyplot as plt


# slice plotting functions
def plot_slice(slice_img, low, high, color='gray'):
    slice_img = slice_img.cpu().detach().numpy()
    plt.figure()
    plt.imshow(slice_img, cmap=color, vmin=low, vmax=high)
    plt.colorbar()


# contrast test function
def contrast_test():
    # Test for 2D images
    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    path = '/Users/dengzhiwei/Desktop/NICR/EyeVessel/DRIVE/test/images/02_test.tif'
    # mask_path = '/Users/dengzhiwei/Desktop/NICR/EyeVessel/DRIVE/test/mask/02_test_mask.gif'
    image = np.array(Image.open(path))[..., 1] / 255.0
    # mask = torch.from_numpy(np.array(Image.open(mask_path)))
    # att_path = '/Users/dengzhiwei/Desktop/NICR/drive_results/adaptive_lc/drive_rad_test_naive.npy'
    # att = np.load(att_path)
    # test_radius = torch.from_numpy(att[2:3]).unsqueeze(0).float()
    image = gaussian_filter(1.0 - image, 0.45)
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    # gradient = torch.gradient(image, dim=(2, 3))
    # gradient = gradient[1]      # + gradient[1]
    test_radius = torch.zeros_like(image)
    test_radius[:, 0, :, :] = 2.0
    img_local_contrast = calc_local_contrast(image, test_radius, 32, 5)
    print(img_local_contrast.mean())

    plot_slice(1.0 - image[0, 0], None, None)
    plot_slice(img_local_contrast[0, 0], None, None)
    # plot_slice(gradient[0, 0], None, None)
    plot_slice(test_radius[0, 0], None, None)
    plt.show()


# swap test function
def swap_test():
    directions = torch.zeros(1, 2, 128, 128)
    for i in range(128):
        for j in range(128):
            directions[0, 0, i, j] = j - 64.0
            directions[0, 1, i, j] = 64.0 - i
    directions = F.normalize(directions, dim=1)

    optimal_dir = directions.permute(0, 2, 3, 1)
    # optimal_dir = swap_order(optimal_dir, [1, 0], dim=-1)
    grid_base = sample_space_to_img_space(get_grid_base(directions), 128, 128)
    sample_grid = img_space_to_sample_space(grid_base + optimal_dir, 128, 128)
    print(sample_grid[0, 50, 72, :] * 128 + 64)
    output = grid_sample(directions, sample_grid, permute=False)
    print(directions[0, :, 60, 60])
    print(output[0, :, 60, 60])

    input_image = torch.arange(4*4).view(1, 1, 4, 4).float()

    # Create grid to up-sample input
    d = torch.linspace(-1, 1, 4)
    meshx, meshy = torch.meshgrid((d, d), indexing='ij')
    grid = torch.stack((meshy, meshx), 2)
    grid = grid.unsqueeze(0)    # add batch dim
    print(grid[0, 0, 2] * 4 + 2)
    output = F.grid_sample(input_image, grid, align_corners=True, padding_mode='border')
    print(output)


if __name__ == '__main__':
    contrast_test()
