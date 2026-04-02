import numpy as np
import torch
import math

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def focal2fov2(focal, pixels):
    return pixels / focal

def focal2halffov2(focal, pixels):
    return pixels / 2 / focal

def fov_sample2ray(fovx, fovy, interval):
    theta_arr = torch.arange(interval / 2, fovx, interval)
    theta_arr, _ = torch.sort(torch.cat((-theta_arr, theta_arr)))
    phi_arr = torch.arange(interval / 2, fovy, interval)
    phi_arr, _ = torch.sort(torch.cat((-phi_arr, phi_arr)))

    return theta_arr.float(), phi_arr.float()

def omni_map_z(m, z, xi=0.0): #1.1
    return m / (1+xi*(z/(torch.abs(z)))*(1+m**2)**0.5)

def omni_tan(Ks, width, height, camera_model, step, fov_mod=1.75, data_device="cuda"):
    # Ks [..., C, 3, 3]
    K = Ks.to("cpu").squeeze() # one image

    focal_length_x = K[0, 0]
    focal_length_y = K[1, 1]

    if camera_model == "pinhole":
        FoVx = focal2fov(focal_length_x, width)
        FoVy = focal2fov(focal_length_y, height)
    elif camera_model == "fisheye":
        # Change the fov to match the undistorted image
        FoVx = min(np.pi, focal2fov2(focal_length_x, width) * fov_mod) #/ 0.8
        FoVy = min(np.pi, focal2fov2(focal_length_y, height) * fov_mod) #/ 0.8

    arr_theta, arr_phi = fov_sample2ray(FoVx/2, FoVy/2, step)

    cos_theta = torch.cos(arr_theta)
    cos_phi = torch.cos(arr_phi)

    cos_theta = torch.where(torch.abs(cos_theta) < 1e-7, torch.full_like(cos_theta, 1e-7), cos_theta).to(data_device)
    cos_phi = torch.where(torch.abs(cos_phi) < 1e-7, torch.full_like(cos_phi, 1e-7), cos_phi).to(data_device)

    tan_theta = torch.tan(arr_theta).to(data_device)
    tan_phi = torch.tan(arr_phi).to(data_device)

    omni_tan_theta = omni_map_z(tan_theta, cos_theta)
    omni_tan_phi = omni_map_z(tan_phi, cos_phi)

    tanfovx = np.tan(FoVx * 0.5)
    tanfovy = np.tan(FoVy * 0.5)

    return omni_tan_theta, omni_tan_phi, tanfovx, tanfovy