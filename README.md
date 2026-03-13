<div align="center">
<h1><code>gsplat-geer</code>: An Open-Source Library for Exact and Efficient Gaussian Rendering</h1>
</div>

⚠️ Work in Progress: This repository is currently undergoing cleanup and restructuring.

`gsplat-geer` is an extension of the open-source [`gsplat`](https://github.com/nerfstudio-project/gsplat) library from [Nerfstudio](https://docs.nerf.studio/) for 3DGEER-based rasterization.

## 📷`gsplat` Rasterization
This repo extends the [`rasterization()`](https://docs.gsplat.studio/main/apis/rasterization.html#gsplat.rasterization) function provided by `gsplat` to rasterize 3D Gaussians to image planes. The argument `with_geer: bool = False` rasterizes Gaussians using the 3DGEER's PBF algorithm when set to True. For users using this function, note:

- `with_geer=True` only works with `with_eval3d=True`.
- `with_geer` only renders one image plane at a time.
- To train/render pinhole camera with distortion, setting the distortion parameters to `radial_coeffs`, `tangential_coeffs`, `thin_prism_coeffs`.
- To train/render fisheye camera with distortion, 
setting the distortion parameters to `radial_coeffs` and set `camera_model="fisheye"`.

These are consistent with `gsplat`'s 3DGUT implementation (`with_ut`).

## 🏃Quick Start
### Training
Passing in `--with_geer --with_ut` to the `simple_trainer.py` arg list will enable training with 3DGEER. Note in `gsplat-geer`, only MCMC densification is supported for 3DGEER training.

#### Install Dependencies
```
pip install -r examples/requirements.txt
```
#### Training Script
```
python examples/simple_trainer.py mcmc --with_geer --with_eval3d ... <OTHER ARGS>
```

### Rendering
Once trained, you can view the 3DGS through the nerfstudio-style viewer to export videos. Play around with the fisheye setting and the FOV!

#### Install Dependencies
```
pip install -r examples/requirements.txt
```
#### Rendering Script
```
CUDA_VISIBLE_DEVICES=0 python simple_viewer.py --with_geer --with_eval3d --ckpt results/benchmark_mcmc_1M_3dgut/garden/ckpt_29999_rank0.pt 
```

## ✨Opensource Community 
### `drivestudio-geer` 
> TBD
### `stormGaussian-geer` 
> TBD
### How to use in your project
> TBD (see [./app](app/) for details).

## 🙏Special `gsplat-geer` Extension OSS Acknowledgments
<p align="left">
  <strong>Core Contributors:</strong><br>
  Edward Lee<sup>1,2*</sup> (GEER Public Integration), <br>
  Zixun Huang<sup>1,‡</sup> (GEER Algorithm Derivation / Implementation), <br>
  Cho-Ying Wu<sup>1</sup> (GEER Implementation)
</p>

<p align="left">
  <strong>Senior Mgmt:</strong><br> 
  Wenbin He<sup>1</sup>, Xinyu Huang<sup>1</sup><br>
</p>

<p align="left">
  <strong>Supervision:</strong><br>
  Liu Ren<sup>1</sup>
</p>

<p align="left">
  <strong>Acknowledgements for additional contributions:</strong><br>
  Hengyuan Zhang<sup>1</sup> (Close-Up Parking Data Calibration)
<br>

### Institution Acknowledgements
<p align="left">
  <img width="200" src="assets/bosch-logo.png" alt="Bosch Logo" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img width="200" src="assets/stanford-logo.png" alt="Stanford Logo" />
</p>

<p align="left">
  <sup>1</sup> <strong>Bosch Center for AI</strong>, Bosch Research North America &nbsp;&nbsp;&nbsp;&nbsp; 
  <sup>2</sup> <strong>Stanford University</strong>
</p>

> The special extension work was performed when <sup>*</sup> worked as an intern at <sup>1</sup> under the mentorship of <sup>‡</sup>.

## 💡License
`gsplat-geer` is released under the AGPL-3.0 License. See the [LICENSE](./LICENSE.md) file for details.
This project is built upon `gsplat` (Apache-2.0 License) by Inria. We thank the authors for their excellent open-source work. The original license and copyright notice are included in this repository, see the file [3rd-party-licenses.txt](./3rd-party-licenses.txt).
