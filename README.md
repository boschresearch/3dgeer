<div align="center">
<h1>[2026 ICLR] 3DGEER: 3D Gaussian Rendering <br> Made Exact and Efficient for Generic Cameras</h1>

[**Zixun Huang**](https://zixunh.github.io/) · [**Cho-Ying Wu**](https://choyingw.github.io/) · [**Yuliang Guo**](https://yuliangguo.github.io/) · [**Xinyu Huang**](https://scholar.google.com/citations?user=cL4bNBwAAAAJ&hl=en) · [**Liu Ren**](https://www.liu-ren.com/)

Bosch Center for AI, Bosch Research North America

<a href="https://arxiv.org/abs/2505.24053"><img src='https://img.shields.io/badge/arXiv-3DGEER-red' alt='ArXiv'></a>
<a href="https://openreview.net/forum?id=4voMNlRWI7"><img src='https://img.shields.io/badge/OpenReview-3DGEER-orange' alt='OpenReview'></a>
<a href="https://iclr.cc/virtual/2026/poster/10011512"><img src='https://img.shields.io/badge/ICLR_2026-3DGEER-blue' alt='ICLR 2026'></a>
<a href='https://zixunh.github.io/3d-geer/'><img src='https://img.shields.io/badge/Project_Page-3DGEER-green' alt='Project Page'></a>
<a href='https://www.youtube.com/watch?v=Grl9jSMIgds'><img src='https://img.shields.io/badge/Video-3DGEER-yellow' alt='Video'></a>

<p align="center">
  <a href='https://zixunh.github.io/3d-geer'>
  <img src="assets/teaser.gif" alt="teaser" style="width: 100%;">
  Check Project Page for More Visuals
  </a>
</p>
</div>

## 🧐Overview
<div class="row">
    <div class="col-md-8 col-md-offset-2">
        <section>
            <p>
              3D Gaussian Splatting (3DGS) has rapidly become one of the most influential paradigms in neural rendering.
              It delivers impressive real-time performance while maintaining high visual fidelity, making it a strong alternative to NeRF-style volumetric methods.
              But there is a fundamental problem hiding beneath its success:
            </p>
            <blockquote style="font-size: 13px;">
                <strong>Splatting doesn't obey exactness in projective geometry.</strong>
            </blockquote>
            <p>
              The splatting approximation is usually harmless for narrow field-of-view (FoV) pinhole cameras.
              However, once we move to fisheye, omnidirectional, or generic camera models — especially those common in robotics and autonomous driving — the approximation error becomes significant.
            </p>
          </section>
    </div>
</div>

## 😺Key Features
This repository contains the official authors implementation associated with the [**ICLR 2026**](https://iclr.cc/virtual/2026/poster/10011512) paper "3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras". The `gsplat-geer` OSS extension can be found [here](#special-extension).
<div class="row">
    <div class="col-md-8 col-md-offset-2">
        <section>
            <ul>
              <li>Projective <strong>exactness</strong> + Real-time <strong>efficiency</strong></li>
              <li>Compatibility with generic camera models (pinhole / fisheye) + Strong generalization to <strong>extreme FoV</strong></li>
              <li>Adaptation to <strong>widely-used</strong> GS frameworks including <code>diff-gaussian-rasterization</code>, <code>gsplat</code>, <code>drivestudio</code></li>
            </ul>
        </section>
    </div>
</div>

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">📰BibTeX</h2>
    <pre><code>@misc{huang20263dgeer3dgaussianrendering,
      title={3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras}, 
      author={Zixun Huang and Cho-Ying Wu and Yuliang Guo and Xinyu Huang and Liu Ren},
      year={2026},
      eprint={2505.24053},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2505.24053}, 
}</code></pre>
  </div>
</section>


## 🎉News
- **TBD**: `drivestudio-geer` and `stormGaussian-geer` will be released [here](#special-extension) as well!
- **2026-03-09**: `gsplat-geer` released [here](#special-extension)!
- **2026-03-09**: Code released! Can Gaussian rendering be both exact and fast without relying on lossy splatting? **Check out 3DGEER**!
- **2026-03-09**: Code release approved. License updated. Requested admin to push code to BoschResearch.
- **2026-01-25**: 3DGEER accepted to **ICLR 2026**, with an [initial review](https://openreview.net/forum?id=4voMNlRWI7) of average **7** (top 1% score).
- **2025-05-29**: Preprint released on [Arxiv](https://arxiv.org/abs/2505.24053).


## 📷3DGEER-CUDA-Rasterizer
The full CUDA implementation can be found here: [./submodules/geer-rasterizer/](./submodules/geer-rasterizer/).
#### Key Insights: Fixing the Math Behind Gaussian Rendering

- Ray–Gaussian Integral (Forward & Backward): Analytical forward rendering and backward gradient computation. (See [paper](https://arxiv.org/pdf/2505.24053) Appendix C for the math.)

  <div align="center">
    <img src="assets/forward2backward.gif" width="60%">
  </div>

- Particle Bounding Frustum: Efficient AABB for ray–particle association. (See [paper](https://arxiv.org/pdf/2505.24053) Appendix D for the math.)

  <div align="center">
    <img src="assets/asso.gif" width="60%">
  </div>

#### Conda Based Installation
Following the 3dgs dependencies https://github.com/graphdeco-inria/gaussian-splatting, and then run the following command to replace the CUDA rasterizer for a geer-version 3dgs environment:
```sh
pip install ./submodules/geer-rasterizer
```
#### Docker Configuration
Set you data path and 3dgeer codebase path in `./docker/init_my_docker.sh`.
```sh
# Build up 3dgs environments for 3DGEER. Example:
bash ./docker/build.sh 4090
# Reset Docker on Terminal 1
bash ./docker/init_my_docker.sh
# If you modify algorithm upon our geer-rasterizer, inside docker container, recompile:
pip install --no-build-isolation ./submodules/geer-rasterizer
```
#### SIBR Viewer Configuration with Docker 
```sh
# Enter Workspace for SIBR Viewer on Terminal 2
bash ./docker/run_my_docker.sh
# Inside docker container, run:
$sibr_rg
```

## 🏃Quick Start
### 1. Data Preparation
Our framework follows the standard COLMAP data structure. For generic cameras (e.g., Fisheye), ensure your `cameras.txt` includes the specific intrinsic parameters. [Link to detailed data format documentation](./data).

**Expected Directory Structure**:
```
|_./data/scnt
    |_datasets # e.g., download data into this folder
        |_1d003b07bd
        |   |_colmap
        |   |   |_images.txt
        |   |   |_points3D.txt
        |   |   |_cameras.txt
        |   |   |_...
        |   |_nerfstudio
        |   |   |_transforms.json
        |   |_resized_images
        |       |_000000.jpg
        |       |_000001.jpg
        |       |_...
        |_e3ecd49e2b
        |_...
```
### 2. Training 3DGEER
To train 3DGEER on scannet++ data:
```bash
bash ./scripts/train_scnt.sh
```
> full training scripts will be released soon.

### 3. Rendering & Evaluation
To render high-quality images and compute PSNR/SSIM/LPIPS:
```bash
bash scripts/render_scnt.sh <SCENE_ID> <DATA_ROOT> <CKPT_DIR> <MODE>
bash scripts/eval_scnt.sh <SCENE_ID> <DATA_ROOT> <CKPT_DIR> <MODE>
```

**Arguments:**

`SCENE_ID` : scene name (e.g. steakhouse, 1d003b07bd/dslr)

`DATA_ROOT` : root directory of the formatted dataset

`CKPT_DIR` : directory containing the trained model checkpoint

`MODE` : rendering backend, (`BEAP`, `KB` or `PH`)

> set `DIST_SCALING` as 0 in the shell to render EQ under KB mode.
> enlarge the value of `FOCAL_SCALING` to test extreme large FoV.
> For fair comparison, we recommend evaluating with `BEAP` mode, which ensures consistent metric computation across different rendering backends.

**Example:**

Aria dataset
```bash
bash scripts/render_scnt.sh steakhouse data/aria/scannetpp_formatted ckpt/aria KB
bash scripts/eval_scnt.sh steakhouse data/aria/scannetpp_formatted ckpt/aria KB
bash scripts/eval_scnt.sh steakhouse data/aria/scannetpp_formatted ckpt/aria BEAP
```
Scannet++ dataset
```bash
bash scripts/render_scnt.sh 1d003b07bd/dslr data/scnt/datasets ckpt/scnt KB
bash scripts/eval_scnt.sh 1d003b07bd/dslr data/scnt/datasets ckpt/scnt KB
bash scripts/eval_scnt.sh 1d003b07bd/dslr data/scnt/datasets ckpt/scnt BEAP
```
Tanks and Temples dataset
```bash
bash scripts/render_scnt.sh truck data/tt/datasets ckpt/tt PH
bash scripts/eval_scnt.sh truck data/tt/datasets ckpt/tt PH
bash scripts/eval_scnt.sh truck data/tt/datasets ckpt/tt BEAP
```

### 4. Available Checkpoints
You can download the pre-trained checkpoints for the scenes shown on our project webpage:
- Scannet++: Kitchen, Lab, Officeroom, Bedroom
- ZipNeRF: Alameda, Berlin, London, NYC
- Aria: Livingroom, Steakhouse
- Tank and Temples: Train, Truck
- Customized Parking: Bosch Center

Download from HuggingFace:
https://huggingface.co/datasets/ZixunH/3DGEER_ckpt

## 🙏Special Extension
<p align="center">
  <a href='https://github.com/boschresearch/3dgeer/tree/gsplat-geer'><img src="assets/deliverypath.jpg" alt="teaser" style="width: 100%;"></a>
  3DGEER supports the opensource community with <code>gsplat</code> integration. <br />Check out our <a href='https://github.com/boschresearch/3dgeer/tree/gsplat-geer'><code>gsplat-geer</code></a> branch for details.
</p>

## ⛽️Contributing
Feel free to drop a pull request whenever!

## 👀Visuals ([More](https://zixunh.github.io/3d-geer))

<div class="col-md-8 col-md-offset-2">
    <h4>
      Out-of-Distribution (OOD) Extreme FoV Results
    </h4>
    <div align="center">
      <img src="assets/our_gif/gut_render.gif" width="47%" style="padding: 2px;">
      <img src="assets/our_gif/geer_render.gif" width="47%" style="padding: 2px;">
    </div>
    <div align="center">
      <img src="assets/our_gif/gut_train_gut_render.gif" width="47%" style="padding: 2px;">
      <img src="assets/our_gif/gut_train_geer_render.gif" width="47%" style="padding: 2px;">
    </div>
    <p align="center">
      Side-by-Side Comparison on Close-Up Parking Data: 3DGEER's PBF association (Right Col.) has less popping issues (First Row) and no grid-line artifacts (Second Row) compared w/ UT (Left Col.).
    </p>
</div>

<div class="col-md-8 col-md-offset-2">
    <h4>
        High-Quality Large FoV Results
    </h4>
    <div align="center">
      <img src="assets/our_gif/alameda3_web.gif" width="32%" style="padding: 2px;">
      <img src="assets/our_gif/alameda1_web.gif" width="32%" style="padding: 2px;">
      <img src="assets/our_gif/alameda2_web.gif" width="32%" style="padding: 2px;">
    </div>
</div>
<div class="col-md-8 col-md-offset-2">
    <h4>
        Highly Distorted & Close-Up Views
    </h4>
    <div align="center">
      <img src="assets/our_gif/berlin4_web.gif" width="32%" style="padding: 2px;">
      <img src="assets/our_gif/berlin8_web.gif" width="32%" style="padding: 2px;">
      <img src="assets/our_gif/nyc2_web.gif" width="32%" style="padding: 2px;">
    </div>
</div>


## 💡License
3DGEER is released under the AGPL-3.0 License. See the [LICENSE](./LICENSE.md) file for details.
This project is built upon [3D Gaussian Splatting by Inria](https://github.com/graphdeco-inria/gaussian-splatting). We thank the authors for their excellent open-source work. The original license and copyright notice are included in this repository, see the file [3dgs-license.txt](./3dgs-license.txt).
