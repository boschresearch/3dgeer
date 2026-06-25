# `drivestudio-geer`

We have released integration with [DriveStudio](https://github.com/ziyc/drivestudio)! In our patch, we provide 3DGEER and 3DGUT training and rendering with a dynamic, temporal Viser viewer for viewing trained representations!
<img src='../../assets/drivestudio_viewer_fisheye_demo.gif' alt='drivestudio-geer' style='width: 100%;'>

## 🏃Quick Start

### `gsplat-geer` Setup
Clone this repo and install the [dependencies](../../README.md#install-dependencies).

### `drivestudio-geer` Setup
Clone the [DriveStudio repo](https://github.com/ziyc/drivestudio).
```bash
git clone --recursive https://github.com/ziyc/drivestudio.git
cd drivestudio
```
Add the patch from `gsplat-geer`.
```bash
git apply path/to/gsplat-geer/0001-Enable-training-with-3DGUT-3DGEER-and-add-viewer.patch
```

### Install Dependencies (including gsplat-geer)
Run the following commands to setup the environment (similar to DriveStudio [Installation](https://github.com/ziyc/drivestudio#-installation)):
```bash
# Create the environment
conda create -n drivestudio python=3.9 -y
conda activate drivestudio
pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast

# Set up gsplat-geer
cd /path/to/gsplat-geer
pip install --no-build-isolation -e .
cd /path/to/drivestudio

# Set up for SMPL Gaussians
cd third_party/smplx/
pip install -e .
cd ../..
```

### Prepare Data
Follow [DriveStudio Prepare Data](https://github.com/ziyc/drivestudio#-prepare-data) or process your own data.

### Train a Model
To train a model with 3DGEER or 3DGUT, in the training YAML configs in the `configs` directory, under `trainer` > `render`, add the `render_mode` parameter, which can be set to be `default` (3DGS splatting), `ut` (3DGUT), or `geer` (3DGEER). An example can be seen in `configs/omnire_geer.yaml`. Then follow [DriveStudio Training](https://github.com/ziyc/drivestudio#training).

*Note*: `gsplat-geer` does not support camera pose optimization at this time, so `model` > `CamPose` may have to be removed from the config.

### View a Model
To view a model checkpoint using our viewer, run
```bash
python tools/viewer.py --ckpt /path/to/ckpt
```
This viewer can
- Render DriveStudio checkpoints
- Move between novel views
- Render with 3DGS, 3DGUT, or 3DGEER
- Change between pinhole and fisheye rendering, with changeable distortion parameters
- Filter Gaussian classes in checkpoint, such as background and RigidNodes
- Snap to dataset camera views and added keyframes
- Create and save trajectories between spatial + temporal keyframes
- Export videos of these trajectories
- And more!
