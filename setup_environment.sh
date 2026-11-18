#!/bin/bash

# Setup script for PyTorch3D environment with uv
set -e

echo "Setting up PyTorch3D environment with uv..."

# Remove existing environment if it exists
if [ -d "omage_env" ]; then
    echo "Removing existing omage_env environment..."
    rm -rf omage_env
fi

# Create new environment with Python 3.11
echo "Creating new environment with Python 3.11..."
uv venv omage_env --python 3.11

# Activate environment
echo "Activating environment..."
source omage_env/bin/activate

# Install PyTorch with CUDA 11.8 support first
echo "Installing PyTorch with CUDA 11.8..."
uv pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies directly (not as editable package)
echo "Installing core dependencies..."
uv pip install numpy scipy scikit-image matplotlib imageio plotly opencv-python pillow tqdm jupyter ipython
uv pip install ninja
uv pip install einops timm
uv pip install fvcore iopath
uv pip install open3d trimesh
uv pip install diffusers transformers
uv pip install omegaconf hydra-core wandb
uv pip install torchdiffeq
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu118.html

# Install packages that need PyTorch to be available during build
echo "Installing libigl (needs PyTorch for compilation)..."
uv pip install libigl==2.5.1

# Blender as a Python module (no longer used; kept to leave the environment unchanged). The wheel is tied to Python 3.11
# and pins numpy==1.26.4.
echo "Installing bpy..."
uv pip install bpy==4.2.0

# Install fpsample from PyPI
echo "Installing fpsample..."
uv pip install fpsample

# # Install PyTorch3D from source (most reliable method)
echo "Installing PyTorch3D from source..."
rm -rf /tmp/pytorch3d
git clone https://github.com/facebookresearch/pytorch3d.git /tmp/pytorch3d
cd /tmp/pytorch3d
uv pip install . --no-build-isolation
cd -

uv pip install torch-cluster --no-build-isolation
uv pip install flash-attn --no-build-isolation
uv pip install accelerate
uv pip install natten==0.17.3+torch240cu118 -f https://shi-labs.com/natten/wheels

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "Testing core packages..."
python -c "import open3d; print(f'Open3D: {open3d.__version__}')" || echo "Open3D: Failed"
python -c "import trimesh; print(f'Trimesh: {trimesh.__version__}')" || echo "Trimesh: Failed"
python -c "import omegaconf; print('OmegaConf: OK')" || echo "OmegaConf: Failed"
python -c "import pytorch3d; print(f'PyTorch3D: {pytorch3d.__version__}')" || echo "PyTorch3D: Failed"
python -c "import igl; print('igl: OK')" || echo "igl: Failed"
python -c "import fpsample; print('fpsample: OK')" || echo "fpsample: Failed"

echo ""
echo "Environment setup complete!"
echo "Activate with: source omage_env/bin/activate"

# Clean up
rm -rf /tmp/pytorch3d

echo ""
echo "Notes:"
echo "- All packages installed directly (not as editable project)"
echo "- If any package failed, try installing it manually after activating the environment"