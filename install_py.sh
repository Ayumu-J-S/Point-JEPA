# !/bin/bash
pip install torch==1.13.1  torchvision==0.14.1

# Need this for pytorch 3d
pip install setuptools==69.5.1
export FORCE_CUDA=1
pip install pytorch3d@git+https://github.com/facebookresearch/pytorch3d.git@799c1cd21beff84e50ac4ab7a480e715780da2de

pip install -r requirements.txt
pip install -U 'jsonargparse[signatures]>=4.17.0'
