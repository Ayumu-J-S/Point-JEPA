source ~/cc_ssl/bin/activate

pip install torch==2.1.1  torchvision==0.16.1
# pip install git+https://github.com/facebookresearch/pytorch3d.git@799c1cd21beff84e50ac4ab7a480e715780da2de

pip install -r requirements_compute_can.txt
pip install -U 'jsonargparse[signatures]>=4.17.0'