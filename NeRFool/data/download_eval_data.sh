#!/usr/bin/env bash

# # deepvoxels
# # download the deepvoxels data [synthetic_scenes.zip]
# # https://drive.google.com/drive/folders/1ScsRlnzy9Bd_n-xw83SP-0t548v63mPH
# gdown https://drive.google.com/uc?id=1lUvJWB6oFtT8EQ_NzBrXnmi25BufxRfl
# unzip synthetic_scenes.zip -d deepvoxels

# # nerf realistic 360
# # download the nerf synthetic data [nerf_synthetic.zip]
# # https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
# gdown https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG
# unzip nerf_synthetic.zip

# # real forward facing
# # download NeRF LLFF test data [nerf_llff_data.zip]
# # https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
# gdown https://drive.google.com/uc?id=16VnMcF1KJYxN9QId6TClMsZRahHNMW5g
# unzip nerf_llff_data.zip





# IBRNet captures
gdown https://drive.google.com/uc?id=1rkzl3ecL3H0Xxf5WTyc2Swv30RIyr1R_
unzip ibrnet_collected.zip

# Spaces dataset
git clone https://github.com/augmentedperception/spaces_dataset

# # RealEstate 10k
# ## make sure to install ffmpeg - sudo apt-get install ffmpeg
# git clone https://github.com/qianqianwang68/RealEstate10K_Downloader
# cd RealEstate10K_Downloader
# python3 generate_dataset.py train
# cd ../

# # Google Scanned Objects
# gdown https://drive.google.com/uc?id=1w1Cs0yztH6kE3JIz7mdggvPGCwIKkVi2
# unzip google_scanned_objects_renderings.zip
