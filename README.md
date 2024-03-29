# Semantic Bird's-Eye-View Map Prediction Using Horizontally-Aware Pyramid Occupancy Network

This repo is a part of senior project titled "Semantic Bird's-Eye-View Map Prediction Using Horizontally-Aware Pyramid Occupancy Network" by Thanapat Teerarattanyu and Tunlaton Wongchai, Data Science and Business Analytics Program, School of Information Technology, King Mongkut's Institute of Technology Ladkrabang.

## Project setup
```
conda create --name myenv python=3.9.0
conda activate myenv
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install --no-deps -r requirements.txt
```


## Label Generation
1. Download [nuScenes](https://www.nuscenes.org/nuscenes) dataset and its map expansion.
2. Extract a dataset folder at root and rename it to "nuscenes".
3. Set `nuscenes_version` in `\configs\configs.yml`.
3. Run `label_generation.py`.
