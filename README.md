# DepACM: A Depth-Aware Consistency Model for Real-Time Underwater Image Restoration[[📖 Paper](https://doi.org/10.1109/TCSVT.2026.3718661)]

## 💥 News
- **[2026.7]** Our paper has been accepted by **TCSVT** 🎉
## 🛠️ Environment Setup
- Python = 3.10.16
- PyTorch = 2.2.2
- torchvision = 0.17.2
- CUDA = 11.8
- flash_attn = 2.4.3.post1

### Installation
```bash
# Clone the repository
git clone https://github.com/GongYuhaoBrad/DepACM.git
cd DepACM

# Create conda environment
conda create -n depacm python=3.10.16
conda activate depacm
```

## 📂 Dataset Preparation
Please organize the data as follows (refer to the [[📖 Paper](https://doi.org/10.1109/TCSVT.2026.3718661)] for details):
```text
dataset/
└── UIEB/
    ├── TRAIN/
    │   ├── input_256/
    │       0001.png
    │       0002.png
    │          ...
    │   └── target_256/
    │       0001.png
    │       0002.png
    │          ...
    └── VAL/
        ├── input_256/
        └── target_256/
```

## 📊 Testing
Please download the pre-trained models [[LUDEN](https://drive.google.com/drive/folders/1Tj1701enAGlDQRL3hw081sKdAjnDZy6Z?usp=sharing)] and [[SDGDN](https://drive.google.com/drive/folders/1Tj1701enAGlDQRL3hw081sKdAjnDZy6Z?usp=sharing)] from Google Drive, and place the files in the `Pretrain_models` folder.

After configuring the environment and dataset as described above, set the checkpoint path in `test.py`, and then run:
```bash
python test.py
```

## 🚀 Training
Download the pre-trained model [[LUDEN](https://drive.google.com/drive/folders/1Tj1701enAGlDQRL3hw081sKdAjnDZy6Z?usp=sharing)] and place it in the `pretrain_models` folder. Then run:
```bash
python train.py
```

## 📜 Citation
If you find our work useful, please cite:
```bibtex
@ARTICLE{11631761,
  author={Wang, Yingbo and Gong, Yuhao and Duan, Bofeng and Liu, Tongfei and Du, Xiaogang and Lei, Tao and Nandi, Asoke K.},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={DepACM: A Depth-Aware Consistency Model for Real-Time Underwater Image Restoration}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2026.3718661}}
```
## Acknowledgement
Our code architecture is based on the [[consistency_models](https://github.com/openai/consistency_models.git)].


## 📧 Contact
If you have any questions, please feel free to contact us:
- 📧 Email: [241612100@sust.edu.cn](mailto:241612100@sust.edu.cn)
