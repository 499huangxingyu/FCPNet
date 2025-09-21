# FCPNet
<div align="center">

# Feature Copy-Paste Network for Lung Cancer EGFR Mutation Status Prediction in CT Images  
**MICCAI 2025**

[Code]  [https://github.com/499huangxingyu/FCPNet](#)
</div>

# Citation
If you find **FCPNet** useful, please cite:

```bibtex
@inproceedings{HuaXin_Feature_MICCAI2025,
  author    = {Huang, Xingyu and Wang, Shuo and Liu, Chengcai and Sang, Haolin and Wu, Yi and Tian, Jie},
  title     = {Feature Copy-Paste Network for Lung Cancer EGFR Mutation Status Prediction in CT Images},
  booktitle = {Proceedings of Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year      = {2025},
  month     = {September},
  publisher = {Springer Nature Switzerland},
  volume    = {LNCS 15974},
  pages     = {208--217}
}

# Usage

## Network Architecture
- The core model is implemented in **`FCPNet.py`**.

## Data Setup
Set your dataset root in **`Utils/Options.py`**:
```python
# Utils/Options.py
data_path = "/path/to/your/dataset"
