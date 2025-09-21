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
```
## Usage

- **Network:** implemented in `FCPNet.py`.
- **Data path:** set `data_path` in `Utils/Options.py`.

### Train
```bash
python FCPNet_train.py
# or
bash FCPNet_train.sh
```

### Test
```bash
python FCPNet_test.py
```

- **Outputs:** Results are saved to `./Result`.
