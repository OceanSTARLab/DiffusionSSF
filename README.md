# Learning SSF Data Distribution with Diffusion Model 

This repository contains the code for the paper "Learning data distribution of three-dimensional ocean sound speed fields via diffusion models".

## Description

Please find the 3DSSF dataset [here](https://drive.google.com/file/d/1M5wtFXQkanNdoM9aLdzPWrzuqe-sL-H0/view?usp=share_link). For model evaluation, you can either train a new model or use the pre-trained model available at [here](https://drive.google.com/file/d/1rxv_PbICrdLhAmBo0UEXjeQy_mDKLvlf/view?usp=share_link).

## Usage

To train the model, modify the data path in `run_lib.py` and run the following command:

```
python main.py --config configs/ve/hycom_ncsnpp_deep_continuous.py --workdir hycomlog --mode train
```

Here the `hycomlog` is the log dir, directory, which stores the model checkpoints and training information. `hycom_ncsnpp_deep_continuous.py` contains the configuration settings for the training procedure. You can modify this file to adjust the training settings as needed.

After the training, you can run the following command for model evaluation:

```
python ssf_sample.py
```

 Note that you need to ensure the correct checkpoint path is specified. 

## References and Acknowledgements

The implementation is based on

```
@inproceedings{
  song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=PxTIG12RRHS}
}
```

For more information about the code, you can refer to <https://github.com/yang-song/score_sde>.

