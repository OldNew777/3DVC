# 6D pose estimation



## requirements

```
numpy
torch torchvision torchaudio >= 2.0	(for new features)
torch-cuda
transform3d
pandas
matplotlib
trimesh
sklearn
cv2
```



## guidance

```
python train.py
```

For coding convenience, all configs are set in `config.py`

- `algo_type: 'icp'/'nn'` to specify the algorithm

- `process: 'train/test'` only for nn

- `data_dir` : the real path of the data directory that contains

  > |-- models
  >
  > |-- testing_data
  >
  > |-- training_data

- `n_seg` : the max segmentation number of each dimension in the ICP initialization



The output directory will be `data_dir/output`



## structures

- `mylogger` : A logger implemented by myself
- `train.py` : The main pipelines (including ICP and learning-based method)
- `config.py` : Config for the algorithm/input/output/...
- `dataset.py` : A lazy dataset loader written by myself
- `eval.py` : File provided by `samples`, to evaluate the transformation answers
- `icp.py` : The main body of ICP algorithm, including iterative part and initialization
- `loss.py` : CD loss
- `model.py` : Neural network model
- `obj_model.py` : Class for object I/O and process
- `utils.py` : File provided by `samples`, adding some basic functions written by myself