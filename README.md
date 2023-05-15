# Simple_MLP_Predictor_NAS
this repo contains predictor based NAS with NAS-Bench-101 dataset, it is just practical simple example 

## dataset.py
dataloader of NAS-Bench-101 dataset. if you download orginal NAS-Bench-101 tfrecord file then you have to change into HDF5 file, this can be done tools/nasbench_tfrecord_converter.py   HDF5 file is faster than original tfrecord file 

Download `nasbench_full.tfrecord` from [NasBench](https://github.com/google-research/nasbench/tree/b94247037ee470418a3e56dcb83814e9be83f3a8), and put it under `data`. Then run

```
python tools/nasbench_tfrecord_converter.py
```

## model.py 
define a predictor model that can predict final accuracy of Architecture
you can build other model, here MLP model that has multiple input(architecture adjancey matrix, operations, mask, num_vertices)

## train.py
train predictor model with NAS-Bench-101 dataset 
we can select split options "172", "334", "860", "91-172", "91-334", "91-860", "denoise-91", "denoise-80", "all"
you can refer original [baseline](https://github.com/ultmaster/neuralpredictor.pytorch) 

### Splits

The following splits are provided 

* `172`, `334`, `860`: Randomly sampled architectures from NasBench.
* `91-172`, `91-334`, `91-860`: The splits above filtered with a threshold (validation accuracy 91% on seed 0).
* `denoise-91`, `denoise-80`: All architectures filtered with threshold 91% and 80%.
* `all`.

### search.py 
this part is predict final acc of all architecture in nasbench or random generate architecture(not implemented)
and then select TOP1 architecture by highest predicted acc 
make top1 architecture into torch model using [nasbench_pytorch](https://github.com/romulus0914/NASBench-PyTorch) <- you can download 
* nasbench architecture is just cell we should stack them for final architecture 

### trainer,py 
training TOP1 architecture on CIFAR10 

### Workflow 
1. prepare dataset (NASbench101)
2. build predictor model in model.py 
3. training predictor model so run 
  ```python train.py``` 
5. then train TOP1 model by 
  ```python search.py```
  
  
# Reference
* [original base line and also code from here](https://github.com/ultmaster/neuralpredictor.pytorch)
* [NASbench](https://github.com/google-research/nasbench/tree/b94247037ee470418a3e56dcb83814e9be83f3a8)
* [nasbench_pytorch](https://github.com/romulus0914/NASBench-PyTorch)

