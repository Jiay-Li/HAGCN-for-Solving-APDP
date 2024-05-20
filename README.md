#  Heterogeneous Attention based Graph Convolutional Network for Solving Asymmetric Pickup and Delivery Problem
HA-GCN model for learning to solve the Asymmetric Pickup and Delivery Problem (APDP) using heterogeneous attention and graph convolutional network. Training with REINFORCE with rollout baseline using real-world geographical information.

## Paper
Our paper, **Heterogeneous Attention-based Graph Convolutional Network for Solving the Asymmetric Pickup and Delivery Problem**, is currently under review at *IEEE Transactions on Automation Science and Engineering*.

## Data
To ascertain the applicability of the proposed method in express hubs or logistics centers scenarios, we collect a large amount of customer information as a pool for a specific region, i.e., Furong District, Changsha City, China. The customer pool (customer information) is crawled from **Amap**, which provides geographical coordinates and a distance matrix of 700 locations in the studied region, with each location corresponding to a potential customer. The coordinates are determined based on the WGS84 international standard coordinate system, and the values in the distance matrix are determined based on the shortest driving routes obtained from actual navigation. Notably, the crawled data exhibits a salient asymmetry in distances between two nodes. 

The experimental data used in this article has been made publicly available, where the geographic coordinates are in `excel/coordinates_700.xlsx` and the distance matrix is in `excel/distance_matrix_700.xlsx`.

## Dependencies
* Python>=3.6
* NumPy
* PyTorch>=1.1
* tensorboard_logger

## Usage
### Training
To train APDP instances with 20 nodes and use rollout as the REINFORCE baseline, you can execute the following command:
```
python run.py --graph_size 20 --baseline rollout
```
By default, training is conducted across all available GPUs. You can specify particular GPUs to use by setting the environment variable `CUDA_VISIBLE_DEVICES`. The command is as follows:
```
CUDA_VISIBLE_DEVICES=1,2 python run.py 
```
### Evaluation
To evaluate a model, you can use `eval.py`. The command is as follows:
```
python eval.py --model 'outputs/pdp_20/pdp20_rollout_{datetime}/epoch-{epoch_number}.pt' --decode_strategy greedy
```
To report the best of 1280 sampled solutions, you can execute the following command:
```
python eval.py --model 'outputs/pdp_20/pdp20_rollout_{datetime}/epoch-{epoch_number}.pt' --decode_strategy sample --width 1280 --eval_batch_size 1
```

 
