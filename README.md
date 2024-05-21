#  Heterogeneous Attention based Graph Convolutional Network for Solving Asymmetric Pickup and Delivery Problem
The HA-GCN model addresses Asymmetric Pickup and Delivery Problem (APDP) by integrating heterogeneous attention (HA) and graph convolutional networks (GCN), aiming to capture both node and edge features present in APDP instances. Training utilizes REINFORCE with rollout baseline, incorporating real-world geographical information.

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
### Dataset
The customer location and distance data used for training and evaluating models are located in the `data` folder. These data will be utilized to randomly generate instances of different sizes throughout the training and evaluation processes.

### Training
To train APDP instances with 20 nodes and utilize rollout as the REINFORCE baseline, execute the following command:
```
python run.py --graph_size 20 --baseline rollout
```
Training is typically conducted across all available GPUs by default. To specify specific GPUs for use, set the environment variable `CUDA_VISIBLE_DEVICES`. The command is outlined below:
```
CUDA_VISIBLE_DEVICES=1,2 python run.py 
```
### Evaluation
Due to the large size of APDP models, the trained models used for evaluating can be downloaded from the following link：
* The model of APDP21: [pdp_20](https://drive.google.com/drive/folders/1do5XWDFNkOtzcydwaJS_JyE9-OfMczq2?usp=sharing)
* The model of APDP51: [pdp_50](https://drive.google.com/drive/folders/1N7b0BAbFjzeMhXFPr_fOv4nwqK9TLtNd?usp=sharing)
* The model of APDP101: [pdp_100](https://drive.google.com/drive/folders/1pm84NpxQIAwQJofjfPtAeGwohrP6O3SL?usp=sharing)
  
The model files can be downloaded into the `outputs` folder for convenient access during evaluation.

To evaluate a model, you can use `eval.py`. Foe example, to test the APDP21 model, the command is as follows:
```
python eval.py --model 'outputs/pdp_20/run_20230413T095145/epoch-49.pt' --decode_strategy greedy
```
To report the best of 1280 sampled solutions, you can execute the following command to test the APDP21 model:
```
python eval.py --model 'outputs/pdp_20/run_20230413T095145/epoch-49.pt' --decode_strategy sample --width 1280 --eval_batch_size 1
```
## Acknowledgements
Thanks to [attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route) for providing the initial codebase for the Attention Model (AM).
