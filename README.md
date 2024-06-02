# FloodGNN-GRU

[![DOI](https://zenodo.org/badge/809192839.svg)](https://zenodo.org/doi/10.5281/zenodo.11416792)

Source code for ["FloodGNN-GRU: A Spatio-Temporal Graph Neural Network for Flood Prediction"](). 

![framework](./figs/floodgnn-gru-framework.png)
![framework](./figs/floodgnn-gru-velocity.png)

## Required Packages
* Python == 3.8
* [PyTorch](https://pytorch.org/) == 1.11
* [PyTorch Geometry](https://pytorch-geometric.readthedocs.io/) == 2.0
* Scikit-learn == 1.1 
* Scipy == 1.10

## Data 
The data can be found [here](https://zenodo.org/records/10787632?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjRmZjFlYjM5LWI0NTItNDVlMy04OTRjLWNiY2M4YTgwZmFlNSIsImRhdGEiOnt9LCJyYW5kb20iOiJmOGI0NDMxNTg2NmMxNzg3YWE3YmQxZmYzMTE2ODRiZSJ9.CFvVViSDO4_Q8CR7mZ5zPzl0qTTYNlRvLs1Li1hbwh80Sz_C1F8pViPXvuToHRJuIK6McjoMuU631q64h-TXtw) and should be placed in the 'data' folder (created by user) after download. 

Details about the data can be found in the jupyter notebook [data_exploration.ipynb](./data_exploration.ipynb)

## Usage
    python train.py   


Optional arguments:

    --time_steps       Total number of time steps.
    --drop             Dropout probability.
    --batch_size       Batch size.
    --weight_decay 
    --n_epochs         Number of epochs.
    --lr               Learing rate.
    --offset           Initial time step.
    --v_h_dim          Vector features - hidden dimensions.
    --s_h_dim          Scalar features - hidden dimensions.


### Note 
Check [Github repository](https://github.com/kanz76/FloodGNN-GRU.git) for updates. 