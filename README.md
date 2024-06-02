# FloodGNN-GRU

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
    python train.py <parameters>  


parameters can 