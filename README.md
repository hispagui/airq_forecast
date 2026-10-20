# airq_forecast

The goal of this project is to predict and analyse the concentration of PM2.5 particules, (smaller than 2.5 micrometers and harmful) in Europe.
To that end, we train learnning/analytical models using data from the European Environment Agency (EEA) and the Copernicus Climate Data Store (CDS).

This project has 3 phases, 
- <i>Phase 1</i> consists in building the ground setup: 
    ....

- <i>Phase 2</i> focuses on using Geometric Deep Learning in order to enhance the , and in 

- <i>Phase 3</i>, we implement quantum layers (maybe overkill), to further improve our models (ex. using Perceval).
We test/evaluate results against GDL and classical model results.



**Contents:**
- [Project Tree Structure](##Project Tree Structure)
- [Phase 1](##Phase-1)
    - [Get the Data](###Get the Data)
    - [Training](###Training)
    - [Evaluations](###Evaluations)
    - [Experiment](###Experiment) 
- [Phase 2](##Phase-2)
    - [Models and Architectures](###Models and Architectures)
        - [Experiment](####Experiment)
    - [Spatial Holdout](###Spatial Holdout)
        - [Experiment](####Experiment)
- [Phase 3](##Phase-3)




## Project Tree Structure
```
├── evaluate.py
├── predict.py
├── forecast_weather.py
├── train.py
├── holdout.py
├── README.md
├── LICENSE
└── architecture
    ├── __init__.py
    ├── common.py
    ├── lstm.py
    ├── mlp.py
    ├── gru.py
    ├── tcn.py
    ├── transformer.py
    ├── gnn.py
    ├── gwnet.py
└── features
    ├── dataset.py
    ├── metrics.py
    ├── graph.py
└── data
    └── processed
    └── raw
        └── eea
        └── era5
    └── get_data
        ├── download_eea.py
        ├── download_weather.py
        ├── prepare_weather.py
        ├── preprocess.py
└── runs
    └── comparisons
    └── model_1
    └── model_2 
    └── ...
└── logs
```

<br>

## Phase 1
### Get the Data
To get the data from EEA and ERA5 (via CDS) respectively run:
- `download_eea.py` 
- `download_weather.py`
Sends everything to `data.raw`.<br>

Then process the downloaded data by running:
- `preprocess.py` (for the EEA data)
- `prepare.py` (for weather statistics)
Sends everything to `data.processed`.<br>

(weather data might take some time to download, the API for ERA5 (copernicus) is quite frustrating, and also you need to setup an account and get a specific personal token.)

### Training
This project has several model architectures, they all share the same frame, taking as input for each forecast; 72 hours \times 55 stations \times 17 numbers per hour and station (PM2.5, an “observed” flag, 8 weather variables, 7 calendar variables). <br>
They output 24 hours \times 55 stations of PM2.5, and they all have a a summary vector of 64 numbers per station that goes into the same `ForecastHead`. <br>
They are all trained using the same call `train.py`, but differ in one thing only; how they turn 72 hours of history into that 64-number summary. 
They are stored in `architectures`, and are:

- `lstm.py` (default), <i>Long Short-Term Memory</i>: reads each station’s 72 hours (previous to wanted prediction) one by one, keeping a memory vector that summarizes level, trend and daily pattern. (~143000 total param.)
- `mlp.py`, <i>Multi-Layer Perceptron</i>: flattens the 72 hours into one vector without any notion of order. It’s a lower bound that shows what the LSTM’s sequential reading adds (~116000 total param.).
- `gru.py`, <i>Gated Recurrent Unit</i>: this is basically a lighter version of the LSTM (~126000 total param.)
- `tcn.py`, <I>Temporal Convolutional Network</i>: here the idea is that instead of reading hour by hour, the TCN slides a small <i>filter</i> along the 72 hours. And a filter is a weighted combination of a few hours (~200000 param.).
- `transformer.py` cuts the 72 hours into 24 <b>chunks</b> of 3 hours that we call tokens and training decides how much each token matters (~147000 total param.).


### Evaluations
Run `evaluate.py runs/modelname_date_time --plots` (which calls `predict.py`, `metrics.py` and `dataset.py`).
It computes statistical analysis of the models you chose to evaluate and save it to `runs.comparison_date_time20`. 
Mainly, it computes the 
- <b>RMSE</b> (root mean square deviation), this will be the main measurement as it punishes big misses more, which is what we want when handeling weather components. 
```math
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \widehat{y}_i)^2}
```
- <b>MAE</b> (mean absolute error), a more explainable metric.
Both compute some kind of average error in $\mu g/m^3$, the lower the better. 
```math
MAE = \frac{1}{n}\sum_{i=1}^n \mid y_i - \widehat{y}_i \mid
```
- <b>BIAS</b>; the average of `forecast`-`observed`. 
Positive means that the model predicts too high on average, negative is the opposite.

- <b>SKILL vs PERSISTENCE</b>; <i>persistence</i> is the laziest forecast it predicts the same as yesterday (no model, no learning, just copy).
This is our minimum bar, if our models cannot beat the persitence model, then they are not effective. This metric is computed as follow: (in this case, the higher the results, the better it outperforms the persistent model)
```math
\frac{1 - RMSE_{model}}{RMSE_{persistence}} 
``` 

`evaluate.py` also produces <b>ALERTS</b> and they depend on certain thresholds.
Basically the idea is to have something behaving like this:
"If tomorrow's forecast says PM2.5 $\ge 15 \mu g / m^3$, we warn the public." 


### Experiment
Having trained the `LSTM`, `MLP`, `TCN`, `GRU` and `Transformr` models, here are our main evaluations:


| Model         | RMSE      |  MAE      | BIAS       | Skil. vs Pers.  | 
| ------------- | ------    | -----     | -----      | -----           |
| LSTM          | 5.25      | 3.21      | +0.74      | 0.25            | 
| MLP           | 5.19      | 3.10      |<b>-0.01</b>| 0.25            | 
| TCN           | 5.19      | 3.14      | +0.38      | 0.25            |
| GRU           |<b>5.05</b>|<b>3.05</b>| +0.09      | <b>0.27</b>     | 
| Transformer   | 6.16      | 3.61      | +1.06      | 0.12            |

From this first experiment we can conclude that the best performing model in almost all categories is the <b>GRU</b>(even if the 3 other models seem to be very close). <br>
`evaluation.py` also outputs many interesting results, like metrics by hours, by seasons, by months, ... <br>
One caveat here is that our training has a "randomness" element to it.
In training, the starting weights of our models, the order of training and the dropout as well are chosen at "random".
For that reason we have included a seed that is modifiable by `train.py --seed X...X`, the default seed is $9999$ and two runs using the same seeds will output similar results.<br>
<br>


One can also make a single `predict.py` call to make a single forecast.
It is possible to specify the date and the model it should use (defaults are going to be next day available from recovered data and last trained model).



<br>

## Phase 2

Phase 2 consists in taking advantage of the fact that air concentration predictions are greatly impacted by geographical location; we can use the proximity of stations in the learning of our models to increase precision, introducing Graph Neural Networks.<br>

In addition, the central idea that we borrowed from [6], is that "wind directions are critical for modeling PM2.5 transport". 
We implement this by letting wind speed and direction decide which edges matter in our graph:




### Models and Architectures
In phase 2, we have added two models (graph models), a <b>Graph Neural Network</b> (GNN) that takes advantage of working over multiple graphs and the <b>Graph WaveNet</b> (GWNet) inspired by [7].
- `gnn.py` is an <i>encode</i> (with GRU) then <i>propagate</i> neural network, with wind-dependnt edges.
The graph layers add about $17000$ parameters which amounts to $\sim 143000$ in total, about the size of the LSTM model, so comparison is fair!
    - `knn.py`, the $k$ neareast neighbours (here by default $k=8$).
    Weights are computed via $A[i,j]=1$ if $j$ is among the 8 neighbours of $i$, then we normalize this.
    - `distance.py`, where closer counts more (this graph is used in Graph WaveNet [7]). Weights are computed via 
    ```math
    A[i,j]=\exp(-d_{ij}^2/\sigma^2)
    ``` 
    with $d_{ij}$ being the distance between station $i$ and $j$ and $\sigma$ the spread of all station distances.
    - `wind.py`, basically listens to whoever is upwind
    ... say more
    ```math
    A[i,j] = \exp(s_{ij}) / \Sigma_j' \exp(s_{ij}')
    ```
    It has two interesting properties, it is a <i>directed graph</i> $A[i,j]\neq A[j,i]$ and it is <i>rotation invariant </i>
    
    - `random.py`, uses the same edges as `distance.py` but shuffles the stations: $A \leftarrow P AP^r$ for a permutation $P$.
    For example, Antwerps' station might now listen to stations near Liège.

    - `none.py`, there is no neighbours here, $A[i,j]=0$. 
    We included this graph this distinguish between : <i>the graph helped</i> and <i>the extra layers helped</i>.

- `gwnet.py` is also a Graph Neural Network, but unlike our first GNN, it is a reference model for forecasting on network sensors. 
Its key idea is that it alternates between time and space at every layer, and learns part of the graph from the data.


<br>
Training and evaluation work the same as in phase 1, the key measurments are the same as well, to ensure we can fairly compare our different models.

#### Experiment

We also evaluate our graph models against the best performing model from phase 1 (i.e. GRU), here is a table of our results:

| Model         | RMSE      |  MAE      | BIAS       | Skil. vs Pers. | 
| ------------- | ------    | -----     | -----      | -----          |
| GNN dist.     | 5.01      | 3.02      | +0.34      | 0.28           | 
| GNN knn       | 5.04      | 3.05      | +0.37      | 0.27           | 
| GNN none      | 5.04      | 3.07      | +0.33      | 0.28           |
| GNN rand.     | 4.95      | 3.01      | +0.39      | 0.29           | 
| GNN wind      |<b>4.94</b>|<b>3.00</b>| +0.39      |<b>0.29</b>     |
| GRU           | 5.05      | 3.05      |<b>+0.09</b>| 0.27           |

GNN wind is the best trained model.




### Spatial Holdout

In this section, wea answer the quesiton: <b>can our model forecast PM2.5 at a station it never saw during training?</b> 
It works as follow:
- It hides stations by folds (for Belgium, 5 folds of 11 stations at a time).
- There are two modes for stations, `history`: a new station whose PM2.5 history becomes available at test time, and a `virtual` station with no PM2.5 history at all, only weather and neighbour data.
- Hidden stations are excluded from the training loss, from model selection, and from the normalisation statistics. And the learned station's identity is replaced by static features (coordinates, altitude, station and area type), an unseen station can’t have a trained identity.<br>

Training is again done by calling `train.py`.
And one can compare (evaluate) different holdouts by calling `holdout.py compare MODELS`

#### Expermient
We compared the holdout's RMSE for 3 trained models, GRU, GNN (random) and GNN (wind), below are our results:

| Model         | RMSE   |  MAE  | h+1   | h+3   | h+6   | h+12  | h+24  |
| ------------- | ------ | ----- | ----- | ----- | ----- | ----- | ----- |
| GRU           | 6.47   | 4.30  | 6.55  | 6.55  | 6.53  | 6.34  | 6.52  |
| GNN (random)  | 5.25   | 3.30  | 4.40  | 4.53  | 4.83  | 5.35  | 5.80  |
| GNN (wind)    | 5.20   | 3.26  | 3.80  | 4.12  | 4.74  | 5.31  | 5.87  |

What is interesting here is that `GNN wind`clearly beats `GNN random` in the first hours.
We can undestand this as; when a station has no measurments, knowing which neighbours are upwind matters a lot for the next hours. <br>
And by $h+24$, `GNN random` becomes slighlty better, this could be explain by one day ahead only the reginonal level matters.
So in conclusion we have that 

| Model      | RMSE with 72h history |  RMSE with no measurment  | Loss  |
| -----------| --------------------- | ------------------------- | ----- |
| GRU        | 5.05                  | 6.47                      | +28.1%|
| GNN (wind) | 4.93                  | 5.20                      | +5.4% |












<br>

## Phase 3

In this phase, we look at the following arguments; with the same encoder, the same data and the same parameter budget, does a photonic quantum head forecast better or worse than a classical head?
Is it more, or less robust to noise?
Does it generalise better to unseen stations?

### How it Works

As a circuit with about 6 inputs can’t process $72$ hours $\times 55$ stations $\times 12$ features, we keep our best trained models and swap only the last layers for a small quantum circuit.



 
quantum phases get their own learning rate (--lr-head 1e-2), higher than the linear layers (1e-3)









## Recommended Installations

PyTorch, 
Numpy
torch-geometric  

## AI 
...


## Litterature

- [1] Jolliffe & Stephenson (eds.), Forecast Verification: A Practitioner’s Guide in Atmospheric Science, 2nd ed., Wiley, 2012.
- [2] Gneiting, “Making and evaluating point forecasts”, Journal of the American Statistical Association, 2011.
- [3] Jacob, Introduction to Atmospheric Chemistry, Princeton University Press, 1999.
- [4] WHO Global Air Quality Guidelines (2021)
-[5] EU Directive 2024/2881
- [6] PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting, Wang S. et al.
- [7] Graph WaveNet for Deep Spatial-Temporal Graph Modeling, Wu Z., et zl
