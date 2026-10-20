# airq_forecast

The goal of this project is to predict and analyse the concentration of PM2.5 particules, (smaller than 2.5 micrometers and harmful) in Europe.
To that end, we train learnning/analytical models using data from the European Environment Agency (EEA) and the Copernicus Climate Data Store (CDS).

This project has 3 phases, 
- <i>Phase 1</i> consists in building the ground setup: 
    ....

- <i>Phase 2</i> focuses on using Geometric Deep Learning in order to enhance the , and in 

- <i>Phase 3</i>, we implement quantum layers (maybe overkill), to further improve our models (ex. using Perceval).
We test/evaluate results against GDL and classical model results.



## Project Tree Structure
```
├── evaluate.py
├── predict.py
├── train.py
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
└── features
    ├── dataset.py
    ├── metrics.py
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


## Phase 1
### Get the Data
To get the data from EEA and ERA5 respectively run:
- `download_eea.py` 
- `download_weather.py`
Sends everything to `data.raw`

Then process the downloaded data by running:
- `preprocess.py` (for the EEA data)
- `prepare.py` (for weather statistics)
Sends everything to `data.processed`

(weather data might take some time to download, the API for ERA5 (copernicus) is quite frustrating, and also you need to setup an account and get a specific personal token.)

### Training
This project has several model architectures, they all share the same frame, taking as input for each forecast; 72 hours \times 55 stations \times 17 numbers per hour and station (PM2.5, an “observed” flag, 8 weather variables, 7 calendar variables). <br>
They output 24 hours \times 55 stations of PM2.5, and they all have a a summary vector of 64 numbers per station that goes into the same `ForecastHead`. <br>
They are all trained via the same call `train.py`, but differ in one thing only; how they turn 72 hours of history into that 64-number summary. 
They are stored in `architectures`, and are:

- `lstm.py` (default), <i>Long Short-Term Memory</i>: reads each station’s 72 hours (previous to wanted prediction) one by one, keeping a memory vector that summarizes level, trend and daily pattern.
- `mlp.py`, <i>Multi-Layer Perceptron</i>: flattens the 72 hours into one vector without any notion of order. It’s a lower bound that shows what the LSTM’s sequential reading adds.
- `gru.py`, <i>Gated Recurrent Unit</i>: this is basically a lighter version of the LSTM 
- `tcn.py`, <I>Temporal Convolutional Network</i>: here the idea is that instead of reading hour by hour, the TCN slides a small <i>filter</i> along the 72 hours. And a filter is a weighted combination of a few hours.
- `transformer.py` cuts the 72 hours into 24 <b>chunks</b> of 3 hours that we call tokens and training decides how much each token matters.


### Evaluations
Run `evaluate.py runs/modelname_date_time --plots` (which calls `predict.py`, `metrics.py` and `dataset.py`).
It computes statistical analysis of the models you chose to evaluate and save it to `runs.comparison_date_time`. 
Mainly, it computes the 
- <i>RMSE</i> (root mean square deviation) and <i>MAE</i> (mean absolute error); which are the average errors in $\mu g/m^3$, the lower the better.
RMSE punishes nig misses more.
- <i>BIAS</i>; the average of `forecast`-`observed`. Positive means that the model predicts too high on average,  
- <i>SKILL vs PERSISTENCE</i>; this is 
```math
\frac{1 - RMSE_{model}}{RMSE_{persistence}} 
``` 
recall that persistence is the <i>laziest forecast</i> it predicts the same as yesterday (no model, no learning, just copy).
This is our minimum bar, if our models cannot beat the persitence model, then they are not effective.

`evaluate.py` also produces <i>ALERTS</i>.
"If tomorrow's forecast says PM2.5 $\ge 15 \mu g / m^3$, we warn the public." 


In our tests, the best performing model was GRU, 





From our evaluations, the GRU seems to perform best, even if the 3 other models seem to be very close.









`prediction.py` returns a forecast_YYYYMMDD_HHMM.csv file

Looks as the data 72h before and predicts data for the next day (24h following).

Each columns corresponds to an air monitoring station, and for each row, it outputs the average for <i>micrograms of fine particules per cubic metre</i> predicted for that hour.





## Phase 2

So phase 2 consists of taking advantage of the fact that air concentration predictions are impacted by geographical location; we can use the proximity of stations in the learning of our models to increase precision. 







Indeed, the central idea that we borrowed from [6], i.e. the winds decides which edges matter, and is implemented as follow:



### Model and Architecture
We have added one model, a <i>Graph Neural Network</i> (GNN) that takes advantage of working over multiple graphs.
- `gnn.py` is an <i>encode</i> (with GRU) then <i>propagate</i> neural network, with wind-dependnt edges.
The graph layers add about $17000$ parameters which amounts to $142920$ in total, about the size of the LSTM model, so comparison is fair!
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
<br>
SAME TRAINING AND EVALUATION 


### Spatial Holdout

With this section, wea answer the quesiton: <b>can our model forecast PM2.5 at a station it never saw during training?</b> 
It works as follow:
- It hides stations by folds (for Belgium, 5 folds of 11 stations at a time).
- There are two modes for stations, `history`: a new station whose PM2.5 history becomes available at test time, and a `virtual` station with no PM2.5 history at all, only weather and neighbour data.
- Hidden stations are excluded from the training loss, from model selection, and from the normalisation statistics. And the learned station's identity is replaced by static features (coordinates, altitude, station and area type), an unseen station can’t have a trained identity.        


What we did not implement from [7], is the 


The graph layers add about 17000 parameters which amounts to 142920 in total, about the size of the LSTM model, so comparison is fair!





## Phase 3

In this phase, with the same encoder, the same data and the same parameter budget, does a photonic quantum head forecast better or worse than a classical head?
Is it more or lesss robust to noise?
Does it generalise better to unseen stations?




 











## Litterature

- [1] Jolliffe & Stephenson (eds.), Forecast Verification: A Practitioner’s Guide in Atmospheric Science, 2nd ed., Wiley, 2012.
- [2] Gneiting, “Making and evaluating point forecasts”, Journal of the American Statistical Association, 2011.
- [3] Jacob, Introduction to Atmospheric Chemistry, Princeton University Press, 1999.
- [4] WHO Global Air Quality Guidelines (2021)
-[5] EU Directive 2024/2881
- [6] PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting, Wang S. et al.
- [7] Graph WaveNet for Deep Spatial-Temporal Graph Modeling, Wu Z., et zl
