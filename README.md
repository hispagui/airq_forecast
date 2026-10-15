# airq_forecast

The goal of this project is to predict and analyse the concentration of PM2.5 particules, (smaller than 2.5 micrometers and harmful) in Europe.
To that end, we train learnning/analytical models using data from the European Environment Agency (EEA) and the Copernicus Climate Data Store (CDS).

This project has 3 phases, 
- <i>Phase 1<i> consists in building the ground setup: 
    ....

- <i>Phase 2<i> focuses on using Geometric Deep Learning in order to enhance the , and in 

- <i>Phase 3<i>, we implement quantum layers (maybe overkill), to further improve our models (ex. using Perceval).
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
### Data
To get the data from wources:
- `download_eea.py`
- `download_weather.py`
Sends everything to `data`.`raw`

To process downloaded data:
- `preprocess.py` (for the eea data)
- `prepare.py` (for weather statistics)
Sends everything to `data`.`processed`

(weather data might take some time to download, furstrating API for ERA5 (copernicus), also you need to setup an account and produce a personal token.)

### Training
This project has several model architectures, that can all be trained via the same call.
They are stored in architectures, and we have:
    - `lstm.py` (default), Long Short-Term Memory: reads each station’s 72 hours one by one, keeping a memory vector that summarizes level, trend and daily pattern.
    - `mlp.py`, Multi-Layer Perceptron: flattens the 72 hours into one vector without any notion of order. It’s a lower bound that shows what the LSTM’s sequential reading adds.
    - `gru.py`, Gated Recurrent Unit,
    - `tcn.py`, Temporal Convolutional Network,
    - `transformer.py`

### Evaluations
Run `evaluate.py runs/modelname_date_time --plots` (which calls `predict.py`, `metrics.py` and `dataset.py`).
It computes statistical analysis of the models you chose to evaluate and save it to `runs.comparison_date_time`. 
Mainly, it computes the 
- <i>RMSE<i> (root mean square deviation) and <i>MAE<i> (mean absolute error); which are the average errors in $\mu g/m^3$, the lower the better.
RMSE punishes nig misses more.
- <i>BIAS<i>; the average of `forecast`-`observed`. Positive means that the model predicts too high on average,  
- <i>SKILL vs PERSISTENCE<i>; this is 
```math
\frac{1 - RMSE_{model}}{RMSE_{persistence}} 
``` 
recall that persistence is the <i>laziest forecast<i> it predicts the same as yesterday (no model, no learning, just copy).
This is our minimum bar, if our models cannot beat the persitence model, then they are not effective.

`evaluate.py` also produces <i>ALERTS<i>.
"If tomorrow's forecast says PM2.5 $\ge 15 \mu g / m^3$, we warn the public." 


In our tests, the best performing model was GRU





From our evaluations, the GRU seems to perform best, even if the 3 other models seem to be very close.









`prediction.py` returns a forecast_YYYYMMDD_HHMM.csv file

Looks as the data 72h before and predicts data for the next day (24h following).

Each columns corresponds to an air monitoring station, and for each row, it outputs the average for <i>micrograms of fine particules per cubic metre<i> predicted for that hour.





## Phase 2




















## Litterature

- Jolliffe & Stephenson (eds.), Forecast Verification: A Practitioner’s Guide in Atmospheric Science, 2nd ed., Wiley, 2012.
- Gneiting, “Making and evaluating point forecasts”, Journal of the American Statistical Association, 2011.
- Jacob, Introduction to Atmospheric Chemistry, Princeton University Press, 1999.
- WHO Global Air Quality Guidelines (2021)
- EU Directive 2024/2881
