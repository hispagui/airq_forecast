# airq_forecast
Train learning model on weather data from the European Environment Agency, to predict concentration of PM2.5 particules (smaller than 2.5 micrometers, harmful).


```
├── dataset.py
├── evaluate.py
├── metrics.py
├── predict.py
├── train.py
├── README.md
├── LICENSE
└── architecture
    ├── common.py
    ├── lstm.py
    ├── mlp.py
└── data
    └── processed
    └── raw
└── get_data
    ├── download_eea.py
└── preprocessing
    ├── preprocess.py
└── runs
```




## 3 Phase project.
### Phase 1
Classically build model base on data retreived from the EEA plateform.
Add multiple metrics, iterations of models and evaluation...
### Phase 2
Implement geometic deep learning in the model.
Make it more complexe, more efficient and ...
### Phase 3
Implement quantum layers (maybe overkill).
Further amelioration of model, using Perceval, test/evaluate results against GDL and classical model.



## Data

To get the data from wources:
    - `download_eea.py`
    - `download_weather.py`
Sends everything to `data` $\rightarrow$ `raw`

To process downloaded data:
    - `preprocess.py` (for the eea data)
    - `prepare.py` (for weather statistics)
Sends everything to `data` $\rightarrow$ `processed`

(weather data might take some time to download, furstrating API for ERA5 (copernicus), also you need to setup an account and produce a personal token.)

## Training
This project has several model architectures, that can all be trained via the same call.
They are stored in architectures, and we have:
    - `lstm.py` (default), Long Short-Term Memory: reads each station’s 72 hours one by one, keeping a memory vector that summarizes level, trend and daily pattern.
    - `mlp.py`, Multi-Layer Perceptron: flattens the 72 hours into one vector without any notion of order. It’s a lower bound that shows what the LSTM’s sequential reading adds.

## Evaluations






## Predictions

`prediction.py` returns a forecast_YYYYMMDD_HHMM.csv file

Looks as the data 72h before and predicts data for the next day (24h following).

Each columns corresponds to an air monitoring station, and for each row, it outputs the average for <i>micrograms of fine particules per cubic metre<i> predicted for that hour.




## Litterature

- Jolliffe & Stephenson (eds.), Forecast Verification: A Practitioner’s Guide in Atmospheric Science, 2nd ed., Wiley, 2012.
- Gneiting, “Making and evaluating point forecasts”, Journal of the American Statistical Association, 2011.
- Jacob, Introduction to Atmospheric Chemistry, Princeton University Press, 1999.
- WHO Global Air Quality Guidelines (2021)
- EU Directive 2024/2881
