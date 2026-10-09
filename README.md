# airq_forecast
Train learning model on weather data from the European Environment Agency, to predict concentration of PM2.5 particules (harmful).


## Data
Run download_eea.py then preprocess.py

## Litterature

- Jolliffe & Stephenson (eds.), Forecast Verification: A Practitioner’s Guide in Atmospheric Science, 2nd ed., Wiley, 2012.
- Gneiting, “Making and evaluating point forecasts”, Journal of the American Statistical Association, 2011.
- Jacob, Introduction to Atmospheric Chemistry, Princeton University Press, 1999.
- WHO Global Air Quality Guidelines (2021)
- EU Directive 2024/2881

## Structure:
...


## Models
- LSTM: reads each station’s 72 hours one by one, keeping a memory vector that summarizes level, trend and daily pattern.
- MLP: flattens the 72 hours into one vector without any notion of order. It’s a lower bound that shows what the LSTM’s sequential reading adds.