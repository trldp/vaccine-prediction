# Prediction Covid vaccinations Belgium
This repository contains code to make a prediction of covid vaccinations for Belgium based on historical vaccination data
and expected deliveries. Note that this code is provided without any warranty. I am not involved in the Belgian vaccination campain and can therefore not make any guarantee 
that these predictions will be accurate, especially not if changes are made to the vaccination strategy in Belgium.

Graphs produced by this code can also be found on [my personal website](https://tobedeprez.be/vaccination.html).

## Running the code
In order to run the code, you need `python 3` with the following packages
```
numpy,
pandas,
plotly.
```
Once these are installed, just run
```
python3 vaccine-prediction.py
```
This should open a web-browser with three plots containing the predictions.

## Data
The following data is used to make the prediction
* `COVID19BE_VACC.csv` Vaccine administrations by manufacturer. Source: [Sciensano](https://epistat.wiv-isp.be/covid/)
* `delivered.csv` Delivered vaccines. Source: [Joris Vaesen's Covid vaccinations dashboard](https://covid-vaccinatie.be/api)
* `predicted-deliveries.csv` The expected deliveries in the next months. Source: https://twitter.com/jorisvaesen/status/1396072474673090565

## License
* `vaccine-prediction.py` Licensed under GPL-3+ (see [here](https://www.gnu.org/licenses/gpl-3.0.html) for more information)
* `data/deliveries.csv` Licensed under CC BY 4.0 (see [here](https://creativecommons.org/licenses/by/4.0/) for more information)
