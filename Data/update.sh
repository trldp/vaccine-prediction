#!/bin/bash

rm COVID19BE_VACC.csv
rm delivered.csv
wget https://covid-vaccinatie.be/api/v1/delivered.csv
wget https://epistat.sciensano.be/Data/COVID19BE_VACC.csv
