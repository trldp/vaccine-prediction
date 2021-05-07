from datetime import datetime, timedelta, time
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

#A few options per manufactuerer
manufacturers = [
        {'manufacturer': 'Moderna',
         'second_dose_reserved': True,
         'weeks_between_doses': timedelta(weeks = 4)},
        {'manufacturer': 'Johnson&Johnson',
         'second_dose_reserved': False},
        {'manufacturer': 'Pfizer/BioNTech',
         'second_dose_reserved': False},
        {'manufacturer': 'AstraZeneca/Oxford',
         'second_dose_reserved': False}
    ]
manufacturers = pd.DataFrame.from_records(manufacturers, index='manufacturer')

def date_range(start, end, interval):
    result = []
    cur = start
    while cur < end:
        result.append(cur)
        cur = cur + interval
    return result

def calculate_administrations_per_delivery(administered, deliveries):
    """Calculate when a delivery was completely administered or if it wasn't completely administered, 
    calculate the doses left of the delivery. Returns the deliveries with the new columns 'completely_administered' and 'doses_left'"""
    manufacturers_with_adminstrations = administered['total'].columns
    manufacturers_no_reservation = manufacturers[~manufacturers['second_dose_reserved']].index & manufacturers_with_adminstrations
    administrations_left = administered.cumsum()['total'][manufacturers_no_reservation].copy()
    manufacturers_with_reservation = manufacturers[manufacturers['second_dose_reserved']].index & manufacturers_with_adminstrations
    first_administrations_left = administered.cumsum()['first_dose'][manufacturers_with_reservation]
    for index, delivery in deliveries.iterrows():
        if delivery['manufacturer'] not in administered['total'].columns:
            continue
        
        if delivery['manufacturer'] in manufacturers_with_reservation:
            #For this manufacturer, the second doses are kept in stock. So only half can be given as first doses
            administrations_for_delivery = first_administrations_left[delivery['manufacturer']]
            first_dose_amount = delivery['amount'] / 2
            administrations_for_delivery = administrations_for_delivery.apply(lambda x: min(x, first_dose_amount))
            if administrations_for_delivery.max() >= first_dose_amount:
                deliveries.loc[index, 'completely_administered'] = administrations_for_delivery[administrations_for_delivery >= first_dose_amount].index.min()
            deliveries.loc[index, 'first_doses_left'] = first_dose_amount - administrations_for_delivery.max()
            
            first_administrations_left[delivery['manufacturer']] -= administrations_for_delivery
        else:
            administrations_for_delivery = administrations_left[delivery['manufacturer']]
            administrations_for_delivery = administrations_for_delivery.apply(lambda x: min(x, delivery['amount']))
            if administrations_for_delivery.max() >= delivery['amount']:
                deliveries.loc[index, 'completely_administered'] = administrations_for_delivery[administrations_for_delivery >= delivery['amount']].index.min()
            deliveries.loc[index, 'doses_left'] = delivery['amount'] - administrations_for_delivery.max()
            
            administrations_left[delivery['manufacturer']] -= administrations_for_delivery

def get_average_pass_through_time(deliveries):
    """Calculate the average pass through time for each vaccine type"""
    pass_through_times = {}
    for t in deliveries['manufacturer'].unique():
        times = deliveries[deliveries['manufacturer'] == t].set_index(['date'])['pass_through_time']
        interval = timedelta(days=1)
        pt = (times / interval).ewm(halflife = 2).mean() * interval
        pass_through_times[t] = pt.iloc[-1]
    return pass_through_times

def get_predicted_administrations_for_delivery(delivery, pass_through_times):
    """Calculate the predicted administrations for the given delivery"""
    complete_administration_date = delivery['date'] + pass_through_times[delivery['manufacturer']].round('1d')
    prediction_start_date = max(prediction_date, delivery['date'])
    time_to_complete_delivery = complete_administration_date - prediction_start_date
    days_to_complete_delivery = time_to_complete_delivery / timedelta(days=1)
    if manufacturers.loc[delivery['manufacturer'], 'second_dose_reserved']:
        doses_left = delivery['first_doses_left']
    else:
        doses_left = delivery['doses_left']
    predicted_administrations_for_delivery = pd.Series({ t: doses_left / days_to_complete_delivery 
                                                        for t in date_range(prediction_start_date, complete_administration_date, timedelta(days=1))})
    predicted_administrations_for_delivery = predicted_administrations_for_delivery.apply(math.floor)
    if predicted_administrations_for_delivery.empty:
        predicted_administrations_for_delivery = pd.Series({prediction_start_date: doses_left})
    else:
        predicted_administrations_for_delivery[predicted_administrations_for_delivery.index.max() + timedelta(days=1)] = doses_left - predicted_administrations_for_delivery.sum()
    
    return predicted_administrations_for_delivery

pd.plotting.register_matplotlib_converters()

administered = pd.read_csv('administered-by-vaccine-type.csv')
administered['date'] = administered['date'].apply(pd.Timestamp)
administered = administered.groupby(['date', 'type']).sum().unstack().fillna(0)
for t in administered.columns.levels[1]:
    administered[('total',t)] = administered[('first_dose',t)] + administered[('second_dose',t)]
    

#Cut of some of the administrations to try out preduction
prediction_date = (datetime.now() - timedelta(weeks = 0)).date()
prediction_date = datetime.combine(prediction_date, time())
administered_complete = administered.copy()
administered = administered[administered.index < prediction_date]

administered_cumsum = administered.cumsum()

deliveries = pd.read_csv('delivered.csv')
deliveries['date'] = deliveries['date'].apply(pd.Timestamp)
predicted_deliveries = pd.read_csv('predicted-deliveries.csv')
predicted_deliveries['date'] = predicted_deliveries['date'].apply(pd.Timestamp)
deliveries = deliveries.append(predicted_deliveries)
deliveries = deliveries.sort_values('date')
deliveries = deliveries.reset_index(drop = True)
delivered = deliveries.groupby(['date', 'manufacturer']).sum()['amount'].unstack().fillna(0).cumsum()

index_min = min(administered.index.min(), delivered.index.min())
index_max = max(administered.index.max(), delivered.index.max())
index_range = date_range(index_min, index_max, timedelta(days=1))
administered_cumsum = administered_cumsum.reindex(index_range, method = 'pad')
delivered = delivered.reindex(index_range, method = 'pad')

calculate_administrations_per_delivery(administered, deliveries)
deliveries['pass_through_time'] = (deliveries['completely_administered'] - deliveries['date'])
pass_through_times = get_average_pass_through_time(deliveries)

#Make the prediction
predicted_administrations = pd.DataFrame()
for index, delivery in deliveries[(deliveries['doses_left'] > 0) | (deliveries['first_doses_left'] > 0)].iterrows():
    predicted_administrations_for_delivery = get_predicted_administrations_for_delivery(delivery, pass_through_times)
    predicted_administrations = predicted_administrations.reindex(predicted_administrations.index.union(predicted_administrations_for_delivery.index), 
                                                                  fill_value = 0.0)
    predicted_administrations_for_delivery = predicted_administrations_for_delivery.reindex(predicted_administrations.index, fill_value = 0.0)
    if delivery['manufacturer'] in predicted_administrations:
        predicted_administrations[delivery['manufacturer']] += predicted_administrations_for_delivery
    else:
        predicted_administrations[delivery['manufacturer']] = predicted_administrations_for_delivery
    predicted_administrations = predicted_administrations.fillna(0.0)

sys.exit()

for manufacturer, details in manufacturers.iterrows():
    if not details['second_dose_reserved']:
        continue
    if manufacturer not in predicted_administrations.columns:
        continue
    
    #For the manufacturers for which the second dose is reserved, we have only predicted the delivery of the first doses until now
    predicted_second_doses = administered['first_dose'][manufacturer].copy()
    predicted_second_doses.index += details['weeks_between_doses']
    predicted_administrations[manufacturer] += predicted_second_doses

sys.exit()

diff = delivered - administered_cumsum['total']

pb = plt.figure()
plt.plot(administered_cumsum[('total', 'Pfizer/BioNTech')])
plt.plot(delivered['Pfizer/BioNTech'])
plt.plot(diff['Pfizer/BioNTech'])
plt.suptitle('Pfizer/BioNTech')
plt.show()

az = plt.figure()
plt.plot(administered_cumsum[('total', 'AstraZeneca/Oxford')])
plt.plot(delivered['AstraZeneca/Oxford'])
plt.plot(diff['AstraZeneca/Oxford'])
plt.suptitle('AstraZeneca/Oxford')
plt.show()

mod = plt.figure()
plt.plot(administered_cumsum[('total', 'AstraZeneca/Oxford')])
plt.plot(delivered['AstraZeneca/Oxford'])
plt.plot(diff['AstraZeneca/Oxford'])
plt.suptitle('Moderna')
plt.show()