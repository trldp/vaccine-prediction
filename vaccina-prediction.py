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
         'time_between_doses': timedelta(weeks = 4)},
        {'manufacturer': 'Johnson&Johnson',
         'second_dose_reserved': False,
         'time_between_doses': None},
        {'manufacturer': 'Pfizer/BioNTech',
         'second_dose_reserved': False,
         'time_between_doses': timedelta(weeks = 5)},
        {'manufacturer': 'AstraZeneca/Oxford',
         'second_dose_reserved': False,
         'time_between_doses': timedelta(weeks = 12)}
    ]
manufacturers = pd.DataFrame.from_records(manufacturers, index='manufacturer')

def date_range(start, end, interval):
    result = []
    cur = start
    while cur < end:
        result.append(cur)
        cur = cur + interval
    return result

def inclusive_date_range(start, end, interval):
    result = []
    cur = start
    while cur <= end:
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
prediction_date = (administered.index.max() + timedelta(days=1) - timedelta(weeks = 0)).date()
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
index_range = inclusive_date_range(index_min, index_max, timedelta(days=1))
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

first_doses_without_second_dose = (administered['first_dose'].cumsum() - administered['second_dose'].sum()).clip(lower = 0).diff()
index_range = inclusive_date_range(first_doses_without_second_dose.index.min(), predicted_administrations.index.max(), timedelta(days = 1))
first_doses_without_second_dose = first_doses_without_second_dose.reindex(index_range)
first_doses_without_second_dose = first_doses_without_second_dose.fillna(0.0)
predicted_first_administrations = pd.DataFrame(index = predicted_administrations.index, columns = predicted_administrations.columns, 
                                               data = 0.0)
predicted_second_administrations = pd.DataFrame(index = predicted_administrations.index, columns = predicted_administrations.columns,
                                                data = 0.0)
for manufacturer, details in manufacturers.iterrows():
    if not details['time_between_doses']:
        continue
    if manufacturer not in predicted_administrations.columns:
        continue
    if details['second_dose_reserved']:
        predicted_first_administrations[manufacturer] = predicted_administrations[manufacturer].copy()
        first_doses_without_second_dose.loc[predicted_first_administrations.index, manufacturer] += predicted_first_administrations[manufacturer]
        second_doses = first_doses_without_second_dose[manufacturer].copy()
        second_doses.index += details['time_between_doses']
        predicted_second_administrations[manufacturer] += second_doses[second_doses.index >= prediction_date]
        predicted_second_administrations.loc[prediction_date, manufacturer] += second_doses[second_doses.index < prediction_date].sum()
        first_doses_without_second_dose[manufacturer] = 0.0
    else:
        for date in predicted_administrations.index:
            first_dose_date = date - details['time_between_doses']
            second_doses = first_doses_without_second_dose.loc[first_doses_without_second_dose.index <= first_dose_date, manufacturer].sum()
            predicted_second_administrations.loc[date, manufacturer] += second_doses
            first_doses_without_second_dose.loc[first_doses_without_second_dose.index <= first_dose_date, manufacturer] = 0.0
            
            doses_left = predicted_administrations.loc[date, manufacturer] - predicted_second_administrations.loc[date, manufacturer]
            predicted_first_administrations.loc[date, manufacturer] += doses_left
            first_doses_without_second_dose.loc[date, manufacturer] += doses_left
            
            #If the predicted second doses are higher than the predicted doses, we have to borrow the predicted first doses from earlier days
            earlier_date = date - timedelta(days = 1)
            while earlier_date > max(first_dose_date, administered.index.max()) and predicted_first_administrations.loc[date, manufacturer] < 0:
                borrowed_doses = min(predicted_first_administrations.loc[earlier_date, manufacturer], 
                                     -predicted_first_administrations.loc[date, manufacturer])
                predicted_first_administrations.loc[earlier_date, manufacturer] -= borrowed_doses
                first_doses_without_second_dose.loc[earlier_date, manufacturer] -= borrowed_doses
                predicted_first_administrations.loc[date, manufacturer] += borrowed_doses
                first_doses_without_second_dose.loc[date, manufacturer] += borrowed_doses
                
                earlier_date -= timedelta(days = 1)
            
            #If we can not borrow first doses from earlier days, subtract them from the second doses and 
            #keep them in first_doses_without_second_dose
            if predicted_first_administrations.loc[date, manufacturer] < 0:
                predicted_second_administrations.loc[date, manufacturer] -= -predicted_first_administrations.loc[date, manufacturer]
                first_doses_without_second_dose.loc[first_dose_date, manufacturer] += -predicted_first_administrations.loc[date, manufacturer]
                predicted_first_administrations.loc[date, manufacturer] = 0.0
                first_doses_without_second_dose.loc[date, manufacturer] = 0.0

predicted_total_administrations = predicted_first_administrations + predicted_second_administrations

#TODO: remove the values below zero in the administrations (i.e. take them from the previous ones)
#TODO: predict more than the period between two vaccines
#TODO: maybe put in oen DF as administrered
administered_total = administered['first_dose'].sum(axis='columns')
prediction = administered['first_dose'].sum() + predicted_first_administrations.cumsum()
prediction_total = prediction.sum(axis='columns')

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