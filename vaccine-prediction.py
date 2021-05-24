/*
 * Copyright (C) 2021 Tobe Deprez
 * 
 * This is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *  
 * This file is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 */ 

from datetime import datetime, timedelta, time
import math
import numpy as np
import pandas as pd
import plotly
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = 'plotly_white'
pio.renderers.default = 'browser'

#The total belgian population
population = 11492641

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

def calculate_administered_per_delivery(administered, deliveries, manufacturers):
    """Calculate the administrations for each delivery. Also adds the columns 'completely_administered and 'doses_left' to deliveres, 
    containing the date when the delivery was completely administered and how many doses are left of the delivery respectively"""
    result = pd.DataFrame(index = administered.index, 
                          columns = deliveries.index,
                          data = 0.0)
    
    manufacturers_with_adminstrations = administered['total'].columns
    manufacturers_no_reservation = manufacturers[~manufacturers['second_dose_reserved']].index & manufacturers_with_adminstrations
    manufacturers_with_reservation = manufacturers[manufacturers['second_dose_reserved']].index & manufacturers_with_adminstrations
    administrations_left = administered['total'][manufacturers_no_reservation].cumsum()
    administrations_left[manufacturers_with_reservation] = administered['first_dose'][manufacturers_with_reservation].cumsum()
    for (manufacturer, date), delivery in deliveries.iterrows():
        if manufacturer not in administered['total'].columns:
            continue
        
        if manufacturer in manufacturers_with_reservation:
            doses = delivery['amount'] / 2
        else:
            doses = delivery['amount']
        administrations_for_delivery = administrations_left[manufacturer].apply(lambda x: min(x, doses))
        result[(manufacturer, date)] = administrations_for_delivery.diff().fillna(0.0)
        
        administrations_left[manufacturer] -= administrations_for_delivery
    
    deliveries.loc[manufacturers_no_reservation, 'doses_left'] = deliveries.loc[manufacturers_no_reservation, 'amount'] - result.sum()
    deliveries.loc[manufacturers_with_reservation, 'doses_left'] = deliveries.loc[manufacturers_with_reservation, 'amount'] / 2 - result.sum()
    deliveries['completely_administered'] = result.cumsum().idxmax().where(deliveries['doses_left'] <= 0, other = None)
    
    return result

def get_average_pass_through_time(deliveries):
    """Calculate the average pass through time for each vaccine type"""
    pass_through_times = {}
    for t in deliveries.index.get_level_values('manufacturer').unique():
        times = deliveries.loc[t]['pass_through_time']
        interval = timedelta(days=1)
        pt = (times / interval).ewm(halflife = 2).mean() * interval
        pass_through_times[t] = pt.iloc[-1]
    return pass_through_times

def get_predicted_administrations_for_delivery(manufacturer, date, delivery, pass_through_times):
    """Calculate the predicted administrations for the given delivery"""
    complete_administration_date = date + pass_through_times[manufacturer].round('1d')
    prediction_start_date = max(prediction_date, date)
    time_to_complete_delivery = complete_administration_date - prediction_start_date
    days_to_complete_delivery = time_to_complete_delivery / timedelta(days=1)
    doses_left = delivery['doses_left']
    predicted_administrations_for_delivery = pd.Series({ t: doses_left / days_to_complete_delivery 
                                                        for t in date_range(prediction_start_date, complete_administration_date, timedelta(days=1))})
    predicted_administrations_for_delivery = predicted_administrations_for_delivery.apply(math.floor)
    if predicted_administrations_for_delivery.empty:
        predicted_administrations_for_delivery = pd.Series({prediction_start_date: doses_left})
    else:
        predicted_administrations_for_delivery[predicted_administrations_for_delivery.index.max() + timedelta(days=1)] = doses_left - predicted_administrations_for_delivery.sum()
    
    return predicted_administrations_for_delivery

def plot(fig, administered, administered_complete, predicted_administrations, label):
    administered = administered.copy()
    administered = administered.cumsum()
    administered['Total'] = administered.sum(axis = 'columns')
    colors = {col: plotly.colors.qualitative.Plotly[i] for i, col in enumerate(administered.columns)}
    manufacturer_line = {manufacturer: f'Manufacturer: {manufacturer}' for manufacturer in manufacturers.index}
    manufacturer_line['Total'] = 'Any manifacturer'
    for name, col in administered.items():
        fig.add_trace(go.Scatter(x = col.index, y = col, name = name, legendgroup = name,
                                 line = go.scatter.Line(color = colors[name]),
                                 mode = 'lines',
                                 hovertemplate = '<b>Date: %{x}</b><br />' +
                                                 label + ': %{y}<br />' + 
                                                 manufacturer_line[name] +
                                                 '<extra></extra>'))
    
    predicted_administrations = predicted_administrations.copy()
    predicted_administrations['Total'] = predicted_administrations.sum(axis = 'columns')
    predicted_administrations.loc[administered.index.max()] = administered.iloc[-1]
    predicted_administrations.sort_index(inplace = True)
    predicted_administrations = predicted_administrations.cumsum()
    for name,col in predicted_administrations.items():
        fig.add_trace(go.Scatter(x = col.index, y = col, name = name, legendgroup = name,
                                 showlegend = False, mode = 'lines',
                                 line = go.scatter.Line(color = colors[name], dash = 'dot'),
                                 hovertemplate = '<b>Date: %{x}</b><br />' +
                                                 label + ' (prediction): %{y}<br />' + 
                                                 manufacturer_line[name] +
                                                 '<extra></extra>'))
    
    if administered_complete.index.max() > administered.index.max():
        administered_complete = administered_complete.copy()
        administered_complete['Total'] = administered_complete.sum(axis = 'columns')
        administered_complete = administered_complete.cumsum()
        administered_complete = administered_complete[administered_complete.index >= administered.index.max()]
        for name,col in administered_complete.items():
            fig.add_trace(go.Scatter(x = col.index, y = col, name = name, legendgroup = name,
                                     showlegend = False, mode = 'lines',
                                     line = go.scatter.Line(color = colors[name], dash = 'dash'),
                                     hovertemplate = '<b>Date: %{x}</b><br />' + 
                                                     label + ': %{y}<br />' +
                                                     manufacturer_line[name] +
                                                     '<extra></extra>'))

pd.plotting.register_matplotlib_converters()

administered = pd.read_csv('administered-by-vaccine-type.csv')
administered['date'] = administered['date'].apply(pd.Timestamp)
administered = administered.groupby(['date', 'type']).sum().unstack().fillna(0)
for t in administered.columns.levels[1]:
    administered[('total',t)] = administered[('first_dose',t)] + administered[('second_dose',t)]

#Cut of some of the administrations to try out preduction
prediction_date = (administered.index.max() + timedelta(days=1) - timedelta(weeks = 0)).date()
prediction_date = datetime.combine(prediction_date, time())
prediction_end_date = datetime(year = 2021, month = 7, day = 4)
administered_complete = administered.copy()
administered = administered[administered.index < prediction_date]

deliveries = pd.read_csv('delivered.csv')
deliveries['date'] = deliveries['date'].apply(pd.Timestamp)
delivered_by_type = deliveries.groupby(['date', 'manufacturer']).sum()['amount'].unstack().fillna(0)
predicted_deliveries = pd.read_csv('predicted-deliveries.csv')
predicted_deliveries['date'] = predicted_deliveries['date'].apply(pd.Timestamp)
predicted_delivered_by_type = predicted_deliveries.groupby(['date', 'manufacturer']).sum()['amount'].unstack().fillna(0)
deliveries = deliveries.append(predicted_deliveries)
deliveries = deliveries.sort_values('date')
deliveries = deliveries.set_index(['manufacturer', 'date'], drop = True)

index_min = min(administered.index.min(), administered.index.min())
index_max = max(administered.index.max(), administered.index.max())
index_range = inclusive_date_range(index_min, index_max, timedelta(days=1))
administered = administered.reindex(index_range, fill_value = 0.0)

#Sometimes we can get more doses out of a vial than the official number. Therefore, we artificially increase the number of delivered doses
#based on an estimation of how often we can do it
for manufacturer, details in manufacturers.iterrows():
    if manufacturer not in administered['total'].columns:
        continue
    if details['second_dose_reserved']:
        estimation = (administered['first_dose'][manufacturer].cumsum() / delivered_by_type[manufacturer].cumsum().reindex_like(administered, method = 'backfill')) * 2
    else:
        estimation = (administered['total'][manufacturer].cumsum() / delivered_by_type[manufacturer].cumsum().reindex_like(administered, method = 'backfill'))
    estimation = estimation[estimation != np.inf]
    factor = max(estimation.max(), 1)
    #There seems to be a bug that makes 
    # > deliveries.loc[manufacturer, 'amount'] *= factor
    #not work
    for i in deliveries.index:
        if i[0] == manufacturer:
            deliveries.loc[i, 'amount'] *= factor
deliveries['amount'] = deliveries['amount'].apply(math.ceil)

administered_per_delivery = calculate_administered_per_delivery(administered, deliveries, manufacturers)
deliveries['pass_through_time'] = (deliveries['completely_administered'] - deliveries.index.get_level_values('date'))
pass_through_times = get_average_pass_through_time(deliveries)

#Make the prediction
predicted_total_administrations = pd.DataFrame()
for (manufacturer, date), delivery in deliveries[deliveries['doses_left'] > 0].iterrows():
    predicted_administrations_for_delivery = get_predicted_administrations_for_delivery(manufacturer, date, delivery, pass_through_times)
    predicted_total_administrations = predicted_total_administrations.reindex(predicted_total_administrations.index.union(predicted_administrations_for_delivery.index), 
                                                                  fill_value = 0.0)
    predicted_administrations_for_delivery = predicted_administrations_for_delivery.reindex(predicted_total_administrations.index, fill_value = 0.0)
    if manufacturer in predicted_total_administrations:
        predicted_total_administrations[manufacturer] += predicted_administrations_for_delivery
    else:
        predicted_total_administrations[manufacturer] = predicted_administrations_for_delivery
    predicted_total_administrations = predicted_total_administrations.fillna(0.0)
predicted_total_administrations = predicted_total_administrations[predicted_total_administrations.index <= prediction_end_date]

first_doses_without_second_dose = (administered['first_dose'].cumsum() - administered['second_dose'].sum()).clip(lower = 0).diff()
index_range = inclusive_date_range(first_doses_without_second_dose.index.min(), predicted_total_administrations.index.max(), timedelta(days = 1))
first_doses_without_second_dose = first_doses_without_second_dose.reindex(index_range)
first_doses_without_second_dose = first_doses_without_second_dose.fillna(0.0)
predicted_administrations = pd.DataFrame(index = predicted_total_administrations.index, columns = administered.columns, data = 0.0)
for manufacturer, details in manufacturers.iterrows():
    if not details['time_between_doses']:
        continue
    if manufacturer not in predicted_total_administrations.columns:
        continue
    if details['second_dose_reserved']:
        predicted_administrations['first_dose', manufacturer] = predicted_total_administrations[manufacturer].copy()
        first_doses_without_second_dose.loc[predicted_administrations.index, manufacturer] += predicted_administrations['first_dose', manufacturer]
        second_doses = first_doses_without_second_dose[manufacturer].copy()
        second_doses.index += details['time_between_doses']
        predicted_administrations['second_dose', manufacturer] += second_doses[second_doses.index >= prediction_date]
        predicted_administrations.loc[prediction_date, ('second_dose', manufacturer)] += second_doses[second_doses.index < prediction_date].sum()
        first_doses_without_second_dose[manufacturer] = 0.0
    else:
        for date in predicted_total_administrations.index:
            first_dose_date = date - details['time_between_doses']
            second_doses = first_doses_without_second_dose.loc[first_doses_without_second_dose.index <= first_dose_date, manufacturer].sum()
            predicted_administrations.loc[date, ('second_dose', manufacturer)] += second_doses
            first_doses_without_second_dose.loc[first_doses_without_second_dose.index <= first_dose_date, manufacturer] = 0.0
            
            doses_left = predicted_total_administrations.loc[date, manufacturer] - predicted_administrations.loc[date, ('second_dose', manufacturer)]
            predicted_administrations.loc[date, ('first_dose', manufacturer)] += doses_left
            first_doses_without_second_dose.loc[date, manufacturer] += doses_left
            
            #If the predicted second doses are higher than the predicted doses, we have to borrow the predicted first doses from earlier days
            earlier_date = date - timedelta(days = 1)
            while earlier_date > max(first_dose_date, administered.index.max()) and predicted_administrations.loc[date, ('first_dose', manufacturer)] < 0:
                borrowed_doses = min(predicted_administrations.loc[earlier_date, ('first_dose', manufacturer)], 
                                     -predicted_administrations.loc[date, ('first_dose', manufacturer)])
                predicted_administrations.loc[earlier_date, ('first_dose', manufacturer)] -= borrowed_doses
                first_doses_without_second_dose.loc[earlier_date, manufacturer] -= borrowed_doses
                predicted_administrations.loc[date, ('first_dose', manufacturer)] += borrowed_doses
                first_doses_without_second_dose.loc[date, manufacturer] += borrowed_doses
                
                earlier_date -= timedelta(days = 1)
            
            #If we can not borrow first doses from earlier days, subtract them from the second doses and 
            #keep them in first_doses_without_second_dose
            if predicted_administrations.loc[date, ('first_dose', manufacturer)] < 0:
                predicted_administrations.loc[date, ('second_dose', manufacturer)] -= -predicted_administrations.loc[date, ('first_dose', manufacturer)]
                first_doses_without_second_dose.loc[first_dose_date, manufacturer] += -predicted_administrations.loc[date, ('first_dose', manufacturer)]
                predicted_administrations.loc[date, ('first_dose', manufacturer)] = 0.0
                first_doses_without_second_dose.loc[date, manufacturer] = 0.0

predicted_administrations['total'] = predicted_administrations['first_dose'] + predicted_administrations['second_dose']

fig = go.Figure()
plot(fig, administered['total'], administered_complete['total'], predicted_administrations['total'], 'Administered')
fig.update_layout(hoverlabel = {'bgcolor': 'black'},
                  xaxis_title = "Date", yaxis_title = "Nr of administered vaccines", 
                  legend = dict(orientation= 'h', 
                                yanchor = 'bottom',
                                y = 1.02,
                                xanchor='right',
                                x = 1))
fig.show()

#TODO: should display this with percentage signs
fig = go.Figure()
plot(fig, administered['first_dose'] / population * 100, administered_complete['first_dose'] / population * 100, 
     predicted_administrations['first_dose'] / population * 100, 'Partially vaccinated')
fig.update_layout(hoverlabel = {'bgcolor': 'black'},
                  xaxis_title = "Date", yaxis_title = "Percentage at least partially vaccinated of complete population",
                  legend = dict(orientation= 'h', 
                                yanchor = 'bottom',
                                y = 1.02,
                                xanchor='right',
                                x = 1))
fig.show()

fig = go.Figure()
plot(fig, administered['second_dose'] / population * 100, administered_complete['second_dose'] / population * 100, 
     predicted_administrations['second_dose'] / population * 100, 'Fully vaccinated')
fig.update_layout(hoverlabel = {'bgcolor': 'black'},
                  xaxis_title = "Date", yaxis_title = "Percentage fully vaccinated of complete population",
                  legend = dict(orientation= 'h', 
                                yanchor = 'bottom',
                                y = 1.02,
                                xanchor='right',
                                x = 1))
fig.show()