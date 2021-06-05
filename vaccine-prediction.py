#!/usr/bin/env python3

#
# Copyright (C) 2021 Tobe Deprez
# 
# This is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
 
# This file is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>. 

import argparse
from datetime import datetime, timedelta, time, date
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

def get_extra_doses_factor(administered, delivered, second_dose_reserved = False):
    """Sometimes, one can get more doses out of a vial than the official amount. This function estimates the ratio of 
    doses one can actually get out of a vial by the official number of doses per vial, based on the total amount of vaccines that have
    been administered and delivered.
    
        administered: a DataFrame or Series containing the amount of administered vaccines per day (first dose and second dose separately if applicable)
        deliveries: the deliveries
    """
    if second_dose_reserved:
        estimation = (administered['first_dose'].cumsum() / delivered.cumsum().reindex_like(administered, method = 'pad')) * 2
    else:
        estimation = (administered.sum(axis = 'columns').cumsum() / delivered.cumsum().reindex_like(administered, method = 'pad'))
    estimation = estimation[estimation != np.inf]
    return max(estimation.max(), 1)

def get_administered_per_delivery(administered, delivered, second_dose_reserved = False):
    """Calculate the administrations for each delivery.
    
    Returns a DataFrame containing the administrations per day for each delivery
    """
    result = pd.DataFrame(index = administered.index, 
                          columns = delivered.index,
                          data = 0.0)
    
    if second_dose_reserved:
        administrations_left = administered['first_dose'].cumsum()
        delivered = delivered / 2
    else:
        administrations_left = administered.sum(axis = 'columns').cumsum()
    administrations_left.loc[administrations_left.index.min() - timedelta(days = 1)] = 0.0
    administrations_left.sort_index(inplace = True)
    
    for d, doses in delivered.iteritems():
        administrations_for_delivery = administrations_left.apply(lambda x: min(x, doses))
        result[d] = administrations_for_delivery.diff().fillna(0.0)
        administrations_left -= administrations_for_delivery
    
    return result

def get_average_pass_through_time(delivered):
    """Calculate the average pass through time for each vaccine type"""
    interval = timedelta(days=1)
    moving_average = (delivered['pass_through_time'] / interval).ewm(halflife = 2).mean() * interval
    return moving_average.iloc[-1]

def get_predicted_administrations_for_delivery(delivery, pass_through_time, prediction_date):
    """Calculate the predicted administrations for the given delivery"""
    complete_administration_date = delivery.name + pass_through_time.round('1d')
    prediction_start_date = max(prediction_date, delivery.name)
    days_to_complete_delivery = (complete_administration_date - prediction_start_date) / timedelta(days=1) + 1
    doses_left = delivery['doses_left']
    predicted_administrations_for_delivery = pd.Series({ t: doses_left / days_to_complete_delivery 
                                                        for t in inclusive_date_range(prediction_start_date, complete_administration_date, timedelta(days=1))})
    predicted_administrations_for_delivery = predicted_administrations_for_delivery.apply(math.floor)
    if predicted_administrations_for_delivery.empty:
        predicted_administrations_for_delivery = pd.Series({prediction_start_date: doses_left})
    else:
        predicted_administrations_for_delivery[predicted_administrations_for_delivery.index.max() + timedelta(days=1)] = doses_left - predicted_administrations_for_delivery.sum()
    
    return predicted_administrations_for_delivery

def predict(administered, delivered, expected_deliveries, prediction_end_date, time_between_doses = None, second_dose_reserved = False):
    """Calculate the predicted administrations for the given vaccine type. If the vaccine only has one dose, the parameter administered
    should have only one column. If the vaccine has two doses, the parameter administered should be a pd.DataFrame
    columns 'first_dose' and 'second_dose' containing the administered first and second doses of the vaccine, respectively.
    
        administrations: a data frame with the administered vaccines until now.
        deliveries: a series with the delivered vaccines
        expected_deliveries: a series with the expected deliveries
        """
    index_range = inclusive_date_range(administered.index.min(), administered.index.max(), timedelta(days=1))
    administered = administered.reindex(index_range, fill_value = 0.0)
    prediction_date = administered.index.max() + timedelta(days = 1)
    
    #Correct for when more doses can be administered from a vial than originally expected
    extra_doses_factor = get_extra_doses_factor(administered, delivered, second_dose_reserved)
    delivered = delivered * extra_doses_factor
    expected_deliveries = expected_deliveries * extra_doses_factor
    delivered = delivered.apply(math.ceil)
    expected_deliveries = expected_deliveries.apply(math.ceil)
    if second_dose_reserved:
        delivered = delivered.apply(lambda x: x + 1 if (x % 2) == 1 else x)
        expected_deliveries = expected_deliveries.apply(lambda x: x + 1 if (x % 2) == 1 else x)
    
    #Calculate the doses left per administrations and the pass through times
    administered_per_delivery = get_administered_per_delivery(administered, delivered, second_dose_reserved)
    delivered = pd.DataFrame(delivered.values, columns = ['amount'], index = delivered.index)    
    if second_dose_reserved:
        delivered['doses_left'] = delivered['amount'] / 2 - administered_per_delivery.sum()
    else:
        delivered['doses_left'] = delivered['amount'] - administered_per_delivery.sum()
    delivered['completely_administered'] = administered_per_delivery.cumsum().idxmax().where(delivered['doses_left'] <= 0, other = None)
    delivered['pass_through_time'] = (delivered['completely_administered'] - delivered.index)
    pass_through_time = get_average_pass_through_time(delivered)
    
    #Predict the total administrations
    expected_deliveries = pd.DataFrame(expected_deliveries.values, columns = ['amount'], index = expected_deliveries.index)
    if second_dose_reserved:
        expected_deliveries['doses_left'] = expected_deliveries['amount'] / 2
    else:
        expected_deliveries['doses_left'] = expected_deliveries['amount']
    to_administer = delivered[delivered['doses_left'] > 0].append(expected_deliveries, sort = True)
    predicted_total_administrations = pd.DataFrame()
    for d, delivery in to_administer.iterrows():
        predicted = get_predicted_administrations_for_delivery(delivery, pass_through_time, prediction_date)
        if predicted_total_administrations.empty:
            date_range = inclusive_date_range(predicted.index.min(), predicted.index.max(), timedelta(days = 1))
        else:
            date_range = inclusive_date_range(min(predicted_total_administrations.index.min(), predicted.index.min()),
                                              max(predicted_total_administrations.index.max(), predicted.index.max()),
                                              timedelta(days = 1))
        predicted_total_administrations = predicted_total_administrations.reindex(date_range)
        predicted_total_administrations[d] = predicted
    predicted_total_administrations = predicted_total_administrations.fillna(0.0).sum(axis = 'columns')
    predicted_total_administrations = predicted_total_administrations[predicted_total_administrations.index < prediction_end_date]
    if pd.isna(time_between_doses):
        #Just one dose
        return predicted_total_administrations
    
    #Predict first and second dose administrations
    first_doses_without_second_dose = (administered['first_dose'].cumsum() - administered['second_dose'].sum()).clip(lower = 0).diff()
    index_range = inclusive_date_range(first_doses_without_second_dose.index.min(), predicted_total_administrations.index.max(), timedelta(days = 1))
    first_doses_without_second_dose = first_doses_without_second_dose.reindex(index_range).fillna(0.0)
    predicted_administrations = pd.DataFrame(index = predicted_total_administrations.index, columns = administered.columns, data = 0.0)
    if second_dose_reserved:
        predicted_administrations['first_dose'] = predicted_total_administrations.copy()
        first_doses_without_second_dose.loc[predicted_administrations.index] += predicted_administrations['first_dose']
        second_doses = first_doses_without_second_dose.copy()
        second_doses.index += time_between_doses
        predicted_administrations['second_dose'] += second_doses[second_doses.index >= prediction_date]
        predicted_administrations.loc[prediction_date, 'second_dose'] += second_doses[second_doses.index < prediction_date].sum()
        
        return predicted_administrations
    else:
        for d in predicted_total_administrations.index:
            first_dose_date = d - time_between_doses
            second_doses = first_doses_without_second_dose.loc[first_doses_without_second_dose.index <= first_dose_date].sum()
            predicted_administrations.loc[d, 'second_dose'] += second_doses
            first_doses_without_second_dose.loc[first_doses_without_second_dose.index <= first_dose_date] = 0.0
            
            doses_left = predicted_total_administrations.loc[d] - predicted_administrations.loc[d, 'second_dose']
            predicted_administrations.loc[d, 'first_dose'] += doses_left
            first_doses_without_second_dose.loc[d] += doses_left
            
            #If the predicted second doses are higher than the predicted doses, we have to borrow the predicted first doses from earlier days
            earlier_date = d - timedelta(days = 1)
            while earlier_date > first_dose_date and earlier_date >= prediction_date and predicted_administrations.loc[d, 'first_dose'] < 0:
                borrowed_doses = min(predicted_administrations.loc[earlier_date, 'first_dose'], -predicted_administrations.loc[d, 'first_dose'])
                predicted_administrations.loc[earlier_date, 'first_dose'] -= borrowed_doses
                first_doses_without_second_dose.loc[earlier_date] -= borrowed_doses
                predicted_administrations.loc[d, 'first_dose'] += borrowed_doses
                first_doses_without_second_dose.loc[d] += borrowed_doses
                
                earlier_date -= timedelta(days = 1)
            
            #If we can not borrow first doses from earlier days, subtract them from the second doses and 
            #keep them in first_doses_without_second_dose
            if predicted_administrations.loc[d, 'first_dose'] < 0:
                predicted_administrations.loc[d, 'second_dose'] -= -predicted_administrations.loc[d, 'first_dose']
                first_doses_without_second_dose.loc[first_dose_date] += -predicted_administrations.loc[d, 'first_dose']
                predicted_administrations.loc[d, 'first_dose'] = 0.0
                first_doses_without_second_dose.loc[d] = 0.0
    return predicted_administrations

def plot(fig, administered, administered_complete, predicted_administrations, predicted_administrations_pessimistic, label, relative = False):
    administered = administered.copy()
    administered = administered.cumsum()
    administered['Total'] = administered.sum(axis = 'columns')
    administered.sort_index(axis = 'columns', inplace = True)
    colors = {col: plotly.colors.qualitative.D3[i] for i, col in enumerate(administered.columns)}
    manufacturer_line = {manufacturer: f'Manufacturer: {manufacturer}' for manufacturer in manufacturers.index}
    manufacturer_line['Total'] = 'Any manifacturer'
    for name, col in administered.items():
        y = col.copy()
        if relative:
            y /= population
            details = '%{customdata:,} (%{y:.2%})'
        else:
            details = '%{y:,}'
        fig.add_trace(go.Scatter(x = col.index, y = y, customdata = col, name = name, legendgroup = name,
                             line = go.scatter.Line(color = colors[name]),
                             mode = 'lines',
                             hovertemplate = '<b>Date: %{x}</b><br />' +
                                             label + ': ' + details + '<br />' + 
                                             manufacturer_line[name] +
                                             '<extra></extra>'))
    
    predicted_administrations = predicted_administrations.copy()
    predicted_administrations['Total'] = predicted_administrations.sum(axis = 'columns')
    predicted_administrations.loc[administered.index.max()] = administered.iloc[-1]
    predicted_administrations.sort_index(inplace = True)
    predicted_administrations = predicted_administrations.cumsum()
    predicted_administrations_pessimistic = predicted_administrations_pessimistic.copy()
    predicted_administrations_pessimistic['Total'] = predicted_administrations_pessimistic.sum(axis = 'columns')
    predicted_administrations_pessimistic.loc[administered.index.max()] = administered.iloc[-1]
    predicted_administrations_pessimistic.sort_index(inplace = True)
    predicted_administrations_pessimistic = predicted_administrations_pessimistic.cumsum()
    for name in predicted_administrations.columns:
        y_min = predicted_administrations_pessimistic[name].copy()
        y_max = predicted_administrations[name].copy()
        diff = (y_min != y_max).any()
        if relative:
            y_min /= population
            y_max /= population
            if diff:
                custom_data = list(zip(predicted_administrations_pessimistic[name], predicted_administrations[name],
                                       y_min, y_max))
                details = 'between %{customdata[0]:,} (%{customdata[2]:.2%}) and %{customdata[1]:,} (%{customdata[3]:.2%})'
            else:
                custom_data = list(zip(predicted_administrations[name], y_max))
                details = '%{customdata[0]:,} (%{customdata[1]:.2%})'
        else:
            if diff:
                custom_data = list(zip(y_min, y_max))
                details = 'between %{customdata[0]:,} and %{customdata[1]:,}'
            else:
                custom_data = None
                details = '%{y:,}'
        fig.add_trace(go.Scatter(x = predicted_administrations[name].index, y = y_max,
                                 customdata = custom_data,
                                 name = name, legendgroup = name,
                                 showlegend = False, mode = 'lines',
                                 line = go.scatter.Line(color = colors[name], dash = 'dot'),
                                 hovertemplate = '<b>Date: %{x}</b><br />' +
                                                 label + ' (prediction): ' + details + '<br />' + 
                                                 manufacturer_line[name] +
                                                 '<extra></extra>'))
        if diff:
            fig.add_trace(go.Scatter(x = predicted_administrations[name].index, y = y_min,
                                     customdata = custom_data,
                                     name = name, legendgroup = name,
                                     showlegend = False, mode = 'lines', fill = 'tonexty',
                                     line = go.scatter.Line(color = colors[name], dash = 'dot'),
                                     hovertemplate = '<b>Date: %{x}</b><br />' +
                                                     label + ' (prediction): ' + details + '<br />' + 
                                                     manufacturer_line[name] +
                                                     '<extra></extra>'))
    
    if administered_complete.index.max() > administered.index.max():
        administered_complete = administered_complete.copy()
        administered_complete['Total'] = administered_complete.sum(axis = 'columns')
        administered_complete = administered_complete.cumsum()
        administered_complete = administered_complete[administered_complete.index >= administered.index.max()]
        for name,col in administered_complete.items():
            y = col.copy()
            if relative:
                y /= population
                details = '%{customdata:,} (%{y:.2%})'
            else:
                details = '%{y:,}'
            fig.add_trace(go.Scatter(x = col.index, y = y, name = name, legendgroup = name,
                                     showlegend = False, mode = 'lines',
                                     line = go.scatter.Line(color = colors[name], dash = 'dash'),
                                     hovertemplate = '<b>Date: %{x}</b><br />' + 
                                                     label + ': ' + details + '<br />' +
                                                     manufacturer_line[name] +
                                                     '<extra></extra>'))
    
    if relative:
        fig.update_layout(yaxis = dict(tickformat = '.0%'))

def show_or_save_plot(fig, name, output_dir = None, suffix = None):
    if output_dir is not None:
        if suffix is not None:
            output_file = f'{output_dir}/{name}-{suffix}.html'
        else:
            output_file = f'{output_dir}/{name}.html'
        fig.write_html(output_file, include_plotlyjs = 'cdn')
    else:
        fig.show()
    

pd.plotting.register_matplotlib_converters()

parser = argparse.ArgumentParser(description='Predict COVID vaccine administrations')
parser.add_argument('-o', '--output-dir', metavar='DIR', 
                    help='Export graphs to the following folder. If not provided, the graphs will be shown in a browser.')
parser.add_argument('-s', '--suffix', metavar='SUFFIX', 
                    help='Add the following suffix to the output filenames')
parser.add_argument('-p', '--prediction-date', metavar='DATE',  type=date.fromisoformat,
                    help='The day to start the prediction from. (default: the day after the last administration.)')
args = parser.parse_args()

administered = pd.read_csv('Data/COVID19BE_VACC.csv')
administered['DATE'] = administered['DATE'].apply(pd.Timestamp)
administered['DOSE'] = administered['DOSE'].replace({'A': 'first_dose', 'B': 'second_dose', 'C': 'dose'})
administered['BRAND'] = administered['BRAND'].replace({'AstraZeneca-Oxford': 'AstraZeneca/Oxford',
                                                       'Pfizer-BioNTech': 'Pfizer/BioNTech'})
administered = administered.groupby(['DATE', 'BRAND', 'DOSE'])['COUNT'].sum().unstack(level = [1, 2]).fillna(0)

#Cut of some of the administrations to try out prediction
if args.prediction_date is None:
    prediction_date = (administered.index.max() + timedelta(days=1)).date()
else:
    prediction_date = args.prediction_date
prediction_date = datetime.combine(prediction_date, time())
prediction_end_date = datetime(year = 2021, month = 7, day = 5)
administered_complete = administered.copy()
administered = administered[administered.index < prediction_date]

deliveries = pd.read_csv('Data/delivered.csv')
deliveries['date'] = deliveries['date'].apply(pd.Timestamp)
deliveries = deliveries.set_index(['manufacturer', 'date'])['amount'].sort_index()
expected_deliveries = pd.read_csv('Data/predicted-deliveries.csv')
expected_deliveries['date'] = expected_deliveries['date'].apply(pd.Timestamp)
expected_deliveries = expected_deliveries.set_index(['manufacturer', 'date'])['amount'].sort_index()
expected_deliveries_pessimistic = pd.read_csv('Data/predicted-deliveries-pessimistic.csv')
expected_deliveries_pessimistic['date'] = expected_deliveries_pessimistic['date'].apply(pd.Timestamp)
expected_deliveries_pessimistic = expected_deliveries_pessimistic.set_index(['manufacturer', 'date'])['amount'].sort_index()

predicted_administrations = pd.DataFrame(columns = administered.columns)
predicted_administrations_pessimistic = pd.DataFrame(columns = administered.columns)
for manufacturer, details in manufacturers.iterrows():
    predicted = predict(administered[manufacturer], deliveries.loc[manufacturer], 
                        expected_deliveries.loc[manufacturer] if manufacturer in expected_deliveries else pd.Series(), prediction_end_date, 
                        details['time_between_doses'], details['second_dose_reserved'])
    predicted_pessimistic = predict(administered[manufacturer], deliveries.loc[manufacturer], 
                                    expected_deliveries_pessimistic.loc[manufacturer] if manufacturer in expected_deliveries_pessimistic else pd.Series(), 
                                    prediction_end_date, details['time_between_doses'], details['second_dose_reserved'])
    
    if isinstance(predicted, pd.Series):
        predicted_administrations[(manufacturer, 'dose')] = predicted
        predicted_administrations_pessimistic[(manufacturer, 'dose')] = predicted_pessimistic
    else:
        for t in predicted.columns:
            predicted_administrations[(manufacturer, t)] = predicted[t]
            predicted_administrations_pessimistic[(manufacturer, t)] = predicted_pessimistic[t]
predicted_administrations = predicted_administrations.fillna(0.0)
predicted_administrations_pessimistic = predicted_administrations_pessimistic.fillna(0.0)

fig = go.Figure()
plot(fig, administered.sum(axis = 'columns', level = 0), administered_complete.sum(axis = 'columns', level = 0), 
     predicted_administrations.sum(axis = 'columns', level = 0), predicted_administrations_pessimistic.sum(axis='columns', level=0), 
     'Administered')
fig.update_layout(hoverlabel = {'bgcolor': 'black'},
                  xaxis_title = "Date", yaxis_title = "Nr of administered vaccines", 
                  legend = dict(orientation= 'h', 
                                yanchor = 'bottom',
                                y = 1.02,
                                xanchor='right',
                                x = 1),
                  separators='. ')
show_or_save_plot(fig, 'administered', args.output_dir, args.suffix)

fig = go.Figure()
plot(fig, administered.swaplevel(axis = 'columns')[['first_dose', 'dose']].droplevel(axis = 'columns', level = 0), 
     administered_complete.swaplevel(axis = 'columns')[['first_dose', 'dose']].droplevel(axis = 'columns', level = 0), 
     predicted_administrations.swaplevel(axis = 'columns')[['first_dose', 'dose']].droplevel(axis = 'columns', level = 0), 
     predicted_administrations_pessimistic.swaplevel(axis = 'columns')[['first_dose', 'dose']].droplevel(axis = 'columns', level = 0),
     'Partially vaccinated', relative = True)
fig.update_layout(hoverlabel = {'bgcolor': 'black'},
                  xaxis_title = "Date", yaxis_title = "Percentage at least partially vaccinated of complete population",
                  legend = dict(orientation= 'h', 
                                yanchor = 'bottom',
                                y = 1.02,
                                xanchor='right',
                                x = 1),
                  separators='. ')
show_or_save_plot(fig, 'partially', args.output_dir, args.suffix)

fig = go.Figure()
plot(fig, administered.swaplevel(axis = 'columns')[['second_dose', 'dose']].droplevel(axis = 'columns', level = 0), 
     administered_complete.swaplevel(axis = 'columns')[['second_dose', 'dose']].droplevel(axis = 'columns', level = 0), 
     predicted_administrations.swaplevel(axis = 'columns')[['second_dose', 'dose']].droplevel(axis = 'columns', level = 0), 
     predicted_administrations_pessimistic.swaplevel(axis = 'columns')[['second_dose', 'dose']].droplevel(axis = 'columns', level = 0), 
     'Fully vaccinated', relative = True)
fig.update_layout(hoverlabel = {'bgcolor': 'black'},
                  xaxis_title = "Date", yaxis_title = "Percentage fully vaccinated of complete population",
                  legend = dict(orientation= 'h', 
                                yanchor = 'bottom',
                                y = 1.02,
                                xanchor='right',
                                x = 1),
                  separators='. ')
show_or_save_plot(fig, 'completely', args.output_dir, args.suffix)