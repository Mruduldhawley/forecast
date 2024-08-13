#@task
def forecast_from_parameter_table(ds_id:str,query:str,Low_level_cols:str, period:int,cat: str, dat: str, metric: str, forecast_type:str, flag:bool, frcst_table:str):
    """Will generate a Forecast Table form Hyper Parameters"""
    try:
        import os
        import re
        import sys
        import math
        import linecache
        import pandas as pd
        import numpy as np
        from conversight import Dataset, Context # ,Flow, Parameter, SmartAnalytics,task,TaskLibrary,FlowLibrary
        from kats.models.prophet import ProphetModel, ProphetParams
        from kats.consts import TimeSeriesData
        from kats.models.holtwinters import HoltWintersParams, HoltWintersModel
        # from pmdarima.arima import auto_arima
        import datetime as dt
        from sklearn import metrics, preprocessing
        from sklearn.metrics import mean_absolute_error,mean_squared_error
        from statsmodels.tsa.seasonal import seasonal_decompose as sdd
        from statsmodels.tsa.stattools import adfuller
        from scipy.stats import kruskal
        from sklearn.preprocessing import StandardScaler
        import statsmodels.tsa.stattools as stt
        import statsmodels.tsa.seasonal as sts
        from pandas.tseries.offsets import DateOffset
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        import tqdm

        import warnings
        warnings.filterwarnings("ignore")
        pd.set_option('display.max_columns', None)


        class suppress_logs(object):
            '''
            A context manager for doing a "deep suppression" of stdout and stderr in
            Python, i.e. will suppress all print, even if the print originates in a
            compiled C/Fortran sub-function.
            This will not suppress raised exceptions, since exceptions are printed
            to stderr just before a script exits, and after the context manager has
            exited (at least, I think that is why it lets exceptions through).
            '''
            def __init__(self):
                # Open a pair of null files
                self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
                # Save the actual stdout (1) and stderr (2) file descriptors.
                self.save_fds = [os.dup(1), os.dup(2)]

            def __enter__(self):
                # Assign the null pointers to stdout and stderr.
                os.dup2(self.null_fds[0], 1)
                os.dup2(self.null_fds[1], 2)

            def __exit__(self, *_):
                # Re-assign the real stdout/stderr back to (1) and (2)
                os.dup2(self.save_fds[0], 1)
                os.dup2(self.save_fds[1], 2)
                # Close the null files
                for fd in self.null_fds + self.save_fds:
                    os.close(fd)

        ctx = Context()
        ctx.log.info('Flow Initiated')
        ds = Dataset(ds_id)
        ctx.log.info('|------------------------------------------------------------|')
        ctx.log.info('|------------------Dataset Loaded ---------------------------|')
        ctx.log.info('|------------------------------------------------------------|')
        P = dt.datetime.today().replace(day=1).strftime("%Y-%m")
        P_query = f"""select * from #Parameter_Table where m_Period = '{P}' """
        ctx.log.info(f'ParaMeter Table Query : {P_query}')
        parameters = pd.read_csv('paramTable.csv')
        #parameters = ds.sqlDf(P_query)
        ctx.log.info('Fetch data from parameter table successful')
        df = ds.sqlDf(query)
        df.columns = [i.replace('m_','') for i in df.columns]
        # df_21 = pd.read_csv('hos_2021.csv')
        # df = pd.concat([df_21,df],ignore_index=True)
        df.dropna(subset=[dat,cat], inplace=True)
        df = df.drop_duplicates(keep='last')

        pattern = r"^m_(.*)"  # Match "m_" at the beginning, followed by any character sequence
        parameters.columns = [re.sub(pattern, r"\1", col) for col in parameters.columns]
        parameters.rename(columns = {cat:'key_column'}, inplace = True)
        #cat = str(parameters.columns[0])
#--------------------------------------------------------------------------------------------------------------------------------------------------
        def printException():
            exc_type, exc_obj, tb = sys.exc_info()
            f = tb.tb_frame
            lineno = tb.tb_lineno
            filename = f.f_code.co_filename
            linecache.checkcache(filename)
            line = linecache.getline(filename, lineno, f.f_globals)
            resp = 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)
            return resp

#--------------------------------------------------------------------------------------------------------------------------------------------------
        def get_preprocessing_params(df: pd.DataFrame, cat: str, dat: str, metric: str) -> dict:
            try:
                if not "float" in str(df[metric].dtype) or "int" in str(df[metric].dtype):
                    return {"RESPONSE": "fail", "message": "Wrong type of metric column"}
                df[dat] = pd.to_datetime(df[dat])
                cat = cat.split(",")
                df['key_column'] = ""
                if len(cat) == 1:
                    df['key_column'] = df[cat[0]]
                else:
                    for i in cat:
                        df['key_column'] += "_" + df[i].astype(str)
                df["key_column"] = df["key_column"].str.lstrip("_")
                df = df[['key_column', dat, metric]]
                df = df.dropna()
                field_Mapping = {'Dat': dat,
                                'Value': metric,
                                'Category': 'key_column'}
                resp = {"RESPONSE": "success",
                        "data": (field_Mapping, df)
                        }
                return resp

            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in get_preprocessing_params ==>> {printException()}   local Expection caused by ==>> {e}'}

#---------------------------------------------------------------------------------------------------------------------------------------------------
        def process_data(field_Mapping,df,mn_date,mx_date): 
            try:
                df[field_Mapping['Dat']] = pd.to_datetime(df[field_Mapping['Dat']]).dt.floor('D')
                df[field_Mapping['Dat']] = df[field_Mapping['Dat']].dt.to_period('W').apply(lambda r: r.start_time)
                #df[field_Mapping['Date']] = ((df[field_Mapping['Dat']] + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1)).dt.floor('d'))
                max_date = str(mx_date)
                dateprocess_max = dt.datetime.strptime(max_date, "%Y-%m-%d %H:%M:%S")
                min_date = str(mn_date)
                dateprocess_min = dt.datetime.strptime(min_date, "%Y-%m-%d %H:%M:%S")
                max_date_limit = (dateprocess_max - pd.DateOffset(weeks = 1)).strftime('%Y-%m-%d')
                min_date_limit = str(dateprocess_min.year) +'-'+ str(dateprocess_min.month) +'-'+ str(dateprocess_min.day)
                df = df.sort_values(field_Mapping['Dat'])
                df[field_Mapping['Dat']] = pd.to_datetime(df[field_Mapping['Dat']]).dt.floor('D')
                training = df.groupby(field_Mapping['Dat'], observed=True)[field_Mapping['Value']].sum().reset_index()
                training.rename(columns={field_Mapping['Dat'] : 'dat'},inplace=True)
                dates_required = pd.date_range(min_date_limit, max_date_limit, freq='W-MON', inclusive='both').strftime("%Y-%m-%d").tolist()
                dates_required = pd.DataFrame(dates_required, columns = ['dat'])
                dates_required.dat = pd.to_datetime(dates_required.dat).dt.floor('D')
                processed_data = dates_required.merge(training,on='dat', how='left')
                processed_data.columns = ['time', field_Mapping['Value']]
                processed_data = processed_data.fillna(0)
                processed_data['time'] = pd.to_datetime(processed_data['time'])
                return processed_data
            
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in process_data ==>> {printException()}   local Expection caused by ==>> {e}'}

#---------------------------------------------------------------------------------------------------------------------------------------------------
        def m1_forecast(data,period,ws,s_type):
            warnings.simplefilter(action='ignore', category=FutureWarning)
            try:
                try:
                    data.columns = ['time','y']
                    data['y'] = np.where(data['y'] < 0, 0.0, data['y'])

                    params = ProphetParams(
                        weekly_seasonality=ws,
                        yearly_seasonality=True,
                        seasonality_mode=s_type)
                    train_p = TimeSeriesData(data)
                    m = ProphetModel(train_p, params)
                    with suppress_logs():
                        m.fit()
                    fcst = m.predict(steps=period, freq="W-MON")
                    model1_result = fcst['fcst']
                    return np.array(model1_result)
                
                except:
                    from prophet import Prophet
                    data.columns = ['ds', 'y']
                    data['y'] = np.where(data['y'] < 0, 0.0, data['y'])

                    params = {
                        'daily_seasonality' : True,
                        'weekly_seasonality' : ws,
                        'yearly_seasonality' : True,
                        'seasonality_mode' : s_type}
                    m = Prophet(**params)
                    with suppress_logs():
                        m.fit(data)
                    future = m.make_future_dataframe(periods=period, freq="W-MON", include_history=False)
                    fcst = m.predict(future)
                    fcst = fcst[['ds', 'yhat']]
                    fcst.columns = ['time','fcst']
                    model1_result = fcst['fcst']
                    return np.array(model1_result)
            
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in m1_forecast ==>> {printException()}   local Expection caused by ==>> {e}'}
#---------------------------------------------------------------------------------------------------------------------------------------------------

        def m2_forecast(data,period,ms,ws,sv):
            try:
                train_data = data
                train_data['unique_id'] = 1

                if ms == True or ws == True: seas = True
                else : seas = False
                if sv == 1: s_v = True
                else : s_v = False
                train_data.columns = ['ds','y','unique_id']

                model = StatsForecast(
                                models = [AutoARIMA(season_length = 52, trace = False, start_p=0, d=1, start_q=0,
                                                max_p=4, max_d=4, max_q=4,
                                                start_P=0, D=1, start_Q=0, max_P=4, max_D=4, max_Q=4 ,seasonal=seas,
                                                stationary=s_v, stepwise=True),
                                        ],
                    freq = "W-MON"
                )

                model.fit(train_data)
                pred = model.predict(h = period)
                pred = pred.reset_index().drop(["unique_id"], axis = 1) 
                pred = pred.set_index("ds")
                pred.columns = ['Pred']

                pred = np.array(pred["Pred"])
                pred = pred.flatten()
                return pred
            
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in m2_forecast ==>> {printException()}   local Expection caused by ==>> {e}'}
#------------------------------------------------------------------------------------------------------------------------------------------------------

        def m3_forecast(data,period,ms,ws,s_type,sv):
            return (m1_forecast(data,period,ws,s_type)[:period] + m2_forecast(data,period,ms,ws,sv)[:period]) / 2
#------------------------------------------------------------------------------------------------------------------------------------------------------
    
        def m4_forecast(data,period,s_type):
            try:
                data.columns = ['time','y']
                data['y'] = np.where(data['y'] < 0, 0.0, data['y'])
                train_p = TimeSeriesData(data)
                try:
                    params = HoltWintersParams(
                        damped = True,
                        trend=s_type.lower()[:3],
                        seasonal=s_type.lower()[:3],
                        seasonal_periods=52,
                    )
                    m = HoltWintersModel(
                    data=train_p, 
                    params=params,)
                    m.fit()
                except:
                    params = HoltWintersParams(
                        damped = True,
                        trend='add',
                        seasonal='add',
                        seasonal_periods=48,
                    )
                    m = HoltWintersModel(
                    data=train_p, 
                    params=params,)
                    m.fit()
                res = m.predict(steps= period,include_history= False)
                model4_result = res['fcst']
                return model4_result
            
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in m4_forecast ==>> {printException()}   local Expection caused by ==>> {e}'}
#------------------------------------------------------------------------------------------------------------------------------------------------------

        def m5_forecast(data,period,s_type,ms,ws,sv):
            return (m4_forecast(data,period,s_type)[:period] + m2_forecast(data,period,ms,ws,sv)[:period]) / 2 
#------------------------------------------------------------------------------------------------------------------------------------------------------

        def m6_forecast(data,period,s_type):
            try:
                data.columns = ['time','y']
                data['y'] = np.where(data['y'] < 0, 0.0, data['y'])
                data.index = data['time']
                data.drop(['time'], axis=1, inplace=True)
                model_data = pd.Series(data['y'], index=data.index)
                try:
                    model = ETSModel(
                        model_data,
                        error=s_type.lower()[:3],
                        trend=s_type.lower()[:3],
                        seasonal=s_type.lower()[:3],
                        damped_trend=True,
                        seasonal_periods=52,
                        freq='W-MON',
                    )
                    with suppress_logs():
                        fit = model.fit()
                except:
                    model = ETSModel(
                        model_data,
                        error="add",
                        trend="add",
                        seasonal="add",
                        damped_trend=True,
                        seasonal_periods=52,
                        freq='W-MON',
                    )
                    with suppress_logs():
                        fit = model.fit()
                
                res6 = fit.forecast(period)
                pred6= pd.DataFrame(res6).reset_index()
                pred6.columns = ['ds','Pred6']
                pred6 = pred6.set_index("ds")
                model6_result = pred6['Pred6'][:period]
                return model6_result
        
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in m4_forecast ==>> {printException()}   local Expection caused by ==>> {e}'}
#------------------------------------------------------------------------------------------------------------------------------------------------------

        def make_forecast_frame(data_with_date,bes_m,max_date,period,field_Mapping,ms,ws,s_type,sv):
            try:
                forecast_frame = pd.DataFrame()
                date_list = [max_date.date() + dt.timedelta(weeks=x) for x in range(period)]
                forecast_frame['time'] = date_list
                rng = (0,10)
                scaler = preprocessing.MinMaxScaler(feature_range=(rng[0], rng[1]))
                normed = scaler.fit(np.array(data_with_date[field_Mapping['Value']]).reshape(-1, 1))
                normed = scaler.transform(np.array(data_with_date[field_Mapping['Value']]).reshape(-1, 1))
                norm_lst = [round(i[0],4) for i in normed]
                data = pd.DataFrame()
                data['time'] = data_with_date['time']
                data[field_Mapping['Value']] = norm_lst
                data[field_Mapping['Value']] = np.where(data[field_Mapping['Value']] < 0 ,0,data[field_Mapping['Value']])
                if bes_m == 'm1':
                    result = m1_forecast(data,period,ws,s_type)
                elif bes_m == 'm2':
                    result = m2_forecast(data,period,ms,ws,sv)
                elif bes_m == 'm3':
                    result = m3_forecast(data,period,ms,ws,s_type,sv)
                elif bes_m == 'm4':
                    result = m4_forecast(data,period,s_type)
                elif bes_m == 'm6':
                    result = m6_forecast(data,period,s_type)
                else:
                    result = m5_forecast(data,period,s_type,ms,ws,sv)
                forecast_frame['Forecasted_Value'] = scaler.inverse_transform(np.array(result).reshape(-1, 1))
                return forecast_frame
            
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in make_forecast_frame ==>> {printException()}   local Expection caused by ==>> {e}'}
#---------------------------------------------------------------------------------------------------------------------------------------------------------

        def start_forecast(data,parameter_df,field_Mapping,period):
            try:
                def get_week_start_date(input_date):
                    input_date = str(input_date)
                    date_object = dt.datetime.strptime(input_date, "%Y-%m-%d %H:%M:%S")
                    day_of_week = date_object.weekday()
                    days_to_subtract = day_of_week
                    week_start_date = date_object - dt.timedelta(days=days_to_subtract)
                    return pd.to_datetime(week_start_date.strftime("%Y-%m-%d"))

                ctx.log.info('Forecast Started')
                # from IPython.display import clear_output
                mn_date,mx_date = data[field_Mapping['Dat']].min(), data[field_Mapping['Dat']].max()
                mn_date = get_week_start_date(mn_date)
                mx_date = get_week_start_date(mx_date)
                unique_Categorys = list(parameter_df[field_Mapping['Category']].unique())
                #field_Mapping,data_with_date,month_seas,week_seas,stationarity,s_type
                forecast_df = pd.DataFrame()
                t = tqdm.tqdm(unique_Categorys)
                for i in t:
                    # clear_output(wait=True)
                    t.set_description(f"ITERATING OVER {i}")
                    df = process_data(field_Mapping,data[data[field_Mapping['Category']] == i],mn_date,mx_date)
                    params = parameter_df[parameter_df[field_Mapping['Category']] == i]
                    params = params.reset_index(drop=True)
                    sationarity,ms,ws,seas_type,best_m = params['Stationarity'][0],params['Monthly_Seasonality'][0],params['Weekly_Seasonality'][0],params['Seasonality_Type'][0],params['Best_Model'][0]
                    if sationarity == 1: sationarity = True
                    else : sationarity = False
                    if ms == 1: ms = True
                    else : ms = False
                    if ws == 1: ws = True
                    else : ws = False
                    frame = make_forecast_frame(df,best_m,mx_date,period,field_Mapping,ms,ws,seas_type,sationarity)
                    frame.time  = pd.to_datetime(frame.time)
                    # frame = (frame.set_index('time')
                    #      .resample('d')
                    #      .ffill()
                    #      .div(7)
                    #      .reset_index()
                    #      )
                    frame[field_Mapping['Category']] = i
                    forecast_df = pd.concat([frame,forecast_df])
                return forecast_df,mx_date
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in start_forecast ==>> {printException()}   local Expection caused by ==>> {e}'}
#--------------------------------------------------------------------------------------------------------------------------------------------------------
        
        def Trickel_down(df_frcst ,df_q,filed_Mapping,lower_level):
                try:
                    #df_frcst.rename(columns = {'key_column':cat}, inplace = True)        
                    df_q.rename(columns = {cat:'key_column'}, inplace = True)
                    cols = lower_level.split(',')
                    if len(cols) > 1:
                        df_q['lower_level'] = df_q[cols].agg('_'.join, axis=1)
                    else : df_q['lower_level'] = df_q[lower_level.strip()]
                    df_q[filed_Mapping['Dat']] = pd.to_datetime(df_q[filed_Mapping['Dat']]).dt.floor('D')
                    df_q[filed_Mapping['Dat']] = df_q[filed_Mapping['Dat']].dt.to_period('W').apply(lambda r: r.start_time)
                    df_q = df_q.groupby(['lower_level',filed_Mapping['Category'],filed_Mapping['Dat']], observed=True)[filed_Mapping['Value']].sum().reset_index()
                    df_frcst['time'] = pd.to_datetime(df_frcst['time']).dt.floor('D')
                    # df_q = df_q[['lower_level',filed_Mapping['Category'],filed_Mapping['Dat'], 'label_Products.Category', 'label_Location.Name']]
                    df_q = df_q[[filed_Mapping['Dat'],filed_Mapping['Category'],filed_Mapping['Value'],'lower_level']]
                    final_forecast = pd.DataFrame()
                    tri = tqdm.tqdm(df_q[filed_Mapping['Category']].unique())
                    for item in tri:
                        tri.set_description(f"ITERATING OVER {item}")
                        df_ = df_q[df_q[filed_Mapping['Category']]==item]
                        df_ = df_[df_[filed_Mapping['Dat']] > df_[filed_Mapping['Dat']].max() - DateOffset(weeks=16)]
                        #df_ = df_[df_['Forecast_Dt'] > df['time'].min() - DateOffset(weeks=16)]
                        df_ = df_.groupby(['lower_level'], observed=True)[filed_Mapping['Value']].sum().reset_index()
                        df_[filed_Mapping['Value']] = df_[filed_Mapping['Value']] / df_[filed_Mapping['Value']].sum()
                        d = dict(df_.values)
                        for x in d.keys():
                            df_f = df_frcst[df_frcst[filed_Mapping['Category']]==item]
                            df_f['Forecasted_Value'] = df_f['Forecasted_Value']*d[x]
                            # df_f['fcst_lower'] = df_f['fcst_lower']*d[x]
                            # df_f['fcst_upper'] = df_f['fcst_upper']*d[x]
                            df_f['lower_level'] = x
                            final_forecast = pd.concat([final_forecast,df_f])
                    final_forecast = final_forecast.round(5)
                    final_forecast[cols]  = final_forecast.lower_level.str.split("_",expand=True)
                    final_forecast.drop(['lower_level'], axis = 1 , inplace = True)
                    df_q.drop(['lower_level'], axis = 1 , inplace = True)
                    del df_q
                    return final_forecast
                
                except Exception as e:
                    return {"RESPONSE": "fail",
                            "message": f'Error in Trickel_down ==>> {printException()}   local Expection caused by ==>> {e}'}
#---------------------------------------------------------------------------------------------------------------------------------------------------------

        def add_actuals(forecast,df,fld_map):
            try:
                forecast.rename(columns = {'time':'Dat'}, inplace = True)
                df.rename(columns = {fld_map['Dat'] : 'Dat'}, inplace = True)
                cols = list(forecast.columns[2:])
                cols.append('Dat')
                
                col = cols.copy()
                # df = df.groupby(col,observed=True)[fld_map['Value']].sum().reset_index()
                df = df.groupby(cols, observed=True)[fld_map['Value']].sum().reset_index()
                col.append(fld_map['Value'])
                df = df[col]
                
                return forecast.merge(df, on = cols , how = 'outer').fillna(0.0)
                # return forecast.merge(df, on = cols , how = 'outer').fillna(0.0)
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in add_actuals ==>> {printException()}   local Expection caused by ==>> {e}'}
#------------------------------------------------------------------------------------------------------------------------------------------------------
        def scale_for_12months(forecast,p):
            try:
                s = sum(forecast)
                d = s + (s * p)/100
                diff = d - s
                a = diff / 12
                l = [a - 9/2 ,a-7/2,a - 5/2, a - 3/2, a -1/2 , a + 1/2, a + 3/2 , a + 5/2, a + 7/2 , a+ 9/2]
                lst = []
                for i,j in zip(l,forecast):
                    lst.append(j + i)
                return lst
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in scale_for_12months ==>> {printException()}   local Expection caused by ==>> {e}'}
#--------------------------------------------------------------------------------------------------------------------------------------------------------
        def custom_round(number):
            decimal_part = number - int(number)
            if decimal_part >= 0.25:
                return int(number) + 1
            else:
                return int(number)
            
#---------------------------------------------------------------------------------------------------------------------------------------------------------
        def process_data_month(field_Mapping, df):
            try:
                cols = Low_level_cols.split(',')
                grp_cols = cols.copy()
                grp_cols.append(field_Mapping['Dat'])
                grp_cols.append(cat)
                
                df[field_Mapping['Dat']] = pd.to_datetime(df[field_Mapping['Dat']]).dt.floor('D')
                df[field_Mapping['Dat']] = ((df[field_Mapping['Dat']] + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1)).dt.floor('d'))

                df = df.sort_values(field_Mapping['Dat'])
                df = df.groupby(grp_cols, observed=True)[field_Mapping['Value']].sum().reset_index()
                
                return {"RESPONSE": "success",
                        "data": df
                        }
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in process_data_month ==>> {printException()}   local Expection caused by ==>> {e}'}
            
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
        def Trickel_down_month(df_frcst ,df_q,filed_Mapping,lower_level):
            try:
                #df_frcst.rename(columns = {'key_column':cat}, inplace = True)        
                df_q.rename(columns = {cat:'key_column'}, inplace = True)
                cols = lower_level.split(',')
                if len(cols) > 1:
                    df_q['lower_level'] = df_q[cols].agg('_'.join, axis=1)
                else : df_q['lower_level'] = df_q[lower_level.strip()]
                df_q[filed_Mapping['Dat']] = pd.to_datetime(df_q[filed_Mapping['Dat']]).dt.floor('D')
                df_frcst['time'] = pd.to_datetime(df_frcst['time']).dt.floor('D')
                df_q = df_q[[filed_Mapping['Dat'],filed_Mapping['Category'],filed_Mapping['Value'],'lower_level']]
                final_forecast = pd.DataFrame()
                tri = tqdm.tqdm(df_q[filed_Mapping['Category']].unique())
                for item in tri:
                    tri.set_description(f"ITERATING OVER {item}")
                    df_ = df_q[df_q[filed_Mapping['Category']]==item]
                    #df_ = df_[df_[filed_Mapping['Dat']] > df_[filed_Mapping['Dat']].max() - DateOffset(month=4)]
                    df_ = df_.groupby(['lower_level'], observed=True)[filed_Mapping['Value']].sum().reset_index()
                    df_[filed_Mapping['Value']] = df_[filed_Mapping['Value']] / df_[filed_Mapping['Value']].sum()
                    d = dict(df_.values)
                    for x in d.keys():
                        df_f = df_frcst[df_frcst[filed_Mapping['Category']]==item]
                        df_f.drop(columns='key_column',inplace=True)
                        df_f = (df_f.set_index('time')
                            .resample('d')
                            .ffill()
                            .div(7)
                            .reset_index()
                            )
                        df_f['time'] = ((df_f['time'] + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1)).dt.floor('d'))
                        df_f = df_f.groupby('time', observed=True)['Forecasted_Value'].sum().reset_index()
                        df_f['Forecasted_Value'] = df_f['Forecasted_Value']*d[x]
                        df_f['lower_level']=x
                        df_f['key_column']=item
                        final_forecast = pd.concat([final_forecast,df_f],ignore_index=True)
                final_forecast = final_forecast.round(5)
                final_forecast[cols]  = final_forecast.lower_level.str.split("_",expand=True)
                final_forecast.drop(['lower_level'], axis = 1 , inplace = True)
                df_q.drop(['lower_level'], axis = 1 , inplace = True)
                del df_q
                return final_forecast
            
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in Trickel_down_month ==>> {printException()} local Expection caused by ==>> {e}'}

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
        def add_actuals_month(forecast,df,fld_map):
            try:
                forecast.rename(columns = {'time':'Dat'}, inplace = True)
                df.rename(columns = {fld_map['Dat'] : 'Dat'}, inplace = True)
                cols = list(forecast.columns[2:])
                cols.append('Dat')

                col = cols.copy()
                df = df.groupby(cols, observed=True)[fld_map['Value']].sum().reset_index()
                col.append(fld_map['Value'])
                df = df[col]
                cur_mnth = dt.datetime.today().replace(day=1).strftime('%Y-%m-%d')

                forecast = forecast.merge(df, on = cols , how = 'outer').fillna(0.0)

                forecast['Forecasted_Value'] = np.where(forecast['Dat'] == cur_mnth,
                                                    forecast['Forecasted_Value']+forecast[metric],
                                                    forecast['Forecasted_Value'])
                forecast[metric] = np.where(forecast['Dat'] == cur_mnth,
                                                    0,
                                                    forecast[metric])
                P = dt.datetime.today().replace(day=1).strftime('%Y-%m')
                forecast['Period'] = P
                return forecast
            
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in add_actuals_month ==>> {printException()}   local Expection caused by ==>> {e}'}

#--------------------------------------------------------------------------------------------------------------------------------------------------------
        r = get_preprocessing_params(df,cat,dat,metric)
        if r["RESPONSE"] == 'fail':
            return r
        FLdMap, df = r['data']
        # FLdMap, df = get_preprocessing_params(df,cat,date,metric)  

        ctx.log.info(f'Field Mapping Generated {FLdMap}')
        ctx.log.info('Starting Preprocessing')
        
        frame,max_date = start_forecast(df,parameters,FLdMap,period + 4)
        frame['Forecasted_Value'] = np.where(frame['Forecasted_Value'] < 0, 0.0, frame['Forecasted_Value'])
        frame.to_csv('no_trik.csv', index = False)
        
        #frame = pd.read_csv('no_trik.csv',)
        
        ds = Dataset(ds_id)
        df = ds.sqlDf(query)
        df.dropna(subset=[dat,cat], inplace=True)
        df = df.drop_duplicates(keep='last')
        df[dat] = pd.to_datetime(df[dat]).dt.floor('D')
        if forecast_type.lower() == 'week':
            df[dat] = df[dat].dt.to_period('W').apply(lambda r: r.start_time)
            final = Trickel_down(frame,df,FLdMap,Low_level_cols)
            ctx.log.info('Post Processing Completed !')
            ctx.log.info('Adding Historical Data to Forecast Frame ')
            final_frame = add_actuals(final,df,FLdMap)
        if forecast_type.lower() == 'month':
            res = process_data_month(FLdMap, df)
            df = res['data']
            final = Trickel_down_month(frame ,df,FLdMap,Low_level_cols)
            ctx.log.info('Post Processing Completed !')
            ctx.log.info('Adding Historical Data to Forecast Frame ')
            final_frame = add_actuals_month(final,df,FLdMap)
        final_frame['Forecasted_Value_Cust_Ceil'] = final_frame['Forecasted_Value'].apply(custom_round)
        final_frame = final_frame.rename(columns={'Dat':'Date','key_column':cat,'Forecasted_Value_Cust_Ceil':'Forecasted_quantity'})
        
        ctx.log.info('Successufully added historical data to forecast ')
        ctx.log.info('Pushing Data to Db in "Smart Analytics"')
        ctx.log.info('|------------------------------------------------------------|')
        ctx.log.info('|----------------- Refreshing Dataset -----------------------|')
        ctx.log.info('|------------------------------------------------------------|')
        
        # final_fore = pd.DataFrame(final_frame,index=[0])
        final_frame = final_frame[(final_frame["Date"] < final_frame["Date"].max() )]
        final_frame.reset_index(drop=True, inplace=True)
        final_frame['Forecasted_quantity'] = np.where(final_frame['Forecasted_quantity'] < 0,0,final_frame['Forecasted_quantity'])
        final_frame.to_csv('fore.csv', index=False)
        
        # Get column names with object data type
        object_columns = final_frame.select_dtypes(include=['object']).columns
        final_frame[object_columns] = final_frame[object_columns].astype(str)
        final_frame['Forecasted_quantity'] = final_frame['Forecasted_quantity'].astype(float)
        final_frame[metric] = final_frame[metric].astype(float)
        print(final_frame.dtypes)
        
        response1 = ds.smartAnalytics.create(frcst_table,final_frame, False, flag, True)
        if ("status" in response1 and response1["status"] == "success"):
            ctx.log.info("smart analytics creation is done -> {}".format(dt.datetime.utcnow()))
            ctx.log.info("|---------------------------------------------------------------------------------|")
            ctx.log.info("| Creation of table was done properly and smart analytics is created successfully |")
            ctx.log.info("|---------------------------------------------------------------------------------|")
        else:
            ctx.log.error("smart analytics creation is failed -> {}".format(dt.datetime.utcnow()))
            return {"status": "failed", "message": "Organization processing failed due to an error {}".format(response1["message"] if "message" in response1 else "internalServerError")}
        return {"status": "success", "message": "- Forecast has been completed successfully! "}

    except Exception as e:
        return {"status": "Failed", "message": f'Error in forecast_from_parameter_table == >> {printException()} Local ERROR == >> {e}'}

