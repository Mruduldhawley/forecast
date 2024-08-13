def create_parms(ds_id: str, query: str, cat: str, date: str, metric_col: str, flag:bool):
    """
    Create a Table of hyperparameters for a given category, dataset ID, and query.

    Args:
        cat: The category column name of data to retrieve.
        date: The date column name of the data to retrieve.
        ds_id: The data set ID.
        metric_col: The metric column name to retrieve.
        query: The query to use to filter the data from db.

    """
    import os
    import linecache
    import sys
    import math
    import pandas as pd
    import numpy as np
    from kats.models.prophet import ProphetModel, ProphetParams
    from kats.consts import TimeSeriesData
    from kats.models.holtwinters import HoltWintersParams, HoltWintersModel
    # from pmdarima.arima import auto_arima
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn import metrics
    from sklearn import preprocessing
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    from sklearn.metrics import explained_variance_score, r2_score
    from statsmodels.tsa.seasonal import seasonal_decompose as sdd
    from statsmodels.tsa.stattools import adfuller
    from scipy.stats import kruskal
    from sklearn.preprocessing import StandardScaler
    import statsforecast
    import statsmodels.tsa.stattools as stt
    import statsmodels.tsa.seasonal as sts
    import datetime as dt
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    import tqdm

    import warnings
    warnings.filterwarnings("ignore")

    from conversight import Dataset, Context
    ctx = Context()
    try:     
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
                self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
                self.save_fds = [os.dup(1), os.dup(2)]

            def __enter__(self):
                os.dup2(self.null_fds[0], 1)
                os.dup2(self.null_fds[1], 2)

            def __exit__(self, *_):
                os.dup2(self.save_fds[0], 1)
                os.dup2(self.save_fds[1], 2)
                for fd in self.null_fds + self.save_fds:
                    os.close(fd)
#-------------------------------------------------------------------------------------------------------------------------------------------------        
        from conversight import Dataset, Context
        ctx = Context()
        ctx.log.info('Flow Initiated')
        # query = " ".join(line.strip() for line in query.splitlines())
        # query = query.replace('(#)', '#')
        # query = query.replace('( #)', '#')
        # query = query.replace('(# )', '#')
        # query = query.replace('(  #)', '#')
        # query = query.replace('(#  )', '#')
#-------------------------------------------------------------------------------------------------------------------------------------------------

        def printException():
            exc_type, exc_obj, tb = sys.exc_info()
            f = tb.tb_frame
            lineno = tb.tb_lineno
            filename = f.f_code.co_filename
            linecache.checkcache(filename)
            line = linecache.getline(filename, lineno, f.f_globals)
            resp = 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)
            return resp
#-------------------------------------------------------------------------------------------------------------------------------------------------

        def get_preprocessing_params(df: pd.DataFrame, cat: str, date_col: str, metric_col: str) -> dict:
            try:
                if not "float" in str(df[metric_col].dtype) or "int" in str(df[metric_col].dtype):
                    return {"RESPONSE": "fail", "message": "Wrong type of metric column"}
                df[date_col] = pd.to_datetime(df[date_col])
                cat = cat.split(",")
                df['key_column'] = ""
                if len(cat) == 1:
                    df['key_column'] = df[cat[0]]
                else:
                    for i in cat:
                        df['key_column'] += "_" + df[i].astype(str)
                df["key_column"] = df["key_column"].str.lstrip("_")
                df = df[['key_column', date_col, metric_col]]
                df = df.dropna()
                field_Mapping = {'Date': date_col,
                                 'Value': metric_col,
                                 'Category': 'key_column'}
                resp = {"RESPONSE": "success",
                        "data": (field_Mapping, df)
                        }
                return resp
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in get_preprocessing_params ==>> {printException()}   local Expection caused by ==>> {e}'}
#-------------------------------------------------------------------------------------------------------------------------------------------------

        def process_data_month(field_Mapping, df, mn_date, mx_date):
            try:
                ##ctx.log.info('process_data_month')
                df[field_Mapping['Date']] = pd.to_datetime(df[field_Mapping['Date']]).dt.floor('D')
                # df[field_Mapping['Date']] = df[field_Mapping['Date']].dt.to_period('W').apply(lambda r: r.start_time)
                df[field_Mapping['Date']] = (
                    (df[field_Mapping['Date']] + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1)).dt.floor('d'))
                max_date = str(mx_date)
                dateprocess_max = dt.datetime.strptime(max_date, "%Y-%m-%d %H:%M:%S")
                min_date = str(mn_date)
                dateprocess_min = dt.datetime.strptime(min_date, "%Y-%m-%d %H:%M:%S")
                max_date_limit = (dateprocess_max - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
                min_date_limit = str(dateprocess_min.year) + '-' + str(dateprocess_min.month) + '-' + str(
                    dateprocess_min.day)

                max_date_limit = dt.datetime.strptime(max_date_limit, '%Y-%m-%d')
                max_date_limit = max_date_limit.replace(day=1)
                max_date_limit = max_date_limit.strftime('%Y-%m-%d')
                min_date_limit = dt.datetime.strptime(min_date_limit, '%Y-%m-%d')
                min_date_limit = min_date_limit.replace(day=1)
                min_date_limit = min_date_limit.strftime('%Y-%m-%d')

                df = df.sort_values(field_Mapping['Date'])
                df[field_Mapping['Date']] = pd.to_datetime(df[field_Mapping['Date']]).dt.floor('D')
                training = df.groupby(field_Mapping['Date'], observed=True)[field_Mapping['Value']].sum().reset_index()
                training.rename(columns={field_Mapping['Date']: 'date'}, inplace=True)

                dates_required = pd.date_range(min_date_limit, max_date_limit, freq='MS',
                                               inclusive='both').strftime(
                    "%Y-%m-%d").tolist()
                dates_required = pd.DataFrame(dates_required, columns=['date'])
                dates_required.date = pd.to_datetime(dates_required.date)
                processed_data = dates_required.merge(training, on='date', how='left')
                processed_data.columns = ['time', field_Mapping['Value']]
                processed_data = processed_data.fillna(0)
                processed_data['time'] = pd.to_datetime(processed_data['time'])

                return {"RESPONSE": "success",
                        "data": processed_data
                        }
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in process_data_month ==>> {printException()}   local Expection caused by ==>> {e}'}
#-------------------------------------------------------------------------------------------------------------------------------------------------

        def process_data(field_Mapping, df, mn_date, mx_date):
            try:
                ##ctx.log.info('in process_data')
                df[field_Mapping['Date']] = pd.to_datetime(df[field_Mapping['Date']]).dt.floor('D')
                df[field_Mapping['Date']] = df[field_Mapping['Date']].dt.to_period('W').apply(
                    lambda r: r.start_time)
                # df[field_Mapping['Date']] = ((df[field_Mapping['Date']] + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1)).dt.floor('d'))
                max_date = str(mx_date)
                dateprocess_max = dt.datetime.strptime(max_date, "%Y-%m-%d %H:%M:%S")
                min_date = str(mn_date)
                dateprocess_min = dt.datetime.strptime(min_date, "%Y-%m-%d %H:%M:%S")
                max_date_limit = (dateprocess_max - pd.DateOffset(weeks=1)).strftime('%Y-%m-%d')
                min_date_limit = str(dateprocess_min.year) + '-' + str(dateprocess_min.month) + '-' + str(
                    dateprocess_min.day)
                df = df.sort_values(field_Mapping['Date'])
                df[field_Mapping['Date']] = pd.to_datetime(df[field_Mapping['Date']]).dt.floor('D')
                training = df.groupby(field_Mapping['Date'], observed=True)[field_Mapping['Value']].sum().reset_index()
                training.rename(columns={field_Mapping['Date']: 'date'}, inplace=True)

                dates_required = pd.date_range(min_date_limit, max_date_limit, freq='W-MON',
                                               inclusive='both').strftime(
                    "%Y-%m-%d").tolist()
                dates_required = pd.DataFrame(dates_required, columns=['date'])
                dates_required.date = pd.to_datetime(dates_required.date)
                processed_data = dates_required.merge(training, on='date', how='left')
                processed_data.columns = ['time', field_Mapping['Value']]
                processed_data = processed_data.fillna(0)
                processed_data['time'] = pd.to_datetime(processed_data['time'])
                return {"RESPONSE": "success",
                        "data": processed_data
                        }
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in process_data ==>> {printException()}   local Expection caused by ==>> {e}'}
#-------------------------------------------------------------------------------------------------------------------------------------------------

        def seasonality_type(data, field_Mapping):  # monthly,weekly
            try:
                ##ctx.log.info('in seasonality_type')
                data.index = data.time
                res = sts.seasonal_decompose(data[field_Mapping['Value']])

                # Get the seasonality component
                seasonality = res.seasonal
                if seasonality.mean() > 0:
                    return {"RESPONSE": "success","data": 1}
                return {"RESPONSE": "success","data": 0}

            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in seasonality_type ==>> {printException()}   local Expection caused by ==>> {e}'}
#-------------------------------------------------------------------------------------------------------------------------------------------------

        def check_stationarity(data, field_Mapping):
            try:
                ##ctx.log.info('in check_stationarity')
                try:
                    data.index = data.time
                    adf = stt.adfuller(data[field_Mapping['Value']])
                    if adf[1] < 0.05:
                        return {"RESPONSE": "success",
                                "data": (1, 'data is stationary. ')
                                }
                    return {"RESPONSE": "success",
                            "data": (0, 'data is not stationary. ')
                            }
                except:
                    return {"RESPONSE": "success",
                            "data": (1, 'data is stationary. ')
                            }
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in check_stationarity ==>> {printException()}   local Expection caused by ==>> {e}'}
#-------------------------------------------------------------------------------------------------------------------------------------------------

        def seasonality_test(series):
            try:
                ##ctx.log.info('in seasonality_test')
                try:
                    seasonal = 0
                    idx = np.arange(len(series.index)) % 12
                    H_statistic, p_value = kruskal(series, idx)
                    if p_value <= 0.05:
                        seasonal = 1
                        return {"RESPONSE": "success",
                                "data": (seasonal, "The series has seasonality. ")
                                }
                    return {"RESPONSE": "success",
                            "data": (seasonal, "No seasonality found in the series. ")
                            }
                except:
                    return {"RESPONSE": "success",
                            "data": (1, 'data is stationary. ')
                            }
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in seasonality_test ==>> {printException()}   local Expection caused by ==>> {e}'}
#-------------------------------------------------------------------------------------------------------------------------------------------------

        def check_seasonality(series, freq):
            try:
                ##ctx.log.info('in check_seasonality')
                result = sdd(series, period=freq)
                is_seasonal = result.seasonal.std() > result.trend.std()
                if is_seasonal:
                    return {"RESPONSE": "success",
                            "data": 'multiplicative'
                            }
                else:
                    return {"RESPONSE": "success",
                            "data": 'additive'
                            }
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in check_seasonality ==>> {printException()}   local Expection caused by ==>> {e}'}
#-------------------------------------------------------------------------------------------------------------------------------------------------

        def check_missing_value(data):
            try:
                ##ctx.log.info('in check_missing_value')
                zero_count = np.count_nonzero(data == 0)
                df = pd.DataFrame(data)
                negative_values = float(df[df < 0.0].count())
                return {"RESPONSE": "success",
                        "data": (zero_count + negative_values, (zero_count + negative_values) / len(
                            data) * 100,
                                 f'The data contains {zero_count + negative_values} (in Percentage = {zero_count + negative_values / len(data) * 100}) inconsistent/missing values. ')
                        }
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in check_missing_value ==>> {printException()}   local Expection caused by ==>> {e}'}
#-------------------------------------------------------------------------------------------------------------------------------------------------

        def check_outlier(data):
            try:
                q = np.quantile(data, [0, 0.25, .5, .75, 1])
                q1 = q[1]
                q3 = q[3]
                iqr = q3 - q1
                upper = q3 + 1.5 * iqr
                count = 0
                for i in data:
                    if i > upper: count += 1
                return {"RESPONSE": "success",
                        "data": (
                            count, f'The data contains {count / len(data) * 100} % outliers. ',
                            count / len(data) * 100)
                        }
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in check_outlier ==>> {printException()}   local Expection caused by ==>> {e}'}
#-------------------------------------------------------------------------------------------------------------------------------------------------

        def check_variance(data):
            try:
                ##ctx.log.info('in check_variance')
                try:
                    cv_before = np.std(data) / np.mean(data) * 100
                    window_size = 12  # 3 months
                    moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode='same')
                    cv_after = np.std(moving_avg) / np.mean(moving_avg) * 100
                    smoothing_percentage = (cv_before - cv_after) / cv_before * 100
                    if smoothing_percentage < 20:
                        return {"RESPONSE": "success",
                                "data": (smoothing_percentage, "The data has less variation. ")
                                }
                    return {"RESPONSE": "success",
                            "data": (smoothing_percentage, "The data has high variation. ")
                            }
                except:
                    smoothing_percentage = 0
                    return {"RESPONSE": "success",
                            "data": (smoothing_percentage, "The data has high variation. ")
                            }

            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f'Error in check_variance ==>> {printException()}   local Expection caused by ==>> {e}'}
#-------------------------------------------------------------------------------------------------------------------------------------------------

        def percent_error(pred, ground):
            try:
                try:
                    return float((abs(ground - pred)) / ground) * 100
                except:
                    return -1
            except Exception as e:
                return {"status": "Failed", "message": f'Error in percent_error ==>> {printException()}'}
#-------------------------------------------------------------------------------------------------------------------------------------------------
            
        def calcsmape(actual, forecast):
            try:
                try:
                    return min(1,1/len(actual) * np.sum(2 * np.abs(forecast-actual) / (np.abs(actual) + np.abs(forecast))))
                except:
                    return 1
            except Exception as e:
                return {"status": "Failed", "message": f'Error in calcsmape ==>> {printException()}'}
#-------------------------------------------------------------------------------------------------------------------------------------------------

        def error_past2_month(field_Mapping, product, data_with_date, month_seas, week_seas, sv, s_type):
            """
            Calculates the error for the past 4 months for a given product.
            """
            try:               
                df3 = data_with_date.copy() 
                ##ctx.log.info('in error_past2_month')
                forecast_frame = pd.DataFrame()
                # forecast_frame['Category'] = product
                if month_seas == 1:
                    ms = True
                else:
                    ms = False

                if week_seas == 1:
                    ws = True
                else:
                    ws = False

                train_data = data_with_date[:-16]
                test_data = data_with_date[-16:]

                forecast_frame["Actual's"] = test_data[field_Mapping['Value']]

                rng = (0, 10)
                scaler = preprocessing.MinMaxScaler(feature_range=(rng[0], rng[1]))
                normed = scaler.fit(np.array(data_with_date[field_Mapping['Value']]).reshape(-1, 1))
                normed = scaler.transform(np.array(train_data[field_Mapping['Value']]).reshape(-1, 1))
                norm_lst = [round(i[0], 4) for i in normed]

                train_data[field_Mapping['Value']] = norm_lst
                train_data[field_Mapping['Value']] = np.where(train_data[field_Mapping['Value']] < 0, 0,
                                                              train_data[field_Mapping['Value']])

#------------------------------------------------Model 1-----------------------------------------------------------------------------------    
                try:
                    params = ProphetParams(
                        daily_seasonality = True,
                        weekly_seasonality=ws,
                        yearly_seasonality=True,
                        seasonality_mode=s_type)
                    train_data.columns = ['time', 'y']
                    train_p = TimeSeriesData(train_data)
    
                    m = ProphetModel(train_p, params)
                    with suppress_logs():
                        m.fit()
    
                    fcst = m.predict(steps=16, freq="W-MON")
                    model1_result = fcst['fcst']
                except:
                    from prophet import Prophet
                    train_data.columns = ['ds', 'y']
                    params = {
                        'daily_seasonality' : True,
                        'weekly_seasonality' : ws,
                        'yearly_seasonality' : True,
                        'seasonality_mode' : s_type}
                    m = Prophet(**params)
                    with suppress_logs():
                        m.fit(train_data)
                    future = m.make_future_dataframe(periods=16, freq="W-MON", include_history=False)
                    fcst = m.predict(future)
                    fcst = fcst[['ds', 'yhat']]
                    fcst.columns = ['time','fcst']
                    model1_result = fcst['fcst']
                    train_data.columns = ['time', 'y']
                    train_p = TimeSeriesData(train_data)
                
#---------------------------------------------------Model 4------------------------------------------------------------------------------------
                try:
                    params = HoltWintersParams(
                        damped=True,
                        trend=s_type.lower()[:3],
                        seasonal=s_type.lower()[:3],
                        seasonal_periods=52,
                    )

                    m = HoltWintersModel(
                        data=train_p,
                        params=params, )

                    with suppress_logs():
                        m.fit()
                except:
                    params = HoltWintersParams(
                        damped=True,
                        trend='add',
                        seasonal='add',
                        seasonal_periods=52,
                    )

                    m = HoltWintersModel(
                        data=train_p,
                        params=params, )

                    with suppress_logs():
                        m.fit()

                res = m.predict(steps=16, include_history=False)

                model4_result = res['fcst']

#-------------------------------------------------Model 2------------------------------------------------------------------------------
                train_data.index = train_data['time']
                train_data.drop(['time'], axis=1, inplace=True)
                if ms == True or ws == True:
                    seas = True
                else:
                    seas = False

                if sv == 1:
                    s_v = True
                else:
                    s_v = False
                
                df3['unique_id'] = 1
                df3.columns = ['ds','y','unique_id']
                
                model = StatsForecast(
                                models = [AutoARIMA(season_length = 52, trace = False, start_p=0, d=1, start_q=0,
                                                   max_p=4, max_d=4, max_q=4,
                                                   start_P=0, D=1, start_Q=0, max_P=4, max_D=4, max_Q=4 ,seasonal=seas,
                                                   stationary=s_v, stepwise=True),
                                         ],
                    freq = "W-MON"
                )
                
                # model = auto_arima(train_data, frequency='W-MON', start_p=0, d=1, start_q=0, max_p=5, max_d=5,
                #                    max_q=5,
                #                    error_action='ignore',
                #                    start_P=0, D=1, start_Q=0, max_P=5, max_D=5, max_Q=5, m=52, seasonal=seas,
                #                    stationary=s_v,
                #                    trace=True, supress_warning=True, stepwise=True, disp=False)
                model.fit(df3)
                p = model.predict(h = 16)
                p = p.reset_index().drop(["unique_id"], axis = 1)                
                pred = p
                pred = pred.set_index("ds")
                #pred = pred.drop(['ds'], axis = 1)
                pred.columns = ['Pred']
                #pred = pred.reset_index()
                #pred.drop(['index'], axis=1, inplace=True)
                model2_result = pred['Pred'][:16]

#-------------------------------------------------Model 6------------------------------------------------------------------------------
                model_data = pd.Series(train_data['y'], index=train_data.index)
        
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
                
                p = fit.forecast(16)
                pred6= pd.DataFrame(p).reset_index()
                pred6.columns = ['ds','Pred6']
                pred6 = pred6.set_index("ds")
                model6_result = pred6['Pred6'][:16]

#-------------------------------------------------Model 3------------------------------------------------------------------------------
                pred['model3_res'] = (np.array(fcst['fcst']) + np.array(pred['Pred']) ) /2
                #pred['model3_res'] = (pred['m2'] + fcst['fcst']) / 2
                model3_result = pred['model3_res'][:16]
                pred['m4'] = np.array(model4_result)

#-------------------------------------------------Model 5------------------------------------------------------------------------------
                pred['model5_res'] = (pred['Pred'] + pred['m4']) / 2
                model5_result = pred['model5_res'][:16]
                                
                orim1 = scaler.inverse_transform(np.array(model1_result).reshape(-1, 1))
                orim2 = scaler.inverse_transform(np.array(model2_result).reshape(-1, 1))
                orim3 = scaler.inverse_transform(np.array(model3_result).reshape(-1, 1))
                orim4 = scaler.inverse_transform(np.array(model4_result).reshape(-1, 1))
                orim5 = scaler.inverse_transform(np.array(model5_result).reshape(-1, 1))
                orim6 = scaler.inverse_transform(np.array(model6_result).reshape(-1, 1))

                del scaler
                forecast_frame['Model 1 Predictions'] = orim1
                forecast_frame['Model 2 Predictions'] = orim2
                forecast_frame['Model 3 Predictions'] = orim3
                forecast_frame['Model 4 Predictions'] = orim4
                forecast_frame['Model 5 Predictions'] = orim5
                forecast_frame['Model 6 Predictions'] = orim6

                m1_res = [round(i[0], 4) for i in orim1]
                m2_res = [round(i[0], 4) for i in orim2]
                m3_res = [round(i[0], 4) for i in orim3]
                m4_res = [round(i[0], 4) for i in orim4]
                m5_res = [round(i[0], 4) for i in orim5]
                m6_res = [round(i[0], 4) for i in orim6]

#-----------------------------------..ERROR CAL..---------------------------------------------------------------------------------------
                model1_mse = mean_squared_error(test_data[field_Mapping['Value']], m1_res)
                model2_mse = mean_squared_error(test_data[field_Mapping['Value']], m2_res)
                model3_mse = mean_squared_error(test_data[field_Mapping['Value']], m3_res)
                model4_mse = mean_squared_error(test_data[field_Mapping['Value']], m4_res)
                model5_mse = mean_squared_error(test_data[field_Mapping['Value']], m5_res)
                model6_mse = mean_squared_error(test_data[field_Mapping['Value']], m6_res)

                model1_mae = mean_absolute_error(test_data[field_Mapping['Value']], m1_res)
                model2_mae = mean_absolute_error(test_data[field_Mapping['Value']], m2_res)
                model3_mae = mean_absolute_error(test_data[field_Mapping['Value']], m3_res)
                model4_mae = mean_absolute_error(test_data[field_Mapping['Value']], m4_res)
                model5_mae = mean_absolute_error(test_data[field_Mapping['Value']], m5_res)
                model6_mae = mean_absolute_error(test_data[field_Mapping['Value']], m6_res)

                res_frame = pd.DataFrame()
                res_frame['Ground'], res_frame['m1_pred'], res_frame['m2_pred'], res_frame['m3_pred'], res_frame[
                    'm4_pred'], res_frame['m5_pred'], res_frame['m6_pred'] = test_data[
                                                           field_Mapping['Value']], m1_res, m2_res, m3_res, m4_res, m5_res, m6_res
                # percent_error_m1 = percent_error(float(res_frame['m1_pred'].sum()),
                #                                  float(res_frame['Ground'].sum()))
                # percent_error_m2 = percent_error(float(res_frame['m2_pred'].sum()),
                #                                  float(res_frame['Ground'].sum()))
                # percent_error_m3 = percent_error(float(res_frame['m3_pred'].sum()),
                #                                  float(res_frame['Ground'].sum()))
                # percent_error_m4 = percent_error(float(res_frame['m4_pred'].sum()),
                #                                  float(res_frame['Ground'].sum()))
                # percent_error_m5 = percent_error(float(res_frame['m5_pred'].sum()),
                #                                  float(res_frame['Ground'].sum()))
                # percent_error_m6 = percent_error(float(res_frame['m6_pred'].sum()),
                #                                  float(res_frame['Ground'].sum()))
                percent_error_m1 = mean_absolute_percentage_error(res_frame["Ground"], res_frame["m1_pred"])
                percent_error_m2 = mean_absolute_percentage_error(res_frame["Ground"], res_frame["m2_pred"])
                percent_error_m3 = mean_absolute_percentage_error(res_frame["Ground"], res_frame["m3_pred"])
                percent_error_m4 = mean_absolute_percentage_error(res_frame["Ground"], res_frame["m4_pred"])
                percent_error_m5 = mean_absolute_percentage_error(res_frame["Ground"], res_frame["m5_pred"])
                percent_error_m6 = mean_absolute_percentage_error(res_frame["Ground"], res_frame["m6_pred"])

                model1_r2 = r2_score(res_frame["Ground"], res_frame["m1_pred"])
                model2_r2 = r2_score(res_frame["Ground"], res_frame["m2_pred"])
                model3_r2 = r2_score(res_frame["Ground"], res_frame["m3_pred"])
                model4_r2 = r2_score(res_frame["Ground"], res_frame["m4_pred"])
                model5_r2 = r2_score(res_frame["Ground"], res_frame["m5_pred"])
                model6_r2 = r2_score(res_frame["Ground"], res_frame["m6_pred"])

                model1_evs = explained_variance_score(res_frame["Ground"], res_frame["m1_pred"])
                model2_evs = explained_variance_score(res_frame["Ground"], res_frame["m2_pred"])
                model3_evs = explained_variance_score(res_frame["Ground"], res_frame["m3_pred"])
                model4_evs = explained_variance_score(res_frame["Ground"], res_frame["m4_pred"])
                model5_evs = explained_variance_score(res_frame["Ground"], res_frame["m5_pred"])
                model6_evs = explained_variance_score(res_frame["Ground"], res_frame["m6_pred"])

                model1_smape = calcsmape(res_frame["Ground"], res_frame["m1_pred"])
                model2_smape = calcsmape(res_frame["Ground"], res_frame["m2_pred"])
                model3_smape = calcsmape(res_frame["Ground"], res_frame["m3_pred"])
                model4_smape = calcsmape(res_frame["Ground"], res_frame["m4_pred"])
                model5_smape = calcsmape(res_frame["Ground"], res_frame["m5_pred"])
                model6_smape = calcsmape(res_frame["Ground"], res_frame["m6_pred"])


                # res_dict = {
                #     percent_error_m1: 'm1',
                #     percent_error_m2: 'm2',
                #     percent_error_m3: 'm3',
                #     percent_error_m4: 'm4',
                #     percent_error_m5: 'm5',
                #     percent_error_m6: 'm6'
                # }
                # min_err = min(res_dict.keys())
                # best_m = res_dict[min_err]
                def sorted_list(list1,list2):
                    sorted_pairs = sorted(zip(list1, list2))
                    l1_sorted, l2_sorted = zip(*sorted_pairs)
                    return l1_sorted, l2_sorted
                
                def max_occurrences(list1):
                    element_counts = {}
                    for element in list1:
                        if element in element_counts:
                            element_counts[element] += 1
                        else:
                            element_counts[element] = 1
                    #most_common_element = max(element_counts, key=element_counts.get)
                    return max(element_counts, key=element_counts.get)
                
                mape_list = [percent_error_m1, percent_error_m2, percent_error_m3, percent_error_m4, percent_error_m5, percent_error_m6]
                mae_list = [model1_mae, model2_mae, model3_mae, model4_mae, model5_mae, model6_mae]
                mse_list = [model1_mse, model2_mse, model3_mse, model4_mse, model5_mse, model6_mse]
                r2_list = [model1_r2, model2_r2, model3_r2, model4_r2, model5_r2, model6_r2]
                evs_list = [model1_evs, model2_evs, model3_evs, model4_evs, model5_evs, model6_evs]
                smape_list = [model1_smape, model2_smape, model3_smape, model4_smape, model5_smape, model6_smape]
                m_list = ["m1", "m2", "m3", "m4", "m5", "m6"]
                
                _ , mape_list_sorted = sorted_list(mape_list,m_list)
                mape_best_m = mape_list_sorted[0]
                _ , mae_list_sorted = sorted_list(mae_list,m_list)
                mae_best_m = mae_list_sorted[0]
                _ , mse_list_sorted = sorted_list(mse_list,m_list)
                mse_best_m = mse_list_sorted[0]
                _ , smape_list_sorted = sorted_list(smape_list,m_list)
                smape_best_m = smape_list_sorted[0]
                _ , r2_list_sorted = sorted_list(r2_list,m_list)
                r2_best_m = r2_list_sorted[-1]
                _ , evs_list_sorted = sorted_list(evs_list,m_list)
                evs_best_m = evs_list_sorted[-1]

                #### Find Best Model ---------------------
                #best_m = mape_list_sorted[0]
                best_m_list = [mape_best_m, mae_best_m, mse_best_m, smape_best_m, r2_best_m, evs_best_m]
                best_m = max_occurrences(best_m_list)
                if best_m == 'm7':
                    best_m = 'm1'
                

                return {"RESPONSE": "success",
                        "data": (
                            percent_error_m1, percent_error_m2, percent_error_m3, percent_error_m4,percent_error_m5,percent_error_m6, mape_best_m,
                            model1_mse,model2_mse,model3_mse,model4_mse,model5_mse,model6_mse, mse_best_m,
                            model1_mae,model2_mae,model3_mae,model4_mae,model5_mae,model6_mae, mae_best_m,
                            model1_smape, model2_smape, model3_smape, model4_smape, model5_smape, model6_smape, smape_best_m,
                            model1_r2, model2_r2, model3_r2, model4_r2, model5_r2, model6_r2, r2_best_m,
                            model1_evs, model2_evs, model3_evs, model4_evs, model5_evs,model6_evs, evs_best_m,
                            best_m, forecast_frame)
                        }
            except Exception as e:
                return {"RESPONSE": "fail",
                        "message": f"Error in error_past2_month ==>> {printException()}   local Expection caused by ==>> {e}"}
#-------------------------------------------------------------------------------------------------------------------------------------------------

        def make_parameter_df(field_Mapping: dict, df):
            """
            Creates a DataFrame of Parameters for Forecasting.

            Args:
                field_Mapping (dict): A dictionary that maps field names to their corresponding values/column.
                df (pd.DataFrame): A DataFrame that contains the raw data.

            Returns:
                pd.DataFrame: A DataFrame of parameters.
            """
            try:
                ctx.log.info("In make_parameter_df")
                min_date = df[field_Mapping["Date"]].min()
                max_date = df[field_Mapping["Date"]].max()
                df2 = pd.DataFrame()
                cat = df[field_Mapping["Category"]].unique()
                ctx.log.info(
                    "Getting All unique Materials to do Hyper Parameter tuning and Validation")  
                # df2.column = ["Category","Stationarity","Seasonality","Inconstitent_Value_Percentage","Percentage of Outliders","Smoothness_Percentage","Explain"]
                res = []
                ctx.log.info("Starting Validation")
                final_frame = pd.DataFrame()
                print(len(cat))
                print("----" * 25)
                t = tqdm.tqdm(cat)
                for i in t:
                    t.set_description(f"ITERATING OVER {i}")
                    seas_exp = ""
                    #data_month = process_data_month(field_Mapping, df[df[field_Mapping['Category']] == i], min_date,max_date)
                    resp = process_data_month(field_Mapping, df[df[field_Mapping["Category"]] == i], min_date,max_date)
                    if resp["RESPONSE"] == "fail":
                        return resp
                    data_month = resp["data"]

                    resp = process_data(field_Mapping, df[df[field_Mapping["Category"]] == i], min_date, max_date)
                    if resp["RESPONSE"] == "fail":
                        return resp
                    data = resp["data"]

                    resp = seasonality_type(data_month, field_Mapping)
                    if resp["RESPONSE"] == "fail":
                        return resp
                    month_seas = resp["data"]

                    resp = seasonality_type(data, field_Mapping)
                    if resp["RESPONSE"] == "fail":
                        return resp
                    week_seas = resp["data"]

                    if month_seas == 1 and week_seas == 1:
                        seas_exp += "Data shows presence of monthly and weekly seasonality "
                    elif month_seas == 1:
                        seas_exp += "Data shows presence of monthly seasonality "
                    elif week_seas == 1:
                        seas_exp += "Data shows presence of weekly seasonality "

                    values = data[field_Mapping["Value"]]

                    resp2 = check_stationarity(data, field_Mapping)
                    if resp2["RESPONSE"] == "fail":
                        return resp2
                    sv, statioary_explain = resp2["data"]

                    resp = check_missing_value(values)
                    if resp["RESPONSE"] == "fail":
                        return resp
                    inconsistent_count, perecentage, inc_explain = resp["data"]

                    resp = check_outlier(values)
                    if resp["RESPONSE"] == "fail":
                        return resp
                    n_outliders, out_explain, out_percentage = resp["data"]

                    resp = check_variance(values)
                    if resp["RESPONSE"] == "fail":
                        return resp
                    smooth_percentage, smooth_explain = resp["data"]

                    resp = check_seasonality(values, freq=52, )
                    if resp["RESPONSE"] == "fail":
                        return resp
                    is_seasonal_additive = resp['data']

                    resp = error_past2_month(field_Mapping, i, data, month_seas,
                                             week_seas, sv, is_seasonal_additive)
                    if resp["RESPONSE"] == "fail":
                        return resp

                    m1_mape, m2_mape, m3_mape, m4_mape, m5_mape, m6_mape, mape_best_m,\
                        m1_mse,m2_mse,m3_mse,m4_mse,m5_mse,m6_mse, mse_best_m,\
                            m1_mae,m2_mae,m3_mae,m4_mae,m5_mae,m6_mae, mae_best_m,\
                                m1_smape,m2_smape,m3_smape,m4_smape,m5_smape,m6_smape, smape_best_m,\
                                    m1_r2,m2_r2,m3_r2,m4_r2,m5_r2,m6_r2, r2_best_m,\
                                        m1_evs,m2_evs,m3_evs,m4_evs,m5_evs,m6_evs, evs_best_m,\
                                            best_m, frame = resp["data"]

                    frame[field_Mapping["Category"]] = i
                    final_frame = pd.concat([frame, final_frame], ignore_index=True)
                    exp = f'{field_Mapping["Category"]} {i} ' + statioary_explain + seas_exp + inc_explain + out_explain + seas_exp + smooth_explain + f" The abs error percentage over past two months is {m1_mape}% ,{m2_mape}% ,{m3_mape}% {m4_mape}% ,{m5_mape}% ,{m6_mape}%for model1, model2, model3, model4, model5 and model6 respectively."     

                    lst = [[i, sv, month_seas, week_seas, perecentage, out_percentage, smooth_percentage,
                            is_seasonal_additive, exp, m1_mape, m2_mape, m3_mape, m4_mape, m5_mape, m6_mape, mape_best_m,
                            m1_mse, m2_mse, m3_mse, m4_mse, m5_mse, m6_mse, mse_best_m, m1_mae, m2_mae, m3_mae, m4_mae, m5_mae, m6_mae, mae_best_m,
                            m1_smape,m2_smape,m3_smape,m4_smape,m5_smape,m6_smape, smape_best_m,
                            m1_r2, m2_r2, m3_r2, m4_r2, m5_r2, m6_r2, r2_best_m, m1_evs, m2_evs, m3_evs, m4_evs, m5_evs, m6_evs, evs_best_m,
                            best_m]]
                    # df2 = df2.append(pd.DataFrame(lst, columns=[field_Mapping['Category'], 'Stationarity','Monthly_Seasonality', 'Weekly_Seasonality','Inconstitent_Value_Percentage',
                    #                                             'Percenage of Outliders', 'Smoothness_Percentage','Seasonality_Type', 'Explain',
                    #                                             'Percent ABS ERROR Model 1','Percent ABS ERROR Model 2','Percent ABS ERROR Model 3','Percent ABS ERROR Model 4','Percent ABS ERROR Model 5','Percent ABS ERROR Model 6','MPAE_Best_Model',
                    #                                             'MSE Model 1', 'MSE Model 2','MSE Model 3','MSE Model 4','MSE Model 5','MSE Model 6', 'MSE_Best_Model',
                    #                                             'MAE Model 1', 'MAE Model 2','MAE Model 3','MAE Model 4','MAE Model 5','MAE Model 6', 'MAE_Best_Model',
                    #                                             'SMAPE Model 1', 'SMAPE Model 2','SMAPE Model 3','SMAPE Model 4','SMAPE Model 5','SMAPE Model 6', 'SMAPE_Best_Model',
                    #                                             'R2 Model 1', 'R2 Model 2','R2 Model 3','R2 Model 4','R2 Model 5','R2 Model 6', 'R2_Best_Model',
                    #                                             'EVS Model 1', 'EVS Model 2','EVS Model 3','EVS Model 4','EVS Model 5','EVS Model 6', 'EVS_Best_Model',
                    #                                             'Best_Model']),ignore_index=True)
                    df2 = df2.append(pd.DataFrame(lst, columns=[field_Mapping['Category'], 'Stationarity','Monthly_Seasonality', 'Weekly_Seasonality','Inconstitent_Value_Percentage',
                                                                'Percenage of Outliders', 'Smoothness_Percentage','Seasonality_Type', 'Explain',
                                                                'MAPE Model 1','MAPE Model 2','MAPE Model 3','MAPE Model 4','MAPE Model 5','MAPE Model 6','MPAE_Best_Model',
                                                                'MSE Model 1', 'MSE Model 2','MSE Model 3','MSE Model 4','MSE Model 5','MSE Model 6', 'MSE_Best_Model',
                                                                'MAE Model 1', 'MAE Model 2','MAE Model 3','MAE Model 4','MAE Model 5','MAE Model 6', 'MAE_Best_Model',
                                                                'SMAPE Model 1', 'SMAPE Model 2','SMAPE Model 3','SMAPE Model 4','SMAPE Model 5','SMAPE Model 6', 'SMAPE_Best_Model',
                                                                'R2 Model 1', 'R2 Model 2','R2 Model 3','R2 Model 4','R2 Model 5','R2 Model 6', 'R2_Best_Model',
                                                                'EVS Model 1', 'EVS Model 2','EVS Model 3','EVS Model 4','EVS Model 5','EVS Model 6', 'EVS_Best_Model',
                                                                'Best_Model']),ignore_index=True)
                    #df2.to_csv("param.csv", index=False)

                P = dt.datetime.today().replace(day=1).strftime("%Y-%m")
                df2["Period"] = P
                final_frame["Period"] = P
                return {"RESPONSE": "success","data": (df2, final_frame)}

            except Exception as e:
                return {"RESPONSE": "fail","message": f"Error in check_missing_value ==>> {printException()}   local Expection caused by ==>> {e}"}
#-------------------------------------------------------------------------------------------------------------------------------------------------

        ds = Dataset(ds_id)
        ctx.log.info("|------------------------------------------------------------|")
        ctx.log.info("|------------------Dataset Loaded ---------------------------|")
        ctx.log.info("|------------------------------------------------------------|")
        df = ds.sqlDf(query)
        #df_21 = pd.read_csv('hos_2021.csv')
        #df = pd.concat([df_21,df],ignore_index=True)
        df.dropna(subset=[date,cat], inplace=True)
        df = df.drop_duplicates(keep='last')

        resp = get_preprocessing_params(df, cat, date, metric_col)
        if resp["RESPONSE"] == "fail":
            return resp
        fld_map, df = resp["data"]

        resp = make_parameter_df(fld_map, df)
        if resp["RESPONSE"] == "fail":
            return resp

        hyper_parameters_data, validation_forecast = resp["data"]

        ctx.log.info("Finished Validation pushing Data into Smart Analytics")
        ctx.log.info("")
        ds = Dataset(ds_id)
        ctx.log.info("|------------------------------------------------------------|")
        ctx.log.info("|----------------- Refreshing Dataset -----------------------|")
        ctx.log.info("|------------------------------------------------------------|")
        ctx.log.info("")
        hyper_parameters_data.to_csv("paramTable.csv", index=False)
        response1 = ds.smartAnalytics.create("Parameter_Table", hyper_parameters_data, False, flag, True)
        response2 = ds.smartAnalytics.create("Validation_Table", validation_forecast, False, flag, True)

        if ("status" in response1 and response1["status"] == "success") & (
                "status" in response2 and response2["status"] == "success"):
            ctx.log.info("smart analytics creation is done -> {}".format(dt.datetime.utcnow()))
            ctx.log.info("")
            ctx.log.info("|---------------------------------------------------------------------------------|")
            ctx.log.info("| Creation of table was done properly and smart analytics is created successfully |")
            ctx.log.info("|---------------------------------------------------------------------------------|")

        else:
            ctx.log.error("smart analytics creation is failed -> {}".format(dt.datetime.utcnow()))
            return {"status": "failed", "message": "Organization processing failed due to an error {}".format(
                response1["message"] if "message" in response1 else "internalServerError")}
        
        return {"status": "success", "message": "Hyper Parameter tuning Completed & Validation Table Generation done"}
    
    except Exception as e:
        return {"status": "Failed", "message": f"Error in hyper_parameter_creation == >> {printException()} Local ERROR == >> {e}"}

