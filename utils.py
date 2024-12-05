from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite import plotting as tp
import matplotlib.pyplot as plt


def check_stationarity_for_dataframe(df):
    """
    Checks the stationarity of all columns in a given DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame whose columns are to be tested for stationarity.
    Returns:
        dict: A dictionary with column names as keys and a boolean indicating stationarity as values.
    """
    stationarity_results = {}
    
    for column in df.columns:
        series = df[column]
        if series.dtype.kind in 'biufc':  # Only check for numeric columns
            result = adfuller(series.dropna())
            print(f'=== Stationarity Test for {column} ===')
            print(f'ADF Statistic: {result[0]}')
            print(f'p-value: {result[1]}')
            print('Critical values:')
            for key, value in result[4].items():
                print(f'\t{key}: {value}')
            is_stationary = result[1] < 0.05
            print(f'Is stationary: {is_stationary}\n')
            
            # Store result
            stationarity_results[column] = is_stationary
        else:
            print(f"Skipping non-numeric column: {column}\n")
    
    return stationarity_results


def analyze_multiple_timeseries_dependencies(df1, df2, cols1, cols2, 
                                             names1=None, names2=None,
                                             tau_max=20):
    """
    Analyze dependencies between multiple time series columns in two DataFrames using PCMCI
    and create lag dependency plots for all pairs of columns.

    Parameters:
    -----------
    df1 : pandas.DataFrame
        First DataFrame containing the time series (with date index)
    df2 : pandas.DataFrame
        Second DataFrame containing the time series (with date index)
    cols1 : list
        List of column names from df1 to analyze
    cols2 : list
        List of column names from df2 to analyze
    names1 : list, optional
        Display names for the first series columns (default: same as cols1)
    names2 : list, optional
        Display names for the second series columns (default: same as cols2)
    tau_max : int, optional
        Maximum time lag to test (default: 20)

    Returns:
    --------
    dict
        A dictionary with column pair names as keys and tuples of 
        (correlations, figure) as values.
    """
    if names1 is None:
        names1 = cols1
    if names2 is None:
        names2 = cols2

    results = {}

    for col1, name1 in zip(cols1, names1):
        for col2, name2 in zip(cols2, names2):
            # Extract and align the time series
            common_dates = df1.index.intersection(df2.index)
            if len(common_dates) == 0:
                raise ValueError(f"No common dates found between '{col1}' and '{col2}'")

            data1 = df1.loc[common_dates, col1].values.reshape(-1, 1)
            data2 = df2.loc[common_dates, col2].values.reshape(-1, 1)

            # Combine the data
            combined_data = np.hstack([data1, data2])

            # Create tigramite dataframe
            var_names = [name1, name2]
            dataframe = pp.DataFrame(combined_data, 
                                     datatime=np.arange(len(common_dates)),
                                     var_names=var_names)

            # Initialize PCMCI with ParCorr independence test
            parcorr = ParCorr(significance='analytic')
            pcmci = PCMCI(dataframe=dataframe, 
                          cond_ind_test=parcorr,
                          verbosity=1)

            # Run bivariate analysis
            correlations = pcmci.run_bivci(tau_max=tau_max, 
                                           tau_min=0,
                                           val_only=True)['val_matrix']

            # Setup plot parameters
            setup_args = {'var_names': var_names,
                          'figsize': (10, 6),
                          'x_base': 5,
                          'y_base': .5}

            # Create plot
            fig = plt.figure(figsize=(10, 6))
            tp.plot_lagfuncs(val_matrix=correlations, setup_args=setup_args)

            plt.title(f'Time Series Dependencies: {name1} vs {name2}')
            plt.tight_layout()

            # Save results
            pair_name = f"{name1} vs {name2}"
            results[pair_name] = (correlations, fig)

    return results
