# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


############Please fill in the blank functions to run the whole program#######
def get_stock_data(path, columns=None):
    """
    :param path: path to the .csv file containing stock information
    :param columns: the columns from the original dataframe to be returned, if None return all data
    :return:
    """
    print("Retrieving Data...")
    df = pd.read_csv(path)
    df['Date']=pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    if columns is None:
        return df
    else:
        return df[columns]


def get_daily_return(df_adj_close):
    """
    :param df_adj_close: pandas series data that contains the adjusted close price of an asset
    :return: the mean and standard deviation as implied in the df_adj_close
    """

    ####### Fill this function to return the daily return mean and std########
    print("Calculating mean and standard deviation...")
    daily_return=df_adj_close.pct_change()
    daily_return=daily_return.dropna()
    mean=daily_return.mean()
    
    #calculate the standard deviation
     
    std=np.std(daily_return)       
    




    ##########################################################################
    return mean, std


def simulate_stock_price(df_adj_close, num_simulations=50):
    """
    :param df_adj_close: pandas series object that contains the adjusted close price
    :param num_simulations: number of simulations to be generated
    :return: df_simulation: the simulated close price dataframe
             df_mean: the dataframe containing the price series should the stock has no volatility and daily return
                      equals the expected daily return implied in df_adj_close
    """
    mean, std = get_daily_return(df_adj_close)
    length = len(df_adj_close)
    ##### Complete this function for price simulation#################
    print("Simulating...")
    
    df_daily=df_adj_close.pct_change()
    df_daily=df_daily.dropna()
    length=len(df_daily)
    
    #get df_simulation
    s0=df_adj_close.iloc[0]
    r=np.random.randn(length,num_simulations)*std+mean
    cumulative_product=np.cumprod(1+r,axis=0)
    absolute_stock_return=s0*cumulative_product
    absolute_stock_return=np.insert(absolute_stock_return,obj=0,values=s0,axis=0)
    df_simulation=pd.DataFrame(data=absolute_stock_return,index=df_adj_close.index)
    
    #get df_mean
    r_mean=np.zeros([length])+mean
    absulote_stock_return_mean=s0*np.cumprod(1+r_mean,axis=0)
    absulote_stock_return_mean=np.insert(absulote_stock_return_mean,obj=0,values=s0,axis=0)
    df_mean=pd.DataFrame(data=absulote_stock_return_mean,index=df_adj_close.index)
        
    ###################################################################

    return df_simulation, df_mean


def plot_simulation_with_mean(df_simulation, df_mean, title="Simulation"):
    """
    :param df_simulation: dataframe containing the simulated close price for an asset
    :param df_mean: dataframe containing the mean-return price series for an asset
    :param title: title of the plot, default "Simulation"
    """
    assert (df_simulation.index == df_mean.index).all()
    ##### Complete this function for plot both the simulated price and mean price#################
    print("Plotting simulation...")
    plt.plot(df_simulation)
    plt.plot(df_mean,color='red',linewidth=5,label='Mean Return')
    plt.legend()
    plt.title(title)
    
    ##############################################################################################
    plt.show()


def plot_volume_data(df_volume, title="Volume"):
    """
    :param df_volume: pandas series that contains volume data, indexed by dates
    :param title: title of the plot, default "Volume"
    :return:
    """
    ##### Complete this function for plot both the simulated price and mean price#################
    print("Plotting volume data...")
    x=df_volume.index
    #Plot the Fill in Picture
    plt.fill_between(x, df_volume)
    plt.title('Volume')
    



    ##############################################################################################
    plt.show()


if __name__ == "__main__":
    np.random.seed(123)
    df_adj_close = get_stock_data('data/S&P500.csv', columns='Adj Close')
    df_volume = get_stock_data('data/S&P500.csv', columns='Volume')
    print("Data Retrived")
    df_simulation, df_mean = simulate_stock_price(df_adj_close, num_simulations=100)
    plot_simulation_with_mean(df_simulation, df_mean)
    plot_volume_data(df_volume)

