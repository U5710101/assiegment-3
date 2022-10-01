#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install yfinance')


# In[2]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


def portfolio_creation_and_optimization(stock_list,num_of_portfolios,risk_free):
    
    #Loading the data
    data = yf.download(stock_list, start= '2012-01-01', end = '2022-01-01', interval = '1mo')
    data = data['Adj Close']
    
    #taking log returns of the monthly data.
    returns = np.log(data).diff()
    
    #These are lists to store the respective metrics of all portfolios
    portfolio_return = []
    portfolio_risk = []
    portfolio_sharpe = []
    portfolio_weights = []
    
    # I have considered negative weights by assigning the minimum value od 0.1 or 10%
    low = -0.1
    k = len(stock_list) #Number of tickers
    
    #Covariance Matrix for the annual returns of each stock 
    covariance_matrix = returns.cov() * 12  # Multiplied by 12 to make annual returns
    for portfolio_len in range(num_of_portfolios):
        a = np.random.rand(k)
        a = (a/a.sum()*(1-low*k)) #division factor to generate random numbers that have both postive and neagtive values sum to 1
        weights = a + low 

        #checking if the sum is 1
        assert np.isclose(weights.sum(), 1)

        portfolio_weights.append(weights)

      #calculating mean returns (multiplying by 12 because we have monthly returns data to make it annualized i am multiplying by 12)
        annualized_returns = np.sum(returns.mean() * weights) * 12
        portfolio_return.append(annualized_returns)

      #matrix covariance and portfolio_variance
        portfolio_variance = np.dot(weights.T,np.dot(covariance_matrix,weights))

      #To calculate sharpe we need portfolio standard deviation 
        portfolio_stdev = np.sqrt(portfolio_variance)
        portfolio_risk.append(portfolio_stdev)

      #sharpe ratio for returns and standard deviation considering risk free rate as 0%
        sharpe_ratio = (annualized_returns - risk_free)/portfolio_stdev
        portfolio_sharpe.append(sharpe_ratio)
    
    #Converting every parameter into a dataframe
    data = pd.DataFrame({'Return': portfolio_return,'Risk': portfolio_risk, 'Sharpe': portfolio_sharpe, 'Weights': portfolio_weights})
    
    #These lines will give the min and max rows which has min risk, max return and max sharpe.
    minimum_risk = data.iloc[data['Risk'].idxmin()]
    maximum_return = data.iloc[data['Return'].idxmax()]
    maximum_sharpe = data.iloc[data['Sharpe'].idxmax()]
    
    #These lines will print the weights and the parameters in a good way above the portfolio visualizations.
    print('Lowest Risk')
    a = []
    for i,j in zip(stock_list,minimum_risk.Weights):
        element = str(i)+' : '+str(j)
        a.append(element)
    print(a)
    print(minimum_risk)
    print('')

    print('Maximum Return')
    b = []
    for i,j in zip(stock_list,maximum_return.Weights):
        element = str(i)+' : '+str(j)
        b.append(element)
    print(b)
    print(maximum_return)
    print('')

    print('Maximum Sharpe')
    c = []
    for i,j in zip(stock_list,maximum_sharpe.Weights):
        element = str(i)+' : '+str(j)
        c.append(element)
    print(c)
    print(maximum_sharpe)
    print('')
    
    #This function will return visualizations of the 10000 portfolios that we created above.
    return portfolio_visualizations(portfolio_return,portfolio_risk,portfolio_sharpe,maximum_sharpe,risk_free)

def portfolio_visualizations(portfolio_return,portfolio_risk,portfolio_sharpe,maximum_sharpe,risk_free):
    plt.figure(figsize=(12,8))
    df_cal = capital_allocation_line(maximum_sharpe,risk_free)
    #Scatter plot x = portfolio_risk, y = portfolio_return and the points are the sharpe ratio.
    plt.scatter(portfolio_risk,portfolio_return,c=portfolio_sharpe, label = 'Sharpe Ratio')
    plt.plot(df_cal.Risk,df_cal.Return,color = 'r', label = 'CAL')
    plt.title('Portfolio Optimization and capital allocaltion line')
    plt.xlabel('Volatality')
    plt.ylabel('Return')
    plt.colorbar(label = 'Sharpe Ratio')
    plt.legend(loc = 'upper left')
    plt.show()
    
def capital_allocation_line(maximum_sharpe,rf):
    weights = [0,1]
    return_p = []
    risk_p = []
    for w in weights:
        returns = w*maximum_sharpe.Return + (1-w)*rf
        risk = w*maximum_sharpe.Risk
        return_p.append(returns)
        risk_p.append(risk)
    cal_df = pd.DataFrame({'Weight': weights, 'Return': return_p, 'Risk': risk_p})
    return cal_df
    
    


# In[4]:


stock_list = ['BA','AAPL','HD', 'BAC', 'SBUX', 'RTX','XOM','BBY','HOG','IHG']


# In[5]:


#parameters 
# stock_list
# number of portfolios
# risk-free rate = 0.0%
portfolio_creation_and_optimization(stock_list,10000,0)


# In[6]:


#parameters 
# stock_list
# number of portfolios
# risk-free rate = 10%
portfolio_creation_and_optimization(stock_list,10000,0.1)


# In[7]:


#parameters 
# stock_list
# number of portfolios
# risk-free rate = 0.2%
portfolio_creation_and_optimization(stock_list,10000,0.002)


# In[ ]:




