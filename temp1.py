import pandas as pd 

bound = pd.read_csv('bound.csv', index_col=False, header=None)
bound = pd.read_csv("bound.csv")
z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')
z_data = z_data.iloc[:,1:]