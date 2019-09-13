
from pandas.io.json import json_normalize
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.tsatools import add_trend
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_white
#import pyodbc
import pandas as pd
import numpy as np


from pandas.io.json import json_normalize

class DFConverter:

    #Converts the input JSON to a DataFrame
    def convertToDF(self,dfJSON):
        return(json_normalize(dfJSON))

    #Converts the input DataFrame to JSON 
    def convertToJSON(self, df):
        resultJSON = df.to_json(orient='records')
        return(resultJSON)
 


''' Get al api del BCRP'''

data = requests.get("https://estadisticas.bcrp.gob.pe/estadisticas/series/api/PN01457BM-PN01652XM-PN01713AM-PN01714AM-PN01715AM-PN01716AM-PN01717AM-PN01718AM-PN01719AM-PN01720AM-PN01721AM-PN01722AM-PN01723AM-PN01724AM-PN01725AM-PN01728AM-PN01731AM-PD04722MM-PN07807NM-PN07816NM-PN01273PM-PN01711BM-PN02196PM-PN01234PM-PN37698XM/json/2009-1/2019-12",\
                    verify=False).json()


conv=DFConverter()
df=conv.convertToDF(data['periods'])
df=pd.DataFrame(df['values'].tolist())

names=conv.convertToDF(data['config']['series'])
df.columns=[names['name'].tolist()]

len(names['name'].tolist())



df.columns=['tasa_ref','tc','var_ipc','balanza_comercial','precio_cobre',\
            'var_ter_inter',\
            'pbi_agro','pbi_agro-agri','pbi_agro_pecu','pbi_pesca','pbi_min',\
            'pbi_min_meta','pbi_min_carbu','pbi_manu','pbi_manu_repri',\
            'pbi_manu_nopri', 'pbi_elec','pbi_constr','pbi_comer','pbi',\
            'pbi_des','ind_dempleo','TAMN','TIPMN','letras_10']

fechas=conv.convertToDF(data['periods'])
fechas=fechas['name'].tolist()

df.index=fechas

df.replace({'n.d.': np.nan}, inplace=True)

df.drop('letras_10',axis=1,inplace=True)

df=df.dropna()


cols = df.columns[df.dtypes.eq('object')]

df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

train=df.tc.pct_change(1).dropna()



from risk_simulator import best_distribution,simulation
#import matplotlib.pyplot as plt

arg,loc,scale,dist=best_distribution(train,plot=True)

result=simulation(1000,arg=arg,loc=loc,scale=scale,dist=dist)

result=pd.DataFrame(result)
print(dist)
VaR=np.quantile(result, 0.05)


result.hist()
plt.axvline(x=VaR,linewidth=4, color='r')
plt.title('VaR')






