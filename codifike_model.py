import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale

#lendo os dados
data = pd.read_csv('./data/BRA.csv', delimiter=',')
#tratando os dados
data = data[data['Season'] == 2023]
condicoes = [
    (data['Res'] == 'D'),
    (data['Res'] == 'H'),
    (data['Res'] == 'A')
]
valores = [0, 1, 2]
data['Res'] = np.select(condicoes, valores, default=np.nan)
matches = data.shape[0]
features = data.shape[1] - 1
home_win = len(data[data.Res == 1])
away_win = len(data[data.Res == 2])
draw = len(data[data.Res == 0])
val = [home_win, away_win, draw]
win_rate = (float(home_win)/(matches)) * 100
print('Total de jogos: ', matches)
print('Total de colunas: ', features)
print('Total de jogos ganhos em casa: ', home_win)
print('Total de jogos ganhos fora de casa: ', away_win)
print('Total de empates: ', draw)
print('Percentual de jogos ganhos em casa: {: .2f}' .format(win_rate))

#visualização gráfica
'''
x = np.arange(3)
plt.bar(x, val)
plt.title('Gráfico de vitórias de time da casa, visitante ou empates')
plt.xticks(x, ('Home', 'Away', 'Draw'))
plt.show()
'''
#resultados por data
'''
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
data_desejada = pd.to_datetime('06/12/2023', format='%d/%m/%Y')
condicao = data['Date'] == data_desejada
resultados = data[condicao]
print(resultados)
'''