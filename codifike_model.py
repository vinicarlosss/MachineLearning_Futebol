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
#Notes for football data

#Country = país do campeonato
#League = nome da liga
#game_id = id do jogo
#Season = temporada
#date = data do jogo
#Time = hora do jogo
#Home = Time da casa
#home_id = id do time da casa
#Away = time visitante
#Away_id =  id do time visitante
#HG = Gols do time da casa
#AG = Gols do time visitante
#Res = Resultado do jogo (D=Draw, H=Home win, A=Away win)
#PH = probablidade vitória time da casa provida pela Pinacle (casa de aposta inglesa)
#PD = probablidade empate provida pela Pinacle (casa de aposta inglesa)
#PA = probablidade vitória time visitante provida pela Pinacle (casa de aposta inglesa)
#MaxH = probablidade vitória time da casa provida pela OddsPortal (casa de aposta inglesa)
#MaxD = probablidade empate provida pela OddsPortal (casa de aposta inglesa)
#MaxA = probablidade vitória time visitante provida pela OddsPortal (casa de aposta ingles
#AvgH = probablidade média de vitória em casa provida pela OddsPortal (casa de aposta inglesa)
#AvgD = probablidade média de empate provida pela OddsPortal (casa de aposta inglesa)
#AvgA = probablidade média de vitória pelo time visitante provida pela OddsPortal (casa de aposta inglesa)
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
'''
print('Total de jogos: ', matches)
print('Total de colunas: ', features)
print('Total de jogos ganhos em casa: ', home_win)
print('Total de jogos ganhos fora de casa: ', away_win)
print('Total de empates: ', draw)
print('Percentual de jogos ganhos em casa: {: .2f}' .format(win_rate))
'''

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
#deixar somente as variáveis númericas
num_data = data.drop(['Country', 'League', 'Season', 'Date', 'Time', 'Home', 'Away'], axis=1)
#separa as features
features = num_data.drop(['Res'], axis=1)
#separa as labels
labels = num_data['Res']
features_list = ()
#Escoolhendo as melhores features com Kbest

features_list = ('HG','AG','PH','PD','PA','MaxH','MaxD','MaxA','AvgH','AvgD','AvgA')

k_best_features = SelectKBest(k='all')
k_best_features.fit_transform(features, labels)
k_best_features_scores = k_best_features.scores_
raw_pairs = zip(features_list[1:], k_best_features_scores)
ordered_pairs = list(reversed(sorted(raw_pairs, key=lambda x: x[1])))

k_best_features_final = dict(ordered_pairs[:15])
best_features = k_best_features_final.keys()
'''
print ('')
print ("Melhores features:")

print (k_best_features_final)
'''
#separa as features com base nas melhores features para treinamento
features = num_data.drop(['Res'],axis=1)
#separa as labels para treinamento
labels = num_data['Res']
# Gerando o novo padrão
scaler = MinMaxScaler().fit(features)
features_scale = scaler.transform(features)
'''
print ('Features: ',features_scale.shape)
print (features_scale)
'''
#Separa em treinamento e teste
#Separação manual para manter a ordem cronológica, uma vez que temos informação temporal. 
#Treino linhas [:1932]
#Teste linhas [1932:2155]
#previsão linhas [2155:2280]
X_train = features_scale[:228]
X_test = features_scale[228:304]
y_train = labels[:228]
y_test = labels[228:304]
#Treinando e testando os modelos

clf_LR = LogisticRegression(multi_class='multinomial',max_iter=2000)
clf_LR.fit(X_train, y_train)
pred= clf_LR.predict(X_test)

lg_acc = accuracy_score(y_test, pred)
f1=f1_score(y_test,pred,average = 'micro')
'''
print ('LogisticRegression')
print ('Acurácia LogisticRegression:{}'.format(lg_acc))
print ('F1 Score:{}'.format(f1) )
'''
#Testando LogistRegression hyper parameters

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid)

search.fit(X_train,y_train)
clf = search.best_estimator_
pred= clf.predict(X_test)
lg_acc = accuracy_score(y_test, pred)


f1=f1_score(y_test,pred,average = 'macro')
'''
print ('Acurácia LogisticRegression:{}'.format(lg_acc))
print ('F1 Score:{}'.format(f1) )

print (clf)
'''
#Treinando e testando os modelos

clf = SVC()
clf.fit(X_train, y_train)
pred= clf.predict(X_test)

svc_acc = accuracy_score(y_test, pred)
f1=f1_score(y_test,pred, average='micro')
'''
print ('SVC')
print ('Acurácia SVC:{}'.format(svc_acc))
print ('F1 Score:{}'.format(f1) )
'''

#Testando SVC hyper parameters

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

search = GridSearchCV(SVC(), param_grid)

search.fit(X_train,y_train)
clf_SVC = search.best_estimator_
pred= clf_SVC.predict(X_test)
acc = accuracy_score(y_test, pred)


f1=f1_score(y_test,pred,average = 'micro')
'''
print('SVC)
print ('F1 Score:{}'.format(f1))

print ('Acurácia LogisticRegression:{}'.format(acc))

print(clf_SVC)
'''
#Treinando e testando os modelos

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred= clf.predict(X_test)

dt_acc = accuracy_score(y_test, pred)
f1=f1_score(y_test,pred, average='macro')
'''
print ('Decision Tree')
print ('Acurácia Tree:{}'.format(dt_acc))
print ('F1 Score:{}'.format(f1) )
'''

n_estimators = [10, 50, 100, 200]
max_depth = [3, 10, 20, 40]

#Treinando e testando os modelos

clf = GaussianNB()
clf.fit(X_train, y_train)
pred= clf.predict(X_test)

nb_acc = accuracy_score(y_test, pred)
f1=f1_score(y_test,pred, average='micro')

print ('Naive baeys')
print ('Acurácia Naive baeys:{}'.format(nb_acc))
print ('F1 Score:{}'.format(f1) )
#Executando previsão
previsao = features_scale[304:]

res_full = data['Res']
res = res_full[304:]
pred = clf.predict(previsao)
df = pd.DataFrame({'real': res, 'previsao': pred})
#matriz de confusão
df=pd.DataFrame(df,columns=['real','previsao' ])
cf_matrix=pd.crosstab(df['real'], df['previsao'], rownames=['real'] , colnames=['previsao'])
sns.heatmap(cf_matrix, annot=True, cmap='Blues')
plt.show()