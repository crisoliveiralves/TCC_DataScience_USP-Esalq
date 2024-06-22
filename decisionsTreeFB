import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np

# Carregar e mostrar informações do dataset
df = pd.read_csv('/content/campFBtrat.csv', delimiter=';')
print(df.head())  # Mostrar as primeiras linhas do dataframe
print(df.info())  # Informações sobre o dataframe

# Tratar dados: converter colunas com valores numéricos em formato string para float
colunas_modificar = ['Spend', 'Frequency', 'CTR']
df[colunas_modificar] = df[colunas_modificar].applymap(lambda x: float(x.replace(',', '.')))
print(df.head())  # Mostrar as primeiras linhas após a modificação
print(df.info())  # Informações sobre o dataframe após a modificação
print(df.describe())  # Estatísticas descritivas do dataframe

# Definir Target (y) e Features (X)
y = df['LeadTotal']
print(y.head())  # Mostrar as primeiras linhas do target

features = ['day', 'month', 'year', 'objective', 'Spend', 'Impressions', 'Reach', 'Frequency', 'Clicks', 'CTR', 'PageView']
X = df[features]
print(X.head())  # Mostrar as primeiras linhas das features

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar e treinar o modelo DecisionTreeRegressor
pre_model = DecisionTreeRegressor(random_state=42)
pre_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
test_pred = pre_model.predict(X_test)

# Calcular MAE (Mean Absolute Error) e MSE (Mean Squared Error)
mae = mean_absolute_error(y_test, test_pred)
print('MAE:', mae)

mse = mean_squared_error(y_test, test_pred)
print(f'Mean Squared Error: {mse}')

# Mostrar as primeiras previsões e os valores reais correspondentes
print("Predições do modelo:", test_pred[:5])
print("Target:", y_test[:5].values)

# Visualizar a árvore de decisão
plt.figure(figsize=(15, 10))  # Ajustar o tamanho da figura conforme necessário
plot_tree(pre_model, filled=True, feature_names=X.columns)  # Plotar a árvore de decisão
plt.show()

# Imprimir o número de nós na árvore
num_nos = pre_model.tree_.node_count
print("Número de nós:", num_nos)

# Avaliar a importância das features
importances = pre_model.feature_importances_
feature_names = pre_model.feature_names_in_

# Ordenar índices das importâncias em ordem decrescente
indices = np.argsort(importances)[::-1]

# Plotar as importâncias das características em um gráfico de barras
plt.figure(figsize=(10, 6))
plt.title("Importância das Características")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), np.array(feature_names)[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()

# Função para calcular MAE com diferentes tamanhos máximos de folhas
def get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds_test)
    return mae

# Testar diferentes tamanhos máximos de folhas
maior_num_nos = [5, 25, 50, 100, 130, 250, 500]
erro = 10

for melhor_num_nos in maior_num_nos:
    my_mae = get_mae(melhor_num_nos, X_train, X_test, y_train, y_test)
    if my_mae < erro:
        erro = my_mae
        melhor_tam_arvore = melhor_num_nos

    print("Melhor tamanho:", melhor_num_nos, "Erro:", my_mae, "Melhor erro geral:", erro)

# Re-treinar o modelo com o melhor tamanho de árvore encontrado
pre_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=42)
pre_model.fit(X_train, y_train)

# Mostrar as primeiras previsões e os valores reais correspondentes novamente
print("Predições do modelo:", test_pred[:5])
print("Target:", y_test[:5].values)
