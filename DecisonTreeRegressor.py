{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "54608047-017c-4b06-a772-3f19f50b4142",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "cfc3f0b2-71e1-49ad-8489-dc7db760deec",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('dataset_proc.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "f721bfcc-ae6b-4216-a3aa-972b4a63cd55",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Dividindo o dataset em treino e teste\n",
    "\n",
    "X = df.drop('Total Leads', axis=1)\n",
    "y = df['Total Leads']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "c9466204-e477-4c7f-8cbe-8df69333c1ac",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Métricas de avaliação:\n",
      "RMSE: 0.3728\n",
      "R²: -0.0770\n",
      "MedAE: 0.0098\n"
     ]
    }
   ],
   "source": [
    "#Criar o modelo\n",
    "dt_regressor = DecisionTreeRegressor(random_state=42)\n",
    "\n",
    "#Treinar o modelo\n",
    "dt_regressor.fit(X_train, y_train)\n",
    "\n",
    "#Fazer previsões\n",
    "y_pred = dt_regressor.predict(X_test)\n",
    "\n",
    "#Avaliar o modelo\n",
    "print(\"Métricas de avaliação:\")\n",
    "print(f\"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}\")\n",
    "print(f\"R²: {r2_score(y_test, y_pred):.4f}\")\n",
    "print(f\"MedAE: {median_absolute_error(y_test, y_pred):.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "063bed79-89ba-45e0-8d01-87ee63e8214a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
