{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "51b012d6-d095-45ab-bb8f-fe16df22d50f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from scipy.stats import skew\n",
    "from sklearn.model_selection import train_test_split\n",
    "from xgboost import XGBRegressor\n",
    "import xgboost as xgb\n",
    "from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "638a5321-1065-48ae-9192-b9b1b6ae91da",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('dataset_proc.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "ae82009e-0175-4c69-9e20-43a5b2b2a598",
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
   "execution_count": 33,
   "id": "202ec102-0ce7-4a68-a099-ce21d3fe9c01",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RMSE: 0.2024\n",
      "R²: 0.6827\n",
      "MedAE: 0.0145\n"
     ]
    }
   ],
   "source": [
    "# Criar o modelo XGBoost\n",
    "xgb_model = xgb.XGBRegressor(\n",
    "    objective='reg:squarederror',  # Para problemas de regressão\n",
    "    random_state=42,\n",
    "    n_estimators=100,\n",
    "    learning_rate=0.1\n",
    ")\n",
    "\n",
    "# Treinar o modelo\n",
    "xgb_model.fit(X_train, y_train)\n",
    "\n",
    "# Fazer previsões\n",
    "y_pred = xgb_model.predict(X_test)\n",
    "\n",
    "# Avaliar o modelo\n",
    "print(f\"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}\")\n",
    "print(f\"R²: {r2_score(y_test, y_pred):.4f}\")\n",
    "print(f\"MedAE: {median_absolute_error(y_test, y_pred):.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "1047e357-1aa6-467f-89d4-6ab13b53d44d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RMSE médio (validação cruzada): 0.4001\n"
     ]
    }
   ],
   "source": [
    "\n",
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "# Modelo XGBoost padrão\n",
    "xgb_model = XGBRegressor(random_state=42)\n",
    "\n",
    "# Validação cruzada (RMSE)\n",
    "scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')\n",
    "rmse_cv = np.mean(-scores)\n",
    "print(f\"RMSE médio (validação cruzada): {rmse_cv:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "1c448535-d134-441c-b2d3-a9f208f3e22d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-04-04 18:10:06,464] A new study created in memory with name: no-name-b773a1fa-60b6-4f4a-b302-c913e694ab42\n",
      "[I 2025-04-04 18:10:06,996] Trial 0 finished with value: 0.30506841257555817 and parameters: {'max_depth': 4, 'learning_rate': 0.0665691714849084, 'n_estimators': 255, 'subsample': 0.7830256620263848, 'colsample_bytree': 0.8282404285235461}. Best is trial 0 with value: 0.30506841257555817.\n",
      "[I 2025-04-04 18:10:07,368] Trial 1 finished with value: 0.3845030878652988 and parameters: {'max_depth': 6, 'learning_rate': 0.03466431607942116, 'n_estimators': 168, 'subsample': 0.9626549110763511, 'colsample_bytree': 0.9776644978251106}. Best is trial 0 with value: 0.30506841257555817.\n",
      "[I 2025-04-04 18:10:07,825] Trial 2 finished with value: 0.4106503097116224 and parameters: {'max_depth': 5, 'learning_rate': 0.04038306752449723, 'n_estimators': 210, 'subsample': 0.9736842881708504, 'colsample_bytree': 0.9584597956584401}. Best is trial 0 with value: 0.30506841257555817.\n",
      "[I 2025-04-04 18:10:08,295] Trial 3 finished with value: 0.33223074650566253 and parameters: {'max_depth': 3, 'learning_rate': 0.10981176889801637, 'n_estimators': 296, 'subsample': 0.8086447956697967, 'colsample_bytree': 0.8076397998650526}. Best is trial 0 with value: 0.30506841257555817.\n",
      "[I 2025-04-04 18:10:08,938] Trial 4 finished with value: 0.3298006153983899 and parameters: {'max_depth': 3, 'learning_rate': 0.014464730843157698, 'n_estimators': 247, 'subsample': 0.9156583980354556, 'colsample_bytree': 0.8096716266147038}. Best is trial 0 with value: 0.30506841257555817.\n",
      "[I 2025-04-04 18:10:09,291] Trial 5 finished with value: 0.336858631941709 and parameters: {'max_depth': 6, 'learning_rate': 0.09298398930025878, 'n_estimators': 174, 'subsample': 0.8344443325181041, 'colsample_bytree': 0.8468446615872519}. Best is trial 0 with value: 0.30506841257555817.\n",
      "[I 2025-04-04 18:10:09,709] Trial 6 finished with value: 0.2891520116293058 and parameters: {'max_depth': 6, 'learning_rate': 0.022212145500616034, 'n_estimators': 231, 'subsample': 0.765882531098614, 'colsample_bytree': 0.7904088127063266}. Best is trial 6 with value: 0.2891520116293058.\n",
      "[I 2025-04-04 18:10:10,094] Trial 7 finished with value: 0.30769687461713985 and parameters: {'max_depth': 3, 'learning_rate': 0.04372579839183567, 'n_estimators': 240, 'subsample': 0.7309310964810334, 'colsample_bytree': 0.767632176455569}. Best is trial 6 with value: 0.2891520116293058.\n",
      "[I 2025-04-04 18:10:10,551] Trial 8 finished with value: 0.3067159124196229 and parameters: {'max_depth': 3, 'learning_rate': 0.06523695760339314, 'n_estimators': 263, 'subsample': 0.7637469165050409, 'colsample_bytree': 0.8328622377603049}. Best is trial 6 with value: 0.2891520116293058.\n",
      "[I 2025-04-04 18:10:10,786] Trial 9 finished with value: 0.3437327605629387 and parameters: {'max_depth': 3, 'learning_rate': 0.09798084703841994, 'n_estimators': 124, 'subsample': 0.9568895191129982, 'colsample_bytree': 0.7744846654845688}. Best is trial 6 with value: 0.2891520116293058.\n",
      "[I 2025-04-04 18:10:11,100] Trial 10 finished with value: 0.3623418057320732 and parameters: {'max_depth': 5, 'learning_rate': 0.16799120984801347, 'n_estimators': 128, 'subsample': 0.8728749191024556, 'colsample_bytree': 0.7150296668025955}. Best is trial 6 with value: 0.2891520116293058.\n",
      "[I 2025-04-04 18:10:11,496] Trial 11 finished with value: 0.30080930323323707 and parameters: {'max_depth': 4, 'learning_rate': 0.06989853637239693, 'n_estimators': 216, 'subsample': 0.7021139299040439, 'colsample_bytree': 0.873135988838329}. Best is trial 6 with value: 0.2891520116293058.\n",
      "[I 2025-04-04 18:10:11,863] Trial 12 finished with value: 0.3111558539113143 and parameters: {'max_depth': 4, 'learning_rate': 0.13941203166601832, 'n_estimators': 209, 'subsample': 0.7033548829726868, 'colsample_bytree': 0.9013673763048582}. Best is trial 6 with value: 0.2891520116293058.\n",
      "[I 2025-04-04 18:10:12,233] Trial 13 finished with value: 0.25464271750310796 and parameters: {'max_depth': 5, 'learning_rate': 0.015491618518409214, 'n_estimators': 183, 'subsample': 0.7001639819180265, 'colsample_bytree': 0.9316100488142729}. Best is trial 13 with value: 0.25464271750310796.\n",
      "[I 2025-04-04 18:10:12,590] Trial 14 finished with value: 0.2720891250695109 and parameters: {'max_depth': 6, 'learning_rate': 0.0184640495133623, 'n_estimators': 176, 'subsample': 0.7541165730121149, 'colsample_bytree': 0.9277791553488123}. Best is trial 13 with value: 0.25464271750310796.\n",
      "[I 2025-04-04 18:10:12,901] Trial 15 finished with value: 0.3157127982602698 and parameters: {'max_depth': 5, 'learning_rate': 0.19911838515753616, 'n_estimators': 160, 'subsample': 0.7399578707201514, 'colsample_bytree': 0.929440377970909}. Best is trial 13 with value: 0.25464271750310796.\n",
      "[I 2025-04-04 18:10:13,276] Trial 16 finished with value: 0.31070230490930867 and parameters: {'max_depth': 6, 'learning_rate': 0.018274253758656517, 'n_estimators': 186, 'subsample': 0.861574126605279, 'colsample_bytree': 0.9947806307959814}. Best is trial 13 with value: 0.25464271750310796.\n",
      "[I 2025-04-04 18:10:13,634] Trial 17 finished with value: 0.3169255623472599 and parameters: {'max_depth': 5, 'learning_rate': 0.054561366265919564, 'n_estimators': 155, 'subsample': 0.8050365078450781, 'colsample_bytree': 0.9349412929241601}. Best is trial 13 with value: 0.25464271750310796.\n",
      "[I 2025-04-04 18:10:13,904] Trial 18 finished with value: 0.3099985319575809 and parameters: {'max_depth': 6, 'learning_rate': 0.13439022024139585, 'n_estimators': 104, 'subsample': 0.7338549493193705, 'colsample_bytree': 0.8913603190033221}. Best is trial 13 with value: 0.25464271750310796.\n",
      "[I 2025-04-04 18:10:14,283] Trial 19 finished with value: 0.3447555737290081 and parameters: {'max_depth': 5, 'learning_rate': 0.08216014225937156, 'n_estimators': 186, 'subsample': 0.8960903264838299, 'colsample_bytree': 0.9384794581561521}. Best is trial 13 with value: 0.25464271750310796.\n",
      "[I 2025-04-04 18:10:14,604] Trial 20 finished with value: 0.3159212924439997 and parameters: {'max_depth': 6, 'learning_rate': 0.03448920973275782, 'n_estimators': 141, 'subsample': 0.8230169545557263, 'colsample_bytree': 0.9072041376481185}. Best is trial 13 with value: 0.25464271750310796.\n",
      "[I 2025-04-04 18:10:15,010] Trial 21 finished with value: 0.27979305407838084 and parameters: {'max_depth': 6, 'learning_rate': 0.017246848621612066, 'n_estimators': 221, 'subsample': 0.7699871086283945, 'colsample_bytree': 0.7485091569416007}. Best is trial 13 with value: 0.25464271750310796.\n",
      "[I 2025-04-04 18:10:15,387] Trial 22 finished with value: 0.2738394606258173 and parameters: {'max_depth': 6, 'learning_rate': 0.010623093316597814, 'n_estimators': 194, 'subsample': 0.7798993639265721, 'colsample_bytree': 0.7306115534517239}. Best is trial 13 with value: 0.25464271750310796.\n",
      "[I 2025-04-04 18:10:15,803] Trial 23 finished with value: 0.2590986418656557 and parameters: {'max_depth': 5, 'learning_rate': 0.01043774627369106, 'n_estimators': 195, 'subsample': 0.7329960190004735, 'colsample_bytree': 0.7073240262350486}. Best is trial 13 with value: 0.25464271750310796.\n",
      "[I 2025-04-04 18:10:16,199] Trial 24 finished with value: 0.3116359972990056 and parameters: {'max_depth': 5, 'learning_rate': 0.05011168080907341, 'n_estimators': 197, 'subsample': 0.7239618458675817, 'colsample_bytree': 0.8761460082502451}. Best is trial 13 with value: 0.25464271750310796.\n",
      "[I 2025-04-04 18:10:16,577] Trial 25 finished with value: 0.280751195281154 and parameters: {'max_depth': 4, 'learning_rate': 0.024991945479049926, 'n_estimators': 174, 'subsample': 0.741987979347623, 'colsample_bytree': 0.9561159644427722}. Best is trial 13 with value: 0.25464271750310796.\n",
      "[I 2025-04-04 18:10:16,901] Trial 26 finished with value: 0.2677335521642068 and parameters: {'max_depth': 5, 'learning_rate': 0.029417018076208216, 'n_estimators': 146, 'subsample': 0.7066276821199571, 'colsample_bytree': 0.7046697644240585}. Best is trial 13 with value: 0.25464271750310796.\n",
      "[I 2025-04-04 18:10:17,218] Trial 27 finished with value: 0.26953159588866177 and parameters: {'max_depth': 5, 'learning_rate': 0.035189029232629954, 'n_estimators': 147, 'subsample': 0.705506303959225, 'colsample_bytree': 0.7009388312146796}. Best is trial 13 with value: 0.25464271750310796.\n",
      "[I 2025-04-04 18:10:17,508] Trial 28 finished with value: 0.29104366060228026 and parameters: {'max_depth': 5, 'learning_rate': 0.054980857660120244, 'n_estimators': 105, 'subsample': 0.7194911287566688, 'colsample_bytree': 0.7408664554557387}. Best is trial 13 with value: 0.25464271750310796.\n",
      "[I 2025-04-04 18:10:17,844] Trial 29 finished with value: 0.2999772494345251 and parameters: {'max_depth': 4, 'learning_rate': 0.07213002479863505, 'n_estimators': 133, 'subsample': 0.792505546042108, 'colsample_bytree': 0.701705740031292}. Best is trial 13 with value: 0.25464271750310796.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Melhores parâmetros: {'max_depth': 5, 'learning_rate': 0.015491618518409214, 'n_estimators': 183, 'subsample': 0.7001639819180265, 'colsample_bytree': 0.9316100488142729}\n"
     ]
    }
   ],
   "source": [
    "#Tunagem fina com Optuna\n",
    "import optuna\n",
    "\n",
    "def objective(trial):\n",
    "    params = {\n",
    "        'max_depth': trial.suggest_int('max_depth', 3, 6),\n",
    "        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),\n",
    "        'n_estimators': trial.suggest_int('n_estimators', 100, 300),\n",
    "        'subsample': trial.suggest_float('subsample', 0.7, 1.0),\n",
    "        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),\n",
    "    }\n",
    "    model = XGBRegressor(**params, random_state=42)\n",
    "    score = cross_val_score(model, X_train, y_train, cv=5, \n",
    "                           scoring='neg_root_mean_squared_error').mean()\n",
    "    return -score\n",
    "\n",
    "study = optuna.create_study(direction='minimize')\n",
    "study.optimize(objective, n_trials=30)\n",
    "print(\"Melhores parâmetros:\", study.best_params)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "6a2c9825-b12c-4101-b515-ec3021046636",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RMSE médio (validação cruzada ajustado): 0.2546\n"
     ]
    }
   ],
   "source": [
    "#Aplicar novos parametros\n",
    "model_xgb = XGBRegressor(\n",
    "    max_depth=5,               # Reduzir complexidade\n",
    "    learning_rate=0.015491618518409214,        # Passos menores\n",
    "    n_estimators=183,          # Mais árvores, mas com taxa de aprendizado baixa\n",
    "    subsample=0.7001639819180265,             # Usar 80% dos dados por árvore\n",
    "    colsample_bytree=0.9316100488142729,      # Usar 80% das features por árvore\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "# Validação cruzada novamente\n",
    "scores = cross_val_score(model_xgb, X_train, y_train, cv=5, \n",
    "                         scoring='neg_root_mean_squared_error')\n",
    "rmse_cv = np.mean(-scores)\n",
    "print(f\"RMSE médio (validação cruzada ajustado): {rmse_cv:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "3fad6e4c-77c9-4cbd-b2ac-122ad9190979",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>XGBRegressor(base_score=None, booster=None, callbacks=None,\n",
       "             colsample_bylevel=None, colsample_bynode=None,\n",
       "             colsample_bytree=0.9316100488142729, device=None,\n",
       "             early_stopping_rounds=None, enable_categorical=False,\n",
       "             eval_metric=None, feature_types=None, feature_weights=None,\n",
       "             gamma=None, grow_policy=None, importance_type=None,\n",
       "             interaction_constraints=None, learning_rate=0.015491618518409214,\n",
       "             max_bin=None, max_cat_threshold=None, max_cat_to_onehot=None,\n",
       "             max_delta_step=None, max_depth=6, max_leaves=None,\n",
       "             min_child_weight=None, missing=nan, monotone_constraints=None,\n",
       "             multi_strategy=None, n_estimators=183, n_jobs=None,\n",
       "             num_parallel_tree=None, ...)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">XGBRegressor</label><div class=\"sk-toggleable__content\"><pre>XGBRegressor(base_score=None, booster=None, callbacks=None,\n",
       "             colsample_bylevel=None, colsample_bynode=None,\n",
       "             colsample_bytree=0.9316100488142729, device=None,\n",
       "             early_stopping_rounds=None, enable_categorical=False,\n",
       "             eval_metric=None, feature_types=None, feature_weights=None,\n",
       "             gamma=None, grow_policy=None, importance_type=None,\n",
       "             interaction_constraints=None, learning_rate=0.015491618518409214,\n",
       "             max_bin=None, max_cat_threshold=None, max_cat_to_onehot=None,\n",
       "             max_delta_step=None, max_depth=6, max_leaves=None,\n",
       "             min_child_weight=None, missing=nan, monotone_constraints=None,\n",
       "             multi_strategy=None, n_estimators=183, n_jobs=None,\n",
       "             num_parallel_tree=None, ...)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "XGBRegressor(base_score=None, booster=None, callbacks=None,\n",
       "             colsample_bylevel=None, colsample_bynode=None,\n",
       "             colsample_bytree=0.9316100488142729, device=None,\n",
       "             early_stopping_rounds=None, enable_categorical=False,\n",
       "             eval_metric=None, feature_types=None, feature_weights=None,\n",
       "             gamma=None, grow_policy=None, importance_type=None,\n",
       "             interaction_constraints=None, learning_rate=0.015491618518409214,\n",
       "             max_bin=None, max_cat_threshold=None, max_cat_to_onehot=None,\n",
       "             max_delta_step=None, max_depth=6, max_leaves=None,\n",
       "             min_child_weight=None, missing=nan, monotone_constraints=None,\n",
       "             multi_strategy=None, n_estimators=183, n_jobs=None,\n",
       "             num_parallel_tree=None, ...)"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Melhores Parametros\n",
    "best_params = {\n",
    "    'max_depth': 6,\n",
    "    'learning_rate': 0.015491618518409214,\n",
    "    'n_estimators': 183,\n",
    "    'subsample': 0.7001639819180265,\n",
    "    'colsample_bytree': 0.9316100488142729\n",
    "}\n",
    "\n",
    "# Modelo final com os melhores parâmetros\n",
    "final_xgb = XGBRegressor(**best_params, random_state=42)\n",
    "final_xgb.fit(X_train, y_train)  # Treina com todos os dados de treino"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "691a59ab-700f-413a-9693-163898f016aa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RMSE no teste: 0.1505\n",
      "R² no teste: 0.8244\n"
     ]
    }
   ],
   "source": [
    "#aplicação no teste\n",
    "\n",
    "y_pred = final_xgb.predict(X_test)\n",
    "rmse_test = mean_squared_error(y_test, y_pred, squared=False)\n",
    "r2_test = r2_score(y_test, y_pred)\n",
    "\n",
    "print(f\"RMSE no teste: {rmse_test:.4f}\")\n",
    "print(f\"R² no teste: {r2_test:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "97204a27-f2e4-4f84-b28f-a79854d1b86b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RMSE no treino: 0.1092\n"
     ]
    }
   ],
   "source": [
    "# Comparar desempenho em treino vs. teste\n",
    "y_pred_train = final_xgb.predict(X_train)\n",
    "rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)\n",
    "print(f\"RMSE no treino: {rmse_train:.4f}\")  # Esperado: próximo ao RMSE do teste (0.1611)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "0f9e4d05-cfdc-4f0b-9529-239cc0d96bad",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Reduzir complexidade\n",
    "best_params['max_depth'] = 4  \n",
    "best_params['learning_rate'] = 0.05  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "7ef740f0-c4f8-4f11-ba7b-61c1b792fc5f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RMSE treino: 0.1092\n",
      "RMSE teste: 0.1505\n",
      "R² teste: 0.8244\n"
     ]
    }
   ],
   "source": [
    "#Aplicar e avaliar\n",
    "y_pred_train = final_xgb.predict(X_train)\n",
    "y_pred_test = final_xgb.predict(X_test)\n",
    "\n",
    "print(f\"RMSE treino: {mean_squared_error(y_train, y_pred_train, squared=False):.4f}\")\n",
    "print(f\"RMSE teste: {mean_squared_error(y_test, y_pred_test, squared=False):.4f}\")\n",
    "print(f\"R² teste: {r2_score(y_test, y_pred_test):.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "1f83646f-2fe4-49b1-8c5f-d902528a5b86",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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U4uc0/jx47Fp0yb4+vpi6tSpaNKkCZ4/f45Dhw6hU6dOaNmyJfr06YP169fDx8cH48aNQ5MmTaCrq4u7d+/i4MGD8PX1RdeuXYscA1GZVNp3ZBERqavcu/VPnjz51n7btm0TLVu2FKampkIulwtHR0fRo0cPsW/fPqnP3bt3Rffu3UX58uWFiYmJaN++vfjnn3/yvQM/JCREODs7C21tbZW745VKpVi0aJGoXLmy0NfXF40aNRIHDhwo8G7933//Pd94Dx06JDp27CjMzc2Frq6uqFChgujYsWOB/XO97W79f//9V6XvoEGDhJGRUZ4xPD09Rc2aNfPEum7dOjF27FhhZWUl5HK5aN68uTh16lSe5bdt2ybc3d2Fvr6+MDIyEq1btxZHjx5V6VNQTEIIkZWVJT777DNhZWUlZDKZACBu3bolhBDizz//FHXr1hX6+vqiQoUKYtKkSWL37t15np7wZg6v5+zo6KjSlpqaKsaNGycqVaokdHV1hbW1tejYsaO4fPmy1EehUIjFixdL6zY2NhYuLi7iiy++ENeuXcuzHqKPnUwIIUqtMiYiIo0WHR2Nli1b4vfff3/rjVpEpDn4KCkiIiIiUhssTomIiIhIbfC0PhERERGpDR45JSIiIiK1weKUiIiIiNQGi1MiIiIiUht8CD+VKqVSifv378PExOS9/tQgERERlR4hBJ4+fQp7e3toaRXvsU4Wp1Sq7t+/DwcHh9IOg4iIiIrgzp07qFixYrGOyeKUSpWJiQkA4NatWzA3Ny/laD4chUKBv/76C+3atYOurm5ph/PBMG/m/bHTxJwB5q2JeW/btg2fffaZ9D1enFicUqnKPZVvYmICU1PTUo7mw1EoFDA0NISpqanG/YfGvJn3x0wTcwaYt6bmDaBELsnjDVFEREREpDZYnBIRERGR2mBxSkRERERqg8UpEREREakNFqdEREREpDZYnBIRERGR2mBxSkRERERqg8UpEREREakNFqdEREREpDZYnBIRERGR2mBxSkRERERqg8UpEREREakNFqdEREREpDZYnBIRERGR2mBxSkRERERqg8UpEREREakNFqdEREREpDZYnBIRERGR2mBxSkRERERqg8UpEREREakNFqdEREREpDZYnBIRERGR2mBxSkRERERqg8UpEREREakNFqdEREREpDZYnBIRERGR2mBxSkRE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lsJs3/dxPLA1LCwsirx8WaNQKBAZGYl/gr015rdtbW1t/PPPP1izZg2CgoJgZ2eHli1bYtOmTTAxMZH6JSYmqnxGmjZtioiICHz99dcICgpClSpVsGnTJj7jlIjoI1XqxWl0dDTq16+v0jZo0CAcPXoUU6ZMgbe3N7KysuDo6Ij27dtDS0tLeozT2LFjUatWLdSoUQPff/89vLy8pDGMjY3x559/Yvjw4ahfvz7c3NywcOFCdO/eXeqjp6eHwMBAJCQkwMDAAM2bN0dEREShY1+wYAEWLlyIM2fOoEqVKti+fbv0IPwKFSogMjISkyZNQt26dWFubi4VckXl5uYGQ0NDTJgwAXfu3IFcLke1atWwatUqDBgwIE9s48aNw7Vr11C3bl3s2LFDeth5nTp1cOjQIUybNg3NmzeHEAJVqlSRbqAqjMJsX6LXH69mYGCAXbt2ITIyEj4+PgUW5W8+kg14dVNUjx49SihKIiJSJ6VanIaHhyM8PLzA+Vu2bClwXps2bXDp0iWVtjevg/zkk0/yHJV9vc/XX3/9n4pFV1fXt57K9vT0xIkTJwqcn9+X8NtUrlw5z7NGC9KsWTP8888/Bc5v3Lgx/vrrrwLnJyQk5Gl7c1u+a/sSERERvS+1fQg/EREREWkeFqf5mDdvHoyNjfP96dChQ4mtd/369QWut2bNmiW2XiIiIiJ1UerXnKqj4cOHo1evXvnOMzAwQIUKFUrk9HWXLl0KvMmjsDfNeHl58dQ6ERERlVksTvNhbm4Oc3PzD75eExMTlbuWiYiIiDQNT+sTERERkdpgcUpEREREaoPFKRERERGpDRanRERERKQ2WJwSERERkdpgcUpEREREaoPFKRERERGpDRanRERERKQ2WJwSERERkdpgcUpEREREaoPFKRERERGpDRanRERERKQ2WJwSERERkdpgcUpEREREaoPFKRERERGpDRanRERERKQ2WJwSERERkdpgcUpEREREaoPFKRERERGpDRanRERERKQ2WJwSERERkdpgcUpEREREaoPFKRERERGpDRanRERERKQ2iq04ffLkSXENRUREREQaqkjF6cKFC7Fp0yZpulevXrCwsECFChVw7ty5YguOiNTb/PnzIZPJMH78eACAQqHAlClTULt2bRgZGcHe3h4DBw7E/fv33znW5s2b4ebmBrlcDjc3N2zdurWE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R6dOneDg4ICePXtCS0sL58+fx4ULFzBnzpwCx5w+fToaNmyImjVrIisrCzt37pQKy6pVqyIxMRERERFo3Lgxdu3aha1bt6osP2nSJPTq1QsNGjRA69at8eeff2LLli3Yt2/fe+X2LqGhoQAALy8vlfawsDD4+/sDAJKSkpCYmCjNc3Z2RmRkJL788kv8+OOPsLe3x/fff8/HSBEREak5Fqf/gbOzM06fPo25c+diwoQJSEpKgpWVFRo2bIjQ0FDIZDJs27YNY8aMQYsWLaClpYX27dvjhx9+eO91tWrVCubm5rhy5Qr69eunMs/b2xs7d+7ErFmzsGjRIujq6sLFxQWfffbZW8fU09NDYGAgEhISYGBggObNmyMiIgIA4Ovriy+//BKjR49GVlYWOnbsiKCgIAQHB0vL+/n54bvvvsM333yDsWPHwtnZGWFhYXmKyMI4HtgaFhYW+c7Lvb73bcLDw/O0eXp64vTp0+8dCxEREZUemSjMNz9RCUlPT4eZmRlSUlIKLE4/RgqFApGRkfDx8dGoU0HMm3l/7DQxZ4B5a2Lef/zxB/r164e0tDSVm6WLA59zSkRERERqg8UpEREREakNFqdEREREpDZYnBIRERGR2mBxSkRERERqg8UpEREREakNFqdEREREpDZYnBIRERGR2mBxSkRERERqg8UpEREREakNFqdEREREpDZYnBIRERGR2mBxSkRERERqg8UpEREREakNFqdEREREpDZYnBIRERGR2mBxSkRERERqg8UpEREREakNFqdEREREpDZYnBIRERGR2mBxSkRERERqg8UpEREREakNFqdEREREpDZYnBIRERGR2mBxSkRERERqg8UpEREREakNFqdEREREpDZYnBIRERGR2mBxSkRERERqg8UpEREREakNFqdEREREpDZYnBIRERGR2mBxSkRERERqg8UpEREREakNndIOgDSbEAIA8PTpU+jq6pZyNB+OQqFAZmYm0tPTmbcGYN6ak7cm5gwwb03NG/i/7/HixOKUStWjR48AAM7OzqUcCREREb2vp0+fwszMrFjHZHFKpcrc3BwAkJiYWOxvbnWWnp4OBwcH3LlzB6ampqUdzgfDvJn3x04TcwaYt6bmfenSJdjb2xf7+CxOqVRpab267NnMzEyjPti5TE1NmbcGYd6aQxNzBpi3pqlQoYL0PV6ceEMU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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Feature Importance\n",
    "from xgboost import plot_importance\n",
    "plot_importance(final_xgb, max_num_features=10)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1df2878d-3c5e-407c-a78a-f8fdf2e828f5",
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
