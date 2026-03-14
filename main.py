import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import yfinance as yf
import matplotlib.pyplot as plt
import math

raw_train = yf.download("^GSPC", start="2012-01-01", end="2019-12-31")
raw_test = yf.download("^GSPC", start="2020-01-01", end="2022-12-31")
base_train = pd.DataFrame(raw_train['Close'])
base_test = pd.DataFrame(raw_test['Close'])
base_train.rename(columns = {'^GSPC' : 'Close'}, inplace = True)
base_test.rename(columns = {'^GSPC' : 'Close'}, inplace = True)

# Computation helper function
def compute_rsi(data, window = 14):
    delta = data.diff()
    avg_gain = delta.clip(lower = 0).rolling(window).mean().shift(1)
    avg_loss = -delta.clip(upper = 0).rolling(window).mean().shift(1)
    rs = avg_gain/avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_zscore(data, window = 20):
    mean = data.rolling(window).mean().shift(1)
    std = data.rolling(window).std().shift(1)
    zscore = (data - mean)/std
    return zscore

# data must have 'Close' column
def add_features(d : pd.DataFrame, vix_close = None, lags = 5, extra_cols = ['MovingAverageRatio', 'Momentum_10d', 'Momentum_20d', 'vol_20d', 'vol_60d', 'vol_regime', 'z_score_20d', 'rsi', 'vix_z', 'drawdown', 'max_drawdown']):
    data = d.copy(deep = True)
    data['returns'] = np.log(data['Close']).diff()
    cols = []
    for lag in range(1, lags+1):
        col = 'lag_{}'.format(lag)
        data[col] = data['returns'].shift(lag)
        cols.append(col)
    cols = cols + extra_cols
    
    # Add Trend -> 5dMA/20dMA ratio, momentum 10d/20d
    data['MovingAverageRatio'] = data['Close'].rolling(5).mean().shift(1)/data['Close'].rolling(window=20).mean().shift(1)-1
    data['Momentum_10d'] = data['Close'].pct_change(periods = 10).shift(1)
    data['Momentum_20d'] = data['Close'].pct_change(periods = 20).shift(1)

    # Add Volatility -> rolling std, vol regime
    data['vol_20d'] = data['returns'].rolling(20).std().shift(1)
    data['vol_60d'] = data['returns'].rolling(60).std().shift(1)
    data['vol_regime'] = data['vol_20d']/data['vol_60d']

    # Mean reversion -> z-score, RSI
    data['z_score_20d'] = compute_zscore(data['returns'])
    data['rsi'] = compute_rsi(data['Close'])

    # Regime -> VIX, drawdown state
    data['drawdown'] = data['Close'] / data['Close'].cummax() - 1
    data['max_drawdown'] = data['drawdown'].cummin()

    if vix_close is not None:
        data['vix'] = vix_close['Close'].reindex(data.index)
        data['vix_z'] = compute_zscore(data['vix'], 20)

    return data, cols

# train baseline strategy
cols = []
rf = RandomForestClassifier(n_estimators = 500, max_depth = None, min_samples_leaf = 20, random_state = 42)
data_train_1, cols = add_features(base_train, vix_close = yf.download("^VIX", start="2012-01-01", end="2019-12-31"))
data_train_1['label'] = np.sign(data_train_1['returns']).shift(-1).replace(0, -1)
data_train_1.dropna(inplace = True)
rf.fit(data_train_1[cols], data_train_1['label'])

# test baseline strategy
data_test_1, cols = add_features(base_test, vix_close = yf.download("^VIX", start="2020-01-01", end="2022-12-31"))
data_test_1['label'] = np.sign(data_test_1['returns']).shift(-1).replace(0, -1)
data_test_1.dropna(inplace = True)
baseline_y_pred = rf.predict(data_test_1[cols])
baseline_proba = rf.predict_proba(data_test_1[cols])
baseline_signal = np.where(baseline_proba[:, 1] > 0.55, 1, np.where(baseline_proba[:, 1] < 0.45, -1, 0))

# Plotting variables for the baseline strategy.
baseline_strategy_return = (pd.Series(baseline_signal, index = data_test_1.index).shift(1) * data_test_1['returns'])
baseline_strategy_return.dropna(inplace = True)
baseline_strategy_cumsum = np.exp(baseline_strategy_return.cumsum())

def compute_yearly_sharpe(data : pd.Series, start_date, end_date):
    # input data series must be a daily return series
    df  = data[start_date:end_date]
    sharpe = np.sqrt(252) * df.mean() / df.std()
    return sharpe

def compute_CAGR (data : pd.Series):
    # input data series must be a strategy cummulative sum series
    years = len(data)/252
    return (data.iloc[-1]/data.iloc[0]) ** (1 / years) - 1

def compute_yearly_turnover(data : pd.Series):
    # input data must be a strategy signal series
    return data.diff().abs().mean()*252

def compute_hit_rate(data : pd.Series, strategy: pd.Series):
    # input data must be the label series for test data, and strategy signal series
    multiple = data * strategy
    zeros = (strategy == 0).sum()
    return ((multiple >= 0).sum() - zeros) / (len(multiple) - zeros)
    
def get_max_drawdown(data : pd.Series):
    # input data must be a pandas dataframe that contains the close price column
    df = np.exp(data.cumsum())
    return df/df.cummax()-1

def print_portfolio_evaluation(strategy_return, test, signal):
    # Print Yearly Sharpe
    print("2020 Sharpe: ", round(compute_yearly_sharpe (strategy_return, "2020-01-01", "2020-12-31"), 4))
    print("2021 Sharpe: ", round(compute_yearly_sharpe (strategy_return, "2021-01-01", "2021-12-31"), 4))
    print("2022 Sharpe: ", round(compute_yearly_sharpe (strategy_return, "2022-01-01", "2022-12-31"), 4))
    
    # Print CAGR
    print("CAGR: ", round(compute_CAGR(np.exp(strategy_return.cumsum())), 4))

    # Print Yearly Turnover
    print("Yearly Turnover: ", round(compute_yearly_turnover(pd.Series(signal, index = test.index)), 4))

    # Print Hit Rate
    print("Hit Rate: ", round(compute_hit_rate(test['label'], pd.Series(signal, index = test.index)), 4))

    # Print Max Drawdown
    print("Max Drawdown: ", round(get_max_drawdown(strategy_return).min(), 4))

# Print portfolio evaluation for the baseline strategy
print_portfolio_evaluation(baseline_strategy_return, data_test_1, baseline_signal)
# 2020 Sharpe:  1.2138
# 2021 Sharpe:  1.0833
# 2022 Sharpe:  -0.9688
# CAGR:  0.0393
# Yearly Turnover:  72.7273
# Hit Rate:  0.4916
# Max Drawdown:  -0.2284


# Plot the curve comparing the baseline strategy and PnL

# t = pd.date_range(start = '2020-01-01', end = '2022-12-31')
# plt.figure(figsize = (16, 6))
# plt.plot(baseline_strategy_cumsum, label = 'Basline Strategy')
# plt.plot(np.exp(data_test['returns'].cumsum()), label = 'PnL')
# plt.title("Baseline Strategy vs PnL returns curve")
# plt.xlabel("Time")
# plt.ylabel("Returns")
# plt.legend()
# plt.show()

pi = permutation_importance(rf, data_test_1[cols], data_test_1['label'], random_state = 42)
pi = pd.DataFrame({"feature" : cols, "mean" : pi.importances_mean, "std" : pi.importances_std}).sort_values("mean", ascending = True)

# Plot the feature importance graph
# plt.figure(figsize = (16, 6))
# plt.barh(pi["feature"], pi["mean"], xerr = pi["std"], capsize = 4, error_kw = {'elinewidth' : 1}, alpha = 0.5)
# plt.title("Feature Importance - Permutation Importance")
# plt.xlabel("Permutation Importance")
# plt.ylabel("Features")
# plt.show()

# Train Lag_1 strategy
rf = RandomForestClassifier(n_estimators = 500, max_depth = None, min_samples_leaf = 20, random_state = 42)
lag1_strategy_train, cols = add_features(base_train, vix_close = yf.download("^VIX", start="2012-01-01", end="2019-12-31"), lags = 1, extra_cols = [])
lag1_strategy_train['label'] = np.sign(lag1_strategy_train['returns']).shift(-1).replace(0, -1)
lag1_strategy_train.dropna(inplace = True)
rf.fit(lag1_strategy_train[cols], lag1_strategy_train['label'])

# Test Lag_1 strategy
lag1_strategy_test, cols = add_features(base_test, vix_close = yf.download("^VIX", start="2020-01-01", end="2022-12-31"), lags = 1, extra_cols = [])
lag1_strategy_test['label'] = np.sign(lag1_strategy_test['returns']).shift(-1).replace(0, -1)
lag1_strategy_test.dropna(inplace = True)
lag1_strategy_test_proba = rf.predict_proba(lag1_strategy_test[cols])
lag1_strategy_test_signal = np.where(lag1_strategy_test_proba[:, 1] > 0.55, 1, np.where(lag1_strategy_test_proba[:, 1] < 0.45, -1, 0))

# Plotting variables for the Lag_1 Strategy
lag1_strategy_return = (pd.Series(lag1_strategy_test_signal, index = lag1_strategy_test.index).shift(1) * lag1_strategy_test['returns'])
lag1_strategy_return.dropna(inplace = True)
lag1_strategy_cumsum = np.exp(lag1_strategy_return.cumsum())


# Train alternative strategy
rf = RandomForestClassifier(n_estimators = 500, max_depth = None, min_samples_leaf = 20, random_state = 42)
alternative_strategy_train, cols = add_features(base_train, vix_close = yf.download("^VIX", start="2012-01-01", end="2019-12-31"), lags = 5, extra_cols = ['vol_regime', 'vol_60d', 'MovingAverageRatio', 'Momentum_20d', 'vix_z'])
alternative_strategy_train['label'] = np.sign(alternative_strategy_train['returns']).shift(-1).replace(0, -1)
alternative_strategy_train.dropna(inplace = True)
rf.fit(alternative_strategy_train[cols], alternative_strategy_train['label'])

# Test alternative strategy
alternative_strategy_test, cols = add_features(base_test, vix_close = yf.download("^VIX", start="2020-01-01", end="2022-12-31"), lags = 5, extra_cols = ['vol_regime', 'vol_60d', 'MovingAverageRatio', 'Momentum_20d', 'vix_z'])
alternative_strategy_test['label'] = np.sign(alternative_strategy_test['returns']).shift(-1).replace(0, -1)
alternative_strategy_test.dropna(inplace = True)
alternative_strategy_test_proba = rf.predict_proba(alternative_strategy_test[cols])
alternative_strategy_test_signal = np.where(alternative_strategy_test_proba[:, 1] > 0.55, 1, np.where(alternative_strategy_test_proba[:, 1] < 0.45, -1, 0))

# Plotting variables for the alternative Strategy
alternative_strategy_return = (pd.Series(alternative_strategy_test_signal, index = alternative_strategy_test.index).shift(1) * alternative_strategy_test['returns'])
alternative_strategy_return.dropna(inplace = True)
alternative_strategy_cumsum = np.exp(alternative_strategy_return.cumsum())


# Plot the curve comparing the baseline strategy, Lag_1 strategy, alternative strategy, and PnL
# t = pd.date_range(start = '2020-01-01', end = '2022-12-31')
# plt.figure(figsize = (16, 6))
# plt.plot(baseline_strategy_cumsum, label = 'Basline Strategy', alpha = 0.7, lw = 2)
# plt.plot(lag1_strategy_cumsum, label = 'Lag_1 Strategy', alpha = 0.7, lw = 2)
# plt.plot(alternative_strategy_cumsum, label = 'Alternative Strategy', alpha = 0.7, lw = 2)
# plt.plot(np.exp(data_test_1['returns'].cumsum()), label = 'PnL', alpha = 0.7, lw = 2)
# plt.title("Baseline Strategy vs PnL returns curve")
# plt.xlabel("Time")
# plt.ylabel("Returns")
# plt.legend()
# plt.show()

# Print portfolio evaluation for the lag_1 strategy
# print_portfolio_evaluation(lag1_strategy_return, lag1_strategy_test, lag1_strategy_test_signal)

# Print portfolio evaluation for the alternative strategy
# print_portfolio_evaluation(alternative_strategy_return, alternative_strategy_test, alternative_strategy_test_signal)
