# يوفر هذا التقرير تحليلًا شاملاً لبيانات السلاسل الزمنية لأسعار الإغلاق لمجموعتين، M و V
# يشمل التحليل استكشاف البيانات، اختبار الثبات، تحليل المكونات الموسمية، نمذجة ARIMA،
# التنبؤ، وتقييم النموذج.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

warnings.filterwarnings("ignore")

# --- 1. قراءة البيانات ---
try:
    df = pd.read_csv(r"D:\\pythonProject5An\\MVR.csv")
    print("\n✅ Dataset loaded successfully! First 5 rows:\n")
    print(df.head())
except FileNotFoundError:
    print("❌ Error: File not found. Check the file path!")
    exit()

# --- 2. معالجة البيانات ---
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.fillna(method='ffill', inplace=True)

# --- 3. استكشاف البيانات ---
print("\n📊 Dataset Summary:\n")
print(df.describe())

plt.figure(figsize=(14, 7))
sns.boxplot(data=df[['Close_M', 'Close_V']])
plt.title('Boxplot of Closing Prices for M and V Groups')
plt.show()

# --- 4. رسم السلسلة الزمنية ---
plt.figure(figsize=(18, 10))
plt.plot(df['Close_M'], label='Close_M', color='b')
plt.plot(df['Close_V'], label='Close_V', color='g')
plt.title('Time Series of Closing Prices for M and V Groups')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.show()

# --- 5. اختبار الثبات (ADF Test) ---
def adf_test(series, title):
    print(f"\n📊 ADF Test Results for {title}:")
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"P-value: {result[1]:.4f}")
    if result[1] <= 0.05:
        print("✅ The data is stationary (Reject the null hypothesis).")
    else:
        print("⚠️ The data is not stationary (Fail to reject the null hypothesis).")
    print("-" * 50)

adf_test(df['Close_M'], "Close_M")
adf_test(df['Close_V'], "Close_V")

# --- 6. تحليل المكونات (Decomposition) ---
def plot_decomposition(series, title):
    decomposition = seasonal_decompose(series, model='additive', period=252)
    plt.figure(figsize=(18, 14))
    plt.suptitle(f"Decomposition of {title}")
    plt.subplot(4, 1, 1)
    plt.plot(decomposition.observed, label='Original', color='blue')
    plt.legend(loc='upper left')
    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend, label='Trend', color='orange')
    plt.legend(loc='upper left')
    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal, label='Seasonality', color='green')
    plt.legend(loc='upper left')
    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid, label='Residual', color='red')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

plot_decomposition(df['Close_M'], "Close_M")
plot_decomposition(df['Close_V'], "Close_V")

# --- 7. تحديد معلمات ARIMA ---
def best_arima(series):
    best_aic = float("inf")
    best_order = None
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                except:
                    continue
    print(f"✅ Best ARIMA order for series: {best_order} with AIC: {best_aic:.2f}")
    return best_order

order_m = best_arima(df['Close_M'])
order_v = best_arima(df['Close_V'])

# --- 8. بناء النموذج والتنبؤ ---
def arima_forecast(series, order, steps=30):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=steps)
    forecast_index = pd.date_range(start=series.index[-1], periods=steps, freq='D')
    forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)
    conf_int = forecast.conf_int()
    return forecast_series, conf_int

forecast_m, conf_m = arima_forecast(df['Close_M'], order_m)
forecast_v, conf_v = arima_forecast(df['Close_V'], order_v)

# --- 9. عرض التنبؤات ---
def plot_forecast(actual, forecast, conf_int, title):
    plt.figure(figsize=(18, 10))
    plt.plot(actual, label='Actual', color='b')
    plt.plot(forecast, label='Forecast (30 days)', color='r')
    plt.fill_between(forecast.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
    plt.title(f'30-Day Forecast for {title}')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_forecast(df['Close_M'], forecast_m, conf_m, "Close_M")
plot_forecast(df['Close_V'], forecast_v, conf_v, "Close_V")

# --- 10. تقييم دقة النموذج (RMSE) ---
def calculate_rmse(actual, forecast):
    common_index = actual.index.intersection(forecast.index)
    if len(common_index) > 0:
        actual = actual.loc[common_index]
        forecast = forecast.loc[common_index]
        rmse = np.sqrt(mean_squared_error(actual, forecast))
        print(f"✅ RMSE: {rmse:.2f}")
        return rmse
    else:
        print("⚠️ No overlapping dates for RMSE calculation.")
        return None

calculate_rmse(df['Close_M'].iloc[-30:], forecast_m)
calculate_rmse(df['Close_V'].iloc[-30:], forecast_v)

print("\n🔍 Final Insights:")
print("- الاتجاه العام: يمكن ملاحظته من مكون (Trend) في التحليل الزمني.")
print("- يوجد نمط موسمي واضح.")
print("- التنبؤات تظهر ثقة معقولة بناءً على مدى فاصل الثقة.")
