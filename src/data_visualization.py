import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DataVisualizer:
    def __init__(self):
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

    def plot_sales_trend(self, df, save_path='output/visualizations/sales_trend.png'):
        plt.figure(figsize=(15, 8))
        daily_sales = df.groupby('Date')['Sales'].mean()
        plt.plot(daily_sales.index, daily_sales.values)
        plt.title('Average Daily Sales Trend', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sales', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_store_types(self, df, store_data, save_path='output/visualizations/store_types.png'):
        df_with_store = pd.merge(df, store_data[['Store', 'StoreType']], on='Store', how='left')
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='StoreType', y='Sales', data=df_with_store)
        plt.title('Sales Distribution by Store Type', fontsize=14)
        plt.xlabel('Store Type', fontsize=12)
        plt.ylabel('Sales', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_holidays(self, df, save_path='output/visualizations/holiday_impact.png'):
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='StateHoliday', y='Sales', data=df)
        plt.title('Sales Distribution by Holiday Type', fontsize=14)
        plt.xlabel('Holiday Type', fontsize=12)
        plt.ylabel('Sales', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_promotions(self, df, save_path='output/visualizations/promo_effect.png'):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Promo', y='Sales', data=df)
        plt.title('Impact of Promotions on Sales', fontsize=14)
        plt.xlabel('Promotion Active', fontsize=12)
        plt.ylabel('Sales', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
