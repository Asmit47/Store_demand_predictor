import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
def load_model():
    with open('model/grid_search_model.pkl', 'rb') as f:
        return pickle.load(f)
model = load_model()

# Load data for analytics
@st.cache_data
def load_data():
    return pd.read_csv('Data and nb/sales_data.csv')
df = load_data()

# Preprocessing for time-based plots
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Day of Week'] = df['Date'].dt.day_name()

# --- Tabs for Prediction and Analytics ---
tab1, tab2 = st.tabs(["Predict Demand", "Explore Analytics"])

with tab1:
    st.title('Store Demand Prediction')
    st.write('Fill in the details below to predict store demand.')

    # --- Feature Inputs ---
    def numeric_input(label, min_value=0, max_value=10000, value=0, step=1, help=None):
        return st.number_input(label, min_value=min_value, max_value=max_value, value=value, step=step, help=help)

    store_options = ['S001', 'S002', 'S003', 'S004', 'S005']
    product_options = [f'P{str(i).zfill(4)}' for i in range(1, 21)]
    category_options = ['Electronics', 'Furniture', 'Groceries', 'Toys']
    region_options = ['East', 'North', 'South', 'West']
    weather_options = ['None', 'Rainy', 'Snowy', 'Sunny']
    seasonality_options = ['Fall', 'Spring', 'Summer', 'Winter']

    with st.form('prediction_form'):
        st.subheader('Basic Information')
        col1, col2, col3 = st.columns(3)
        with col1:
            inventory = numeric_input('Inventory Level', help='Current inventory level')
            price = numeric_input('Price', help='Product price')
            discount = numeric_input('Discount', help='Discount applied')
        with col2:
            month = numeric_input('Month', min_value=1, max_value=12, value=1, help='Month (1-12)')
            competitor_pricing = numeric_input('Competitor Pricing', help='Competitor product price')
            promotion = numeric_input('Promotion', help='Promotion value')
        with col3:
            day = numeric_input('Day', min_value=1, max_value=31, value=1, help='Day of month (1-31)')
            units_sold_lag1 = numeric_input('Units Sold Lag1', help='Units sold previous day')
            units_ordered_lag1 = numeric_input('Units Ordered Lag1', help='Units ordered previous day')

        st.subheader('Categorical Information')
        col4, col5, col6 = st.columns(3)
        with col4:
            store_id = st.selectbox('Store ID', store_options, index=0, help='Select store')
            category = st.selectbox('Category', ['None'] + category_options, index=0, help='Product category')
            region = st.selectbox('Region', region_options, index=0, help='Store region')
        with col5:
            product_id = st.selectbox('Product ID', product_options, index=0, help='Select product')
            weather = st.selectbox('Weather Condition', weather_options, index=0, help='Current weather')
            seasonality = st.selectbox('Seasonality', seasonality_options, index=0, help='Current season')
        with col6:
            st.write('')
            st.write('')
            st.write('')

        submitted = st.form_submit_button('Predict')

    feature_names = [
        'Inventory Level', 'Price', 'Discount', 'Promotion', 'Competitor Pricing', 'Month', 'Day', 'Units Sold Lag1',
        'Units Ordered Lag1', 'Store ID_S002', 'Store ID_S003', 'Store ID_S004', 'Store ID_S005',
        'Product ID_P0002', 'Product ID_P0003', 'Product ID_P0004', 'Product ID_P0005', 'Product ID_P0006',
        'Product ID_P0007', 'Product ID_P0008', 'Product ID_P0009', 'Product ID_P0010', 'Product ID_P0011',
        'Product ID_P0012', 'Product ID_P0013', 'Product ID_P0014', 'Product ID_P0015', 'Product ID_P0016',
        'Product ID_P0017', 'Product ID_P0018', 'Product ID_P0019', 'Product ID_P0020',
        'Category_Electronics', 'Category_Furniture', 'Category_Groceries', 'Category_Toys',
        'Region_North', 'Region_South', 'Region_West',
        'Weather Condition_Rainy', 'Weather Condition_Snowy', 'Weather Condition_Sunny',
        'Seasonality_Spring', 'Seasonality_Summer', 'Seasonality_Winter'
    ]

    def make_feature_vector():
        features = [
            inventory, price, discount, promotion, competitor_pricing, month, day, units_sold_lag1, units_ordered_lag1
        ]
        for s in ['S002', 'S003', 'S004', 'S005']:
            features.append(1 if store_id == s else 0)
        for p in [f'P{str(i).zfill(4)}' for i in range(2, 21)]:
            features.append(1 if product_id == p else 0)
        for c in category_options:
            features.append(1 if category == c else 0)
        for r in ['North', 'South', 'West']:
            features.append(1 if region == r else 0)
        for w in ['Rainy', 'Snowy', 'Sunny']:
            features.append(1 if weather == w else 0)
        for s in ['Spring', 'Summer', 'Winter']:
            features.append(1 if seasonality == s else 0)
        return np.array(features).reshape(1, -1)

    if submitted:
        X = make_feature_vector()
        try:
            prediction = model.predict(X)[0]
            st.success(f'Predicted Demand: {prediction:.2f}')
        except Exception as e:
            st.error(f'Prediction failed: {e}')

with tab2:
    st.header('📊 Explore Store Demand Demographics')
    st.subheader('Filter Data')
    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        regions = st.multiselect('Region', options=sorted(df['Region'].dropna().unique()), help='Filter by region')
        stores = st.multiselect('Store ID', options=sorted(df['Store ID'].dropna().unique()), help='Filter by store')
        categories = st.multiselect('Category', options=sorted(df['Category'].dropna().unique()), help='Filter by product category')
    with colf2:
        seasons = st.multiselect('Seasonality', options=sorted(df['Seasonality'].dropna().unique()), help='Filter by season')
        weather = st.multiselect('Weather Condition', options=sorted(df['Weather Condition'].dropna().unique()), help='Filter by weather')
        promotions = st.selectbox('Promotion', options=['All', 0, 1], help='Promotion applied?')
    with colf3:
        discount_min, discount_max = int(df['Discount'].min()), int(df['Discount'].max())
        discount_range = st.slider('Discount Range', min_value=discount_min, max_value=discount_max, value=(discount_min, discount_max), help='Filter by discount range')
        if 'Date' in df.columns:
            date_min, date_max = df['Date'].min(), df['Date'].max()
            date_range = st.date_input('Date Range', value=(date_min, date_max), min_value=date_min, max_value=date_max, help='Filter by date range')
        else:
            date_range = None

    filtered_df = df.copy()
    if regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    if stores:
        filtered_df = filtered_df[filtered_df['Store ID'].isin(stores)]
    if categories:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
    if seasons:
        filtered_df = filtered_df[filtered_df['Seasonality'].isin(seasons)]
    if weather:
        filtered_df = filtered_df[filtered_df['Weather Condition'].isin(weather)]
    if promotions != 'All':
        filtered_df = filtered_df[filtered_df['Promotion'] == promotions]
    filtered_df = filtered_df[(filtered_df['Discount'] >= discount_range[0]) & (filtered_df['Discount'] <= discount_range[1])]
    if date_range and 'Date' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['Date'] >= pd.to_datetime(date_range[0])) & (filtered_df['Date'] <= pd.to_datetime(date_range[1]))]

    graph_options = {
        'Demand by Region': 'region',
        'Demand by Product Category': 'category',
        'Monthly Demand Trend': 'month',
        'Seasonal Demand': 'season',
        'Demand by Store': 'store',
        'Demand by Product': 'product',
        'Demand by Weather Condition': 'weather',
        'Demand by Promotion': 'promotion',
        'Demand by Discount': 'discount',
        'Demand by Day of Week': 'dayofweek',
        'Demand Over Time': 'time',
    }
    selected_graph = st.selectbox('Select a demographic/business graph to view:', list(graph_options.keys()))

    def plot_graph(kind, data):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 4))
        if data.empty:
            ax.text(0.5, 0.5, 'No data for selected filters', ha='center', va='center', fontsize=14)
            ax.axis('off')
            st.pyplot(fig)
            return
        if kind == 'region':
            d = data.groupby('Region')['Demand'].sum().sort_values(ascending=False)
            sns.barplot(x=d.index, y=d.values, ax=ax)
            ax.set_title('Total Demand by Region')
            ax.set_ylabel('Total Demand')
        elif kind == 'category':
            d = data.groupby('Category')['Demand'].sum().sort_values(ascending=False)
            sns.barplot(x=d.index, y=d.values, ax=ax)
            ax.set_title('Total Demand by Product Category')
            ax.set_ylabel('Total Demand')
        elif kind == 'month':
            d = data.groupby('Month')['Demand'].sum()
            sns.lineplot(x=d.index, y=d.values, marker='o', ax=ax)
            ax.set_title('Monthly Demand Trend')
            ax.set_xlabel('Month')
            ax.set_ylabel('Total Demand')
        elif kind == 'season':
            d = data.groupby('Seasonality')['Demand'].sum().sort_values(ascending=False)
            sns.barplot(x=d.index, y=d.values, ax=ax)
            ax.set_title('Total Demand by Season')
            ax.set_ylabel('Total Demand')
        elif kind == 'store':
            d = data.groupby('Store ID')['Demand'].sum().sort_values(ascending=False)
            sns.barplot(x=d.index, y=d.values, ax=ax)
            ax.set_title('Total Demand by Store')
            ax.set_ylabel('Total Demand')
        elif kind == 'product':
            d = data.groupby('Product ID')['Demand'].sum().sort_values(ascending=False).head(20)
            sns.barplot(x=d.index, y=d.values, ax=ax)
            ax.set_title('Top 20 Products by Total Demand')
            ax.set_ylabel('Total Demand')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        elif kind == 'weather':
            d = data.groupby('Weather Condition')['Demand'].mean().sort_values(ascending=False)
            sns.barplot(x=d.index, y=d.values, ax=ax)
            ax.set_title('Average Demand by Weather Condition')
            ax.set_ylabel('Average Demand')
        elif kind == 'promotion':
            d = data.groupby('Promotion')['Demand'].mean()
            sns.barplot(x=d.index.astype(str), y=d.values, ax=ax)
            ax.set_title('Average Demand by Promotion (0=No, 1=Yes)')
            ax.set_ylabel('Average Demand')
        elif kind == 'discount':
            bins = [0, 5, 10, 20, 50, 100]
            labels = ['0-5', '6-10', '11-20', '21-50', '51+']
            data['Discount Bin'] = pd.cut(data['Discount'], bins=bins, labels=labels, right=False, include_lowest=True)
            d = data.groupby('Discount Bin')['Demand'].mean()
            sns.barplot(x=d.index, y=d.values, ax=ax)
            ax.set_title('Average Demand by Discount Bin')
            ax.set_ylabel('Average Demand')
        elif kind == 'dayofweek':
            order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            d = data.groupby('Day of Week')['Demand'].mean().reindex(order)
            sns.barplot(x=d.index, y=d.values, ax=ax)
            ax.set_title('Average Demand by Day of Week')
            ax.set_ylabel('Average Demand')
        elif kind == 'time':
            d = data.groupby('Date')['Demand'].sum()
            sns.lineplot(x=d.index, y=d.values, ax=ax)
            ax.set_title('Demand Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel('Total Demand')
        st.pyplot(fig)

    plot_graph(graph_options[selected_graph], filtered_df) 