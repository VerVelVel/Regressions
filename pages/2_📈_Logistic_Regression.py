import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
# import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

st.title="Logictic regression"

st.markdown("# Logictic regression")
st.sidebar.header("Logictic regression")

uploaded_file = st.file_uploader("Select CSV-file", type=["csv"])



class LogReg:
    def __init__(self, lr, n_iters):
        self.lr = lr
        self.n_iters = n_iters
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        y = np.array(y)
        n_samples, n_features = X.shape     #768x2
        self.coef_ = np.random.uniform(low=-1, high=1, size=n_features)      #2x1
        self.intercept_ = np.random.uniform(low=-1, high=1, size=1)     # 1x1

        for _ in range(self.n_iters):
            linear_reg = np.dot(X, self.coef_) + self.intercept_    #768x1
            pred = 1/(1+np.exp(-linear_reg)) #768x1

            dw = (1/n_samples) * np.dot(X.T, (pred-y)) # 2x1
            db = (1/n_samples) * np.sum(pred-y)

            self.coef_ = self.coef_ - self.lr * dw
            self.intercept_ = self.intercept_ - self.lr*db
            
    def predict(self, X):
        linear_reg = X@self.coef_ + self.intercept_
        pred = 1/(1+np.exp(-linear_reg))
        res_pred = [1 if y>0.5 else 0 for y in pred]
        return res_pred

# Если файл загружен, считываем его как DataFrame
if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        return df
    
    # Передаем загруженный файл в функцию load_data
    df_train = load_data(uploaded_file)

    target = st.selectbox('Select target', df_train.columns.unique(), index=len(df_train.columns.unique())-1)

    # if target:
    y_train = df_train[target]
    X_train = df_train.drop(columns=target)
    # else:
    #     st.warning("Please select at least one target column.")
    features = X_train.columns.to_list()

    rs = RobustScaler()
    X_train = rs.fit_transform(X_train)

    lr = st.number_input("Enter learning rate:", value=0.01)
    n_iters = int(st.number_input("Enter count of iterations:", value=1000, min_value=1, max_value=10000))

    my_lr = LogReg(lr=lr, n_iters=n_iters)
    my_lr.fit(X_train, y_train)

    st.subheader("Result:")
    weights_dict = dict(zip(features, my_lr.coef_)) # type: ignore
    
    st.write(weights_dict)

    y_pred = my_lr.predict(X_train)

    def accuracy(y_pred, y_train):
        return np.sum(y_pred==y_train)/len(y_train)

    acc = accuracy(y_pred, y_train)
    st.subheader(f'Accuracy = {np.round(acc, 3)}')   

    st.subheader("2D Graphic interpretation:")
    X_label = st.selectbox('Select feature for x-axis', features)
    features_copy = features.copy()
    features_copy.remove(X_label) # type: ignore
    Y_label = st.selectbox('Select feature for y-axis', features_copy)
   


    X = X_train[:, features.index(X_label)] # type: ignore
    Y = X_train[:, features.index(Y_label)] # type: ignore
    Pl = my_lr.predict(X_train)

    fig = px.scatter(x=X, y=Y, color=Pl, color_continuous_scale=['blue', 'red'],labels={'x':X_label, 'y':Y_label})
    fig.update_coloraxes(showscale=False)
    fig.update_layout(title=f'Dependency graph {X_label} и {Y_label} with Prediction color label')
    st.plotly_chart(fig)
    st.write('RED points - p>0.5, BLUE points - p<=0.5')

