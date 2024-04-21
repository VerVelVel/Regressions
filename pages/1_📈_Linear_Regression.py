import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
# import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

st.title="📈Linear regression"

st.markdown("# Linear regression")
st.sidebar.header("Linear regression")

uploaded_file = st.file_uploader("Select CSV-file", type=["csv"])



class LinReg:
    def __init__(self, learning_rate, n_epochs): #  Занесите все необходимые вещи в атрибуты класса(коэффициенты регуляризации, вид, инициализацию весов)
        self.learning_rate = learning_rate # learning_rate
        self.n_epochs = n_epochs
            
    def fit(self, X, y): # Пример для линейной регрессии без регуляризации
        X = np.array(X)# Переведем в numpy для матричных преобразований
        y = np.array(y)
        
        self.coef_ = np.random.normal(size=X.shape[1]) # Инициализируем веса
        self.intercept_ =  np.random.normal() # Инициализируем свободный член w0
        
        for epoch in range(self.n_epochs):

            y_pred = self.intercept_ + X@self.coef_ # (162, 1)
            
            # 1. Посчитаем отклонение нового результата от обучающего:
            error = (y - y_pred)
            w0_grad = -2 * error # Здесь посчитана частная производная w0
            w_grad = -2 * X * error.reshape(-1, 1) #Здесь посчитаны все частные производные (w1, ..., wn) для всех 162 объектов

            
            # 3. обновляем параметры, используя коэффициент скорости обучения
            self.coef_ = self.coef_ - self.learning_rate * w_grad.mean(axis=0) # Усредняем по 162 объектам каждую из координат и обновляем нашу текущую точку
            self.intercept_ = self.intercept_ - self.learning_rate * w0_grad.mean() # Усредняем intercept_(свободный член w0) и обновляем
                       
    def predict(self, X): # Предсказания, для будущих точек
        X = np.array(X) # (n,2) (2, 1)
        return X@self.coef_ + self.intercept_
    
    def score(self, X, y):
        return r2_score(y, X@self.coef_ + self.intercept_) 

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

    rs = StandardScaler()
    X_train = rs.fit_transform(X_train)

    lr = st.number_input("Enter learning rate:", value=0.1)
    n_iters = int(st.number_input("Enter count of iterations:", value=1000, min_value=1, max_value=10000))

    my_lr = LinReg(learning_rate=lr, n_epochs=n_iters)
    my_lr.fit(X_train, y_train)

    st.subheader("Result:")
    weights_dict = dict(zip(features, my_lr.coef_)) # type: ignore
    
    st.write(weights_dict)

    y_pred = my_lr.predict(X_train)

    # def accuracy(y_pred, y_train):
    #     return np.sum(y_pred==y_train)/len(y_train)

    acc = my_lr.score(X_train, y_train)
    st.subheader(f'R2 csore = {np.round(acc, 3)}')  


    st.subheader("3D Graphic interpretation:")
    X_label = st.selectbox('Select feature for x-axis', features)
    features_copy = features.copy()
    features_copy.remove(X_label) # type: ignore
    Y_label = st.selectbox('Select feature for y-axis', features_copy)
    


    X = X_train[:, features.index(X_label)] # type: ignore
    Y = X_train[:, features.index(Y_label)] # type: ignore
    Pl = my_lr.predict(X_train)
    

    fig = go.Figure()

    # Создание набора точек для предсказания
    x_grid, y_grid = np.meshgrid(np.linspace(X.min(), X.max(), 10),
                                np.linspace(Y.min(),Y.max(), 10))
    X_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    # Получение предсказаний для созданного набора точек
    z_pred = my_lr.predict(X_grid).reshape(x_grid.shape)

    fig.add_trace(go.Surface(x=x_grid, y=y_grid, z=z_pred, opacity=0.5, colorscale='Viridis', name='Prediction'))

    # Добавление тренировочных данных
    fig.add_trace(go.Scatter3d(
        x=X,
        y=Y,
        z=y_train,
        mode='markers',
        marker=dict(
            color='red',
            size=2,
            opacity=0.5
        ),
        name='True',
        showlegend=True
    ))

    # Добавление предсказанных данных
    fig.add_trace(go.Scatter3d(
        x=X,
        y=Y,
        z=Pl,
        mode='markers',
        marker=dict(
            color='blue',
            size=1,
            opacity=0.5
        ),
        name='Pred',
        showlegend=True
    ))
       # Определение меток осей
    fig.update_layout(showlegend=True)
    fig.update_layout(
    legend=dict(
        orientation="h",
        traceorder='normal',
        font=dict(
            family="Courier",
            size=18,
            color="black"
        ),
        itemsizing='constant'
    )
)

    st.plotly_chart(fig)