import chardet
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import numpy.random as random
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
import scipy as sc
import scipy as sp
from sklearn import linear_model
from scipy import linalg
from scipy import spatial
from scipy import stats
import scipy.spatial.distance
from scipy.optimize import minimize_scalar
from scipy.optimize import newton
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager
import pylab
import time
import os
import glob
import datetime
from scipy import signal
import requests, zipfile
from io import StringIO
import io
#import shap
import streamlit as st
from PIL import Image

st.title('解析アプリ')

st.sidebar.markdown("### csvファイルを入力してください")
st.sidebar.markdown("### ※1行目はラベルにしてください")
file = st.sidebar.file_uploader("ファイルアップロード", type='csv')
if file:
    df = pd.read_csv(file)
    df_columns = df.columns
    st.markdown("### データ一覧")
    st.dataframe(df)
    st.markdown("### 可視化 単変量")
    x = st.selectbox("X軸", df_columns)
    y = st.selectbox("Y軸", df_columns)
    fig = plt.figure(figsize= (12,8))
    plt.scatter(df[x],df[y])
    plt.xlabel(x,fontsize=18)
    plt.ylabel(y,fontsize=18)
    st.pyplot(fig)
    #seabornのペアプロットで可視化。複数の変数を選択できる。
    st.markdown("### 可視化 ペアプロット")
    #データフレームのカラムを選択肢にする。複数選択
    item = st.multiselect("可視化するカラム", df_columns)
    #散布図の色分け基準を１つ選択する。カテゴリ変数を想定
    hue = st.selectbox("色の基準", df_columns)
    
    #実行ボタン（なくてもよいが、その場合、処理を進めるまでエラー画面が表示されてしまう）
    execute_pairplot = st.button("ペアプロット描画")
    #実行ボタンを押したら下記を表示
    if execute_pairplot:
            df_sns = df[item]
            df_sns["hue"] = df[hue]
            
            #streamlit上でseabornのペアプロットを表示させる
            fig = sns.pairplot(df_sns, hue="hue")
            st.pyplot(fig)


    st.markdown("### モデリング")
    #説明変数は複数選択式
    ex = st.multiselect("説明変数を選択してください（複数選択可）", df_columns)

    #目的変数は一つ
    ob = st.selectbox("目的変数を選択してください", df_columns)

    #機械学習のタイプを選択する。
    ml_menu = st.selectbox("実施する機械学習のタイプを選択してください", ["重回帰分析","ロジスティック回帰分析"])
    
    #機械学習のタイプにより以下の処理が分岐
    if ml_menu == "重回帰分析":
            st.markdown("#### 機械学習を実行します")
            execute = st.button("実行")
            
            lr = linear_model.LinearRegression()
            #実行ボタンを押したら下記が進む
            if execute:
                  df_ex = df[ex]
                  df_ob = df[ob]
                  X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size = 0.3)
                  lr.fit(X_train, y_train)
                  #プログレスバー（ここでは、やってる感だけ）
                  my_bar = st.progress(0)
                  
                  for percent_complete in range(100):
                        time.sleep(0.02)
                        my_bar.progress(percent_complete + 1)
                  
                  #metricsで指標を強調表示させる
                  col1, col2 = st.columns(2)
                  col1.metric(label="トレーニングスコア", value=lr.score(X_train, y_train))
                  col2.metric(label="テストスコア", value=lr.score(X_test, y_test))
                  
    #ロジスティック回帰分析を選択した場合
    elif ml_menu == "ロジスティック回帰分析":
            st.markdown("#### 機械学習を実行します")
            execute = st.button("実行")
            
            lr = LogisticRegression()

            #実行ボタンを押したら下記が進む
            if execute:
                  df_ex = df[ex]
                  df_ob = df[ob]
                  X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size = 0.3)
                  lr.fit(X_train, y_train)
                  #プログレスバー（ここでは、やってる感だけ）
                  my_bar = st.progress(0)
                  for percent_complete in range(100):
                        time.sleep(0.02)
                        my_bar.progress(percent_complete + 1)

                  col1, col2 = st.columns(2)
                  col1.metric(label="トレーニングスコア", value=lr.score(X_train, y_train))
                  col2.metric(label="テストスコア", value=lr.score(X_test, y_test))
                  







