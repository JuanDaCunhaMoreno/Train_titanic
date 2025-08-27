import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

df = pd.read_csv("train.csv")
print(df.head())
print(df.info())

df["Age"].fillna(df["Age"].median(), inplace = True)

sobrevivencia_geral = df["Survived"].mean() *100
st.metric("Taxa Sobrevivência Geral", f"{sobrevivencia_geral:.2f}%")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Sobrevivência por Sexo")
    fig, ax = plt.subplots(figsize = (6, 4))
    sns.countplot(data = df, x = "Sex", hue = "Survived", ax = ax)
    ax.set_title("Distribuição de sobrevivência por sexo")
    st.pyplot(fig)

with col2:
    st.subheader("Sobrevivência por Classe")
    fig, ax = plt.subplots(figsize = (6, 4))
    sns.countplot(data = df, x = "Pclass", hue = "Survived", ax = ax)
    ax.set_title("Distribuição de sobrevivência por classe")
    st.pyplot(fig)

with col3:
    df["Child"] = df["Age"].apply(lambda x: "Child" if x < 18 else "Adult") 
    st.subheader("Sobrevivência por Faixa Etária")
    fig, ax = plt.subplots(figsize = (6, 4))
    sns.countplot(data = df, x = "Child", hue = "Survived", ax = ax)
    ax.set_title("Distribuição de sobrevicência por Faixa Etária Criança/Adulto")
    st.pyplot(fig)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sobrevivência por sexo e Classe")
    fig, ax = plt.subplots(figsize = (6, 4))
    sns.barplot(data = df, x = "Pclass", y = "Survived", hue = "Sex", ax = ax, ci = None)
    ax.set_title("Taxa de sobrevivência por sexo e classe")
    st.pyplot(fig)

with col2:
    st.subheader("Correlação entre variáveis numéricas")
    corr = df.corr(numeric_only = True)
    fig, ax = plt.subplots(figsize = (6, 4))
    sns.heatmap(corr, annot = True, cmap = "coolwarm", ax = ax)
    ax.set_title("Mapa de calor de correlações")
    st.pyplot(fig)