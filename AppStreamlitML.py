import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def main():
    st.title("Machine Learning - Classificação")
    st.markdown("Previsão de Câncer de mama usando Machine Learning.")

    st.sidebar.title("ML - Classificação Binária")
    st.sidebar.markdown("Previsão de Câncer de mama")

    @st.cache(persist = True)
    def load_data():
        data = pd.read_csv("Cancer_Wisconsin.csv", encoding = "ISO-8859-1", delimiter = ";")
        label = LabelEncoder()

        for classe in data.columns:
            data[classe] = label.fit_transform(data[classe])

            return data   
    @st.cache(persist = True)
    def split(df):
        y = df.iloc[:, 10].values
        x = df.iloc[:, 1:9].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            st.markdown("A **Confusion Matrix** resume a quantidade de erros e acertos do modelo em ambas as classes.")
            plot_confusion_matrix(model, x_test, y_test, display_labels = class_names)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
    
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            st.markdown("A **ROC Curve** resume a compensação entre a taxa de verdadeiros positivos e a taxa de falsos positivos para um modelo preditivo usando diferentes limites de probabilidade.")
            plot_roc_curve(model, x_test, y_test)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            st.markdown("A **Precision-Recall Curve** resume a compensação entre a taxa de verdadeiro positivo e o valor preditivo positivo para um modelo preditivo usando diferentes limites de probabilidade.")
            plot_precision_recall_curve(model, x_test, y_test)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['Não Câncer','Câncer']
    st.sidebar.subheader("Escolha o modelo de classificação:")
    Classificador = st.sidebar.selectbox("Classificador:", ("Logistic Regression", "Random Forest", "Decision Tree", "K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)"))

    if Classificador == "Logistic Regression":
        st.sidebar.subheader("Hiperparâmetros do modelo")
        st.sidebar.markdown("Interaja com o aplicativo ajustando os hiperparâmentros: \n - **C:** Parâmetro de regularização do modelo (0.01 a 10.0). \n - **max_iter:** Número máximo de iterações (3 a 500).")
        C = st.sidebar.number_input("Parâmetro de regularização:", 0.01, 10.0, step = 0.10, key = 'C')
        max_iter = st.sidebar.slider("Número máximo de iterações:", 3, 500, step = 1, key = 'max_iter')
        metrics = st.sidebar.multiselect("Selecione as métricas:", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
    
        if st.sidebar.button("Aplicar", key = 'Aplicar'):
            st.subheader("Resultados Algoritmo Logistic Regression")
            model = LogisticRegression(C = C, max_iter = max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(3))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names, pos_label= 4).round(3))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names, pos_label= 4).round(3))
            plot_metrics(metrics)


    if Classificador == "Random Forest":
        st.sidebar.subheader("Hiperparâmetros do modelo")
        st.sidebar.markdown("Interaja com o aplicativo ajustando os hiperparâmentros: \n - **n_estimators:** Número de árvores na floresta (1 a 1000). \n - **max_depth:** Profundidade máxima da árvore (1 a 20). \n - **bootstrap:** Amostras de bootstrap ao construir árvores (*True* ou *False*).")        
        n_estimators = st.sidebar.number_input("Número de árvores na floresta:", 1, 1000, step = 1, key = 'n_estimators')
        max_depth = st.sidebar.number_input("Profundidade máxima da árvore:", 1, 20, step = 1, key = 'max_depth')
        bootstrap = st.sidebar.radio("Amostras de bootstrap ao construir árvores:", ('True', 'False'), key = 'bootstrap')
        metrics = st.sidebar.multiselect("Selecione as métricas:", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
    

        if st.sidebar.button("Aplicar", key = 'Aplicar'):
            st.subheader("Resultados Algoritmo Random Forest")
            model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, bootstrap = bootstrap, n_jobs = -1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(3))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names, pos_label= 4).round(3))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names, pos_label= 4).round(3))
            plot_metrics(metrics)

    if Classificador == "Decision Tree":
        st.sidebar.subheader("Hiperparâmetros do modelo")
        st.sidebar.markdown("Interaja com o aplicativo ajustando os hiperparâmentros: \n - **max_features:** Número de recursos a ser utilizado (1 a 20). \n - **max_depth:** Profundidade máxima da árvore (1 a 20).")
        max_features = st.sidebar.slider("Número de recursos a ser utilizado:", 1, 20, step = 1, key = 'max_features')
        max_depth = st.sidebar.number_input("Profundidade máxima da árvore:", 1, 20, step = 1, key = 'max_depth')
        metrics = st.sidebar.multiselect("Selecione as métricas:", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Aplicar", key = 'Aplicar'):
            st.subheader("Resultados Algoritmo Decision Tree")
            model = DecisionTreeClassifier(max_features = max_features, max_depth = max_depth)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(3))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names, pos_label= 4).round(3))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names, pos_label= 4).round(3))
            plot_metrics(metrics)

    if Classificador == "K-Nearest Neighbors (KNN)":
        st.sidebar.subheader("Hiperparâmetros do modelo")
        st.sidebar.markdown("Interaja com o aplicativo ajustando os hiperparâmentros: \n - **n_neighbors:** Número de vizinhos a considerar (1 a 20). \n - **weights:** Pesos (*uniform*  ou *distance*),  o padrão é *uniform*.")
        n_neighbors = st.sidebar.slider("Número de vizinhos a considerar:", 1, 20, step = 1, key = 'n_neighbors')
        weights = st.sidebar.selectbox("Pesos: ", ["uniform", "distance"], key = 'weights')
        metrics = st.sidebar.multiselect("Selecione as métricas:", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Aplicar", key = 'Aplicar'):
            st.subheader("Resultados Algoritmo K-Nearest Neighbors (KNN)")
            model = KNeighborsClassifier(n_neighbors= n_neighbors, weights= weights)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(3))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names, pos_label= 4).round(3))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names, pos_label= 4).round(3))
            plot_metrics(metrics)


    if Classificador == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Hiperparâmetros do modelo")
        st.sidebar.markdown("Interaja com o aplicativo ajustando os hiperparâmentros: \n - **C:** Parâmetro de regularização (0.01 a 10.0). \n - **kernel:** Especifica o tipo de kernel (*rbf*, *linear*, *sigmoid*, *poly*). \n - **gamma:** Coeficientes do Kernel (*scale*, *auto*).")
        C = st.sidebar.number_input("Parâmetro de regularização:", 0.01, 10.0, step = 0.10, key = 'C')
        kernel = st.sidebar.radio("Kernel:", ("rbf", "linear", "sigmoid", "poly"), key = 'kernel')
        gamma = st.sidebar.radio("Gamma (Coeficientes do Kernel):", ("scale", "auto"), key = 'gamma')
        metrics = st.sidebar.multiselect("Selecione as métricas:", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Aplicar", key = 'Aplicar'):
            st.subheader("Resultados Algoritmo Support Vector Machine (SVM)")
            model = SVC(C = C, kernel = kernel, gamma = gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(3))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names, pos_label= 4).round(3))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names, pos_label= 4).round(3))
            plot_metrics(metrics)


    if st.sidebar.checkbox("Sobre as métricas", False):
        st.markdown("**Accuracy:** A proporção de casos de câncer e não câncer que o modelo classificou corretamente.")
        st.markdown("**Precision:** Dentre todas as classificações de casos de câncer que o modelo fez, a proporção de classificações corretas.")
        st.markdown("**Recall:** Dentre todas as classificações que realmente são casos de câncer, a proporção que está correta.")

    if st.sidebar.checkbox("Visualizar dados brutos", False):
        st.subheader("Conjunto de dados de Câncer de mama (Classificação)")
        st.markdown("Os dados refere-se a casos de pacientes com Tumor na mama, sendo o atributo *classe* binário, onde: \n - **2:** Para tumor benigno (Não câncer) \n - **4:** Para tumor maligno (Câncer). \n As variáveis explicativas são: Espessura de aglomerado, Uniformidade do tamanhoda célula, Uniformidade da forma celular, Adesão marginal, Tamanho de célula epitelial única, Núcleos nus, Cromatina Suave, Nucléolos normais, Mitoses. As variáveis preditoras estão em escala ordinal de 0 a 10. \n **Fonte:** https://archive.ics.uci.edu/")
        st.write(df)  
      

    st.sidebar.markdown('[GitHub](https://github.com/Fagna)')
    st.sidebar.markdown("[LinkedIn](https://br.linkedin.com/in/maria-fagna-8116a8218)")

                

if __name__ == '__main__':
    main()