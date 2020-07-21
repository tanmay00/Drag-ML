# Importing various Packages

# for implementation


import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')


def main():
    """Drag and Drop ML App"""
    st.title("Drag and Drop ML App")
    st.text("Using the Streamlit Python Library")

    activities = ["EDA", "Plot", "Model Building", "About ML","About Me"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    ######################################################################
    if choice == 'EDA':
        st.subheader("Exploratory Data Analysis")
        data = st.file_uploader("Upload Your Dataset",type=["csv","txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox("Show Shape"):
                st.write(df.shape)

            if st.checkbox("Show Columns"):
                all_columns = df.columns.to_list()
                st.write(all_columns)

            if st.checkbox("Show Specific Column"):
                all_columns = df.columns.to_list()
                selected_columns = st.multiselect("Select Columns", all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)

            if st.checkbox("Show Summary"):
                st.write(df.describe())

            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:,-1].value.counts())

            if st.checkbox("Correlation Using Seaborn"):
                st.write(sns.heatmap(df.corr(), annot=True))
                st.pyplot()
            if st.checkbox("Pie Chart"):
                all_columns = df.columns.to_list()
                columns_to_plot = st.selectbox("Select Any Column", all_columns)
                pie_chart = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pie_chart)
                st.pyplot()

    ###################################################################

    elif choice == 'Plot':
        st.subheader("Data Visualisation")

        data = st.file_uploader("Upload Your Dataset", type=["csv", "txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
        if st.checkbox("Correlation Using Seaborn"):
            st.write(sns.heatmap(df.corr(),annot = True))
            st.pyplot()
        if st.checkbox("Pie Chart"):
            all_columns = df.columns.to_list()
            columns_to_plot = st.selectbox("Select Any Column" , all_columns)
            pie_chart = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
            st.write(pie_chart)
            st.pyplot()

        all_columns_names = df.columns.to_list()
        type_of_plot = st.selectbox("Select type of Plot",["Area","Bar","Line","Histogram","Box","KDE"])
        selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)
        if st.button("Generate Plot"):
            st.success("Generating Custom Plot of {} for {} ".format(type_of_plot,selected_columns_names))


            if type_of_plot == 'Area':
                custom_data = df[selected_columns_names]
                st.area_chart(custom_data)

            elif type_of_plot == 'Bar':
                custom_data = df[selected_columns_names]
                st.bar_chart(custom_data)

            elif type_of_plot == 'Line':
                custom_data = df[selected_columns_names]
                st.line_chart(custom_data)

            elif type_of_plot :
                custom_plot = df[selected_columns_names].plot(kind = type_of_plot)
                st.write(custom_plot)
                st.pyplot()



#####################################################################3

    elif choice == 'Model Building':
        st.subheader("Build ML Model")
        data = st.file_uploader("Upload Your Dataset", type=["csv", "txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            # Building Model
            X = df.iloc[:,0:-1]
            Y = df.iloc[:,-1]
            seed = 7

            #model
            models=[]
            models.append(("Logistic Regression",LogisticRegression()))
            models.append(("Linear Discriminant Analysis", LinearDiscriminantAnalysis()))
            models.append(("K-Nearest Neighbours", KNeighborsClassifier()))
            models.append(("Decision Tree", DecisionTreeClassifier()))
            models.append(("Naive Bayes", GaussianNB()))
            models.append(("SVM", SVC()))


            model_names =[]
            model_mean =[]
            model_std =[]
            all_models=[]
            scoring = 'accuracy'

            for name,model in models:
                kfold = model_selection.KFold(n_splits=10,random_state=seed)
                cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
                model_names.append(name)
                model_mean.append(cv_results.mean())
                model_std.append(cv_results.std())

                accuracy_results = {"Model Name:":name,"Model Accuracy:":cv_results.mean(),"Standard Deviation:":cv_results.std()}
                all_models.append(accuracy_results)
            if st.checkbox("Metrics as Table"):
                st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Model Name","Model Accuracy","Standard Deviation"]))
            if st.checkbox("Metrics as JSON"):
                st.json(all_models)

###################################################################################
    elif choice == 'About ML':
        st.subheader("About Machine Learning")
        st.write("Machine learning is a segment of artificial intelligence. It is designed to make computers learn by themselves and perform operations without human intervention, when they are exposed to new data. It means a computer or a system designed with machine learning will identify, analyse and change accordingly and give the expected output when it comes across a new pattern of data, without any need of humans.\n The power behind machine learning’s self-identification and analysis of new patterns, lies in the complex and powerful ‘pattern recognition’ algorithms that guide them in where to look for what. Thus, the demand for machine learning programmers who have extensive knowledge on working with complex mathematical calculations and applying them to big data and AI is growing year after year .")
    elif choice == 'About Me':
        st.subheader("Tanmay Kansal")
        st.write("")

if __name__ == '__main__':
    main()












