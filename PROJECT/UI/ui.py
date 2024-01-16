from UI.other import *
from algorithms.SupervisedModel.DescisionArbre import DecisionTree
from algorithms.SupervisedModel.KNN import *
from algorithms.SupervisedModel.RandomForest import RandomForest
from algorithms.UnsupervisedModel.Apriori import Apriori
from algorithms.UnsupervisedModel.DBSCAN import DBSCAN
from algorithms.UnsupervisedModel.Kmeans import KMeans
from algorithms.metrics import *
import tkinter as tk
from sklearn.model_selection import train_test_split
from preprocessing import *
import pandas as pd
import TKinterModernThemes as TKMT


class App(TKMT.ThemedTKinterFrame):
    def __init__(self, theme, mode, usecommandlineargs=True, usethemeconfigfile=True):
        self.theme = theme
        self.mod = mode
        super().__init__("DATA MINING APPLICATION", theme, mode,
                         usecommandlineargs=usecommandlineargs, useconfigfile=usethemeconfigfile)
        # variable
        self.option_missing_List = ["Leave Values as Is", "Replace with Mean", "Replace with Median",
                                    "Replace with Mod", "Delete"]
        self.option_missing_Var = tk.StringVar(value=self.option_missing_List[0])

        self.option_outliers_List = ["Leave Values as Is", "Replace with Mean", "Replace with Median",
                                     "Replace with Mod", "Replace with IQR-Min", "Replace with IQR-Max",
                                     "Replace with NAN", "Delete"]
        self.option_outliers_Var = tk.StringVar(value=self.option_outliers_List[0])

        self.option_normalization_List = ["Leave Values as Is", "Min-Max", "Z-score"]
        self.option_normalization_var = tk.StringVar(value=self.option_normalization_List[0])

        self.option_descritization_list = ["Leave Values as Is", "Equal-Frequency", "Equal-width"]
        self.option_descritization_var = tk.StringVar(value=self.option_descritization_list[0])

        self.option_delete_list = ["No", "Yes"]
        self.option_delete_var = tk.StringVar(value=self.option_delete_list[0])

        self.option_redection_list = ["No", "Yes"]
        self.option_redection_var = tk.StringVar(value=self.option_redection_list[0])

        self.spinboxQ = tk.IntVar(value=3)

        self.option_distance_list = ["euclidean", "manhattan", "minkowski", 'cosine']
        self.option_distance_var = tk.StringVar(value=self.option_distance_list[0])

        self.option_method_init_list = ["Random", "K-means++"]
        self.option_method_init_var = tk.StringVar(value=self.option_method_init_list[0])

        self.welcome_page()

    def welcome_page(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.welcome = self.addLabelFrame("Welcome", rowspan=1, colspan=2, padx=150)
        welcome_txt = "Welcome to our Data Mining Application!"
        self.welcome.Label(text=welcome_txt, size=17, colspan=2)
        descibe_txt_1 = "This application allows you to study two types of data: \n\n" \
                        "                 Static Data And Dynamic Data    "
        self.welcome.Label(text=descibe_txt_1, size=15, colspan=2)

        descibe_txt_2 = """  
            You can implement both supervised and unsupervised models to explore and analyze  data.

            Start by importing the datasets and explore the various features offered by the application.

                                                   Happy mining! 
       """
        self.welcome.Label(text=descibe_txt_2, size=15, colspan=2)

        # self.menu = self.addLabelFrame("Menu", rowspan=1, colspan=2,padx=500)
        self.Button(text="     Dataset1     ", command=self.dataset1, colspan=2, padx=400)
        self.Button(text="     Dataset2     ", command=self.dataset2, colspan=2, padx=400)
        self.Button(text="     Dataset3     ", command=self.dataset3, colspan=2, padx=400)


        self.run()

    def dataset1(self):
        file_path = "../dataset/Dataset1.csv"
        self.df = read_data(file_path)
        self.manipulate_dataset1()

    def dataset3(self):
        file_path = "../dataset/Dataset3.csv"
        self.df = read_data(file_path)
        self.manipulate_dataset3()

    def dataset2(self):
        file_path = "../dataset/Dataset2.csv"
        self.df = read_data(file_path)
        self.options_area_list = self.df['zcta'].unique().astype(str)
        self.options__interval_list = ['weekly', 'monthly', 'yearly']

        self.options_period_list = self.df['time_period'].unique().astype(int)
        self.period_var = tk.IntVar(value=self.options_period_list[0])

        self.option_area_Var = tk.StringVar(value=self.options_area_list[0])
        self.options__interval_var = tk.StringVar(value=self.options__interval_list[0])
        self.manipulate_dataset2()

    def create_menu_dataset3(self):
        self.menu = self.addLabelFrame("Menu", row=0, col=1)

        self.menu.Button(text="Welcome Page", command=self.welcome_page, padx=150, pady=30)
        self.menu.Button(text="Visualize Data", command=self.manipulate_dataset3, padx=150, pady=10)
        self.menu.Button(text="Preprocess Data", command=self.preprocess_dataset3, padx=150, pady=20)
        self.menu.Button(text="Apriori Algorithm", command=self.apriori_button, padx=150)

    def manipulate_dataset3(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.create_data_display()
        self.create_menu_dataset3()

        self.displayDataInfo = self.addLabelFrame("Description for Each Attribute", colspan=2)

        self.displayDataInfo.Treeview(columnnames=self.df_info.columns.tolist(),
                                      columnwidths=[1000 // self.df.shape[1]] * self.df_info.shape[1], height=8,
                                      data=self.df_info.to_dict(orient='records'), subentryname=None)

    def manipulate_dataset2(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        #  self.menu = self.addLabelFrame("Menu", row=0, col=1)
        self.Button(text="Return to welcome page", command=self.welcome_page, row=0, col=0, padx=200)
        self.displayData = self.addLabelFrame("Data Display ", rowspan=1, colspan=2, row=1)
        self.df_info, self.count_missValues = dataset_describe(self.df)
        text_shape_data = "Data Shape: " + str(self.df.shape[0]) + " x " + str(self.df.shape[1])
        text_missing_val = "                                Number of missing Values : " + str(
            self.count_missValues)
        text_data_info = text_shape_data + text_missing_val
        self.displayData.Label(text=text_data_info, row=1, col=0, size=9)
        self.displayData.Treeview(row=0, col=0, columnnames=self.df.columns.tolist(),
                                  columnwidths=[1100 // self.df.shape[1]] * self.df.shape[1], height=8,
                                  data=self.df.to_dict(orient='records'), subentryname=None)
        self.preprocess = self.addLabelFrame("Preprocess Data", row=2, col=1)
        self.option_pre_att_list = ["All"] + self.df.columns.tolist()
        self.option_prep_att_var = tk.StringVar(value=self.option_pre_att_list[0])

        self.preprocess.Label(text="Selected attribute to preprocess :", row=0, col=0, size=9)
        self.preprocess.OptionMenu(self.option_pre_att_list, self.option_prep_att_var, lambda x: print("Menu:", x),
                                   row=0, col=1)

        self.preprocess.Label(text="Handle Missing Values :", row=1, col=0, size=9)
        self.preprocess.Label(text='Handle Outliers :', row=2, col=0, size=9)

        self.preprocess.OptionMenu(self.option_missing_List, self.option_missing_Var, lambda x: print("Menu:", x),
                                   row=1, col=1)
        self.preprocess.OptionMenu(self.option_outliers_List, self.option_outliers_Var, lambda x: print("Menu:", x),
                                   row=2, col=1)
        self.preprocess.Button(text="Preprocess", command=self.run_preprocessdataset2, row=3, col=1, colspan=2)

        self.visualize = self.addLabelFrame("Visualize Data", row=2, col=0)
        self.option_questions_list = [
            "The distribution of the total number of confirmed cases and positive tests by zones",
            "The evolution of COVID-19 tests, positive test results, and the number of cases over \n"
            " time (weekly, monthly, and annually) for a selected area.",
            "Distribution of positive COVID cases by zone and by year",
            "The ratio between the population and the number of tests conducted.",
            "The 5 zones most heavily impacted by the coronavirus.",
            "The relationship between confirmed cases, tests conducted, and positive tests over\n time for each zone depends on the chosen period."
        ]
        self.visualize.Label("Choose question :", row=0, col=0, size=9)

        self.option_questions_var = tk.StringVar(value=self.option_questions_list[0])
        self.visualize.OptionMenu(self.option_questions_list, self.option_questions_var,
                                  lambda x: print("Menu:", x), row=0, col=1, colspan=3)
        self.visualize.Label("Choose Area :", row=1, col=0, size=9)
        self.visualize.OptionMenu(self.options_area_list, self.option_area_Var, lambda x: print("Menu:", x),
                                   row=1, col=1, colspan=1, padx=50)
        self.visualize.Label("Choose Interval :", row=1, col=2, size=9)
        self.visualize.OptionMenu(self.options__interval_list, self.options__interval_var, lambda x: print("Menu:", x),
                                   row=1, col=3)
        self.visualize.Label("Choose Period :", row=2, col=0, size=9)
        self.visualize.OptionMenu(self.options_period_list, self.period_var, lambda x: print("Menu:", x),
                                  row=2, col=1, colspan=1, padx=50)
        self.visualize.Button(text="Show graphe", command=self.visualize_dataset2, row=2, col=2, padx=100)

    def textupdate(self, _var, _indx, _mode):
        print("Current text status:", self.textinputvar.get())


    def visualize_dataset2(self):
        GrapheDataset2(self.theme, self.mode, True, True, self)

    def showGraphe_method(self):
        if self.df is None:
            print("There is no data")
            return
        if self.option_selected_att_var.get() == "All":
            if self.radiobuttonvar.get() == "Box-Plot":
                graphe = Show_Graph(self.theme, self.mode, True, True, self.df, "BOX-Plot")
            else:
                graphe = Show_Graph(self.theme, self.mode, True, True, self.df, "Histogram")
        else:
            name_column = self.option_selected_att_var.get()
            self.canvas, fig, self.ax, background, self.accent = self.graphframe.matplotlibFrame("Graph Frame Test",
                                                                                                 toolbar=False,
                                                                                                 figsize=(3, 2), row=0,
                                                                                                 col=0)
            if self.radiobuttonvar.get() == "Box-Plot":
                self.ax.boxplot(self.df[name_column])
            else:
                self.ax.hist(self.df[name_column], bins=10, color='skyblue', edgecolor='black')

            self.canvas.draw()

    def manipulate_dataset1(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        # variable
        self.radiobuttonvar = tk.StringVar(value='Box-Plot')
        self.option_selected_att_list = ["All"] + self.df.columns.tolist()
        self.option_selected_att_var = tk.StringVar(value=self.option_selected_att_list[0])

        self.check_frame = self.addLabelFrame("Manipulate dataset", row=0, col=0)
        self.check_frame.Label(text="Selected attribute :", row=0, size=9, col=0)
        self.check_frame.OptionMenu(self.option_selected_att_list, self.option_selected_att_var,
                                    lambda x: print("Menu:", x), row=0, col=1)
        # Separator
        self.Seperator()
        self.check_frame.Radiobutton("Box-Plot", self.radiobuttonvar, value="Box-Plot", row=0, col=2)
        self.check_frame.Radiobutton("Histogram", self.radiobuttonvar, value="Histogram", row=1, col=2)
        self.check_frame.Button("Show Graphe", row=1, command=self.showGraphe_method, col=3)

        self.menu = self.addLabelFrame("Menu", rowspan=1, row=0, col=1)
        self.menu.Button(text="Welcome page", command=self.welcome_page, row=0, colspan=2, padx=100)
        self.menu.Button(text="Preprocess Data", command=self.preprocess_page, row=1, colspan=2, padx=100)

        self.displayData = self.addLabelFrame("Data Display ", rowspan=1, colspan=1, row=1, col=0)
        self.graphframe = self.addLabelFrame("Show Graphe", rowspan=1, colspan=1, row=1, col=1)
        self.displayDataInfo = self.addLabelFrame("Description for Each Attribute", rowspan=2, colspan=2)

        # data information
        self.df_info, self.count_missValues = dataset_describe(self.df)
        self.displayData.Treeview(row=0, col=0, columnnames=self.df.columns.tolist(),
                                  columnwidths=[800 // self.df.shape[1]] * self.df.shape[1], height=5,
                                  data=self.df.to_dict(orient='records'), subentryname=None)
        text_shape_data = "Data Shape: " + str(self.df.shape[0]) + " x " + str(self.df.shape[1])
        text_missing_val = "                                Number of missing Values : " + str(self.count_missValues)
        text_data_info = text_shape_data + text_missing_val
        self.displayData.Label(text=text_data_info, row=1, col=0, size=9)
        self.displayDataInfo.Treeview(columnnames=self.df_info.columns.tolist(),
                                      columnwidths=[1000 // self.df.shape[1]] * self.df_info.shape[1], height=3,
                                      data=self.df_info.to_dict(orient='records'), subentryname=None)

    def create_data_display(self):
        self.displayData = self.addLabelFrame("Data Display ", rowspan=1, colspan=1, row=0, col=0)
        if self.df.shape[1] == 0:
            self.displayData.Label(text="There is no data here ", size=17, colspan=2)
        else:
            self.df_info, self.count_missValues = dataset_describe(self.df)
            text_shape_data = "Data Shape: " + str(self.df.shape[0]) + " x " + str(self.df.shape[1])
            text_missing_val = "                                Number of missing Values : " + str(
                self.count_missValues)
            text_data_info = text_shape_data + text_missing_val
            self.displayData.Label(text=text_data_info, row=1, col=0, size=9)
            self.displayData.Treeview(row=0, col=0, columnnames=self.df.columns.tolist(),
                                      columnwidths=[700 // self.df.shape[1]] * self.df.shape[1], height=10,
                                      data=self.df.to_dict(orient='records'), subentryname=None)

    def create_preprocessing_menu(self):
        self.menu = self.addLabelFrame("Menu", row=0, col=1)
        self.menu.analyse = self.menu.addLabelFrame("Analyze and Visualize data", row=0, col=0)
        self.menu.analyse.Button(text="Welcome page", command=self.welcome_page, row=0, col=0)
        self.menu.analyse.Button(text="Manipulate data", command=self.manipulate_dataset1, row=0, col=1)

        self.menu.supervised = self.menu.addLabelFrame("Supervised model", row=1)
        self.menu.supervised.Button(text="KNN", command=self.knn_button, row=2, col=0)
        self.menu.supervised.Button(text="Random forest", command=self.random_forest_button, row=2, col=1)
        self.menu.supervised.Button(text="Tree decision", command=self.desicion_tree_button, row=2, col=2)

        self.menu.unsupervised = self.menu.addLabelFrame("UnsupervisedModel model", row=2)
        self.menu.unsupervised.Button(text="K-means", command=self.kmeans_button, row=2, col=0)
        self.menu.unsupervised.Button(text="DBSCAN", command=self.dbscam_button, row=2, col=1)

    def preprocess_page(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.create_data_display()
        self.create_preprocessing_menu()

        self.option_pre_att_list = ["All"] + self.df.columns.tolist()
        self.option_prep_att_var = tk.StringVar(value=self.option_pre_att_list[0])

        self.input_frame = self.addLabelFrame("Preprocess to each attribute", row=1, col=0)
        self.processesall = self.addLabelFrame("Processes to all the data", row=1, col=1, rowspan=2)
        self.input_frame.Label(text="Selected attribute to preprocess :", row=0, col=0, size=9)
        self.input_frame.OptionMenu(self.option_pre_att_list, self.option_prep_att_var, lambda x: print("Menu:", x),
                                    row=0, col=1)

        self.input_frame.Label(text="Handle Missing Values :", row=1, col=0, size=9)
        self.input_frame.Label(text='Handle Outliers :', row=1, col=2, size=9)
        self.processesall.Label(text='Normalization :', row=0, col=1, size=9)
        self.input_frame.Label(text="Delete Attributes :", row=3, col=0, size=9)
        self.input_frame.Label(text="Discretization :", row=2, col=0, size=9)
        self.processesall.Label(text="Removes duplicate row :", row=1, col=1, size=9)

        self.input_frame.OptionMenu(self.option_missing_List, self.option_missing_Var, lambda x: print("Menu:", x),
                                    row=1, col=1)
        self.input_frame.OptionMenu(self.option_outliers_List, self.option_outliers_Var, lambda x: print("Menu:", x),
                                    row=1, col=3)
        self.processesall.OptionMenu(self.option_normalization_List, self.option_normalization_var,
                                     lambda x: print("Menu:", x),
                                     row=0, col=2)

        self.processesall.OptionMenu(self.option_redection_list, self.option_redection_var,
                                     lambda x: print("Menu:", x),
                                     row=1, col=2)

        self.input_frame.OptionMenu(self.option_delete_list, self.option_delete_var,
                                    lambda x: print("Menu:", x), row=3, col=1)

        self.input_frame.OptionMenu(self.option_descritization_list, self.option_descritization_var,
                                    lambda x: print("Menu:", x), row=2, col=1)

        self.input_frame.Label(text=" with K equal to :    ", row=2, col=2, size=9)

        self.input_frame.NumericalSpinbox(1, 100, 1, self.spinboxQ, row=2, col=3, padx=5)

        self.input_frame.Button(text="Run", command=self.preprocess_run, row=3, col=3, colspan=2)
        self.processesall.Button(text="Run", command=self.preprocessall_run, row=2, col=2, colspan=2)

    def preprocess_dataset3(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.create_data_display()
        self.create_menu_dataset3()

        self.option_pre_att_list = ["All"] + self.df.columns.tolist()
        self.option_prep_att_var = tk.StringVar(value=self.option_pre_att_list[0])
        self.option_descritization_list = ["Leave Values as Is", "Equal-Frequency", "Equal-width"]
        self.option_descritization_var = tk.StringVar(value=self.option_descritization_list[0])

        self.input_frame = self.addLabelFrame("Preprocess to each attribute")

        self.input_frame.Label(text="Selected attribute to preprocess :", row=0, col=0, size=9)
        self.input_frame.OptionMenu(self.option_pre_att_list, self.option_prep_att_var, lambda x: print("Menu:", x),
                                    row=0, col=1)

        self.spinboxQ = tk.IntVar(value=3)

        self.input_frame.Label(text="Discretization :", row=2, col=0, size=9)

        self.input_frame.OptionMenu(self.option_descritization_list, self.option_descritization_var,
                                    lambda x: print("Menu:", x), row=2, col=1)

        self.input_frame.Label(text=" with K equal to :    ", row=2, col=2, size=9)

        self.input_frame.NumericalSpinbox(1, 100, 1, self.spinboxQ, row=2, col=3, padx=5)

        self.input_frame.Button(text="Run", command=self.preprocessdataset3_run, row=3, col=3, colspan=2)

    def preprocess_missing_values(self, name_column):
        if self.option_missing_Var.get() == "Replace with Mod":
            self.df[name_column] = replace_missing_values(self.df[name_column], method="mod")
        elif self.option_missing_Var.get() == "Replace with Median":
            if type_attribute(self.df[name_column][0]) != 'str':
                self.df[name_column] = replace_missing_values(self.df[name_column], method="median")
            else:
                print("Methode invalid")
        elif self.option_missing_Var.get() == "Replace with Mean":
            if type_attribute(self.df[name_column][0]) != 'str':
                self.df[name_column] = replace_missing_values(self.df[name_column], method="mean")
            else:
                print("Methode invalid")
        elif self.option_missing_Var.get() == "Delete":
            data_type = type_attribute(self.df[name_column][0])
            if data_type != 'str':
                self.df[name_column] = pd.to_numeric(self.df[name_column], errors='coerce')
            else:
                self.df[name_column].replace([" ", "?"], np.nan, inplace=True)
            index_nan = self.df.loc[self.df[name_column].isna()].index
            self.df.drop(index_nan, inplace=True)

    def preprocess_outliers(self, name_column):
        if self.option_outliers_Var.get() == "Replace with Mean":
            self.df[name_column] = replace_outliers(self.df[name_column], method='mean')
        elif self.option_outliers_Var.get() == "Replace with Median":
            self.df[name_column] = replace_outliers(self.df[name_column], method='median')
        elif self.option_outliers_Var.get() == "Replace with Mod":
            self.df[name_column] = replace_outliers(self.df[name_column], method='mod')
        elif self.option_outliers_Var.get() == "Replace with IQR-Min":
            self.df[name_column] = replace_outliers(self.df[name_column], method='IQR-min')
        elif self.option_outliers_Var.get() == "Replace with IQR-Max":
            self.df[name_column] = replace_outliers(self.df[name_column], method='IQR-max')
        elif self.option_outliers_Var.get() == "Replace with NAN":
            self.df[name_column] = replace_outliers(self.df[name_column], method='null')
            self.preprocess_missing_values(name_column)
        elif self.option_outliers_Var.get() == "Delete":
            self.df[name_column] = replace_outliers(self.df[name_column], method='null')
            index_nan = self.df.loc[self.df[name_column].isna()].index
            self.df.drop(index_nan, inplace=True)

    def descritization_column(self, name_column):
        if self.option_descritization_var.get() == "Equal-Frequency":
            Q = self.spinboxQ.get()
            if Q == 3:
                interval = ['low', 'medium', 'high']
            elif Q == 4:
                interval = ['low', 'medium', 'high', 'very high']
            else:
                interval = None
            self.df[name_column] = equalFrequencyDiscretization(self.df[name_column], Q, interval)
        elif self.option_descritization_var.get() == "Equal-width":
            Q = self.spinboxQ.get()
            if Q == 3:
                interval = ['low', 'medium', 'high']
            elif Q == 4:
                interval = ['low', 'medium', 'high', 'very high']
            else:
                interval = None
            self.df[name_column] = equalWidthDiscretization(self.df[name_column], Q, interval)

    def normalize_data(self):
        if self.option_normalization_var.get() == "Min-Max":
            self.df.iloc[:, :-1] = min_max_normalisation(self.df.iloc[:, :-1], min=0, max=1)
        elif self.option_normalization_var.get() == 'Z-score':
            self.df.iloc[:, :-1] = z_score_normalisation(self.df.iloc[:, :-1])

    def preprocess_attribute(self, name_column):
        if self.option_delete_var.get() == "Yes":
            self.df.drop(name_column, axis=1, inplace=True)
            return
        self.preprocess_missing_values(name_column)
        self.preprocess_outliers(name_column)
        self.descritization_column(name_column)

    def remove_duplicate_row(self):
        if self.option_redection_var.get() == "Yes":
            self.df = delete_duplicate_rows(self.df)

    def preprocessall_run(self):
        self.remove_duplicate_row()
        self.normalize_data()
        self.preprocess_page()

    def preprocess_run(self):
        name_column = self.option_prep_att_var.get()
        if name_column == "All":
            for name_column in self.df.columns:
                self.preprocess_attribute(name_column)
        else:
            self.preprocess_attribute(name_column)
        self.preprocess_page()

    def preprocessdataset3_run(self):
        name_column = self.option_prep_att_var.get()
        if name_column == "All":
            for name_column in self.df.columns:
                self.descritization_column(name_column)
        else:
            self.descritization_column(name_column)
        self.preprocess_dataset3()

    def run_preprocessdataset2(self):
        name_column = self.option_prep_att_var.get()
        if name_column == "All":
            for name_column in self.df.columns:
                self.preprocess_missing_values(name_column)
                self.preprocess_outliers(name_column)
        else:
            self.preprocess_missing_values(name_column)
            self.preprocess_outliers(name_column)
        self.manipulate_dataset2()

    def create_model_menu(self):
        self.menu = self.addLabelFrame("Menu", row=0, col=1)
        self.menu.analyse = self.menu.addLabelFrame("Analyze and Visualize data", row=0, col=0)
        #       self.menu.analyse.Button(text="Welcome page", command=self.welcome, row=0, col=0)
        self.menu.analyse.Button(text="Welcome page", command=self.welcome_page, row=0, col=0)
        self.menu.analyse.Button(text="Manipulate data", command=self.preprocess_page, row=0, col=1)
        self.menu.analyse.Button(text="Preprocess data", command=self.preprocess_page, row=0, col=2)

        self.menu.supervised = self.menu.addLabelFrame("Supervised model", row=1)
        self.menu.supervised.Button(text="KNN", command=self.knn_button, row=2, col=0)
        self.menu.supervised.Button(text="Random forest", command=self.random_forest_button, row=2, col=1)
        self.menu.supervised.Button(text="Tree decision", command=self.desicion_tree_button, row=2, col=2)

        self.menu.unsupervised = self.menu.addLabelFrame("UnsupervisedModel model", row=2)
        self.menu.unsupervised.Button(text="K-means", command=self.kmeans_button, row=2, col=1)
        self.menu.unsupervised.Button(text="DBSCAN", command=self.dbscam_button, row=2, col=2)

    def knn_button(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.create_data_display()
        self.create_model_menu()

        self.knn_frame = self.addLabelFrame("KNN algorithm", colspan=2, rowspan=2)
        self.knn_frame.parameter = self.knn_frame.addLabelFrame("Fixe parameter", row=0, col=0)

        self.split = tk.DoubleVar(value=0.2)
        self.knn_frame.parameter.Label(text="Test size:", row=0, col=0, size=9)
        self.knn_frame.parameter.NumericalSpinbox(0, 1, 0.1, self.split, row=0, col=1, padx=5)

        self.knn_frame.parameter.Label(text="Normalization", row=0, col=2, size=9)
        self.knn_frame.parameter.OptionMenu(self.option_normalization_List, self.option_normalization_var,
                                                      lambda x: print("Menu:", x), row=0, col=3)

        self.knn_frame.parameter.Label(text="Chose distance methode:", row=1, col=0, size=9)
        self.knn_frame.parameter.OptionMenu(self.option_distance_list, self.option_distance_var,
                                            lambda x: print("Menu:", x), row=1, col=1)
        self.p = tk.IntVar(value=2)
        self.knn_frame.parameter.Label(text="p:", row=1, col=2, size=9)
        self.knn_frame.parameter.NumericalSpinbox(1, 50, 1, self.p, row=1, col=3, padx=2)

        self.knn_frame.parameter.Label(text="K-visions:", row=2, col=0, size=9)
        self.kvoisin = tk.IntVar(value=3)
        self.knn_frame.parameter.NumericalSpinbox(0, len(self.df), 1, self.kvoisin, row=2, col=1, padx=5)
        self.knn_frame.parameter.Button(text="Run", command=self.run_knn, row=2, col=3)

    def run_knn(self):
        self.knn_frame.evaluation = self.knn_frame.addLabelFrame("Evaluation", row=0, col=1, rowspan=2)
        k = self.kvoisin.get()
        distance = self.option_distance_var.get()
        p = self.p.get()
        model = KNNClassifier(n_neighbors=k, distance=distance, p=p)
        data = self.df.to_numpy()
        X, y = data[:, :-1], data[:, -1].reshape(-1, 1)
        X = X.astype(float)
        y = y.astype(int)
        y = y.ravel()
        test_size = self.split.get()
        X = np.where(np.isnan(X), 0, X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        if self.option_normalization_var.get() == 'Min-Max':
            X_train = min_max_normalisation(pd.DataFrame(X_train)).to_numpy()
            X_test = min_max_normalisation(pd.DataFrame(X_test)).to_numpy()
        elif self.option_normalization_var.get() == 'Z-score':
            X_train = z_score_normalisation(pd.DataFrame(X_train)).to_numpy()
            X_test = z_score_normalisation(pd.DataFrame(X_test)).to_numpy()
        model.fit(X_train, y_train)
        self.evaluate_matrix, self.predections = model.evaluate(X_test, y_test)
        self.y_true = y_test
        self.knn_frame.evaluation.Treeview(columnnames=self.evaluate_matrix.columns.tolist(), col=0, colspan=2, row=0,
                                           columnwidths=[270 // self.evaluate_matrix.shape[1]] *
                                                        self.evaluate_matrix.shape[1],
                                           height=2,
                                           data=self.evaluate_matrix.to_dict(orient='records'), subentryname=None)
        self.knn_frame.evaluation.Button(text="show confusion matrix", command=self.show_matrix_button, row=1, col=1,
                                         padx=200)

    def show_matrix_button(self):
        matrix = np_to_pd_confusion_matrix(self.y_true, self.predections)
        showMatrixConfusion(self.theme, self.mode, True, True, matrix)

    def random_forest_button(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.create_data_display()
        self.create_model_menu()

        self.random_forest_frame = self.addLabelFrame("Random Forest algorithm", colspan=2, rowspan=2)
        self.random_forest_frame.parameter = self.random_forest_frame.addLabelFrame(" Fixe parameter ", row=0, col=0)

        self.split = tk.DoubleVar(value=0.2)
        self.random_forest_frame.parameter.Label(text="Test size:", row=0, col=0, size=9)
        self.random_forest_frame.parameter.NumericalSpinbox(0, 1, 0.1, self.split, row=0, col=1, padx=5)

        self.random_forest_frame.parameter.Label(text="Normalization", row=0, col=2, size=9)
        self.random_forest_frame.parameter.OptionMenu(self.option_normalization_List, self.option_normalization_var,
                                               lambda x: print("Menu:", x), row=0, col=3)

        self.maxtree = tk.IntVar(value=2)
        self.random_forest_frame.parameter.Label(text="Max tree:", row=1, col=0, size=9)
        self.random_forest_frame.parameter.NumericalSpinbox(0, len(self.df), 1, self.maxtree, row=1, col=1, padx=5)

        self.max_depth = tk.IntVar(value=2)
        self.random_forest_frame.parameter.Label(text="Max depth:", row=1, col=2, size=9)
        self.random_forest_frame.parameter.NumericalSpinbox(0, len(self.df), 1, self.max_depth, row=1, col=3, padx=5)

        self.min_samples_split = tk.IntVar(value=2)
        self.random_forest_frame.parameter.Label(text="Min samples split:", row=2, col=0, size=9)
        self.random_forest_frame.parameter.NumericalSpinbox(0, len(self.df), 1, self.min_samples_split, row=2, col=1, padx=5)

        self.random_forest_frame.parameter.Button(text="Run", command=self.rundom_forest_run, row=2, col=3)

    def desicion_tree_button(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.create_data_display()
        self.create_model_menu()

        self.descision_tree_frame = self.addLabelFrame("Decision Tree algorithm", colspan=2, rowspan=2)
        self.descision_tree_frame.parameter = self.descision_tree_frame.addLabelFrame(" Fixe parameter ", row=0, col=0)

        self.split = tk.DoubleVar(value=0.2)
        self.descision_tree_frame.parameter.Label(text="Test size:", row=0, col=0, size=9)
        self.descision_tree_frame.parameter.NumericalSpinbox(0, 1, 0.1, self.split, row=0, col=1, padx=5)

        self.descision_tree_frame.parameter.Label(text="Normalization", row=0, col=2, size=9)
        self.descision_tree_frame.parameter.OptionMenu(self.option_normalization_List, self.option_normalization_var,
                                                      lambda x: print("Menu:", x), row=0, col=3)

        self.max_depth = tk.IntVar(value=2)
        self.descision_tree_frame.parameter.Label(text="Max depth:", row=1, col=0, size=9)
        self.descision_tree_frame.parameter.NumericalSpinbox(0, len(self.df), 1, self.max_depth, row=1, col=1, padx=5)

        self.min_samples_split = tk.IntVar(value=2)
        self.descision_tree_frame.parameter.Label(text="Min samples split:", row=1, col=2, size=9)
        self.descision_tree_frame.parameter.NumericalSpinbox(0, len(self.df), 1, self.min_samples_split, row=1, col=3,
                                                            padx=5)

        self.descision_tree_frame.parameter.Button(text="Run", command=self.run_descision_tree, row=2, col=3)

    def run_descision_tree(self):
        self.descision_tree_frame.evaluation = self.descision_tree_frame.addLabelFrame("Evaluation", row=0, col=1, rowspan=2)
        min_samples_split = self.min_samples_split.get()
        max_depth = self.max_depth.get()
        model = DecisionTree(min_samples_split, max_depth)
        data = self.df.to_numpy()
        X, y = data[:, :-1], data[:, -1].reshape(-1, 1)
        X = X.astype(float)
        y = y.astype(int)
        y = y.ravel()
        test_size = self.split.get()
        X = np.where(np.isnan(X), 0, X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        if self.option_normalization_var.get() == 'Min-Max':
            X_train = min_max_normalisation(pd.DataFrame(X_train)).to_numpy()
            X_test = min_max_normalisation(pd.DataFrame(X_test)).to_numpy()
        elif self.option_normalization_var.get() == 'Z-score':
            X_train = z_score_normalisation(pd.DataFrame(X_train)).to_numpy()
            X_test = z_score_normalisation(pd.DataFrame(X_test)).to_numpy()
        model.fit(X_train, y_train)
        self.evaluate_matrix, self.predections = model.evaluate(X_test, y_test)
        self.y_true = y_test
        self.descision_tree_frame.evaluation.Treeview(columnnames=self.evaluate_matrix.columns.tolist(), col=0, colspan=2, row=0,
                                           columnwidths=[270 // self.evaluate_matrix.shape[1]] *
                                                        self.evaluate_matrix.shape[1],
                                           height=2,
                                           data=self.evaluate_matrix.to_dict(orient='records'), subentryname=None)
        self.descision_tree_frame.evaluation.Button(text="show confusion matrix", command=self.show_matrix_button, row=1, col=1,
                                         padx=200)

    def rundom_forest_run(self):
        self.random_forest_frame.evaluation = self.random_forest_frame.addLabelFrame("Evaluation", row=0, col=1,
                                                                                       rowspan=2)
        min_samples_split = self.min_samples_split.get()
        max_depth = self.max_depth.get()
        max_tree = self.maxtree.get()
        model = RandomForest(max_tree, min_samples_split, max_depth)
        data = self.df.to_numpy()
        X, y = data[:, :-1], data[:, -1].reshape(-1, 1)
        X = X.astype(float)
        y = y.astype(int)
        y = y.ravel()
        test_size = self.split.get()
        X = np.where(np.isnan(X), 0, X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        if self.option_normalization_var.get() == 'Min-Max':
            X_train = min_max_normalisation(pd.DataFrame(X_train)).to_numpy()
            X_test = min_max_normalisation(pd.DataFrame(X_test)).to_numpy()
        elif self.option_normalization_var.get() == 'Z-score':
            X_train = z_score_normalisation(pd.DataFrame(X_train)).to_numpy()
            X_test = z_score_normalisation(pd.DataFrame(X_test)).to_numpy()
        forest = model.make_forest(pd.DataFrame(X_train), y_train)
        self.evaluate_matrix, self.predections = model.evaluate(forest, X_test, y_test)
        self.y_true = y_test
        self.random_forest_frame.evaluation.Treeview(columnnames=self.evaluate_matrix.columns.tolist(), col=0,
                                                      colspan=2, row=0,
                                                      columnwidths=[270 // self.evaluate_matrix.shape[1]] *
                                                                   self.evaluate_matrix.shape[1],
                                                      height=2,
                                                      data=self.evaluate_matrix.to_dict(orient='records'),
                                                      subentryname=None)
        self.random_forest_frame.evaluation.Button(text="show confusion matrix", command=self.show_matrix_button,
                                                    row=1, col=1,
                                                    padx=200)

    def apriori_button(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.create_data_display()
        self.create_menu_dataset3()

        self.apprioriFrame = self.addLabelFrame("Appriori Alogithme", colspan=2)
        self.apprioriFrame.parameter = self.apprioriFrame.addLabelFrame("Fixe parameter:", row=0, col=0)
        self.apprioriFrame.parameter.Label(text="Min sup :", row=0, col=0, size=9)
        self.apprioriFrame.parameter.Label(text="Min conf :", row=1, col=0, size=9)
        self.spinboxminconf = tk.DoubleVar(value=0.1)
        self.apprioriFrame.parameter.NumericalSpinbox(0, 1, 0.1, self.spinboxminconf, row=1, col=1, padx=5)
        self.spinboxminsup = tk.DoubleVar(value=0.1)
        self.apprioriFrame.parameter.NumericalSpinbox(0, 1, 0.1, self.spinboxminsup, row=0, col=1, padx=5)

        self.apprioriFrame.Button(text="Run", command=self.run_apriori, row=1, col=0)

    def run_apriori(self):
        self.apprioriFrame.reusult = self.apprioriFrame.addLabelFrame("Result", row=0, col=1)
        min_sup = self.spinboxminsup.get()
        min_conf = self.spinboxminconf.get()
        apriori = Apriori(data=self.df, min_conf=min_conf, min_sup=min_sup)
        self.df_result = apriori.fit()
        if type(self.df_result) == int:
            text = "There are no recommendations with these values of min-sup and min-conf. \n" \
                   "Please change it!"
            self.apprioriFrame.reusult.Label(text=text, row=1, col=0, size=17)
        else:
            self.apprioriFrame.reusult.Treeview(columnnames=self.df_result.columns.tolist(),
                                                columnwidths=[800 // self.df_result.shape[1]] * self.df_result.shape[1],
                                                height=2,
                                                data=self.df_result.to_dict(orient='records'), subentryname=None)

    def kmeans_button(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.create_data_display()
        self.create_model_menu()
        self.kmeans_frame = self.addLabelFrame("K-means algorithm", colspan=2)
        self.kmeans_frame.parameter = self.kmeans_frame.addLabelFrame(" Fixe parameter ", row=0, col=0)
        self.kmeans_frame.parameter.Label(text="Number of cluster:", row=0, col=0, size=9)
        self.n_cluster = tk.IntVar(value=2)
        self.kmeans_frame.parameter.NumericalSpinbox(0, len(self.df), 1, self.n_cluster, row=0, col=1, padx=5)
        self.kmeans_frame.parameter.Label(text="Methode initialisations:", row=0, col=2, size=9)
        self.kmeans_frame.parameter.OptionMenu(self.option_method_init_list, self.option_method_init_var,
                                               lambda x: print("Menu:", x), row=0, col=3)
        self.kmeans_frame.parameter.Label(text="Chose distance methode:", row=1, col=0, size=9)
        self.kmeans_frame.parameter.OptionMenu(self.option_distance_list, self.option_distance_var,
                                            lambda x: print("Menu:", x), row=1, col=1)
        self.kmeans_frame.parameter.Label(text="Number of initialization:", row=2, col=0, size=9)
        self.n_init = tk.IntVar(value=2)
        self.kmeans_frame.parameter.NumericalSpinbox(1, 50, 1, self.n_init, row=2, col=1, padx=2)
        self.p = tk.IntVar(value=2)
        self.kmeans_frame.parameter.Label(text="p:", row=1, col=2, size=9)
        self.kmeans_frame.parameter.NumericalSpinbox(1, 50, 1, self.p, row=1, col=3, padx=2)

        self.kmeans_frame.parameter.Button(text="Run", command=self.run_kmeans, row=2, col=3)

    def dbscam_button(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.create_data_display()
        self.create_model_menu()
        self.dbscan_frame = self.addLabelFrame("DBSCAN algorithm", colspan=2)
        self.dbscan_frame.parameter = self.dbscan_frame.addLabelFrame(" Fixe parameter ", row=0, col=0)
        self.dbscan_frame.parameter.Label(text="eps:", row=0, col=0, size=9)
        if self.df['N'][0] > 1:
            eps0 = 19.02
            pas = 0.01
            fin = 30
        else:
            eps0 = 0.38
            pas = 0.001
            fin = 1
        self.eps = tk.DoubleVar(value=eps0)
        self.dbscan_frame.parameter.NumericalSpinbox(0, fin, pas, self.eps, row=0, col=1, padx=5)
        self.dbscan_frame.parameter.Label(text="min samples:", row=0, col=2, size=9)
        self.min_samples = tk.IntVar(value=20)
        self.dbscan_frame.parameter.NumericalSpinbox(1, 50, 1, self.min_samples, row=0, col=3, padx=2)

        self.dbscan_frame.parameter.Label(text="Chose distance methode:", row=1, col=0, size=9)
        self.dbscan_frame.parameter.OptionMenu(self.option_distance_list, self.option_distance_var,
                                               lambda x: print("Menu:", x), row=1, col=1)

        self.p = tk.IntVar(value=2)
        self.dbscan_frame.parameter.Label(text="p:", row=1, col=2, size=9)
        self.dbscan_frame.parameter.NumericalSpinbox(1, 50, 1, self.p, row=1, col=3, padx=2)

        self.dbscan_frame.parameter.Button(text="Run", command=self.run_dbscan, row=2, col=3)

    def run_kmeans(self):
        self.kmeans_frame.evaluation = self.kmeans_frame.addLabelFrame("Evaluation", row=0, col=1, rowspan=2)
        k = self.n_cluster.get()
        p = self.p.get()
        n_init = self.n_init.get()
        distance = self.option_distance_var.get()
        if self.option_method_init_var.get() == 'Random':
            init = 'random'
        else: init = 'k-means++'
        model = KMeans(n_clusters=k, p=p, distance=distance, n_init=n_init, max_iter=50, init=init)
        X = self.df.to_numpy()
        X = X.astype(float)
        self.X = np.where(np.isnan(X), 0, X)
        model.fit(self.X)
        self.labels = model.labels_
        self.centroid = model.cluster_centers_
        self.evaluate_matrix = model.evaluate()
        self.type = 'K-means'
        self.kmeans_frame.evaluation.Treeview(columnnames=self.evaluate_matrix.columns.tolist(), col=0, colspan=2, row=0,
                                           columnwidths=[300 // self.evaluate_matrix.shape[1]] *
                                                        self.evaluate_matrix.shape[1],
                                           height=1,
                                           data=self.evaluate_matrix.to_dict(orient='records'), subentryname=None)
        self.kmeans_frame.evaluation.Button(text="Visualiz Clusters", command=self.visualize_clusters, row=1, col=1,
                                         padx=200)

    def run_dbscan(self):
        self.dbscan_frame.evaluation = self.dbscan_frame.addLabelFrame("Evaluation", row=0, col=1, rowspan=2)
        eps = self.eps.get()
        p = self.p.get()
        min_sample = self.min_samples.get()
        distance = self.option_distance_var.get()
        model = DBSCAN(eps=eps, min_samples=min_sample, distance=distance, p=p)
        X = self.df.to_numpy()
        X = X.astype(float)
        self.X = np.where(np.isnan(X), 0, X)
        model.fit(self.X)
        self.labels = model.labels_
        self.evaluate_matrix = model.evaluate()
        self.type = 'DBSCAN'
        self.dbscan_frame.evaluation.Treeview(columnnames=self.evaluate_matrix.columns.tolist(), col=0, colspan=2,
                                              row=0,
                                              columnwidths=[300 // self.evaluate_matrix.shape[1]] *
                                                           self.evaluate_matrix.shape[1],
                                              height=1,
                                              data=self.evaluate_matrix.to_dict(orient='records'), subentryname=None)
        self.dbscan_frame.evaluation.Button(text="Visualize Clusters", command=self.visualize_clusters, row=1, col=1,
                                            padx=200)

    def visualize_clusters(self):
        try:        visualizeClusters(self.theme, self.mode, True, True, self)
        except: print()





