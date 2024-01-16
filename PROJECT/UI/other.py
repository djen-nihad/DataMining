import TKinterModernThemes as TKMT
import numpy as np
import pandas as pd
import tkinter as tk

from sklearn.decomposition import PCA


class GrapheDataset2(TKMT.ThemedTKinterFrame):
    def __init__(self, theme, mode, usecommandlineargs=True, usethemeconfigfile=True, app=None):
        self.theme = theme
        self.mod = mode
        self.data = app.df
        self.app = app
        super().__init__("Graphe matpolib", theme, mode,
                         usecommandlineargs=usecommandlineargs, useconfigfile=usethemeconfigfile)

        self.graphframe = self.addLabelFrame(app.option_questions_var.get())
        if app.option_questions_var.get() == app.option_questions_list[4]:
            self.graphe_qst5()
        else:
            self.canvas, fig, self.ax, background, self.accent = self.graphframe.matplotlibFrame("Graph Frame Test",
                                                                                             toolbar=True,
                                                                                             figsize=(9, 5), row=1,
                                                                                             colspan=4)

        if app.option_questions_var.get() == app.option_questions_list[0]:
            self.graphe_qst1()
        elif app.option_questions_var.get() == app.option_questions_list[1]:
            self.graphe_qst2()
        elif app.option_questions_var.get() == app.option_questions_list[2]:
            self.graphe_qst3()
        elif app.option_questions_var.get() == app.option_questions_list[3]:
            self.graphe_qst4()

        elif app.option_questions_var.get() == app.option_questions_list[5]:
            self.graphe_qst6()

        self.run()

    def graphe_qst1(self):
        zone_totals = self.data.groupby('zcta')[['case count', 'positive tests']].sum()
        zone_totals.plot(kind='bar', figsize=(8, 6), ax=self.ax)
        self.ax.set_title('Distribution du nombre total de cas confirmés et tests positifs par zones')
        self.ax.set_xlabel('Zone')
        self.ax.set_ylabel('Nombre total')
        self.ax.legend(["Cas Confirmés", "Tests Positifs"])
        self.canvas.draw()

    def graphe_qst2(self):
        zone = int(self.app.option_area_Var.get())
        interval = self.app.options__interval_var.get()
        data = self.data.copy()
        Zone = data[data['zcta'] == zone]
        Zone.loc[:, 'Start date'] = pd.to_datetime(Zone['Start date'])
        # Set the 'date' column as the index
        Zone.set_index('Start date', inplace=True)
        # Resample data based on the specified interval
        if interval == 'weekly':
            interval_data = Zone.resample('W').sum()
        elif interval == 'monthly':
            interval_data = Zone.resample('M').sum()
        elif interval == 'yearly':
            interval_data = Zone.resample('Y').sum()
        else:
            raise ValueError("Invalid interval. Choose 'weekly', 'monthly', or 'yearly'.")

        self.ax.plot(interval_data.index, interval_data['test count'], label='Tests COVID-19', marker='o')
        self.ax.plot(interval_data.index, interval_data['positive tests'], label='Tests Positifs', marker='o')
        self.ax.plot(interval_data.index, interval_data['case count'], label='Nombre de Cas', marker='o')
        self.ax.set_title(f'Évolution des tests COVID-19, des tests positifs et du nombre de cas pour la zone {zone}')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Nombre')
        self.ax.legend()

    def graphe_qst3(self):
        def add_year_to_date(date_str):
            try:
                # Essayez de convertir la date directement
                return pd.to_datetime(date_str)
            except ValueError:
                # Si la conversion échoue, ajoutez une année et convertissez à nouveau
                date_with_year = date_str + '-2022'
                return pd.to_datetime(date_with_year, errors='coerce')

        # Appliquer la fonction à la colonne Start date
        self.data['Start date'] = self.data['Start date'].apply(add_year_to_date)
        # Extract the year from the 'date' column
        self.data['year'] = self.data['Start date'].dt.year

        # Group data by 'zone' and 'year' and sum the positive cases
        grouped_data = self.data.groupby(['zcta', 'year'])['positive tests'].sum().unstack()

        # Plotting
        grouped_data.plot(kind='bar', figsize=(10, 6), ax=self.ax)
        self.ax.set_title('Distribution des cas COVID-19 positifs par zone et par année')
        self.ax.set_xlabel('Zone')
        self.ax.set_ylabel('Nombre de Cas Positifs')
        self.ax.legend(title='Année')

    def graphe_qst4(self):
        region_sum = self.data.groupby('zcta')['test count'].sum()

        ratio_per_region = region_sum / self.data.groupby('zcta')['population'].max()

        # Tracer le graphique à barres
        ratio_per_region.plot(kind='bar', color='skyblue', edgecolor='black', ax=self.ax)
        self.ax.set_title('Rapport entre la population et le nombre de tests effectués par zone')
        self.ax.set_xlabel('Zone')
        self.ax.set_ylabel('Rapport population et tests')

    def graphe_qst5(self):
        # Regroupez les données par 'zone' et faites la somme des tests positifs durant toutes les années
        grouped_data = self.data.groupby('zcta')['positive tests'].sum().reset_index()
        # Triez le dataframe par 'positive tests' de manière décroissante
        zones_plus = grouped_data.sort_values(by='positive tests', ascending=False).head(5)
        txt = " Les 5 zones les plus fortement impactées par le coronavirus :" + str(zones_plus[['zcta', 'positive tests']])
        self.graphframe.Label(text=txt, row=1, col=2, size=17)


    def graphe_qst6(self):
        time_period = self.app.period_var.get()
        selected_data = self.data[self.data['time_period'] == time_period]

        grouped_data = selected_data.groupby(['zcta']).agg({
            'case count': 'sum',
            'test count': 'sum',
            'positive tests': 'sum'
        }).reset_index()

        grouped_data.set_index('zcta').plot(kind='bar', ax=self.ax)
        self.ax.set_title(f'Rapport entre les cas confirmés, les tests effectués et les tests positifs par zone ({time_period})')
        self.ax.set_xlabel('Zone')
        self.ax.set_ylabel('Numbre')




class Show_Graph(TKMT.ThemedTKinterFrame):
    def __init__(self, theme, mode, usecommandlineargs=True, usethemeconfigfile=True, df=None, type=None):
        self.theme = theme
        self.mod = mode
        self.df = df
        super().__init__("Matplotlib Graphe", theme, mode,
                         usecommandlineargs=usecommandlineargs, useconfigfile=usethemeconfigfile)
        text = str(type) + " for each attribute"
        self.graphframe = self.addLabelFrame(text)
        i = j = 0
        for column in self.df.columns:
            self.canvas, fig, self.ax, background, self.accent = self.graphframe.matplotlibFrame("Graph Frame Test",
                                                                                                 toolbar=False,
                                                                                                 figsize=(10, 10),
                                                                                                 row=i, col=j)

            if type == "BOX-Plot":
                self.ax.boxplot(self.df[column])
            else:
                self.ax.hist(self.df[column], bins=10, color='skyblue', edgecolor='black')
            self.ax.set_title("Attribute :" + column)
            self.canvas.draw()
            j = j + 1
            if j > 5:
                j = 0
                i = i + 1

        self.run()


class showMatrixConfusion(TKMT.ThemedTKinterFrame):
    def __init__(self, theme, mode, usecommandlineargs=True, usethemeconfigfile=True, matrix=None):
        self.theme = theme
        self.mod = mode
        super().__init__("Confusion matrix", theme, mode,
                         usecommandlineargs=usecommandlineargs, useconfigfile=usethemeconfigfile)
        if matrix is None:
            print('There is no matrix here !')
        else:
            self.matrix = matrix
            self.label = self.addLabelFrame("Display confuion matrix", row=0, rowspan=2)

            self.label.Treeview(columnnames=self.matrix.columns.tolist(), col=0, colspan=2, row=0,
                                columnwidths=[600 // self.matrix.shape[1]] * self.matrix.shape[1],
                                height=2,
                                data=self.matrix.to_dict(orient='records'), subentryname=None)
        self.run()


class visualizeClusters(TKMT.ThemedTKinterFrame):
    def __init__(self, theme, mode, usecommandlineargs=True, usethemeconfigfile=True, app=None):
        self.theme = theme
        self.mod = mode
        self.labels = app.labels
        self.X = app.X
        super().__init__("Clusters Visualizations", theme, mode,
                         usecommandlineargs=usecommandlineargs, useconfigfile=usethemeconfigfile)

        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(self.X)


        self.graphframe = self.addLabelFrame("Cluster dataset 1")
        if app.type == 'DBSCAN':
            n_cluster = len(np.unique(self.labels))
            noise = np.count_nonzero(self.labels == -1)
            text = 'Number cluster = ' + str(n_cluster) + '           Number of noise = ' + str(noise)
            self.graphframe.Label(text=text, size=17)
        self.canvas, fig, self.ax, background, self.accent = self.graphframe.matplotlibFrame("Graph Frame Test",
                                                                                             toolbar=True,
                                                                                             figsize=(9, 5), row=1,
                                                                                             colspan=4)

        self.ax.scatter(data_pca[:, 0], data_pca[:, 1], c=self.labels, cmap='viridis', s=30, label='Data Points')
        if app.type == 'K-means':
            self.centroid = app.centroid
            centroids = np.array(self.centroid)
            centroids_pca = pca.transform(centroids)
            self.ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, label='Centroids')
        self.ax.set_title('Clusters Visualizations ')
        self.ax.set_xlabel('PCA Component 1')
        self.ax.set_ylabel('PCA Component 2')
        self.canvas.draw()
        self.run()


