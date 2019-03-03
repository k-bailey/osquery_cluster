import argparse
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as graph_obj
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA 
from sklearn.feature_selection import VarianceThreshold


def parse_data(filename, cron, kernel, pkg):
    df = pd.read_csv(filename)
    print("[+] Data successfully ingested")
    
    df_dict = []

    if pkg == True:
        dummy_df = df['package'].str.get_dummies()
        df_dict.append(dummy_df)
    if kernel == True:
        dummy_df_km = df['kernel_module'].str.get_dummies()
        df_dict.append(dummy_df_km)
    if cron == True:
        dummy_df_cron = df['cron_command'].str.get_dummies(sep='`')
        df_dict.append(dummy_df_cron)

    
    features_df = pd.concat(df_dict, axis=1, join_axes=[df.index])

    print("[+] Feature dataframe created of size:" + str(features_df.shape))
    return features_df, df


def cluster(features_df, csv_df, clusters):
    print("[+] Removing low variance features")
    sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
    features_df = sel.fit_transform(features_df)
    print("[+] Feature dataframe now of size: " + str(features_df.shape))

    print("[+] Starting K-means using " + clusters + " clusters: " )
    km = KMeans(n_clusters=int(clusters), init='k-means++', max_iter=100, n_init=1, verbose=True)
    km.fit(features_df)

    labels_km = km.labels_

    # build a df that makes it easy to see the cluster results
    csv_df['cluster_kmeans'] = labels_km
    columnsData = csv_df.loc[ : , ['host', 'os_major', 'os_minor', 'cluster_kmeans'] ]
    print("[+] Cluster results: ") 
    print(columnsData)

    print("[+] Number of hosts per cluster: ") 
    print(columnsData['cluster_kmeans'].value_counts())

    return columnsData

### This method has been depricated for plot_plotly but can be reused for matplotlib plots if desired
def plot(features_df, cluster_results_df):
    print("[+] Starting to plot") 

    pca = PCA(n_components=2).fit(features_df)
    pca_2d = pca.transform(features_df)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(17, 9))

    #plot kmeans by os minor
    x_pos, y_pos = pca_2d[:, 0], pca_2d[:, 1]
    plot_df_dbscan = pd.DataFrame(dict(x= x_pos, y= y_pos, label=cluster_results_df['cluster_kmeans'], title=cluster_results_df['os_minor']))
    groups = plot_df_dbscan.groupby('label')

    for name, group in groups:
        ax1.plot(group.x, group.y, marker='D', linestyle='solid', label=name)
        ax1.set_aspect('auto')
        ax1.tick_params(axis= 'x', which='both')
        ax1.tick_params(axis= 'y', which='both')
    ax1.legend(numpoints=1)
    ax1.set_title('Kmeans (OS Version)')

    # Make the labels on each point
    for i in range(len(plot_df_dbscan)):
        ax1.text(plot_df_dbscan.loc[i]['x'], plot_df_dbscan.loc[i]['y'], plot_df_dbscan.loc[i]['title'], size=9)


    #plot kmeans results
    x_pos, y_pos = pca_2d[:, 0], pca_2d[:, 1]
    plot_df_kmeans = pd.DataFrame(dict(x=x_pos, y=y_pos, label=cluster_results_df['cluster_kmeans'], title=cluster_results_df['host']))
    groups = plot_df_kmeans.groupby('label')

    for name, group in groups:
        ax2.plot(group.x, group.y, marker='D', label=name)
        ax2.set_aspect('auto')
        ax2.tick_params(axis= 'x', which='both')
        ax2.tick_params(axis= 'y', which='both')
    
    ax2.legend(numpoints=1)
    ax2.set_title('Kmeans (Hostname)')
    ax2.yaxis.tick_right()

    # Make the labels on each point
    for i in range(len(plot_df_kmeans)):
        ax2.text(plot_df_kmeans.loc[i]['x'], plot_df_kmeans.loc[i]['y'], plot_df_kmeans.loc[i]['title'], size=9)

    fig.tight_layout()
    print("[+] Rendering graphs... done. ") 
    plt.show()


def plot_plotly(features_df, cluster_results_df, filename):
    print("[+] Starting to plot") 

    pca = PCA(n_components=2).fit(features_df)
    pca_2d = pca.transform(features_df)

    x_pos, y_pos = pca_2d[:, 0], pca_2d[:, 1]
    plot_df_kmeans = pd.DataFrame(dict(x=x_pos, y=y_pos, 
        label=cluster_results_df['cluster_kmeans'], 
        host=cluster_results_df['host'], 
        os_major=cluster_results_df['os_major'], 
        os_minor=cluster_results_df['os_minor']),
    )
    plot_df_kmeans['hover_data'] = "Host: " + plot_df_kmeans["host"] + "; os_ver: " + plot_df_kmeans["os_major"].map(str) + "." + plot_df_kmeans["os_minor"].map(str) + "; kmeans_cluster: " + plot_df_kmeans["label"].map(str)

    l = []
    
    trace0 = graph_obj.Scatter(
        x = plot_df_kmeans['x'],
        y = plot_df_kmeans['y'],
        mode = 'markers+text',
        marker = dict(size = 28, line = dict(width=4), 
            color = plot_df_kmeans['label'], cauto=True, 
            colorscale='Hot', opacity = 0.8),
        name = "hostname", 
        text = plot_df_kmeans['label'],
        hovertext = plot_df_kmeans['hover_data'],
        hoverinfo = "text",
        unselected = dict(textfont=dict(color='white')),
        hoveron = 'points'
        )

    layout= graph_obj.Layout(title = ('Kmeans Clusters: ' + filename) , hovermode= 'closest', showlegend = False)

    data = [trace0]
    fig = graph_obj.Figure(data=data, layout=layout)
    plot_url = plotly.offline.plot(fig, filename='basic-scatter.html', auto_open=True)
    print('[*] Plot successful! raw HTML is located @ ' + plot_url)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', action="store_true", default=False, help='Use host package install features')
    parser.add_argument('-k', action="store_true", default=False, help='Use kernel module features')
    parser.add_argument('-c', action="store_true", default=False, help='Use crontab features')
    parser.add_argument('csv', help='The full or relative path to the CSV you would like to ingest')
    parser.add_argument('clusters', help='The number of clusters to feed to kmeans')
    args = parser.parse_args()

    if args.c==False and args.k==False and args.p==False:
        print("[!] No features were added... quitting. Please see --help")
        exit()

    feature_df, csv_df = parse_data(args.csv, args.c, args.k, args.p)
    cluster_results_df = cluster(feature_df, csv_df, args.clusters)

    ### This method has been depricated in favor of plot_plotly but can be reused for matplotlib plots if desired
    #plot(features_df, cluster_results_df)
    plot_plotly(feature_df, cluster_results_df, args.csv)


if __name__ == "__main__": main()


