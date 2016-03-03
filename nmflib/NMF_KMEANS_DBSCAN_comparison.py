import numpy as np
import math
from sklearn import datasets
from sklearn import decomposition
from sklearn import cluster
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
from cluster import NMFClustering

def plot_gng_pca(X, centroids, datapoint_cluster_index, pc=np.array([0, 1]), title="Students", show_centroids=True,
                 save_pdf=False, save_title='plot.pdf'):

    #colors = cm.rainbow(np.linspace(0, 1, 15))
    colors = []
    for c in mpl_colors.cnames:
        colors.append(c)

    #colors = mpl_colors.cnames #['r','g','b','k','c','b']
    gs2 = gridspec.GridSpec(1, 1)
    fig = plt.figure(facecolor='white', edgecolor='white')
    ax = fig.add_subplot(gs2[0])

    datapoint_colors = []
    for i in range(len(datapoint_cluster_index)):
        cluster_index = datapoint_cluster_index[i][1]
        #datapoint_colors.append(get_color(cluster_index))
        datapoint_colors.append(colors[cluster_index])
    #ax2.scatter(data[:, pc[0]], data[:, pc[1]], zs=data[:, pc[2]], marker='x', s=2, color=datapoint_colors, picker=True)
    ax.scatter(X[:, pc[0]], X[:, pc[1]], s=2, color=datapoint_colors)

    if show_centroids:
        # plot centroids
        ax.scatter(centroids[:, pc[0]], centroids[:, pc[1]], s=10, color='r')
        # plot cluster centers
        #for center_iter in range(num_clusters):
        #    center = cluster_centers[center_iter]
        #    ax2.plot([center[pc[0]]], [center[pc[1]]], [center[pc[2]]], marker='o', markersize=8, color=get_color(center_iter))

    ax.set_xlabel('Principle Component {0}'.format(pc[0]+1))
    ax.set_ylabel('Principle Component {0}'.format(pc[1]+1))
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_zticks([])
    plt.title(title)
    plt.gca().grid(False)
    if save_pdf:
        plt.savefig(save_title, format="pdf")
    plt.show()


def plot_scatter2d(x, labels, centres, num_clusters, title="", axis_titles=["PC1", "PC2"], save_fig=False, fig_file_name='fig.pdf'):
    gs1 = gridspec.GridSpec(1, 1)
    fig = plt.figure(facecolor='white', edgecolor='white')
    ax = fig.add_subplot(gs1[0])

    colors = []
    for c in mpl_colors.cnames:
        colors.append(c)
    datapoint_colors = []
    for label_idx in labels:
        datapoint_colors.append(colors[label_idx])
    #ax2.scatter(data[:, pc[0]], data[:, pc[1]], zs=data[:, pc[2]], marker='x', s=2, color=datapoint_colors, picker=True)
    #ax.scatter(X[:, pc[0]], X[:, pc[1]], s=2, color=datapoint_colors)



    #ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # converts x axis background from grey to white
    #ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # converts y axis background from grey to white

    #ax.set_xlim3d(0, 10)  # this is to specify the min and max on the x axis
    #ax.set_ylim3d(0, 10)  # this is to specify the min and max on the Y axis
    #ax.set_zlim3d(0, 10)  # this is to specify the min and max on the Z axis

    ax.set_xlabel(axis_titles[0])
    ax.set_ylabel(axis_titles[1])

    ax.scatter(x[:, 0], x[:, 1], s=1, color=datapoint_colors)

    if centres is not None:
        ax.scatter(centres[:, 0], centres[:, 1], s=30, marker='x',color='black')

    plt.title(title)
    plt.gca().grid(False)
    if save_fig == True:
        plt.savefig(fig_file_name, format="pdf")
    plt.show()


def normalize_features2(patches):

    temp_data = np.array(patches)

    ncols = temp_data.shape[1]

    printables = np.zeros([ncols, ncols*2])

    #for i in range(ncols):
    #    mu = np.mean(temp_data[:, i])
    #    sigma = np.std(temp_data[:, i])
    #    printables[i, 0] = mu
    #    printables[i, 1] = sigma
    #    print "{0}  & {1}     & {2}".format(i+1, mu, sigma)
    #    print "mean col: {0} mean val: {1}".format(i, np.mean(temp_data[:,i]))
    #    print "std col: {0} std val: {1}".format(i, np.std(temp_data[:,i]))

    # normalize
    mean1_ = np.mean(temp_data, axis=0)
    std1_ = np.std(temp_data, axis=0)

    temp_data -= mean1_
    temp_data /= std1_

    ## calculate
    for i in range(ncols):
        mu = np.mean(temp_data[:, i])
        sigma = np.std(temp_data[:, i])
        printables[i, 2] = mu
        printables[i, 3] = sigma

    for i in range(ncols):
        print "{0}  & {1}     & {2}     & {3}   & {4}".format(i+1, printables[i, 0], printables[i, 1],
                                                              printables[i, 2], printables[i, 3])
    return temp_data


def print_pca_variance(pca):
    pc_ratios = []
    for ratio in pca.explained_variance_ratio_:
        pc_ratios.append(int(ratio * 10000) / 100.0)
    print "Retained PCA variances for PC1, PC2, PC3, PCn...(%): {0}".format(pc_ratios)
    return pc_ratios

def run_pca_with_variance(data, num_princomps=2):
    pca = decomposition.PCA(n_components=num_princomps) ## specify 0 < x < 1 percentage variance required.
    # Will return components up to the variance specified
    pca = pca.fit(data)
    data_reduced = pca.transform(data)

    print_pca_variance(pca) # %age variance for each principle component

    return pca, data_reduced

def get_pca_variance(pca, princomps):
    pc_ratios = []
    for princomp in princomps:
        ratio = pca.explained_variance_ratio_[princomp]
        pc_ratios.append(int(ratio * 10000) / 100.0)
    #for ratio in pca.explained_variance_ratio_:
    #    pc_ratios.append(int(ratio * 10000) / 100.0)
    pc_ratios = np.array(pc_ratios)
    return pc_ratios.sum()

def calc_centres(X, labels):
    centres = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        datapoints = X[mask]
        means = np.mean(datapoints, axis=0)
        centres.append(means)
    return np.array(centres)

def test_hc(X, n_components):

    return

def test_dbscan(X):
    db = cluster.DBSCAN(eps=0.5, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    centres = calc_centres(X, labels)
    return labels, centres

def test_kmeans(X, n_components):
    c = cluster.MiniBatchKMeans(n_clusters=n_components)
    res = c.fit(X)
    labels = res.predict(X)
    return labels, res.cluster_centers_

def test_nmf(X, n_components):

    obj = NMFClustering(n_components, "nmf") #, maxiter=50
    labels, result = obj.fit_predict(X)

    W = result.matrices[0]
    H = result.matrices[1]

    # calc centres
    centres = calc_centres(X, labels)

    #obj = NMFClustering(n_components, "nmf")
    #obj = NMFClustering(n_components, "spectral")
    #obj = NMFClustering(n_components, "projective")
    #obj = NMFClustering(n_components, "sparse")
    #obj = NMFClustering(n_components, "cluster")

    return labels, centres, W, H


def main():
    # generate test data
    n_samples = 10000
    n_components=10
    n_features=20
    K = 10
    input_data, labels = datasets.make_blobs(n_samples=n_samples, centers=n_components, cluster_std=0.5, n_features=n_features, random_state=6) #random_state=8,

    normed_data = normalize_features2(input_data)
    normed_data += math.fabs(normed_data.min())
    print input_data.shape

    dbscan_labels, dbscan_centres = test_dbscan(normed_data)
    #print "end dbscan"
    kmeans_labels, kmeans_centres = test_kmeans(normed_data, K)
    print "end kmeans"
    normed_data2 = np.array(normed_data, copy=True)
    nmf_labels, nmf_centres, W, H = test_nmf(normed_data2, K)
    print "end nmf"

    pca, X = run_pca_with_variance(normed_data, num_princomps=2)

    pca_kmeans_centres = pca.transform(kmeans_centres)
    pca_nmf_centres = pca.transform(nmf_centres)
    pca_dbscan_centres = pca.transform(dbscan_centres)

    plot_scatter2d(X, kmeans_labels, pca_kmeans_centres, n_components, "Mini-batch KMeans visualisation", save_fig=True, fig_file_name='pca_kmeans.pdf')  # plot the data in a scatter graph
    plot_scatter2d(X, nmf_labels, pca_nmf_centres, n_components, "NMF visualisation", save_fig=True, fig_file_name='pca_nmf.pdf')  # plot the data in a scatter graph
    plot_scatter2d(X, dbscan_labels, pca_dbscan_centres, n_components, "DBSCAN visualisation", save_fig=True, fig_file_name='pca_dbscan.pdf')  # plot the data in a scatter graph

    # FAST NMF LIbarary -- also spark
    # http://stackoverflow.com/questions/13814907/is-there-good-library-to-do-nmf-fast



if __name__ == "__main__":
    main()