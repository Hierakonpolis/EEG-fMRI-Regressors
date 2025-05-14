import numpy as np
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from scipy.signal import savgol_filter
from tqdm.autonotebook import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from .efp import mh_common_model, get_mh_multi_run_selected_channels_dataset, reshape_data_for_inference, \
    get_mh_single_run_selected_channels_dataset
import math
from sklearn.metrics import silhouette_score


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def calculate_wcss(D, labels, n_clusters):
    # Compute Within-Cluster Sum of Squares
    wcss = 0

    if len(D.shape) == 2 and D.shape[0] == D.shape[1]:
        for i in range(n_clusters):
            cluster_points = np.where(labels == i)[0]
            intra_cluster_distances = D[np.ix_(cluster_points, cluster_points)]
            wcss += np.sum(intra_cluster_distances) / (2 * len(cluster_points))
    else:
        for i in range(n_clusters):
            cluster_points = D[labels == i, :]
            centroid = cluster_points.mean(axis=0)
            wcss += np.sum((cluster_points - centroid) ** 2)
    return wcss


def projection_plot_clusters(distance_matrix, clusters, subjects, special_markers=()):
    if len(distance_matrix.shape) == 2 and distance_matrix.shape[1] == distance_matrix.shape[0]:
        print('Using MDS for cluster visualization')
        explainer = MDS(
            n_components=2,
            dissimilarity='precomputed',
            random_state=93
        )
    else:
        print('Using t-SNE for cluster visualization')
        # explainer = PCA(n_components=2)
        explainer = TSNE(n_components=2, random_state=93)

    coordinates = explainer.fit_transform(distance_matrix)

    colors = cm.get_cmap('tab20', np.max(clusters) + 1)
    point_colors = [colors(cluster) for cluster in clusters]

    plt.figure(figsize=(8, 6))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c=point_colors)
    extra_markers_x = []
    extra_markers_y = []

    # Annotate points
    for i, (x, y) in enumerate(coordinates):
        plt.text(x, y, f'{subjects[i]}', fontsize=12)
        if subjects[i] in special_markers:
            extra_markers_x.append(x)
            extra_markers_y.append(y)
        plt.scatter(extra_markers_x, extra_markers_y, marker='2', color='black')


def agglomerative_clustering(n_clusters, data,
                             subject_labels=None, linkage='average', metric='precomputed', plot=True,
                             special_markers=(),
                             ):
    if type(data) is list:
        data = np.stack([k.ravel() for k in data])
    if subject_labels is None:
        subject_labels = list(range(data.shape[0]))
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=metric,
        compute_full_tree=True,
        compute_distances=True,
        linkage=linkage,
    )
    clusters = clustering.fit_predict(data)
    if plot:
        plot_dendrogram(clustering, labels=subject_labels)
        projection_plot_clusters(data, clusters, subject_labels, special_markers)

    return calculate_wcss(data, clusters, n_clusters), clustering, clusters


def plot_elbow(data, linkage='average', metric='precomputed', show=True, show_vline=True, min_size=10):
    x = []
    y = []
    sizes = []
    if hasattr(data, 'shape'):
        n_samples = data.shape[0]
    else:
        n_samples = len(data)
    for n_clusters in range(1, n_samples):
        try:
            wcss, _, labels = agglomerative_clustering(n_clusters, data, None, linkage, metric, plot=False)
            biggest_cluster_size = max(list(labels).count(k) for k in labels)
            sizes.append(biggest_cluster_size)

            x.append(n_clusters)
            y.append(wcss)
        except ValueError as e:
            print(e)
            break

    x = np.array(x)
    y = np.array(y)
    sizes = np.array(sizes)
    for wl in reversed(range(3, 10)):
        try:
            y = savgol_filter(y, wl, 2)
            break
        except ValueError:
            continue

    first_derivative = np.diff(y) / np.diff(x)

    # Calculate the second derivative (discrete difference of the first derivative)
    second_derivative = np.diff(first_derivative) / np.diff(x[:-1])

    # Interpolate second_derivative to match the length of x

    second_derivative_full = np.zeros_like(x, dtype=float)
    second_derivative[0] = 0
    second_derivative_full[1:-1] = second_derivative
    second_derivative_full[0] = second_derivative[0]  # or np.nan, depending on how you want to handle boundaries
    second_derivative_full[-1] = second_derivative[-1]

    # Find the index of the maximum absolute second derivative
    optimal_index = np.argmax(np.abs(second_derivative_full[sizes >= min_size]))

    second_derivative_full = np.abs(second_derivative_full)
    y = y / y.max() * sizes.max()
    optimal_clusters = x[optimal_index] + 1

    if show:
        plt.plot(x, y, c='teal')
        plt.plot(x, sizes, c='red')
        if show_vline: plt.axvline(optimal_clusters, linestyle='--', c='purple')
    return optimal_clusters


def plot_elbow_and_select_clusters(data, linkage='average', metric='precomputed', show=True, show_vline=True,
                                   max_clusters=15):
    n_clusters = 2
    n_clusters_range = range(2, max_clusters + 1)
    x = []
    y = []
    scores_dbi = []
    scores_chi = []

    if hasattr(data, 'shape') and data.shape[0] == data.shape[1]:
        np.fill_diagonal(data, 0)
    else:
        data = [k.ravel() for k in data]

    for n_clusters in n_clusters_range:
        wcss, _, labels = agglomerative_clustering(n_clusters, data, None, linkage, metric, plot=False)
        dbi = davies_bouldin_score(data, labels)
        scores_dbi.append(dbi)
        chi = calinski_harabasz_score(data, labels)
        scores_chi.append(chi)
        optimal_dbi_clusters = n_clusters_range[np.argmin(scores_dbi)]
        optimal_chi_clusters = n_clusters_range[np.argmax(scores_chi)]
        x.append(n_clusters)
        y.append(wcss)

    if show:
        plt.plot(x, y, c='teal')
        if show_vline:
            plt.axvline(optimal_dbi_clusters, linestyle='--', c='purple', label='DBI')
            plt.axvline(optimal_chi_clusters, linestyle='--', c='orange', label='CHI')

    return (optimal_chi_clusters + optimal_chi_clusters) // 2


def greedy_max_min_selection(d_largest, n):
    # Step 1: Initialize with the pair having the smallest distance
    num_samples = d_largest.shape[0]
    best_pair = np.unravel_index(np.argmin(d_largest + np.eye(num_samples) * np.inf), d_largest.shape)
    selected_indices = list(best_pair)
    min_max_distance = np.max(d_largest[np.ix_(selected_indices, selected_indices)])
    # Step 2: Iteratively add samples
    while len(selected_indices) < n:
        remaining_indices = [i for i in range(num_samples) if i not in selected_indices]
        if len(remaining_indices) == 0:
            break
        best_candidate = None
        min_max_distance = np.inf

        for candidate in remaining_indices:
            new_subset = selected_indices + [candidate]
            max_distance = np.max(d_largest[np.ix_(new_subset, new_subset)])
            if max_distance < min_max_distance:
                min_max_distance = max_distance
                best_candidate = candidate

        selected_indices.append(best_candidate)

    return selected_indices, min_max_distance


def exact_max_min_selection(d_largest, n, disable_bar=True):
    best_combination = None
    min_max_distance = np.inf
    total_iters = math.comb(d_largest.shape[0], n)

    for comb in tqdm(combinations(range(d_largest.shape[0]), n), total=total_iters, disable=disable_bar):
        comb_distances = d_largest[np.ix_(comb, comb)]
        max_distance = np.max(comb_distances)
        if max_distance < min_max_distance:
            min_max_distance = max_distance
            best_combination = comb

    return list(best_combination), min_max_distance


def max_min_selection(d_largest, n, max_iter=2000000):
    total_iters = math.comb(d_largest.shape[0], n)
    if total_iters > max_iter:
        print(f"These look like a lot of iterations ({total_iters}), using greedy algorithm")
        return greedy_max_min_selection(d_largest, n)
    else:
        return exact_max_min_selection(d_largest, n)


def get_closest_n_elements(
        n, clustering_data, cluster_labels, subject_labels, plot=True, metric='euclidean', max_iter=2000000
):
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(counts)]
    indices_largest_cluster = np.where(cluster_labels == largest_cluster_label)[0]
    if len(indices_largest_cluster) < n:
        raise ValueError(f"Number of elements to select {n} larger than the number of elements"
                         f" in the largest cluster {len(indices_largest_cluster)}")

    # Handle vectors by calculating the distance matrix for visualization and selection
    if type(clustering_data) is list or clustering_data.shape[0] != clustering_data.shape[1]:
        if metric == 'l1':
            metric = 'cityblock'
        dist_matrix = squareform(pdist(np.stack([k.ravel() for k in clustering_data]), metric=metric))
        d_largest_cluster = dist_matrix[np.ix_(indices_largest_cluster, indices_largest_cluster)]
        clustering_data = np.stack([k.ravel() for k in clustering_data])
    else:
        d_largest_cluster = clustering_data[np.ix_(indices_largest_cluster, indices_largest_cluster)]

    selected_samples_in_cluster, max_min_distance = max_min_selection(d_largest_cluster, n, max_iter)
    selected_samples = indices_largest_cluster[selected_samples_in_cluster]
    selected_dataset_indices = [subject_labels[k] for k in selected_samples]

    if plot:
        projection_plot_clusters(
            clustering_data, cluster_labels, subject_labels, selected_dataset_indices
        )

    return selected_samples, max_min_distance, selected_dataset_indices


def get_trained_model_on_cluster(clustering_data, dataset, linkage, metric, band_sharing, subjects, channel,
                                 test_mode=False,
                                 delay_time_seconds=12,
                                 target_str=None,
                                 max_iterations=100000000,
                                 override_n_clusters=None,
                                 print_indices=False,
                                 stockwell_save_folder=None,
                                 window_type='kazemi',
                                 gamma=15,
                                 fmax=100,
                                 pca=None
                                 ):
    if stockwell_save_folder is None:
        raise ValueError("stockwell_save_folder must be specified")
    if override_n_clusters is None:
        n_clusters = plot_elbow(clustering_data, linkage=linkage, metric=metric, show=False)
    else:
        n_clusters = override_n_clusters
    wcss, _, labels = agglomerative_clustering(n_clusters, clustering_data, subjects, linkage=linkage, metric=metric,
                                               plot=False)
    n_to_select = 10
    done = False
    while n_to_select > 0 and not done:
        try:
            selected_samples, max_min_distance, selected_dataset_indices = get_closest_n_elements(
                n_to_select, clustering_data, labels, subjects, max_iter=max_iterations, metric=metric, plot=False)
            done = True
        except ValueError as e:
            print(e)
            n_to_select = n_to_select - 1
            max_min_distance = 0
    if band_sharing != 'from start':
        subsample_dataset = dataset.get_subsample(selected_dataset_indices)
        subsample_bands = subsample_dataset.get_dataset_shared_bands(fmax=fmax)
        subsample_dataset.get_mh_features_and_targets(
            nf_key_bold=('roimean', 'bgmean'),
            overwrite=True,
            included_channels=[channel], band_boundaries=subsample_bands,
            save_folder=stockwell_save_folder,
            window_type=window_type,
            gamma=gamma,
            disable_bar=True
        )
    else:
        subsample_dataset = dataset.get_subsample(selected_dataset_indices)
    common_model_results, estimator = mh_common_model(subsample_dataset, None, channel,
                                                      delay_time_seconds=delay_time_seconds,
                                                      train_only=test_mode,
                                                      target_str=target_str,
                                                      pca=pca
                                                      )
    training_subjects = [k['subj'] for k in subsample_dataset.dataset]
    if print_indices:
        print(selected_dataset_indices)
    if test_mode:
        assert target_str is not None
        test_data = dataset.get_subsample(selected_dataset_indices, opposite=True)
        results = []
        for sample_n in range(len(test_data)):
            if test_data[sample_n]['subj'] in training_subjects: continue
            test_sample = test_data[sample_n]
            test_samples, test_targets = get_mh_single_run_selected_channels_dataset(channel, delay_time_seconds,
                                                                                     target_str, test_sample,
                                                                                     pca=pca
                                                                                     )
            if test_samples.shape[-1] != test_targets.shape[-1]: continue
            test_predictions = estimator.predict(reshape_data_for_inference(test_samples, None))
            pearson_r, pearson_p = pearsonr(test_targets, test_predictions)
            results.append(
                {'pearson r': pearson_r, 'pearson p': pearson_p}
            )
        return results
    else:
        return common_model_results, estimator, wcss, n_clusters, n_to_select, selected_dataset_indices, max_min_distance
