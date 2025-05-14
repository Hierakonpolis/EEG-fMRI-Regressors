from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt
import mne
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pandas as pd


def standardize(x, axis=None):
    return (x - x.mean(axis=axis, keepdims=True)) / x.std(axis=axis, keepdims=True)


def cubic_spline_interpolator(signal, source_f, target_f, axis=-1):
    interpolator = CubicSpline(np.arange(0, signal.shape[axis]), signal, axis=axis)
    target_grid = np.arange(0, signal.shape[axis], source_f / target_f)
    return interpolator(target_grid)


def plot_on_topomap(electrode_scores, subject_data, vlim=None, order=None, axes=None, plot_bar=True):
    if vlim is None:
        vlim = (min(electrode_scores), max(electrode_scores))
    if type(electrode_scores) is pd.core.series.Series and order is None:
        print('Using order based on dataframe channel info')
        order = list(electrode_scores.index)
    if order is not None:
        order_mapping = {o: e for o, e in zip(order, electrode_scores)}
        electrode_scores = [order_mapping[label] for label in subject_data['mh_ch_names']]
    if axes is None:
        fig, ax = plt.subplots()
    else:
        ax = axes
    im, _ = mne.viz.plot_topomap(
        electrode_scores,
        subject_data['eeg_data'].info,
        axes=ax,
        size=4,
        names=subject_data['mh_ch_names'],
        show=False,
        cmap='viridis',
        vlim=vlim,
    )
    # Create the colorbar
    if plot_bar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    if axes is None:
        plt.show()


def select_grid_results_by_metric(
        grid_search_results, dataset, corr_cutoff=0.55, selection_metric='pearson val', plot=True, axes=None, vlim=None,
        plot_bar=True):
    included_subjects = []
    for subject in grid_search_results['subject'].unique():
        max_achieved_score = grid_search_results[grid_search_results['subject'] == subject][selection_metric].max()
        if max_achieved_score > corr_cutoff:
            included_subjects.append(subject)
    print(
        f'Subjects:\n{included_subjects}\nlen: {len(included_subjects)},'
        f' {np.round(len(included_subjects) / len(dataset), 3) * 100}% of the total.')
    selected_subjects_results = grid_search_results[[k in included_subjects for k in grid_search_results['subject']]]
    selected_subjects_results_grouped = selected_subjects_results.groupby('channel').mean()
    channel_name = selected_subjects_results_grouped.index[
        np.argmax(selected_subjects_results_grouped['pearson r test'])]
    print(f'Max mean correlation channel: {channel_name}')
    if plot:
        plot_on_topomap(selected_subjects_results_grouped['pearson r test'], dataset[0], axes=axes, vlim=vlim,
                        plot_bar=plot_bar)
    return included_subjects, channel_name


def plot_channel_coefficients(
        included_subjects,
        dataset,
        channel_name,
        models,
        n_bands=10,
        plotting_std_range=4,
):
    all_coefficients = []
    channels = []
    if type(models) is dict:
        for subject in included_subjects:
            channel = np.argmax([k == channel_name for k in dataset.dataset[subject]['mh_ch_names']])
            channels.append(channel)
            model = models[subject][channel]
            all_coefficients.append(model.coef_.reshape(n_bands, -1))

        channels = set(channels)
        if len(channels) > 1:
            print(f"Warning! Channel positions are inconsistent! {channels}")
        all_coefficients = np.stack(all_coefficients).mean(0)
    else:
        all_coefficients = models.coef_.reshape(n_bands, -1)
    ax = sns.heatmap(
        all_coefficients,
        cmap='vlag',
        vmin=0 - all_coefficients.std() * plotting_std_range,
        vmax=0 + all_coefficients.std() * plotting_std_range
    )
    current_labels = ax.get_xticklabels()
    new_labels = [(float(label.get_text()) - all_coefficients.shape[1]) / dataset.dataset[0]['f_features'] for
                  label in current_labels]
    _ = ax.set_xticklabels(new_labels)
