import copy, pickle
from stockwell import st
import time

import numpy as np
import os
import contextlib
import mne
import tqdm
from datasets.utils import cubic_spline_interpolator, standardize
from scipy.interpolate import interpn, RegularGridInterpolator
from itertools import product
from scipy.stats import pearsonr


def get_band_centers(band_boundaries_dict):
    band_centers = {}
    for channel, band_boundaries in band_boundaries_dict.items():
        band_boundaries = np.array(band_boundaries)
        band_centers[channel] = (band_boundaries[1:] + band_boundaries[:-1]) / 2
    return band_centers


def get_band_boundaries(sample, num_bands, fmin, fmax):
    psd = sample.compute_psd(fmin=fmin, fmax=fmax)
    psd_data = psd.get_data()
    band_boundaries = {}

    for channel in range(psd.shape[0]):
        log_psd = np.log10(psd_data[channel, :])
        log_psd = log_psd - log_psd.min()
        cumulative_energy = np.cumsum(log_psd)
        energy_per_band = cumulative_energy[-1] / num_bands
        band_boundaries[channel] = [0]
        for i in range(1, num_bands):
            boundary_index = np.argmin(np.abs(cumulative_energy - i * energy_per_band))
            band_boundaries[channel].append(psd.freqs[boundary_index])
        band_boundaries[channel].append(psd.freqs[-1])
    return band_boundaries


def get_padded_signal(data, start_index, chunk_size, padding_size):
    lenght = data.shape[-1]

    left_space = start_index
    left_take = np.min([left_space, padding_size])
    right_space = lenght - start_index - chunk_size
    right_take = np.min([right_space, padding_size])

    left_pad = data[:, :, start_index - left_take: start_index]
    right_pad = data[:, :, start_index + chunk_size: start_index + chunk_size + right_take]

    selected_data = data[:, :, start_index: start_index + chunk_size]
    padded_data = np.concatenate(
        [left_pad, selected_data, right_pad], axis=-1
    )
    padded_data = np.pad(
        padded_data,
        ((0, 0), (0, 0), (padding_size - left_take, padding_size - right_take)),
        'reflect'
    )
    return padded_data


def average_into_bands(stock_power, stock_freqs, band_boundaries, do_log=False):
    band_means = []
    band_freqs = []
    if len(stock_power.shape) == 2:
        stock_power = stock_power.reshape(1, stock_power.shape[0], stock_power.shape[1])
    for band_idx in range(len(band_boundaries) - 1):
        band_indexes = (stock_freqs >= band_boundaries[band_idx]) * (stock_freqs < band_boundaries[band_idx + 1])
        selected = stock_power[:, band_indexes, :]
        if do_log:
            selected[selected == 0] = selected[selected != 0].min()
            selected = np.log10(selected)
        band_means.append(
            selected.mean(axis=1).reshape([1, 1, -1])
        )
        band_freqs.append(
            stock_freqs[band_indexes].mean()
        )

    band_means = np.concatenate(band_means, axis=1)
    return band_means


def stockwell_full(
        array,
        time_array,
        fmin,
        fmax,
        freq_sampling_points=None,
        save_path=None,
        gamma=1,
        window_type='gauss'
):
    if save_path is not None and os.path.exists(save_path):
        try:
            result = pickle.load(open(save_path, 'rb'))
            if np.any(np.isinf(result[0])):
                print(f'Infinities found in {save_path}, recalculating')
                del result
            else:
                return result
        except Exception as e:
            print('Exception raised while attempting to open', save_path)
            print(e)
            print('Recalculating')
            # If the process was interrupted while saving, we get EOF error when opening
            # We just ignore it, recalculate, and overwrite.
            pass

    t = time_array.ravel()
    w = array.ravel()
    df = 1. / (t[-1] - t[0])  # sampling step in frequency domain (Hz)
    fmin_samples = int(fmin / df)
    fmax_samples = int(fmax / df)
    stock = st.st(w, fmin_samples, fmax_samples, gamma, win_type=window_type)
    stock = np.abs(stock)
    freqs = np.arange(stock.shape[0]) * df

    if freq_sampling_points:
        grid = np.linspace(fmin, fmax, freq_sampling_points + 1)
        stock = average_into_bands(stock, freqs, grid, True)
        freqs = grid[:-1]

    result = stock, freqs
    if save_path is not None:
        pickle.dump(result, open(save_path, 'wb'))
    return result


def get_stockwell_multichannel(sample,
                               fmax=60,
                               fmin=0,
                               f_features=4,
                               test_mode=False,
                               channel_names: list = None,
                               start_delay=None,
                               freq_sampling_points=500,
                               folder=None,
                               UID=None,
                               band_boundaries=None,
                               n_bands=None,
                               window_type='gauss',
                               gamma=1,
                               no_banding=False,
                               return_stats=False,
                               ):
    if start_delay is not None:
        time.sleep(start_delay)
    sample = sample.pick(['eeg'])
    ch_names = sample.ch_names
    sample.load_data()
    data = sample.get_data()
    assert len(data.shape) == 2
    channels = data.shape[0]
    included_channels = []
    if sample.info['sfreq'] <= fmax:
        fmax = sample.info['sfreq']

    channel_stockwells = []
    if band_boundaries is None and not no_banding:
        band_boundaries = get_band_boundaries(sample, n_bands, fmin, fmax)
    elif not no_banding:
        n_bands = len(band_boundaries)
    else:
        n_bands = None

    if test_mode:
        if type(test_mode) is str:
            test_mode = [test_mode]
        print('FEATURES ONLY COMPUTED FOR', test_mode)
        channels = [k for k in range(channels) if ch_names[k] in test_mode]
        ch_names = test_mode
    else:
        channels = range(channels)
    for channel, ch_name in zip(channels, ch_names):
        if channel_names is not None and ch_name not in channel_names:
            continue
        else:
            save_path = os.path.join(
                folder,
                f'{UID}{channel}{fmin}{fmax}{freq_sampling_points}{window_type}{gamma}.p'
            )

            stockwell, stock_frequencies = stockwell_full(
                data[channel, :],
                sample[channel][1],
                fmin,
                fmax,
                freq_sampling_points,
                save_path,
                window_type=window_type,
                gamma=gamma,
            )

            if n_bands is not None:
                stockwell = average_into_bands(stockwell, stock_frequencies, band_boundaries[channel], do_log=False)

            channel_stockwells.append(stockwell)
            included_channels.append(ch_name)

    channel_stockwells = np.concatenate(channel_stockwells, axis=0)
    channel_stockwells = cubic_spline_interpolator(channel_stockwells, sample.info['sfreq'], f_features)
    means = channel_stockwells.mean(axis=2)
    stds = channel_stockwells.std(axis=2)
    channel_stockwells = standardize(channel_stockwells, axis=2)
    if not return_stats:
        return channel_stockwells, included_channels, band_boundaries
    else:
        return channel_stockwells, included_channels, band_boundaries, means, stds


def get_mh_single_run_dataset(channel_features, targets, sample_time_window, f_features, pca=None):
    times_per_sample = int(sample_time_window * f_features)
    sample_shape = (channel_features.shape[0], channel_features.shape[1], times_per_sample, 1)
    features = []
    for k in range(times_per_sample, channel_features.shape[-1]):
        features.append(
            channel_features[:, :, k - times_per_sample:k].reshape(sample_shape)
        )
    data = np.concatenate(features, axis=-1)
    time_len = data.shape[-1]
    if pca is not None:
        data = np.split(data, data.shape[0], 0)
        data = [pca.transform(k.reshape(-1, time_len).T).T for k in data]
        data = np.stack(data, axis=0)
        # will generate a shape of len 3 instead of 4

    return data, targets[times_per_sample:]


def get_shared_runs_boundaries(runs, num_bands=10, fmin=0, fmax=60):
    runs[0] = copy.deepcopy(runs[0])
    joined_raws = mne.concatenate_raws(
        runs
    )
    return get_band_boundaries(joined_raws, num_bands, fmin, fmax)


def interpolate_coefficients(coefficients_as_bands, source_band_centers, target_band_centers, method='linear'):
    interpolator = RegularGridInterpolator(
        (source_band_centers, np.arange(coefficients_as_bands.shape[-1])),
        coefficients_as_bands,
        method=method,
        bounds_error=False,
        fill_value=None,
    )

    return interpolator(
        np.array(list(product(target_band_centers,
                              np.arange(coefficients_as_bands.shape[-1]))))).reshape(
        -1, coefficients_as_bands.shape[-1]
    )


def mh_upsampled_grid(coefficients_as_bands, source_band_boundaries, resolution_hz, max_f):
    upsampled_grid = np.empty((int(max_f // resolution_hz), coefficients_as_bands.shape[-1]))
    for b, bandmax in enumerate(reversed(source_band_boundaries[1:])):
        upsampled_grid[:int(bandmax // resolution_hz), :] = coefficients_as_bands[-1 - b, :]
    return upsampled_grid


def mh_interpolate_coefficients(
        coefficients_as_bands, source_band_boundaries, target_band_boundaries, resolution_hz=.1):
    max_f = target_band_boundaries[-1]
    min_f = target_band_boundaries[0]
    upsampled_grid = mh_upsampled_grid(coefficients_as_bands, source_band_boundaries, resolution_hz, max_f)
    upsampled_grid_points = np.linspace(min_f, max_f, int(max_f // resolution_hz))
    resampled = np.empty((len(target_band_boundaries) - 1, coefficients_as_bands.shape[-1]))
    for b in range(1, len(source_band_boundaries)):
        positions = np.argwhere((upsampled_grid_points >= target_band_boundaries[b - 1]) * (
                upsampled_grid_points <= target_band_boundaries[b]))
        resampled[b - 1, :] = upsampled_grid[positions, :].mean(
            axis=0, keepdims=True)
    return resampled


def correlation_distance(a, b):
    return 1 - pearsonr(a, b)[0]


covariance_matrices_temp = {
    True: {},
    False: {}
}
covariance_matrices = {}


def get_distance_matrix(
        dataset, models, channel, shared_comparison_bands=None, method='mh', distance_metric=correlation_distance,
        fmin=0, fmax=60, num_bands=10, shared_channel_names=None, use_covariance=False, delay=12, pca=None
):
    def get_shared_if_exists(x):
        if shared_channel_names is None:
            return x
        else:
            return np.argmax([k == channel for k in shared_channel_names])
    if channel not in covariance_matrices:
        covariance_matrices[channel] = copy.deepcopy(covariance_matrices_temp)
    n_bands = len(dataset[0]['mh_band_centers'][0])
    if type(distance_metric) is list:
        distances = [np.ones((len(dataset), len(dataset))) * np.nan for _ in range(len(distance_metric))]
    else:
        distances = np.ones((len(dataset), len(dataset))) * np.nan
    resampled_runs = []

    if shared_comparison_bands is not None:
        shared_band_centers = get_band_centers(shared_comparison_bands)
        to_calculate_bands = False
    else:
        to_calculate_bands = True
    subj_index = list(models.keys())
    for subj_one in tqdm.tqdm(subj_index):
        if type(channel) is str:
            channel_id_1 = np.argmax([k == channel for k in dataset.dataset[subj_one]['mh_ch_names']])
        else:
            channel_id_1 = channel
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull):
                banded_coefficients_one = models[subj_one][channel_id_1].coef_.reshape(n_bands, -1)
                if method == 'mh' and not to_calculate_bands:
                    resampled_one = mh_interpolate_coefficients(
                        banded_coefficients_one,
                        dataset[subj_one]['mh_band_boundaries'][channel_id_1],
                        shared_comparison_bands[get_shared_if_exists(channel_id_1)]
                    )
                elif not to_calculate_bands:
                    resampled_one = interpolate_coefficients(
                        banded_coefficients_one,
                        dataset[subj_one]['mh_band_centers'][channel_id_1],
                        shared_band_centers[get_shared_if_exists(channel_id_1)]
                    )

                for subj_two in subj_index:
                    if type(channel) is str:
                        channel_id_2 = np.argmax([k == channel for k in dataset.dataset[subj_two]['mh_ch_names']])
                    else:
                        channel_id_2 = channel
                    if to_calculate_bands:
                        shared_comparison_bands = {}
                        shared_comparison_bands_1 = get_shared_runs_boundaries(
                            [dataset[subj_one]['eeg_data'], dataset[subj_two]['eeg_data']],
                            fmin=fmin, fmax=fmax, num_bands=num_bands)
                        shared_comparison_bands_2 = get_shared_runs_boundaries(
                            [dataset[subj_two]['eeg_data'], dataset[subj_one]['eeg_data']],
                            fmin=fmin, fmax=fmax, num_bands=num_bands)
                        for k in shared_comparison_bands_1.keys():
                            shared_comparison_bands[k] = list((np.array(shared_comparison_bands_1[k]) +
                                                               np.array(shared_comparison_bands_2[k])) / 2)
                        shared_band_centers = get_band_centers(shared_comparison_bands)

                    banded_coefficients_two = models[subj_two][channel_id_2].coef_.reshape(n_bands, -1)

                    if method == 'mh':
                        resampled_two = mh_interpolate_coefficients(
                            banded_coefficients_two,
                            dataset[subj_two]['mh_band_boundaries'][channel_id_2],
                            shared_comparison_bands[get_shared_if_exists(channel_id_2)]
                        )
                        if to_calculate_bands:
                            resampled_one = mh_interpolate_coefficients(
                                banded_coefficients_one,
                                dataset[subj_one]['mh_band_boundaries'][channel_id_1],
                                shared_comparison_bands[get_shared_if_exists(channel_id_1)]
                            )
                    else:
                        resampled_two = interpolate_coefficients(
                            banded_coefficients_two,
                            dataset[subj_two]['mh_band_centers'][channel_id_2],
                            shared_band_centers[get_shared_if_exists(channel_id_2)]
                        )
                        if to_calculate_bands:
                            resampled_one = interpolate_coefficients(
                                banded_coefficients_one,
                                dataset[subj_one]['mh_band_centers'][channel_id_1],
                                shared_band_centers[get_shared_if_exists(channel_id_1)]
                            )
                    if use_covariance:
                        if f'{subj_one}_{subj_two}' not in covariance_matrices[channel][to_calculate_bands]:
                            features_one, _ = get_mh_single_run_dataset(
                                dataset[subj_one]['mh_features'],
                                dataset[subj_one]['target_sma'],
                                delay,
                                dataset[subj_one]['f_features'],
                                pca=pca
                                # Select one channel based on channel name and MNE channel info
                            )
                            features_one = features_one[channel_id_1, ...]
                            features_one = features_one.reshape(features_one.shape[0] * features_one.shape[1], -1)

                            features_two, _ = get_mh_single_run_dataset(
                                dataset[subj_two]['mh_features'],
                                dataset[subj_two]['target_sma'],
                                delay,
                                dataset[subj_two]['f_features'],
                                pca=pca
                                # Select one channel based on channel name and MNE channel info
                            )
                            features_two = features_two[channel_id_1, ...]
                            features_two = features_two.reshape(features_two.shape[0] * features_two.shape[1], -1)
                            covariance_mat = np.cov(np.concatenate((features_one, features_two), axis=1))

                            covariance_matrices[channel][to_calculate_bands][f'{subj_one}_{subj_two}'] = covariance_mat
                            covariance_matrices[channel][to_calculate_bands][f'{subj_two}_{subj_one}'] = covariance_mat
                        else:
                            covariance_mat = covariance_matrices[channel][to_calculate_bands][f'{subj_one}_{subj_two}']
                    if type(distance_metric) is list:
                        for k in range(len(distance_metric)):
                            if use_covariance:
                                distances[k][subj_one, subj_two] = distance_metric[k](
                                    resampled_one.ravel(), resampled_two.ravel(), covariance_mat
                                )
                            else:
                                distances[k][subj_one, subj_two] = distance_metric[k](
                                    resampled_one.ravel(), resampled_two.ravel(),
                                )
                    else:
                        if use_covariance:
                            distances[subj_one, subj_two] = distance_metric(
                                resampled_one.ravel(), resampled_two.ravel(), covariance_mat
                            )
                        else:
                            distances[subj_one, subj_two] = distance_metric(
                                resampled_one.ravel(), resampled_two.ravel(),
                            )

        resampled_runs.append(resampled_one)
    missing = [k for k in range(len(dataset)) if k not in models]
    if type(distance_metric) is list:
        for k in range(len(distance_metric)):
            distances[k] = np.delete(distances[k], missing, axis=0)
            distances[k] = np.delete(distances[k], missing, axis=1)
    else:
        distances = np.delete(distances, missing, axis=0)
        distances = np.delete(distances, missing, axis=1)

    if to_calculate_bands:
        return distances, subj_index
    else:
        return distances, subj_index, resampled_runs


def get_corr_matrix_distances(
        dataset, models, distance_metric=None,
        matrix_type='correlation',  # correlation or covariance
        selected_channels=None,
        bands=False,
):
    if type(distance_metric) is list:
        distances = [np.ones((len(dataset), len(dataset))) * np.nan for _ in range(len(distance_metric))]
    else:
        distances = np.ones((len(dataset), len(dataset))) * np.nan
    matrices = {}

    subj_index = list(models.keys())
    for subj_one in tqdm.tqdm(subj_index, desc=f'Building {matrix_type} matrices'):
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull):
                if type(selected_channels) is str:
                    ch_index = np.argwhere(np.array(dataset[subj_one]['mh_ch_names']) == selected_channels).item()
                    eeg_data = dataset[subj_one]['mh_features'][ch_index, :, :]
                elif selected_channels is not None:
                    eeg_data = copy.deepcopy(dataset[subj_one]['eeg_data'])
                    eeg_data.drop_channels([c for c in eeg_data.ch_names if c.upper() not in selected_channels])
                    eeg_data = eeg_data.pick('eeg').get_data()
                else:
                    eeg_data = dataset[subj_one]['eeg_data']
                    eeg_data = eeg_data.pick('eeg').get_data()

                if matrix_type == 'correlation':
                    matrices[subj_one] = np.corrcoef(eeg_data)
                elif matrix_type == 'covariance':
                    matrices[subj_one] = np.cov(eeg_data)
                else:
                    raise ValueError('matrix_type must be either "correlation" or "covariance"')

    for subj_one in tqdm.tqdm(subj_index, desc='Calculating distance metrics'):
        for subj_two in subj_index:
            if type(distance_metric) is list:
                for k in range(len(distance_metric)):
                    distances[k][subj_one, subj_two] = distance_metric[k](
                        matrices[subj_one], matrices[subj_two],
                    )
            else:
                distances[subj_one, subj_two] = distance_metric(
                    matrices[subj_one], matrices[subj_two],
                )

    missing = [k for k in range(len(dataset)) if k not in models]
    if type(distance_metric) is list:
        for k in range(len(distance_metric)):
            distances[k] = np.delete(distances[k], missing, axis=0)
            distances[k] = np.delete(distances[k], missing, axis=1)
    else:
        distances = np.delete(distances, missing, axis=0)
        distances = np.delete(distances, missing, axis=1)

    return distances, missing
