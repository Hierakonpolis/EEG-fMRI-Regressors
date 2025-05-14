import os
import pickle

import numpy as np
import pandas as pd
import tqdm
from scipy import stats
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold, GridSearchCV

from datasets.mh_features import get_mh_single_run_dataset


class RidgeWithStats(Ridge):
    def fit(self, X, y, n_jobs=1):
        self = super(Ridge, self).fit(X, y, n_jobs)

        ssr = np.sum((self.predict(X) - y) ** 2)
        mse = ssr / (X.shape[0] - X.shape[1] - 1)

        se = np.array(
            np.sqrt(np.diagonal(mse * np.linalg.inv(np.dot(X.T, X))))
            # for i in range(mse.shape[0])
        )

        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        return self


class MaxCorrEstimator(BaseEstimator):
    def __init__(self):
        self.most_correlated_band = None
        self.is_fitted_ = False

    def fit(self, X, y, **kwargs):
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')
        correlations = [
            pearsonr(X[k, :], y)[0] for k in range(X.shape[0])
        ]
        self.most_correlated_band = np.argmax(correlations)
        self.is_fitted_ = True

    def predict(self, X):
        return X[self.most_correlated_band, :]


def pearson_scorer(estimator, X, y_true):
    return pearsonr(estimator.predict(X), y_true)[0]


def get_regularization_grid(data, grid_size):
    _, singular_values, _ = np.linalg.svd(data)
    smallest = singular_values.min()
    biggest = singular_values.max() ** 2

    grid = np.linspace(np.log(smallest), np.log(biggest), grid_size)
    grid = np.exp(grid)
    return {'alpha': grid}


def reshape_data_for_inference(data, channel):
    if len(data.shape) == 4:
        return data[channel, :, :, :].reshape(data.shape[1] * data.shape[2], data.shape[-1]).T
    elif len(data.shape) == 3:
        return data.reshape(data.shape[0] * data.shape[1], data.shape[-1]).T
    elif len(data.shape) == 2:
        return data.T
    else:
        raise ValueError(f'data shape {data.shape} not supported')


def train_efp_model(
        data,
        target,
        inner_k=5,
        outer_test_ratio=.2,
        regularization_grid_search=10,
        range_channels=None,
        metric='MSE',
        verbose=False,
        estimator=None,
):
    # CHANNEL, FREQ, DELAY, TIME
    channels, time = data.shape[0], data.shape[-1]
    inner_splitter = KFold(n_splits=inner_k, shuffle=False)
    test_set_index = int((time * (1 - outer_test_ratio)))
    train_data = data[..., :test_set_index]
    test_data = data[..., test_set_index:]
    # times_per_sample = int(sample_time_window * f_features)
    train_targets = target[:test_set_index]
    test_targets = target[test_set_index:]

    results = []
    models = []
    if estimator is None:
        estimator = Ridge

    if range_channels is None:
        range_channels = range(channels)

    for channel in tqdm.tqdm(range_channels, desc='Training a regressor for each channel', disable=(not verbose)):
        # Reshape it to be (FEATURES, TIME)
        grid_search = grid_search_with_alpha_grid(channel, estimator, inner_splitter, metric,
                                                  regularization_grid_search, train_data, train_targets)

        test_results = grid_search.predict(reshape_data_for_inference(test_data, channel))

        pearson_r, pearson_p = pearsonr(test_targets, test_results)
        MSE = mean_squared_error(test_targets, test_results)
        r2 = r2_score(test_targets, test_results)

        results.append({
            'channel': channel,
            'pearson r test': pearson_r,
            'pearson p test': pearson_p,
            'MSE test': MSE,
            'nMSE test': MSE / np.std(test_targets),
            'r2 test': r2,
            'MSE val': -grid_search.cv_results_['mean_test_MSE'].mean(),
            'r2 val': grid_search.cv_results_['mean_test_r2'].mean(),
            'pearson val': grid_search.cv_results_['mean_test_pearson'].mean(),
        })

        models.append(
            grid_search.best_estimator_
        )

    return results, models


def grid_search_with_alpha_grid(channel, estimator, inner_splitter, metric, regularization_grid_search, train_data,
                                train_targets, verbose=0):
    one_ch_data = reshape_data_for_inference(train_data, channel)
    if type(regularization_grid_search) is int:
        alpha_grid = get_regularization_grid(one_ch_data, regularization_grid_search)
    else:
        alpha_grid = regularization_grid_search
    grid_search = GridSearchCV(
        estimator=estimator(),
        param_grid=alpha_grid,
        cv=inner_splitter,
        verbose=verbose,
        scoring={
            'pearson': pearson_scorer,
            'MSE': make_scorer(mean_squared_error, greater_is_better=False),
            'r2': make_scorer(r2_score)
        },
        refit=metric,
        n_jobs=-1,
    )
    grid_search.fit(one_ch_data, train_targets)
    return grid_search


def get_correlation_model(
        data,
        target,
        outer_test_ratio=.2,
        range_channels=None
):
    # CHANNEL, FREQ, DELAY, TIME
    channels, bands, time = data.shape
    test_set_index = int((time * (1 - outer_test_ratio)))
    train_data = data[:, :, :test_set_index]
    test_data = data[:, :, test_set_index:]
    # times_per_sample = int(sample_time_window * f_features)
    train_targets = target[:test_set_index]
    test_targets = target[test_set_index:]

    results = []
    models = []

    if range_channels is None:
        range_channels = range(channels)

    for channel in range_channels:
        estimator = MaxCorrEstimator()
        estimator.fit(train_data[channel, :, :], train_targets)

        test_results = estimator.predict(test_data[channel, :, :])

        pearson_r, pearson_p = pearsonr(test_targets, test_results)
        MSE = mean_squared_error(test_targets, test_results)
        r2 = r2_score(test_targets, test_results)

        results.append({
            'channel': channel,
            'pearson r test': pearson_r,
            'pearson p test': pearson_p,
            'MSE test': MSE,
            'r2 test': r2,
        })
        # models.append(estimator)

    return results, models


def mh_grid_search(
        dataset, search_save, models_save,
        target_str='target_m1',
        metric='MSE',
        test_ratio=.2,
        delay_time_seconds=12,
        feature_key='mh_features',
        regularization_grid_search=50,
        estimator=None,
        pca=None
):
    if os.path.isfile(search_save):
        all_grid_search_results = pickle.load(open(search_save, 'rb'))
        all_models = pickle.load(open(models_save, 'rb'))
    else:
        all_grid_search_results = []
        errors = {}
        all_models = {}
        for subject in tqdm.tqdm(range(len(dataset))):
            try:
                source_data, targets_data = get_mh_single_run_dataset(
                    dataset.dataset[subject][feature_key],
                    dataset.dataset[subject][target_str],
                    delay_time_seconds,
                    dataset.dataset[subject]['f_features'],
                    pca=pca
                )
                grid_search_results, all_models[subject] = train_efp_model(
                    source_data,
                    targets_data,
                    regularization_grid_search=regularization_grid_search,
                    metric=metric,
                    outer_test_ratio=test_ratio,
                    estimator=estimator
                )
                for k in grid_search_results:
                    k['subject'] = subject
                    k['channel'] = dataset.dataset[subject]['mh_ch_names'][k['channel']]
                all_grid_search_results.extend(grid_search_results)
            except Exception as e:
                errors[subject] = e
        all_grid_search_results = pd.DataFrame(all_grid_search_results)
        pickle.dump(all_grid_search_results, open(search_save, 'wb'))
        pickle.dump(all_models, open(models_save, 'wb'))
    grouped = all_grid_search_results.groupby('channel').mean()
    return all_grid_search_results, all_models, grouped


def mh_common_model(
        dataset, selected_runs, channel_name, target_str=None,
        metric='MSE', delay_time_seconds=12,
        regularization_grid_search: int = 50, inner_k: int = 5, estimator=None,
        train_only: bool = False,
        features_key=None,
        verbosity_final_fit=0,
        test_on_one=False,
        return_scores=False,
        pca=None,
):
    if features_key is None:
        features_key = 'mh_features'
    results = []
    inner_splitter = KFold(n_splits=inner_k, shuffle=False)
    if estimator is None:
        estimator = Ridge
    if selected_runs is not None:
        selected_samples = dataset.get_subsample(selected_runs, copy_data=False)
    else:
        selected_samples = dataset
    accumulated_evals = []
    accumulated_targets = []
    for test_sample_index in tqdm.tqdm(range(len(selected_samples)), desc='Performing LOO'):
        if train_only:
            print('Skipped')
            break
        # get train and test samples for LOO validation

        test_sample = selected_samples[test_sample_index]
        train_samples = [k for i, k in enumerate(selected_samples)
                         if i != test_sample_index and k['subj'] != test_sample['subj']]

        # Samples are CHANNEL, FREQ, DELAY, TIME.
        test_sample, test_targets = get_mh_single_run_selected_channels_dataset(channel_name, delay_time_seconds,
                                                                                target_str, test_sample, features_key,
                                                                                pca=pca
                                                                                )
        # Data is now FREQ, DELAY, TIME for channel channel_name

        train_samples, train_targets = get_mh_multi_run_selected_channels_dataset(channel_name, delay_time_seconds,
                                                                                  train_samples, target_str,
                                                                                  test_sample_index, features_key,
                                                                                  pca=pca
                                                                                  )
        grid_search = grid_search_with_alpha_grid(None, estimator, inner_splitter, metric,
                                                  regularization_grid_search, train_samples, train_targets)

        test_results = grid_search.predict(reshape_data_for_inference(test_sample, None))
        accumulated_evals.append(test_results)
        accumulated_targets.append(test_targets)
        pearson_r, pearson_p = pearsonr(test_targets, test_results)
        MSE = mean_squared_error(test_targets, test_results)
        r2 = r2_score(test_targets, test_results)

        results.append({
            'pearson r test': pearson_r,
            'pearson p test': pearson_p,
            'MSE test': MSE,
            'nMSE test': MSE / np.std(test_targets),
            'r2 test': r2,
            'MSE val': -grid_search.cv_results_['mean_test_MSE'].mean(),
            'r2 val': grid_search.cv_results_['mean_test_r2'].mean(),
            'pearson val': grid_search.cv_results_['mean_test_pearson'].mean(),
        })
        if test_on_one:
            return results[-1], grid_search
    print('Training final model.')
    # Finally, retrain one last time on the entire dataset before returning the model
    train_samples, train_targets = get_mh_multi_run_selected_channels_dataset(channel_name, delay_time_seconds,
                                                                              selected_samples, target_str, None,
                                                                              features_key, pca=pca)
    grid_search_final = grid_search_with_alpha_grid(None, estimator, inner_splitter, metric,
                                                    regularization_grid_search, train_samples, train_targets,
                                                    verbose=verbosity_final_fit)
    if return_scores:
        return results, grid_search_final, (accumulated_evals, accumulated_targets)
    else:
        return results, grid_search_final


def get_mh_multi_run_selected_channels_dataset(channel_name, delay_time_seconds, selected_samples, target_str,
                                               test_sample_index=None, features_keys=None, stack=False,
                                               standardize_individually=False, return_subj=False,
                                               pca=None):
    if test_sample_index is None:
        train_samples_raw = selected_samples
    else:
        train_samples_raw = [selected_samples[a] for a in range(len(selected_samples)) if a != test_sample_index]
    # After extracting samples, we pick one specific channel and concatenate through time
    samples = []
    targets = []
    subjs = []
    for sample in train_samples_raw:
        data = get_mh_single_run_selected_channels_dataset(
            channel_name, delay_time_seconds,
            target_str, sample, features_keys,
            stack=stack,
            standardize_here=standardize_individually,
            return_subj=return_subj,
            pca=pca
        )
        if return_subj:
            train_data, train_targets, subj = data
            subjs.extend(subj)
        else:
            train_data, train_targets = data
        samples.append(train_data)
        targets.append(train_targets)
    samples = np.concatenate(samples, axis=-1)
    targets = np.concatenate(targets, axis=-1)
    if return_subj:
        return samples, targets, subjs
    else:
        return samples, targets


def get_mh_single_run_selected_channels_dataset(channel_selection, delay_time_seconds, target_str, test_sample,
                                                features_keys=None, stack=False, standardize_here=False,
                                                return_subj=False, pca=None):
    if features_keys is None:
        features_keys = 'mh_features'
    test_data, test_targets = get_mh_single_run_dataset(
        test_sample[features_keys],
        test_sample[target_str],
        delay_time_seconds,
        test_sample['f_features'],
        pca=pca
        # Select one channel based on channel name and MNE channel info
    )
    if type(channel_selection) is str:
        channel_selection = [channel_selection]
    selected_data = []
    for channel_name in channel_selection:
        data = test_data[np.argmax([k == channel_name for k in test_sample['mh_ch_names']]), ...]
        selected_data.append(data)
    if stack:
        test_data = np.stack(selected_data, axis=0)
    else:
        test_data = np.concatenate(selected_data, axis=0)
    if standardize_here:
        test_data = (test_data - test_data.mean(axis=-1, keepdims=True)) / test_data.std(axis=-1, keepdims=True)
    if not return_subj:
        return test_data, test_targets
    else:
        return test_data, test_targets, [test_sample['subj'] for _ in test_targets]
