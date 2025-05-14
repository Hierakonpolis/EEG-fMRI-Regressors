import copy
import time
import os
import pickle
import warnings

import h5py
import mne
import numpy as np
import tqdm
from datasets.mh_features import (get_stockwell_multichannel, get_shared_runs_boundaries, get_band_boundaries,
                                  get_band_centers, stockwell_full)
import multiprocessing as mp

from datasets.utils import standardize, cubic_spline_interpolator

included_no_ecg = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'C1', 'C2', 'FC3', 'FC4',
                   'CP3', 'CP4', 'C5', 'C6', 'CPz', 'P1', 'P2', 'Pz', 'F1', 'F2', 'Fz']
included_with_ecg = included_no_ecg + ['ECG']


def get_xp2_dataset(root):
    dataset = []
    for parent, dirs, files in os.walk(root):
        for file in files:
            if not file.endswith('.vhdr') or 'derivatives' not in parent:
                continue
            if 'run' not in file:
                continue
            path_elements = parent.replace(root, '').split(os.sep)
            subj_dir = [d for d in path_elements if 'sub' in d][0]
            subj = subj_dir.replace('sub-', '')
            run = [k for k in file.split('_') if 'run' in k][0].replace('run-', '')

            eeg_nf = os.path.join(parent.replace('eeg_pp', 'NF_eeg'),
                                  file.replace('.vhdr', '_scores.mat').replace('eeg_pp', 'NFeeg'))
            bold_nf = os.path.join(parent.replace('eeg_pp', 'NF_bold'),
                                   file.replace('.vhdr', '_scores.mat').replace('eeg_pp', 'NFbold'))
            if not os.path.isfile(eeg_nf):
                warnings.warn(f"File not found, skipping: {eeg_nf}")
                continue
            if not os.path.isfile(bold_nf):
                warnings.warn(f"File not found, skipping: {bold_nf}")
                continue
            bold_vol = os.path.join(
                parent.replace('derivatives' + os.sep, '').replace('eeg_pp', 'func'),
                file.replace('d_', '').replace('eeg_pp.vhdr', 'bold.nii.gz')
            )

            if not os.path.exists(bold_nf):
                warnings.warn(f"Bold NF not found for subject {subj}, skipping...")
                continue
            dataset.append(
                {
                    'subj': subj,
                    'run': run,
                    'vhdr_path': os.path.join(parent, file),
                    'nf_bold_path': bold_nf,
                    'bold_vol': bold_vol,
                    'nf_eeg_path': eeg_nf,
                }
            )

    return dataset


class Xp2Dataset:
    def add_feature_transformer(
            self,
            transformer,
            decomposer_parameters,
            transformer_name: str,
            samples: list = None,
            channels: list = None,
            stockwell_folder: str = '/media/storage1/stockwells',
            fmin: float = 0,
            fmax: float = 60,
            freq_sampling_points: int = 500,
            window_type: str = 'kazemi',
            gamma: float = 15,
            f_features: float = 4,
            overwrite: bool = False,
            disable_bar=False,
    ):
        if transformer_name not in self.transformers or overwrite:
            self.transformers[transformer_name] = {}
        if samples is not None:
            assert all(type(k) is int for k in samples)
            samples = [self.dataset[k] for k in samples]
        else:
            samples = self.dataset

        if channels is not None:
            assert all(type(k) is str for k in channels)
        else:
            channels = self.dataset[0]['mh_ch_names']
        self.transformers[transformer_name]['parameters'] = {
            'fmin': fmin,
            'fmax': fmax,
            'freq_sampling_points': freq_sampling_points,
            'stockwell_folder': stockwell_folder,
            'window_type': window_type,
            'f_features': f_features,
            'gamma': gamma,
        }

        for channel in channels:
            collected_stockwells = []
            if channel in self.transformers[transformer_name]:
                continue
            for run in tqdm.tqdm(samples, desc=channel, disable=disable_bar):
                sample = run['eeg_data'].pick(['eeg'])
                data = sample.get_data()
                channel_indexing = [k.lower() == channel.lower() for k in sample.ch_names]
                assert np.sum(channel_indexing) == 1
                channel_idx = np.argmax(channel_indexing)
                UID = run['subj'] + '_' + run['run'] + '_' + str(f_features) + '_'

                save_path = os.path.join(
                    stockwell_folder,
                    f'{UID}{channel_idx}{fmin}{fmax}{freq_sampling_points}{window_type}{gamma}.p'
                )

                stockwell, stock_freqs = stockwell_full(
                    data[channel_idx, :],
                    sample[channel_idx][1],
                    fmin,
                    fmax,
                    freq_sampling_points,
                    save_path,
                    window_type=window_type,
                    gamma=gamma,
                )

                collected_stockwells.append(stockwell)
            collected_stockwells = np.concatenate(collected_stockwells, axis=-1).squeeze()
            self.transformers[transformer_name][channel] = transformer(**decomposer_parameters).fit(
                collected_stockwells.T)
            self.save()

    def adapt_to_new_path(self, root):
        new_set = Xp2Dataset(root, None)
        for run in self:
            broke = 0
            for new_run in new_set:
                if new_run['subj'] == run['subj']:
                    broke = 1
                    break
            if not broke: print(f"{run['subj']} not found")
            run['vhdr_path'] = new_run['vhdr_path']
            run['nf_bold_path'] = new_run['nf_bold_path']
            run['bold_vol'] = new_run['bold_vol']
            run['nf_eeg_path'] = new_run['nf_eeg_path']

    def __init__(self, root=None, save_path=None, samples=None):
        self.save_path = save_path
        if samples is not None:
            self.dataset = samples
        else:
            if save_path is not None and os.path.exists(save_path):
                loaded = pickle.load(open(save_path, 'rb'))
                if type(loaded) is list:
                    self.dataset = loaded
                else:
                    for k, v in loaded.items():
                        setattr(self, k, v)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.dataset = get_xp2_dataset(root)
        if not hasattr(self, 'transformers'):
            self.transformers = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    def get_subsample(self, indexes, opposite=False, copy_data=True):
        if not opposite:
            selected_samples = [self[k] for k in indexes]
        else:
            selected_samples = [self[k] for k in range(len(self)) if k not in indexes]

        if copy_data:
            return copy.deepcopy(Xp2Dataset(samples=selected_samples))
        else:
            return Xp2Dataset(samples=selected_samples)

    def get_dataset_shared_bands(self, num_bands=10, fmin=0, fmax=60):
        runs = [k['eeg_data'] for k in self]
        return get_shared_runs_boundaries(runs, num_bands, fmin, fmax)

    def get_eeg_data(self, overwrite=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for sample in self.dataset:
                if 'eeg_data' not in sample or overwrite:
                    data_eeg = mne.io.read_raw_brainvision(
                        sample['vhdr_path'],
                        misc=['ECG'],
                    )
                    data_eeg.set_channel_types({
                        'ECG': 'ecg'
                    })

                    bads = [k for k in data_eeg.annotations if k['description'].lower().startswith('bad')]
                    if len(bads) > 0:
                        warnings.warn(f"´bad´ annotations detected for {sample['subj']} run {sample['run']}. "
                                      f"Currently unhandled.")

                    sample['eeg_data'] = data_eeg
                    self.save()

    def get_mh_features_and_targets(self,
                                    fmax=100,
                                    fmin=0,
                                    n_bands=10,
                                    f_features=4,
                                    nf_key_bold='smoothnf',
                                    TR=1,
                                    overwrite=False,
                                    workers=1,
                                    band_boundaries=None,
                                    included_channels=None,
                                    assumed_total_time=40,
                                    save_folder=None,
                                    base_freq_grid_size=500,
                                    window_type='kazemi',
                                    gamma=1,
                                    test_mode=False,
                                    disable_bar=False,
                                    ):
        fmri_f = 1 / TR
        # times_per_sample = int(sample_time_window * f_features)
        args = []
        # Computing these can be slow. Prepare tasks for parallel processing
        r = 0
        if save_folder is not None and not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for run in self.dataset:
            if not overwrite and 'mh_features' in run:
                args.append(None)
            else:
                if r < workers:
                    delay = r / workers * assumed_total_time
                    r += 1
                else:
                    delay = None
                args.append([
                    run['eeg_data'],
                    fmax,
                    fmin,
                    f_features,
                    test_mode,
                    included_channels,
                    delay,
                    base_freq_grid_size,
                    save_folder,
                    run['subj'] + '_' + run['run'] + '_' + str(f_features) + '_',
                    band_boundaries,
                    n_bands,
                    window_type,
                    gamma
                ])

        processes = []
        # Define pool and start tasks
        print('Starting workers and distributing tasks...')
        with mp.Pool(processes=workers) as pool:
            for arg in args:
                if arg is not None:
                    processes.append(pool.apply_async(
                        get_stockwell_multichannel, arg
                    ))
                    time.sleep(0.01)
                else:
                    processes.append(None)
            with tqdm.tqdm(desc='Preparing features and targets...', total=len(processes), disable=disable_bar) as pbar:
                for process, run in zip(processes, self.dataset):
                    if process is not None:
                        # Build the dataset with the results and from the bold NF signals
                        result = process.get()
                        channel_stockwells, ch_names, channel_boundaries = result
                        # channel_features, band_boundaries, band_centers, ch_names = result

                        if band_boundaries is None:
                            band_boundaries = get_band_boundaries(run['eeg_data'], n_bands, fmin, fmax)
                        run['mh_features'] = channel_stockwells

                        run['mh_band_boundaries'] = channel_boundaries
                        run['mh_band_centers'] = get_band_centers(band_boundaries)
                        run['mh_ch_names'] = ch_names
                        self.save(silent=True)

                    if type(nf_key_bold) in [tuple, list]:
                        nf_bold_sma = (h5py.File(run['nf_bold_path'])['NF_bold']['sma'][nf_key_bold[0]][:].ravel() /
                                       h5py.File(run['nf_bold_path'])['NF_bold']['sma'][nf_key_bold[1]][:].ravel())
                        nf_bold_m1 = (h5py.File(run['nf_bold_path'])['NF_bold']['m1'][nf_key_bold[0]][:].ravel() /
                                      h5py.File(run['nf_bold_path'])['NF_bold']['m1'][nf_key_bold[1]][:].ravel())
                    else:
                        nf_bold_sma = h5py.File(run['nf_bold_path'])['NF_bold']['sma'][nf_key_bold][:].ravel()
                        nf_bold_m1 = h5py.File(run['nf_bold_path'])['NF_bold']['m1'][nf_key_bold][:].ravel()
                    nf_bold_sma = cubic_spline_interpolator(nf_bold_sma, fmri_f, f_features, axis=0)
                    nf_bold_m1 = cubic_spline_interpolator(nf_bold_m1, fmri_f, f_features, axis=0)
                    target_m1 = standardize(nf_bold_m1)
                    target_sma = standardize(nf_bold_sma)

                    run['target_m1'] = target_m1
                    run['target_sma'] = target_sma
                    run['f_features'] = f_features

                    pbar.update()

    def save(self, silent=False):
        if self.save_path is not None:
            to_save = {'dataset': self.dataset}
            if hasattr(self, 'transformers'):
                to_save['transformers'] = self.transformers
            pickle.dump(to_save, open(self.save_path, 'wb'))
            if not silent:
                print(f"Written {self.save_path}")
