import numpy as np
import pandas as pd
from neuropy.utils import mathutil, signal_process
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import scipy.signal as sg
from neuropy.core import Signal, ProbeGroup, Epoch, Ratemap
from neuropy.utils.signal_process import WaveletSg, filter_sig,ThetaParams
from neuropy.io import BinarysignalIO
from copy import deepcopy

import matplotlib.pyplot as plt

def _detect_freq_band_epochs(
    signals,
    freq_band,
    thresh,
    edge_cutoff,
    mindur,
    maxdur,
    mergedist,
    fs,
    sigma,
    ignore_times=None,
    return_power=False,
):
    """Detects epochs of high power in a given frequency band

    Parameters
    ----------
    thresh : tuple, optional
        low and high threshold for detection
    mindur : float, optional
        minimum duration of epoch
    maxdur : float, optional
    chans : list
        channels used for epoch detection, if None then chooses best chans
    """

    lf, hf = freq_band
    dt = 1 / fs
    smooth = lambda x: gaussian_filter1d(x, sigma=sigma / dt, axis=-1)
    lowthresh, highthresh = thresh

    # Because here one shank is selected per shank, based on visualization:
    # mean: very conservative in cases where some shanks may not have that strong ripple
    # max: works well but may have occasional false positives

    # First, bandpass the signal in the range of interest
    power = np.zeros(signals.shape[1])
    for sig in signals:
        yf = signal_process.filter_sig.bandpass(sig, lf=lf, hf=hf, fs=fs)
        # zsc_chan = smooth(stats.zscore(np.abs(signal_process.hilbertfast(yf))))
        # zscsignal[sig_i] = zsc_chan
        power += np.abs(signal_process.hilbertfast(yf))

    # Second, take the mean and smooth the signal with a sigma wide gaussian kernel
    power = smooth(power / signals.shape[0])

    # Third, exclude any noisy periods due to motion or other artifact
    # ---------setting noisy periods zero --------
    if ignore_times is not None:
        assert ignore_times.ndim == 2, "ignore_times should be 2 dimensional array"
        noisy_frames = np.concatenate(
            [
                (np.arange(start * fs, stop * fs)).astype(int)
                for (start, stop) in ignore_times
            ]
        )

        # edge case: remove any frames that might extend past end of recording
        noisy_frames = noisy_frames[noisy_frames < len(power)]
        power[noisy_frames] = 0

    # Fourth, identify candidate epochs above edge_cutoff threshold
    # ---- thresholding and detection ------
    power = stats.zscore(power)
    # power_thresh = np.where(power >= edge_cutoff, power, 0)
    power_thresh = np.where(power >= edge_cutoff, power, -100)  # NRK bugfix

    # Fifth, refine candidate epochs to periods between lowthresh and highthresh
    peaks, props = sg.find_peaks(
        power_thresh, height=[lowthresh, highthresh], prominence=0
    )
    starts, stops = props["left_bases"], props["right_bases"]
    peaks_power = power_thresh[peaks]

    # ----- merge overlapping epochs ------
    # Last, merge any epochs that overlap into one longer epoch
    n_epochs = len(starts)
    ind_delete = []
    for i in range(n_epochs - 1):
        if starts[i + 1] - stops[i] < 1e-6:
            # stretch the second epoch to cover the range of both epochs
            starts[i + 1] = min(starts[i], starts[i + 1])
            stops[i + 1] = max(stops[i], stops[i + 1])

            peaks_power[i + 1] = max(peaks_power[i], peaks_power[i + 1])
            peaks[i + 1] = [peaks[i], peaks[i + 1]][
                np.argmax([peaks_power[i], peaks_power[i + 1]])
            ]

            ind_delete.append(i)

    epochs_arr = np.vstack((starts, stops, peaks, peaks_power)).T
    starts, stops, peaks, peaks_power = np.delete(epochs_arr, ind_delete, axis=0).T

    epochs_df = pd.DataFrame(
        dict(
            start=starts, stop=stops, peak_time=peaks, peak_power=peaks_power, label=""
        )
    )
    epochs_df[["start", "stop", "peak_time"]] /= fs  # seconds
    epochs = Epoch(epochs=epochs_df)

    # ------duration thresh---------
    epochs = epochs.duration_slice(min_dur=mindur, max_dur=maxdur)
    print(f"{len(epochs)} epochs remaining with durations within ({mindur},{maxdur})")

    epochs.metadata = {
        "params": {
            "lowThres": lowthresh,
            "highThresh": highthresh,
            "freq_band": freq_band,
            "mindur": mindur,
            "maxdur": maxdur,
            # "mergedist": mergedist,
        },
    }
    if not return_power:
        return epochs
    else:
        return epochs, power


def detect_hpc_delta_wave_epochs(
    signal: Signal,
    freq_band=(0.2, 5),
    min_dur=0.15,
    max_dur=0.5,
    ignore_epochs: Epoch = None,
):
    """Detect delta waves epochs.

    Method
    -------
    Maingret, Nicolas, Gabrielle Girardeau, Ralitsa Todorova, Marie Goutierre, and Michaël Zugaro. “Hippocampo-Cortical Coupling Mediates Memory Consolidation during Sleep.” Nature Neuroscience 19, no. 7 (July 2016): 959–64. https://doi.org/10.1038/nn.4304.

    -> filtered singal in 0.5-4 Hz (Note: Maingret et al. used 0-6 Hz for cortical LFP)
    -> remove noisy epochs if provided
    -> z-scored the filtered signal, D(t)
    -> flip the sign of signal to be consistent with cortical LFP
    -> calculate derivative, D'(t)
    -> extract upward-downward-upward zero-crossings which correspond to start,peak,stop of delta waves, t_start, t_peak, t_stop
    -> discard sequences below 150ms and above 500ms
    -> Delta waves corresponded to epochs where D(t_peak) > 2, or D(t_peak) > 1 and D(t_stop) < -1.5.

    Parameters
    ----------
    signal : Signal object
        signal trace to be used for detection
    freq_band : tuple, optional
        frequency band in Hz, by default (0.5, 4)
    min_dur: float, optional
        minimum duration for delta waves, by default 0.15 seconds
    max_dur: float, optional
        maximum duration for delta waves, by default 0.5 seconds
    ignore_epochs: core.Epoch, optional
        ignore timepoints within these epochs, primarily used for noisy time periods if known already, by default None

    Returns
    -------
    Epoch
        delta wave epochs. In addition, peak_time, peak_amp_zsc, stop_amp_zsc are also returned as columns
    """

    assert freq_band[1] <= 6, "Upper limit of freq_band can not be above 6 Hz"
    assert signal.n_channels == 1, "Signal should have only 1 channel"

    delta_signal = signal_process.filter_sig.bandpass(
        signal, lf=freq_band[0], hf=freq_band[1]
    ).traces[0]
    time = signal.time

    # ----- remove timepoints provided in ignore_epochs ------
    if ignore_epochs is not None:
        noisy_bool = ignore_epochs.get_indices_for_time(time)
        time = time[~noisy_bool]
        delta_signal = delta_signal[~noisy_bool]

    # ---- normalize and flip the sign to be consistent with cortical lfp ----
    delta_zsc = -1 * stats.zscore(delta_signal)

    # ---- finding peaks and trough for delta oscillations
    delta_zsc_diff = np.diff(delta_zsc).squeeze()
    zero_crossings = np.diff(np.sign(delta_zsc_diff))
    troughs_indx = np.where(zero_crossings > 0)[0]
    peaks_indx = np.where(zero_crossings < 0)[0]

    if peaks_indx[0] < troughs_indx[0]:
        peaks_indx = peaks_indx[1:]

    if peaks_indx[-1] > troughs_indx[-1]:
        peaks_indx = peaks_indx[:-1]

    n_peaks_in_troughs = np.histogram(peaks_indx, troughs_indx)[0]
    assert n_peaks_in_troughs.max() == 1, "Found multiple peaks within troughs"

    troughs_time, peaks_time = time[troughs_indx], time[peaks_indx]
    trough_pairs = np.vstack((troughs_time[:-1], troughs_time[1:])).T
    trough_peak_trough = np.insert(trough_pairs, 1, peaks_time, axis=1)
    duration = np.diff(trough_pairs, axis=1).squeeze()
    peak_amp = delta_zsc[peaks_indx]
    stop_amp = delta_zsc[troughs_indx[1:]]

    # ---- filtering based on duration and z-scored amplitude -------
    good_duration_bool = (duration >= min_dur) & (duration <= max_dur)
    good_amp_bool = (peak_amp > 2) | ((peak_amp > 1.5) & (stop_amp < -1.5))
    good_bool = good_amp_bool & good_duration_bool

    delta_waves_time = trough_peak_trough[good_bool]

    print(f"{delta_waves_time.shape[0]} delta waves detected")

    epochs = pd.DataFrame(
        {
            "start": delta_waves_time[:, 0],
            "stop": delta_waves_time[:, 2],
            "peak_time": delta_waves_time[:, 1],
            "peak_amp_zsc": peak_amp[good_bool],
            "stop_amp_zsc": stop_amp[good_bool],
            "label": "delta_wave",
        }
    )
    params = {"freq_band": freq_band, "channel": signal.channel_id}

    return Epoch(epochs=epochs, metadata=params)


def detect_beta_epochs(
    signal: Signal,
    probegroup: ProbeGroup = None,
    freq_band=(15, 40),
    thresh=(0, 0.5),
    mindur=0.25,
    maxdur=5,
    mergedist=0.5,
    sigma=0.125,
    edge_cutoff=-0.25,
    ignore_epochs: Epoch = None,
    return_power=False,
):
    if probegroup is None:
        selected_chan = signal.channel_id
        traces = signal.traces
    else:
        if isinstance(probegroup, np.ndarray):
            changrps = np.array(probegroup, dtype="object")
        if isinstance(probegroup, ProbeGroup):
            changrps = probegroup.get_connected_channels(groupby="shank")
        channel_ids = np.concatenate(changrps).astype("int")

        duration = signal.duration
        t1, t2 = signal.t_start, signal.t_start + np.min([duration, 3600])
        signal_slice = signal.time_slice(channel_id=channel_ids, t_start=t1, t_stop=t2)
        hil_stat = signal_process.hilbert_amplitude_stat(
            signal_slice.traces,
            freq_band=freq_band,
            fs=signal.sampling_rate,
            statistic="mean",
        )
        selected_chan = channel_ids[np.argmax(hil_stat)].reshape(-1)
        traces = signal.time_slice(channel_id=selected_chan).traces.reshape(1, -1)

    print(f"Best channel for beta: {selected_chan}")
    if ignore_epochs is not None:
        ignore_times = ignore_epochs.as_array()
    else:
        ignore_times = None

    epochs = _detect_freq_band_epochs(
        signals=traces,
        freq_band=freq_band,
        thresh=thresh,
        mindur=mindur,
        maxdur=maxdur,
        mergedist=mergedist,
        fs=signal.sampling_rate,
        ignore_times=ignore_times,
        sigma=sigma,
        edge_cutoff=edge_cutoff,
        return_power=return_power,
    )

    if not return_power:
        epochs = epochs.shift(dt=signal.t_start)
        epochs.metadata = dict(channels=selected_chan)
        return epochs
    else:
        beta_power = epochs[1]
        epochs = epochs[0].shift(dt=signal.t_start)
        epochs.metadata = dict(channels=selected_chan)
        return epochs, beta_power


def detect_ripple_epochs(
    signal: Signal,
    probegroup: ProbeGroup = None,
    freq_band=(150, 250),
    thresh=(2.5, None),
    edge_cutoff=0.5,
    mindur=0.05,
    maxdur=0.450,
    mergedist=0.05,
    sigma=0.0125,
    ignore_epochs: Epoch = None,
    ripple_channel: int or list = None,
    return_power: bool = False,
):
    # TODO chewing artifact frequency (>300 Hz) or emg based rejection of ripple epochs

    if ripple_channel is None:  # auto-detect ripple channel
        if probegroup is None:
            selected_chans = signal.channel_id
            traces = signal.traces

        else:
            if isinstance(probegroup, np.ndarray):
                changrps = np.array(probegroup, dtype="object")
            if isinstance(probegroup, ProbeGroup):
                changrps = probegroup.get_connected_channels(groupby="shank")
                # if changrp:
            selected_chans = []
            for changrp in changrps:
                signal_slice = signal.time_slice(
                    channel_id=changrp.astype("int"),
                    t_start=0,
                    t_stop=np.min((3600, signal.duration)),
                )
                hil_stat = signal_process.hilbert_amplitude_stat(
                    signal_slice.traces,
                    freq_band=freq_band,
                    fs=signal.sampling_rate,
                    statistic="mean",
                )
                selected_chans.append(changrp[np.argmax(hil_stat)])

            traces = signal.time_slice(channel_id=selected_chans).traces
    else:
        assert isinstance(ripple_channel, (list, int))
        selected_chans = (
            [ripple_channel] if isinstance(ripple_channel, int) else ripple_channel
        )
        traces = signal.time_slice(channel_id=selected_chans).traces

    print(f"Selected channels for ripples: {selected_chans}")
    if ignore_epochs is not None:
        ignore_times = ignore_epochs.as_array()
    else:
        ignore_times = None

    epochs = _detect_freq_band_epochs(
        signals=traces,
        freq_band=freq_band,
        thresh=thresh,
        edge_cutoff=edge_cutoff,
        mindur=mindur,
        maxdur=maxdur,
        mergedist=mergedist,
        fs=signal.sampling_rate,
        sigma=sigma,
        ignore_times=ignore_times,
        return_power=return_power,
    )

    if not return_power:
        epochs = epochs.shift(dt=signal.t_start)
        epochs.metadata = dict(channels=selected_chans)
        return epochs
    else:
        ripple_power = epochs[1]
        epochs = epochs[0].shift(dt=signal.t_start)
        epochs.metadata = dict(channels=selected_chans)
        return epochs, ripple_power


def detect_sharpwave_epochs(
    signal: Signal,
    probegroup: ProbeGroup = None,
    freq_band=(2, 50),
    thresh=(2.5, None),
    edge_cutoff=0.5,
    mindur=0.05,
    maxdur=0.450,
    mergedist=0.05,
    sigma=0.0125,
    ignore_epochs: Epoch = None,
    sharpwave_channel: int or list = None,
):
    if sharpwave_channel is None:
        if probegroup is None:  # auto-detect sharpwave channel
            selected_chans = signal.channel_id
            traces = signal.traces

        else:
            if isinstance(probegroup, np.ndarray):
                changrps = np.array(probegroup, dtype="object")
            if isinstance(probegroup, ProbeGroup):
                changrps = probegroup.get_connected_channels(groupby="shank")
                # if changrp:
            selected_chans = []
            for changrp in changrps:
                signal_slice = signal.time_slice(
                    channel_id=changrp.astype("int"),
                    t_start=0,
                    t_stop=np.min((3600, signal.duration)),
                )
                hil_stat = signal_process.hilbert_amplitude_stat(
                    signal_slice.traces,
                    freq_band=freq_band,
                    fs=signal.sampling_rate,
                    statistic="mean",
                )
                selected_chans.append(changrp[np.argmax(hil_stat)])

            traces = signal.time_slice(channel_id=selected_chans).traces
    else:
        assert isinstance(sharpwave_channel, (list, int))
        selected_chans = (
            [sharpwave_channel]
            if isinstance(sharpwave_channel, int)
            else sharpwave_channel
        )
        traces = signal.time_slice(channel_id=selected_chans).traces

    print(f"Selected channels for sharp-waves: {selected_chans}")
    if ignore_epochs is not None:
        ignore_times = ignore_epochs.as_array()
    else:
        ignore_times = None

    epochs = _detect_freq_band_epochs(
        signals=traces,
        freq_band=freq_band,
        thresh=thresh,
        edge_cutoff=edge_cutoff,
        mindur=mindur,
        maxdur=maxdur,
        mergedist=mergedist,
        fs=signal.sampling_rate,
        sigma=sigma,
        ignore_times=ignore_times,
    )
    epochs = epochs.shift(dt=signal.t_start)
    epochs.metadata = dict(channels=selected_chans)
    return epochs


def detect_theta_epochs(
    signal: Signal,
    probegroup: ProbeGroup = None,
    freq_band=(5, 12),
    thresh=(0, 0.5),
    mindur=0.25,
    maxdur=5,
    mergedist=0.5,
    sigma=0.125,
    edge_cutoff=-0.25,
    ignore_epochs: Epoch = None,
    return_power=False,
):
    if probegroup is None:
        selected_chan = signal.channel_id
        traces = signal.traces
    else:
        if isinstance(probegroup, np.ndarray):
            changrps = np.array(probegroup, dtype="object")
        if isinstance(probegroup, ProbeGroup):
            changrps = probegroup.get_connected_channels(groupby="shank")
        channel_ids = np.concatenate(changrps).astype("int")

        duration = signal.duration
        t1, t2 = signal.t_start, signal.t_start + np.min([duration, 3600])
        signal_slice = signal.time_slice(channel_id=channel_ids, t_start=t1, t_stop=t2)
        hil_stat = signal_process.hilbert_amplitude_stat(
            signal_slice.traces,
            freq_band=freq_band,
            fs=signal.sampling_rate,
            statistic="mean",
        )
        selected_chan = channel_ids[np.argmax(hil_stat)].reshape(-1)
        traces = signal.time_slice(channel_id=selected_chan).traces.reshape(1, -1)

    print(f"Best channel for theta: {selected_chan}")
    if ignore_epochs is not None:
        ignore_times = ignore_epochs.as_array()
    else:
        ignore_times = None

    epochs = _detect_freq_band_epochs(
        signals=traces,
        freq_band=freq_band,
        thresh=thresh,
        mindur=mindur,
        maxdur=maxdur,
        mergedist=mergedist,
        fs=signal.sampling_rate,
        ignore_times=ignore_times,
        sigma=sigma,
        edge_cutoff=edge_cutoff,
        return_power=return_power,
    )

    if not return_power:
        epochs = epochs.shift(dt=signal.t_start)
        epochs.metadata = dict(channels=selected_chan)
        return epochs
    else:
        theta_power = epochs[1]
        epochs = epochs[0].shift(dt=signal.t_start)
        epochs.metadata = dict(channels=selected_chan)
        return epochs, theta_power


def detect_spindle_epochs(
    signal: Signal,
    probegroup: ProbeGroup = None,
    freq_band=(8, 16),
    thresh=(1, 5),
    mindur=0.35,
    maxdur=4,
    mergedist=0.05,
    ignore_epochs: Epoch = None,
    method="hilbert",
):
    if probegroup is None:
        selected_chans = signal.channel_id
        traces = signal.traces

    else:
        if isinstance(probegroup, np.ndarray):
            changrps = np.array(probegroup, dtype="object")
        if isinstance(probegroup, ProbeGroup):
            changrps = probegroup.get_connected_channels(groupby="shank")
            # if changrp:
        selected_chans = []
        for changrp in changrps:
            signal_slice = signal.time_slice(
                channel_id=changrp.astype("int"), t_start=0, t_stop=3600
            )
            hil_stat = signal_process.hilbert_amplitude_stat(
                signal_slice.traces,
                freq_band=freq_band,
                fs=signal.sampling_rate,
                statistic="mean",
            )
            selected_chans.append(changrp[np.argmax(hil_stat)])

        traces = signal.time_slice(channel_id=selected_chans).traces

    print(f"Selected channels for spindles: {selected_chans}")

    if ignore_epochs is not None:
        ignore_times = ignore_epochs.as_array()
    else:
        ignore_times = None

    epochs, metadata = _detect_freq_band_epochs(
        signals=traces,
        freq_band=freq_band,
        thresh=thresh,
        mindur=mindur,
        maxdur=maxdur,
        mergedist=mergedist,
        fs=signal.sampling_rate,
        ignore_times=ignore_times,
    )
    epochs["start"] = epochs["start"] + signal.t_start
    epochs["stop"] = epochs["stop"] + signal.t_start

    metadata["channels"] = selected_chans
    return Epoch(epochs=epochs, metadata=metadata)


def detect_gamma_epochs():
    pass


class Ripple:
    """Events and analysis related to sharp-wave ripple oscillations"""

    @staticmethod
    def detect_ripple_epochs(**kwargs):
        return detect_ripple_epochs(**kwargs)

    @staticmethod
    def detect_sharpwave_epochs(**kwargs):
        return detect_sharpwave_epochs(**kwargs)

    @staticmethod
    def get_peak_ripple_freq(eegfile: BinarysignalIO, rpl_epochs: Epoch, lf=100, hf=300):
        """Detect peak ripple frequency"""

        # Load in relevant metadata
        sampling_rate = eegfile.sampling_rate
        rpl_channels = rpl_epochs.metadata["channels"]

        # Specify frequencies and get signal
        freqs = np.linspace(100, 250, 100)
        signal = eegfile.get_signal(rpl_channels)

        # Bandpass signal in ripple range
        lfp = filter_sig.bandpass(signal, lf=lf, hf=hf).traces.mean(axis=0)

        # Build up arrays to grab every 1000 ripples
        n_rpls = len(rpl_epochs)
        rpls_window = np.arange(0, n_rpls, np.min([1000, n_rpls - 1]))
        rpls_window[-1] = n_rpls
        peak_freqs = []

        # Loop through each set of 1000 ripples, concatenate signal for each together, run Wavelet and get peak frequency
        # at time of peak power
        buffer_frames = int(.1 * sampling_rate)  # grab 100ms either side of peak power
        for i in range(len(rpls_window) - 1):
            # Get blocks of ripples and their peak times
            rpl_df = rpl_epochs[rpls_window[i] : rpls_window[i + 1]].to_dataframe()
            peakframe = (rpl_df["peak_time"].values * sampling_rate).astype("int")

            rpl_frames = [np.arange(p - buffer_frames, p + buffer_frames) for p in peakframe]  # Grab 100ms either side of peak frame
            rpl_frames = np.concatenate(rpl_frames)

            # Grab signal for ripples only
            new_sig = Signal(lfp[rpl_frames].reshape(1, -1), sampling_rate=sampling_rate)

            # Run Wavelet and get peak frequency for each ripple
            wvlt = WaveletSg(signal=new_sig, freqs=freqs, ncycles=10).traces
            peak_freqs.append(
                freqs[
                    np.reshape(wvlt, (len(freqs), len(peakframe), -1))
                    .max(axis=2)
                    .argmax(axis=0)
                ]
            )

        # Concatenate all peak frequencies found
        peak_freqs = np.concatenate(peak_freqs)
        assert len(peak_freqs) == len(rpl_epochs), "# peak frequencies found does not match size of input 'rpl_epochs', check code"
        new_epochs = rpl_epochs.add_column("peak_frequency_bp", peak_freqs)

        return new_epochs


class Gamma:
    """Events and analysis related to gamma oscillations"""

    def get_peak_intervals(
        self,
        lfp,
        band=(40, 80),
        lowthresh=0,
        highthresh=1,
        minDistance=300,
        minDuration=125,
        return_amplitude=False,

    ):
        """Returns strong theta lfp. If it has multiple channels, then strong theta periods are calculated from that
        channel which has highest area under the curve in the theta frequency band. Parameters are applied on z-scored lfp.

        Parameters
        ----------
        lfp : array like, channels x time
            from which strong periods are concatenated and returned
        lowthresh : float, optional
            threshold above which it is considered strong, by default 0 which is mean of the selected channel
        highthresh : float, optional
            [description], by default 0.5
        minDistance : int, optional
            minimum gap between periods before they are merged, by default 300 samples
        minDuration : int, optional
            [description], by default 1250, which means theta period should atleast last for 1 second

        Returns
        -------
        2D array
            start and end frames where events exceeded the set thresholds
        """

        # ---- filtering --> zscore --> threshold --> strong gamma periods ----
        gammalfp = signal_process.filter_sig.bandpass(lfp, lf=band[0], hf=band[1])
        hil_gamma = signal_process.hilbertfast(gammalfp)
        gamma_amp = np.abs(hil_gamma)

        zsc_gamma = stats.zscore(gamma_amp)
        peakevents = mathutil.threshPeriods(
            zsc_gamma,
            lowthresh=lowthresh,
            highthresh=highthresh,
            minDistance=minDistance,
            minDuration=minDuration,
        )
        if not return_amplitude:
            return peakevents
        else:
            return peakevents, gamma_amp

    def csd(self, period, refchan, chans, band=(40, 80), window=1250):
        """Calculating current source density using laplacian method

        Parameters
        ----------
        period : array
            period over which theta cycles are averaged
        refchan : int or array
            channel whose theta peak will be considered. If array then median of lfp across all channels will be chosen for peak detection
        chans : array
            channels for lfp data
        window : int, optional
            time window around theta peak in number of samples, by default 1250

        Returns:
        ----------
        csd : dataclass,
            a dataclass return from signal_process module
        """
        lfp_period = self._obj.geteeg(chans=chans, timeRange=period)
        lfp_period = signal_process.filter_sig.bandpass(
            lfp_period, lf=band[0], hf=band[1]
        )

        gamma_lfp = self._obj.geteeg(chans=refchan, timeRange=period)
        nChans = lfp_period.shape[0]
        # lfp_period, _, _ = self.getstrongTheta(lfp_period)

        # --- Selecting channel with strongest theta for calculating theta peak-----
        # chan_order = self._getAUC(lfp_period)
        # gamma_lfp = signal_process.filter_sig.bandpass(
        #     lfp_period[chan_order[0], :], lf=5, hf=12, ax=-1)
        gamma_lfp = signal_process.filter_sig.bandpass(
            gamma_lfp, lf=band[0], hf=band[1]
        )
        peak = sg.find_peaks(gamma_lfp)[0]
        # Ignoring first and last second of data
        peak = peak[np.where((peak > 1250) & (peak < len(gamma_lfp) - 1250))[0]]

        # ---- averaging around theta cycle ---------------
        avg_theta = np.zeros((nChans, window))
        for ind in peak:
            avg_theta = avg_theta + lfp_period[:, ind - window // 2 : ind + window // 2]
        avg_theta = avg_theta / len(peak)

        _, ycoord = self._obj.probemap.get(chans=chans)

        csd = signal_process.Csd(lfp=avg_theta, coords=ycoord, chan_label=chans)
        csd.classic()

        return csd
    
class Theta:
    """Events and analysis related to theta oscillations
    
    example signal creation:
    theta_sig = signal.time_slice(channel_id=theta_channel,t_start = pos.t_start,t_stop=pos.t_stop)
    
    """

    @staticmethod
    def detect_theta_epochs(**kwargs):
        return detect_theta_epochs(**kwargs)

    def get_theta_sequence_data(
            signal: Signal, #neuropy.core.signal, one ch (best theta power)
            pf: Ratemap,#neuropy.core.ratemap - comes in as pf1d or circularPf1
            theta_epochs = None, #will generate if not passed in
        ):

        assert signal.n_channels == 1, "Pass in one channel"
        fs = signal.sampling_rate

        if theta_epochs == None:
            theta_epochs = detect_theta_epochs(signal)

        theta_params = ThetaParams(signal.traces,fs=fs) #use hilbert to get troughs, etc

        #------------ get theta cycles --------------- #
        theta_cycles = []

        for ep in range(len(theta_epochs.starts)):
            ep_start = theta_epochs.starts[ep]*fs
            ep_stop = theta_epochs.stops[ep]*fs

            #get the theta troughs just within that high theta epoch
            th_troughs = theta_params.trough[(theta_params.trough > ep_start) & (theta_params.trough < ep_stop)]
            th_troughs_sec = th_troughs/fs
            #define theta cycle as one trough to the next
            ep_cycles = [(th_troughs_sec[t],th_troughs_sec[t+1]) for t in range(len(th_troughs_sec) -1)]

            theta_cycles.extend(ep_cycles)

        ntheta_cycles = len(theta_cycles) #for statistics purposes

        # ----------- get all spikes within each theta cycle from all cells -------- #
        spktrains = pf.ratemap_spiketrains #in seconds

        pf.estimate_theta_phases(signal)
        spkphases = pf.ratemap_spiketrains_phases

        # Initialize an empty list to collect data for the DataFrame
        cycle_data = []
        all_data = []
        # Iterate through theta cycles and cells
        for cycle_idx, (start, end) in enumerate(theta_cycles):
            for cell_idx, (cell_spikes, cell_phases) in enumerate(zip(spktrains, spkphases)):

                cell_id = pf.neuron_ids[cell_idx]
                
                # Filter spikes within the current theta cycle
                in_epoch = (cell_spikes >= start) & (cell_spikes < end)
                cycle_spikes = cell_spikes[in_epoch]
                cycle_phases = cell_phases[in_epoch]
                
                # ensure that it's sorted by spike time within a cell
                sorted_indices = cycle_spikes.argsort()
                sorted_spikes = cycle_spikes[sorted_indices]
                sorted_phases = cycle_phases[sorted_indices]
                
                # Append to the data list
                for spike, phase in zip(sorted_spikes, sorted_phases):
                    cycle_data.append([cycle_idx, cell_id, spike, phase])
            
            unique_cells = len(set(row[1] for row in cycle_data))
            if len(cycle_data) < 3 or unique_cells < 2:
                cycle_data = []
                continue #need at least 3 spikes from at least 2 cells
            else:
 #               add_data = deepcopy(cycle_data)
                all_data.extend(cycle_data)
                cycle_data = []

        # Create a DataFrame that summarizes over all cycles
       # flattened_data = [row for cycle_data in all_data for row in cycle_data]
        theta_sequence_data = pd.DataFrame(all_data, columns=['Theta_Cycle', 'Cell_ID', 'Spiketime', 'Phase'])
       
        return theta_sequence_data #,ntheta_cycles
    
    def plot_theta_sequences(theta_sequence_data,ax=None,sortby='median'):
       
        """

        run get_theta_sequence_data first, then pass output into this func

        sortby is for determining how to sort which cell comes first in the sequence. 
        options are median (default) or first_spike

        """
        if sortby == 'median':
            resorted_data = (
                theta_sequence_data.groupby(['Theta_Cycle','Cell_ID','Phase'])['Spiketime']
                .median()
                .reset_index()
            )

        elif sortby == 'first_spike':    
            resorted_data = (
                theta_sequence_data.groupby(['Theta_Cycle','Cell_ID','Phase'])['Spiketime']
                .min()
                .reset_index()
            )

        # plot
        uniquecycles = resorted_data['Theta_Cycle'].unique()
        n_figs = int(np.ceil(len(uniquecycles)/100))

        for f in range(n_figs):
            fig, axes = plt.subplots(10, 10, figsize=(15, 15))# Create a new figure for each 100 cycles
            axes = axes.flatten()  # Flatten to make indexing easier
            cycles_to_plot = uniquecycles[f * 100 : (f + 1) * 100] #which cycles for this fig

            for i, cycle in enumerate(cycles_to_plot):
                ax = axes[i]  # Select the current subplot axis

                # Get the data for this cycle
                cycle_data = resorted_data[resorted_data['Theta_Cycle'] == cycle]

                # Iterate over each unique Cell_ID in this cycle
                for j, cell_id in enumerate(cycle_data['Cell_ID'].unique()):
                    # Get the spike times for this cell in the current cycle
                    spike_times = cycle_data[cycle_data['Cell_ID'] == cell_id]['Spiketime']
                    phases = cycle_data[cycle_data['Cell_ID'] == cell_id]['Phase']
                   # ax.plot(spike_times, [j] * len(spike_times), 'k.', markersize=5)
                    ax.plot(phases, [j] * len(phases), 'k.', markersize=5)
                        # Set the title and labels for the subplot
                ax.set_title(f'Cycle {cycle}')
              #  ax.set_xlabel('Spiketime')
                ax.set_xlabel('Phase')
                ax.set_ylabel('Cell ID')
                ax.set_xticks([])
                ax.set_yticks([])

            # Adjust layout to avoid overlap
        plt.tight_layout()
        plt.show()   
        # for cycle in resorted['Theta_Cycle'].unique():
           
        #    spikes_to_plot = resorted[resorted['Theta_Cycle'] == cycle]
        #     # Filter only the current cycle and sort by median spike time
        #     # cell_id_order = (
        #     #     resorted[resorted['Theta_Cycle'] == cycle]
        #     #     .sort_values(['Spiketime','Phase'])['Cell_ID']
        #     #     .tolist()
        #     # )

        #     plt.




            #plot raster for just one cycle, each row one cell, ordered by cell_id_order

    # if ax is None:
    #             if subplots is None:
    #                 Fig = plotting.Fig(nrows=1, ncols=1, size=(10, 5))
    #                 ax = plt.subplot(Fig.gs[0])
    #                 ax.spines["right"].set_visible(True)
    #                 axphase = ax.twinx()
    #                 widgets.interact(
    #                     plot_,
    #                     cell=widgets.IntSlider(
    #                         min=0,
    #                         max=nCells - 1,
    #                         step=1,
    #                         description="Cell ID:",
    #                     ),
    #                     ax=widgets.fixed(ax),
    #                     axphase=widgets.fixed(axphase),
    #                 )

if __name__ == "__main__":
    from neuropy.io import BinarysignalIO
    from neuropy.core import ProbeGroup, Epoch

    # eegfile = BinarysignalIO(
    #     "/data/Working/Trace_FC/Recording_Rats/Finn/2022_01_18_habituation/Finn_habituation2_denoised.eeg",
    #     n_channels=35,
    #     sampling_rate=1250,
    # )
    # signal = eegfile.get_signal()
    # prbgrp = ProbeGroup().from_file(
    #     "/data/Working/Trace_FC/Recording_Rats/Finn/2022_01_18_habituation/Finn_habituation2_denoised.probegroup.npy"
    # )
    # art_epochs = Epoch(
    #     epochs=None,
    #     file="/data/Working/Trace_FC/Recording_Rats/Finn/2022_01_18_habituation/Finn_habituation2_denoised.art_epochs.npy",
    # )
    # ripple_epochs = detect_ripple_epochs(
    #     signal, prbgrp, thresh=(2.5, None), ignore_epochs=art_epochs, mindur=0.025
    # )
    eegfile = BinarysignalIO('/data/Clustering/sessions/RatU/RatUDay2NSD/RatU_Day2NSD_2021-07-24_08-16-38_thetachan.eeg',
                             n_channels=1, sampling_rate=1250)
    signal = eegfile.get_signal()
    art_epochs = Epoch(epochs=None, file='/data/Clustering/sessions/RatU/RatUDay2NSD/RatU_Day2NSD_2021-07-24_08-16-38.artifact.npy')
    detect_theta_epochs(signal, ignore_epochs=art_epochs)
