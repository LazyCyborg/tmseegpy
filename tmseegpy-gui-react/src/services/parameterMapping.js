// services/parameterMapping.js

/**
 * Helper functions to exactly match Python's type conversion
 */
const ensureFloat = (value, defaultValue) => {
    if (value === undefined || value === null || value === '') {
        return defaultValue;
    }
    const num = Number(value);
    return isNaN(num) ? defaultValue : num;
};

const ensureInt = (value, defaultValue) => {
    if (value === undefined || value === null || value === '') {
        return defaultValue;
    }
    const num = parseInt(value);
    return isNaN(num) ? defaultValue : num;
};

const ensureBoolean = (value, defaultValue) => {
    if (value === undefined || value === null) {
        return defaultValue;
    }
    return Boolean(value);
};

/**
 * Maps frontend parameters to backend parameters exactly matching Python backend
 */
export const mapParametersToBackend = (frontend_params) => {
    // Validate required parameters
    if (!frontend_params.dataFormat) {
        throw new Error("Missing required parameter: dataFormat");
    }

    // Handle data directory path
    let data_dir = frontend_params.dataDir || '';
    if (data_dir.includes('TMSEEG/TMSEEG')) {
        data_dir = data_dir.replace('TMSEEG/TMSEEG', 'TMSEEG');
    }

    return {
        // Core Processing Parameters
        data_dir: String(data_dir),
        output_dir: frontend_params.outputDir || 'output',
        data_format: frontend_params.dataFormat || 'neurone',
        no_preproc_output: ensureBoolean(frontend_params.noPreprocessOutput, false),
        no_pcist: ensureBoolean(frontend_params.noPcist, false),
        eeglab_montage_units: frontend_params.eeglabMontageUnits || 'auto',
        stim_channel: frontend_params.stimChannel || 'STI 014',
        save_preproc: ensureBoolean(frontend_params.savePreproc, false),
        random_seed: ensureInt(frontend_params.randomSeed, 42),
        substitute_zero_events_with: ensureInt(frontend_params.substituteZeroEventsWith, 10),

        // Sampling and Window Parameters
        initial_sfreq: ensureFloat(frontend_params.initialSfreq, 1000),
        final_sfreq: ensureFloat(frontend_params.finalSfreq, 725),
        initial_window_start: ensureFloat(frontend_params.initialWindowStart, -2),
        initial_window_end: ensureFloat(frontend_params.initialWindowEnd, 10),
        extended_window_start: ensureFloat(frontend_params.extendedWindowStart, -2),
        extended_window_end: ensureFloat(frontend_params.extendedWindowEnd, 15),
        initial_interp_window: ensureFloat(frontend_params.initialInterpWindow, 1.0),
        extended_interp_window: ensureFloat(frontend_params.extendedInterpWindow, 5.0),
        interpolation_method: frontend_params.interpolationMethod || 'cubic',

        // Processing Options
        skip_second_artifact_removal: ensureBoolean(frontend_params.skipSecondArtifactRemoval, false),
        mne_filter_epochs: ensureBoolean(frontend_params.mneFilterEpochs, false),
        plot_raw: ensureBoolean(frontend_params.plotRaw, false),
        filter_raw: ensureBoolean(frontend_params.filterRaw, false),

        // Filtering Parameters
        l_freq: ensureFloat(frontend_params.lFreq, 0.1),
        h_freq: ensureFloat(frontend_params.hFreq, 45),
        raw_h_freq: ensureFloat(frontend_params.rawHFreq, 250),
        notch_freq: ensureFloat(frontend_params.notchFreq, 50),
        notch_width: ensureFloat(frontend_params.notchWidth, 2),

        // Epoch Parameters
        epochs_tmin: ensureFloat(frontend_params.epochsTmin, -0.41),
        epochs_tmax: ensureFloat(frontend_params.epochsTmax, 0.41),

        // Artifact Detection Parameters
        bad_channels_threshold: ensureFloat(frontend_params.badChannelsThreshold, 3),
        bad_epochs_threshold: ensureFloat(frontend_params.badEpochsThreshold, 3),
        amplitude_threshold: ensureFloat(frontend_params.amplitudeThreshold, 300.0),

        // ICA Parameters
        ica_method: frontend_params.icaMethod || 'fastica',
        first_ica_manual: ensureBoolean(frontend_params.firstIcaManual, true),
        second_ica_manual: ensureBoolean(frontend_params.secondIcaManual, true),
        no_first_ica: ensureBoolean(frontend_params.noFirstIca, false),
        no_second_ica: ensureBoolean(frontend_params.noSecondIca, false),
        second_ica_method: frontend_params.secondIcaMethod || 'fastica',

        // Artifact Thresholds
        blink_thresh: ensureFloat(frontend_params.blinkThresh, 2.5),
        lat_eye_thresh: ensureFloat(frontend_params.latEyeThresh, 2.0),
        noise_thresh: ensureFloat(frontend_params.noiseThresh, 4.0),
        tms_muscle_thresh: ensureFloat(frontend_params.tmsMuscleThresh, 2.0),
        muscle_thresh: ensureFloat(frontend_params.muscleThresh, 0.6),

        // Muscle Artifact Parameters
        parafac_muscle_artifacts: ensureBoolean(frontend_params.parafacMuscleArtifacts, false),
        muscle_window_start: ensureFloat(frontend_params.muscleWindowStart, 0.005),
        muscle_window_end: ensureFloat(frontend_params.muscleWindowEnd, 0.030),
        threshold_factor: ensureFloat(frontend_params.thresholdFactor, 1.0),
        n_components: ensureInt(frontend_params.nComponents, 5),

        // SSP and CSD Parameters
        apply_ssp: ensureBoolean(frontend_params.applySsp, false),
        ssp_n_eeg: ensureInt(frontend_params.sspNEeg, 2),
        apply_csd: ensureBoolean(frontend_params.applyCsd, false),
        lambda2: ensureFloat(frontend_params.lambda2, 1e-3),
        stiffness: ensureInt(frontend_params.stiffness, 4),

        // TEP Analysis Parameters
        save_evoked: ensureBoolean(frontend_params.saveEvoked, false),
        analyze_teps: ensureBoolean(frontend_params.analyzeTeps, true),
        save_validation: ensureBoolean(frontend_params.saveValidation, false),
        tep_analysis_type: frontend_params.tepAnalysisType || 'gmfa',
        tep_roi_channels: frontend_params.tepRoiChannels || ['C3', 'C4'],
        tep_method: frontend_params.tepMethod || 'largest',
        tep_samples: ensureInt(frontend_params.tepSamples, 5),

        // Window Parameters
        baseline_start: ensureInt(frontend_params.baselineStart, -400),
        baseline_end: ensureInt(frontend_params.baselineEnd, -50),
        response_start: ensureInt(frontend_params.responseStart, 0),
        response_end: ensureInt(frontend_params.responseEnd, 299),

        // PCIst Parameters
        k: ensureFloat(frontend_params.k, 1.2),
        min_snr: ensureFloat(frontend_params.minSnr, 1.1),
        max_var: ensureFloat(frontend_params.maxVar, 99.0),
        embed: ensureBoolean(frontend_params.embed, false),
        n_steps: ensureInt(frontend_params.nSteps, 100),
        pre_window_start: ensureInt(frontend_params.preWindowStart, -400),
        pre_window_end: ensureInt(frontend_params.preWindowEnd, -50),
        post_window_start: ensureInt(frontend_params.postWindowStart, 0),
        post_window_end: ensureInt(frontend_params.postWindowEnd, 300),

        // Research Mode
        research: ensureBoolean(frontend_params.research, false),
    };
};

export default mapParametersToBackend;