clear; clc;

% 1. LOAD THE TRAINED MODEL
if isfile('HeartClassifier.mat')
    disp('Loading trained model...');
    load('HeartClassifier.mat'); % Loads 'rf_model', 'afe', 'fs'
else
    error('Model file not found! Please run your training script first.');
end

% 2. DEFINE CLASS MAP (Must match your training labels)
class_map = {1, 'Normal'; 2, 'AF'; 3, 'LSM'; 4, 'AVB'; 5, 'ESM'; ...
             6, 'LDM'; 7, 'MSM'; 8, 'S3'; 9, 'S4'; 10, 'T'};

% 3. SELECT FILE
[fname, fpath] = uigetfile('*.wav', 'Select a Heart Sound to Diagnose');

if isequal(fname,0)
    disp('User canceled selection.');
else
    fullpath = fullfile(fpath, fname);
    fprintf('\nAnalyzing: %s ...\n', fname);
    
    % --- PRE-PROCESSING (Identical to Training) ---
    [audio_in, fs_in] = audioread(fullpath);
    audio_in = audio_in(:,1); % Force Mono
    
    % Resample if necessary
    if fs_in ~= fs
        audio_in = resample(audio_in, fs, fs_in); 
    end
    
    % Normalize
    audio_in = audio_in / (max(abs(audio_in)) + eps);
    
    % Filter (Re-create filter to match training range)
    bpFilt = designfilt('bandpassiir', 'FilterOrder', 4, ...
        'HalfPowerFrequency1', 25, 'HalfPowerFrequency2', 900, 'SampleRate', fs);
    audio_in = filtfilt(bpFilt, audio_in);
    
    % --- SLICING & PREDICTION LOOP ---
    X_Single = [];
    
    % Use parameters from the saved 'afe' object implicitly or define standard
    % (Standard window 1.0s, Step 0.5s used in training)
    win_samples = round(1.0 * fs);       
    step_samples = round(0.5 * fs);      
    
    for start_idx = 1 : step_samples : (length(audio_in) - win_samples)
        slice = audio_in(start_idx : start_idx + win_samples - 1);
        
        % Extract using the LOADED extractor (guarantees consistency)
        feats = extract(afe, slice);
        
        % Condense to Stats Row
        stats = [mean(feats,1), std(feats,0,1), range(feats,1)];
        X_Single = [X_Single; stats];
    end
    
    % --- FINAL DIAGNOSIS LOGIC ---
    if isempty(X_Single)
        disp('ERROR: Audio is too short (< 1 second).');
    else
        [pred_str, ~] = predict(rf_model, X_Single);
        pred_ids = str2double(pred_str);
        
        % A. Calculate Statistics
        total_slices = length(pred_ids);
        normal_votes = sum(pred_ids == 1);
        disease_votes = total_slices - normal_votes;
        disease_ratio = disease_votes / total_slices;
        
        fprintf('--------------------------------------\n');
        
        % B. "Safety First" Threshold Logic
        THRESHOLD = 0.30; % If >30% is abnormal, flag it.
        
        if disease_ratio > THRESHOLD
            % Identify the specific disease
            disease_ids = pred_ids(pred_ids ~= 1);
            primary_disease_id = mode(disease_ids);
            final_label = class_map{primary_disease_id, 2};
            
            fprintf(' FINAL DIAGNOSIS:  %s (High Risk)\n', final_label);
            fprintf(' CONFIDENCE:       %.1f%% of slices flagged this disease.\n', ...
                (sum(pred_ids == primary_disease_id)/total_slices)*100);
            fprintf(' REASONING:        Total Abnormal Activity: %.1f%%\n', disease_ratio*100);
            
        else
            % Normal
            final_label = 'Normal';
            fprintf(' FINAL DIAGNOSIS:  Normal\n');
            fprintf(' CONFIDENCE:       %.1f%% (Slices classified as Normal)\n', ...
                (normal_votes/total_slices)*100);
        end
        fprintf('--------------------------------------\n');
        
        % C. Visualization
        figure('Name', ['Results: ' fname]);
        histogram(pred_ids, 'BinEdges', 0.5:10.5, 'FaceColor', [0 0.447 0.741]);
        xticks(1:10);
        xticklabels(class_map(:,2));
        title(['Prediction Distribution for ' fname]);
        ylabel('Count of 1-sec Slices');
        grid on;
    end
end