clear; clc;

fs = 44100;
%FEATURE EXTRACTOR
afe = audioFeatureExtractor('SampleRate', fs, ...
    'Window', hann(round(0.03*fs), 'periodic'), ...
    'OverlapLength', round(0.02*fs), ...
    'mfcc', true, ... 
    'spectralCentroid', true, ...
    'spectralEntropy', true, ...  
    'spectralFlux', true, ...
    'pitch', true);

X_Final = [];
Y_Final = [];

%DEFINE FILES (Balanced to ~9 files per class)
normal_files = {'n1.wav', 'n2.wav', 'n3.wav', 'n4.wav', 'n5.wav', 'n6.wav', 'n7.wav', 'n8.wav', 'n9.wav'};
af_files     = {'af1.wav', 'af2.wav', 'af3.wav', 'af4.wav', 'af1_noise_1.wav', 'af1_spd105_5.wav', 'af2_spd91_2.wav', ...
                'af3_vol_3.wav', 'af4_noise_4.wav'};
lsm_files    = {'lsm1.wav', 'lsm2.wav', 'lsm3.wav', 'lsm4.wav', 'lsm5.wav', 'lsm1_noise_1.wav','lsm2_spd91_2.wav', ...
                'lsm3_vol_3.wav', 'lsm4_noise_4.wav'};
avb_files    = {'avb1.wav', 'avb2.wav', 'avb3.wav','avb1_noise_1.wav', 'avb1_noise_4.wav', 'avb2_spd91_5.wav',...
                'avb2_spd107_2.wav', 'avb3_vol_3.wav', 'avb_vol_6.wav'}; 
esm_files    = {'esm1.wav', 'esm2.wav', 'esm3.wav', 'esm4.wav', 'esm5.wav', 'esm6.wav', 'esm1_noise_1.wav', ...
                'esm2_spd105_2.wav', 'esm_vol_3.wav'};
ldm_files    = {'ldm1.wav', 'ldm2.wav', 'ldm3.wav', 'ldm4.wav', 'ldm5.wav', 'ldm6.wav', 'ldm1_noise_1', 'ldm2_spd93_2.wav',...
                'ldm3_vol_3.wav'};
msm_files    = {'msm1.wav', 'msm2.wav', 'msm3.wav', 'msm4.wav', 'msm5.wav', 'msm6.wav', 'msm7.wav', 'msm1_noise_1.wav', ...
                'msm2_spd95_2.wav'};
s3_files     = {'s3_1.wav', 's3_2.wav', 's3_3.wav', 's3_4.wav', 's3_5.wav', 's3_1_noise_1.wav', 's3_2_spd94_2.wav',...
                's3_3_vol_3.wav', 's3_4_noise_4.wav'};
s4_files     = {'s4_1.wav', 's4_2.wav','s4_1_noise_1.wav','s4_1_noise_7.wav', 's4_1_spd93_5.wav','s4_1_vol_3.wav',...
                's4_2_noise_4.wav', 's4_2_spd107_2.wav', 's4_2_vol_6.wav'};
t_files      = {'t1.wav', 't2.wav', 't3.wav', 't1_noise_1.wav', 't1_noise_4.wav', 't2_spd102_2.wav', 't2_spd103_5.wav',...
                't3_vol_3.wav', 't3_vol_6.wav'};

%SLICING LOOP
disp('Extracting Features with 50% Overlap...');
[X_n, Y_n]     = processAudio(normal_files, 1, afe, fs);
[X_af, Y_af]   = processAudio(af_files, 2, afe, fs);
[X_lsm, Y_lsm] = processAudio(lsm_files, 3, afe, fs);
[X_avb, Y_avb] = processAudio(avb_files, 4, afe, fs);
[X_esm, Y_esm] = processAudio(esm_files, 5, afe, fs);
[X_ldm, Y_ldm] = processAudio(ldm_files, 6, afe, fs);
[X_msm, Y_msm] = processAudio(msm_files, 7, afe, fs);
[X_s3, Y_s3]   = processAudio(s3_files, 8, afe, fs);
[X_s4, Y_s4]   = processAudio(s4_files, 9, afe, fs);
[X_t, Y_t]     = processAudio(t_files, 10, afe, fs);

% Combine
X_Final = [X_n; X_af; X_lsm; X_avb; X_esm; X_ldm; X_msm; X_s3; X_s4; X_t];
Y_Final = [Y_n; Y_af; Y_lsm; Y_avb; Y_esm; Y_ldm; Y_msm; Y_s3; Y_s4; Y_t];
fprintf('Total Dataset Size: %d samples\n', length(Y_Final));

% --- PARTITIONING ---
cv = cvpartition(Y_Final, 'HoldOut', 0.30); % 70% Train, 30% Test
XTrain = X_Final(training(cv), :);
YTrain = Y_Final(training(cv), :);
XTest  = X_Final(test(cv), :);
YTest  = Y_Final(test(cv), :);

disp('--- BALANCING TRAINING SAMPLES ---');

% 1. Find the target size (Size of the majority class)
[counts, ~] = groupcounts(YTrain);
max_samples = max(counts);
disp(['Target Samples per Class: ' num2str(max_samples)]);

X_Balanced = [];
Y_Balanced = [];

% 2. Loop through each class and fill gaps
for c = 1:10
    % Find all samples belonging to this class
    idx = find(YTrain == c);
    current_count = length(idx);
    
    % Get the actual data for this class
    X_class = XTrain(idx, :);
    Y_class = YTrain(idx, :);
    
    % Calculate how many we need to add
    num_needed = max_samples - current_count;
    
    if num_needed > 0
        % Randomly pick samples to duplicate
        aug_idx = randsample(current_count, num_needed, true); % 'true' means replacement allowed
        
        % Append original + duplicates
        X_Balanced = [X_Balanced; X_class; X_class(aug_idx, :)];
        Y_Balanced = [Y_Balanced; Y_class; Y_class(aug_idx, :)];
    else
        % If it's already the majority, just take it as is
        X_Balanced = [X_Balanced; X_class];
        Y_Balanced = [Y_Balanced; Y_class];
    end
end

% 3. Overwrite Training Data with Balanced Data
XTrain = X_Balanced;
YTrain = Y_Balanced;

% 4. Shuffle the data (Important so batches aren't ordered by class)
shuf_idx = randperm(length(YTrain));
XTrain = XTrain(shuf_idx, :);
YTrain = YTrain(shuf_idx, :);

disp(' Training Set is now perfectly balanced.');

disp('--- CHECKING TRAINING SET BALANCE ---');
class_counts = zeros(10, 1);
class_names = {'Normal', 'AF', 'LSM', 'AVB', 'ESM', 'LDM', 'MSM', 'S3', 'S4', 'T'};

fprintf('\n%-10s | %-15s\n', 'Class ID', 'Training Samples');
fprintf('--------------------------\n');
for c = 1:10
    class_counts(c) = sum(YTrain == c);
    fprintf('Class %-2d   | %d\n', c, class_counts(c));
end
fprintf('--------------------------\n');

% Plot Distribution
figure('Name', 'Training Data Distribution');
bar(class_counts);
xlabel('Class ID');
ylabel('Number of Samples');
xticklabels(class_names);
title('Training Set Balance Check');
grid on;
% =========================================================

%TRAIN RANDOM FOREST
disp('Training Random Forest (50 Trees)...');
rf_model = TreeBagger(50, XTrain, YTrain, ...
    'Method', 'classification', ...
    'OOBPrediction','on', ...
    'MinLeafSize', 1); 

%Testing
[pred_str, scores] = predict(rf_model, XTest);
pred = str2double(pred_str);

% Calculate Global Accuracy
acc = sum(pred == YTest) / length(YTest) * 100;

fprintf('\n============================\n');
fprintf('FINAL GLOBAL ACCURACY: %.2f%%\n', acc);
fprintf('============================\n');

% --- ADVANCED CONFUSION MATRIX ---
figure('Name', 'Confusion Matrix with Class Accuracy');
cm = confusionchart(YTest, pred, ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
title(sprintf('Random Forest Results (Global Acc: %.1f%%)', acc));

figure('Name', 'Error vs. Number of Trees');

% 1. Get Out-of-Bag Error (This acts as Training/Validation Error)
oobErr = oobError(rf_model);

% 2. Get Test Error (Performance on XTest for every tree count)
% The 'error' function calculates misclassification rate as trees are added
testErr = error(rf_model, XTest, YTest);

% 3. Plotting
plot(oobErr, 'LineWidth', 2, 'Color', [0 0.447 0.741]); % Blue
hold on;
plot(testErr, 'LineWidth', 2, 'Color', [0.85 0.325 0.098]); % Red/Orange

% Formatting
xlabel('Number of Trees');
ylabel('Classification Error (1 - Accuracy)');
title('Learning Curve: Training (OOB) vs. Test Error');
legend('Training Error (OOB)', 'Test Error', 'Location', 'NorthEast');
grid on;

% Mark the final error rates
final_oob = oobErr(end);
final_test = testErr(end);
text(40, final_oob + 0.02, sprintf('Train: %.2f', final_oob), 'Color', 'b');
text(40, final_test + 0.02, sprintf('Test: %.2f', final_test), 'Color', 'r');

% SAVE TRAINED MODEL
disp('Saving Model to disk...');
% We must save 'afe' (to know how to extract features) and 'fs' (for resampling)
save('HeartClassifier.mat', 'rf_model', 'afe', 'fs'); 
disp('Success! Model saved as "HeartClassifier.mat"');

%Slicer function
function [X, Y] = processAudio(file_list, label, afe, fs)
    X = []; Y = [];
    
    % Slicing Configuration
    window_sec = 1.0; 
    overlap_sec = 0.50; 
    win_samples = round(window_sec * fs);
    step_samples = round((window_sec - overlap_sec) * fs);
    
    % Create a Bandpass Filter (25-700 Hz)
    bpFilt = designfilt('bandpassiir', ...
        'FilterOrder', 4, ...
        'HalfPowerFrequency1', 25, ...
        'HalfPowerFrequency2', 700, ...  
        'SampleRate', fs);
    
    for f = 1:length(file_list)
        fname = file_list{f}; 
        if isfile(fname)
            [audio, fs_curr] = audioread(fname);
            audio = audio(:,1); 
            if fs_curr ~= fs, audio = resample(audio, fs, fs_curr); end
            
            % --- PRE-PROCESSING---
            audio = audio / (max(abs(audio)) + eps);
            audio = filtfilt(bpFilt, audio);
            
            % --- SLICING LOOP ---
            for start_idx = 1 : step_samples : (length(audio) - win_samples)
                slice = audio(start_idx : start_idx + win_samples - 1);
                
                % Extraction
                feats = extract(afe, slice);
                
                % Condense
                stats = [mean(feats,1), std(feats,0,1), range(feats,1)];
                X = [X; stats]; 
                Y = [Y; label];
            end
        end
    end
end