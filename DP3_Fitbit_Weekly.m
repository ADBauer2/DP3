%% Load Data

data = readtable("C:\Users\alexb\Downloads\DP3_playset.csv");



% % Find rows containing NaN values in the numeric column
% nanRowsNumeric = isnat(data.fitbit_data_date);
% 
% 
% clean = data(~nanRowsNumeric, :);
% 
nanTimePoints = isnan(data.timepoint);
% 
clean = data(~nanTimePoints, :);
%% Clean data
timepoints = 1:1:365; %in days

control_data  = clean(clean.compY_n == 0, :);
test_data  = clean(clean.compY_n == 1, :);

uniques_control = unique(control_data.record_id);
uniques_test = unique(test_data.record_id);


test_steps = nan(length(uniques_test), 365);
control_steps = nan(length(uniques_control), 365);
test_dist = nan(length(uniques_test), 365);
control_dist = nan(length(uniques_control), 365);

control = cell(numel(uniques_control), 1); % Initialize cell arrays to store the columns for each record ID
test = cell(numel(uniques_test), 1);

% Loop through each unique record ID
for i = 1:numel(uniques_control)
    % Get the rows corresponding to the current record ID
    rows = strcmp(control_data.record_id, uniques_control{i});
    
    % Extract the desired columns for the current record ID
    control{i, 1} = control_data.fb_act_summ_steps(rows);
    control{i, 2} = control_data.timepoint(rows) + 1;
    control{i, 3} = control_data.fb_act_summ_totaldistances(rows);
end

for i = 1:numel(uniques_test)
    % Get the rows corresponding to the current record ID
    rows = strcmp(test_data.record_id, uniques_test{i});
    
    % Extract the desired columns for the current record ID
    test{i, 1} = test_data.fb_act_summ_steps(rows);
    test{i, 2} = test_data.timepoint(rows) + 1;
    test{i, 3} = test_data.fb_act_summ_totaldistances(rows);
end
%% Convert to long
for i = 1:numel(uniques_control)
    disp(i)
    control_steps(i, control{i, 2}) = control{i, 1}';
end
control_steps = control_steps(:, 1:365);
for i = 1:numel(uniques_test)
    disp(i)
    test_steps(i, test{i, 2}) = test{i, 1}';
end
test_steps = test_steps(:, 1:365);
for i = 1:numel(uniques_control)
    disp(i)
    control_dist(i, control{i, 2}) = control{i, 3}';
end
control_dist = control_dist(:, 1:365);
for i = 1:numel(uniques_test)
    disp(i)
    test_dist(i, test{i, 2}) = test{i, 3}';
end
test_dist = test_dist(:, 1:365);

%%

i = 1;
count = 1;
test_weekly_s = [];
control_weekly_s = [];
for n = 7:7:365
    test_weekly_s = [test_weekly_s, sum(test_steps(:, i:n), 2, "omitmissing")]
    control_weekly_s = [control_weekly_s, sum(control_steps(:, i:n), 2, "omitmissing")]
    count = count + 1;
    i = i + 7;
    disp(n)
    disp(i)
end

%% Prep Gaussian smoothing

sigma_w  = 1; %day
binsize = 1;

tau = -5*sigma_w : binsize : 5*sigma_w;
w = binsize* (1/(sqrt(2*pi)*sigma_w)) * exp(-tau.^2/(2*sigma_w^2));
%% Steps
timepoints = 1:52;
control_smooth_s = [];
for entry = 1:length(uniques_control)
    rate_w = imfilter(control_weekly_s(entry, :), w, 'conv');
    control_smooth_s = [control_smooth_s;rate_w];
end

control_psth = sum(control_weekly_s(:, 1:52), 1, 'omitnan')/size(control_weekly_s, 1);

rate_control = imfilter(control_psth, w, 'conv');
figure(1)
hold on
plot(timepoints, control_smooth_s(:, 1:52), color = 'black')
plot(timepoints, rate_control, color = 'red', LineWidth=3)
ylabel('Steps (in 10,000s)')
xlabel('Time (Days)')
hold off

test_smooth_s = [];
for entry = 1:length(uniques_test)
    rate_w = imfilter(test_weekly_s(entry, :), w, 'conv');
    test_smooth_s = [test_smooth_s;rate_w];
end

test_psth = sum(test_weekly_s(:, 1:52), 1, 'omitnan')/size(test_weekly_s, 1);

rate_test = imfilter(test_psth, w, 'conv');

%% Comparison of Groups Steps

num_time_points = size(control_weekly_s, 2); % Assuming control_long and test_long have the same number of time points

p_values = zeros(1, num_time_points);
    
for i = 1:num_time_points
    % Extract the data for the current time point
    control_data = control_weekly_s(:, i);
    test_data = test_weekly_s(:, i);

    % Remove NaN values and ensure equal length
    control_data = control_data(~isnan(control_data));
    test_data = test_data(~isnan(test_data));

    % Perform Welch's ANOVA
    [~, p_values(i)] = vartest2(control_data, test_data);
end

figure(3);
% Assuming p_values is a vector of p-values obtained from statistical analysis
significance_threshold = 0.05; % Adjust as needed

% Identify significant time points based on the significance threshold
significant_time_points = p_values < significance_threshold;

% Reshape the binary vector into a matrix for heatmap plotting
% Assuming num_time_points is the total number of time points
binary_matrix = reshape(significant_time_points(1, 1:52), [], length(timepoints))';

times = find(binary_matrix == 1);

%times([2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 75, 76, 77, 78, 79, 81, 82, 83, 84, 86, 87, 88, 89 ...
%    , 91, 92, 93, 94, 96, 97, 98, 99, 101, 102, 103, 104, 106, 107, 108, 109, 111, 112, 113, 114, 116, ...
 %   117, 118, 119, 121, 122, 123,124, 126, 127, 128, 129, 131, 132, 133, 134, 136, 137, 138, 139]) = [];


figure(1);
%TEST
subplot(2, 1, 1)
hold on
plot(timepoints, test_smooth_s(:, 1:52), Color = [0,0,0,0.5]);
line(1:52, mean(test_smooth_s, "all")* ones(1, 52), LineWidth = 3)
ylim([0,100000])
xlim([11, 41])
for i = 1:numel(significant_time_points(1:52))
    diff = rate_control(i) - rate_test(i);
    if diff > 0 && significant_time_points(i) > 0 
        col = 'g';
        x = plot([(i) * significant_time_points(i), (i) * significant_time_points(i)], ylim, '-', 'LineWidth', 3, color = col);
        x.Color(4) = 0.2;
    elseif diff < 0 && significant_time_points(i) > 0 
        col = 'b';
        x = plot([(i) * significant_time_points(i), (i) * significant_time_points(i)], ylim, '-', 'LineWidth', 3, color = col);
        x.Color(4) = 0.2;
    elseif diff == 0 && significant_time_points(i) > 0 
        col = 'r';
        x = plot([(i) * significant_time_points(i), (i) * significant_time_points(i)], ylim, '-', 'LineWidth', 3, color = col);
        x.Color(4) = 0.2;  
    end  
end
plot(timepoints, rate_test, color = 'red', LineWidth=3)


ylabel('Steps (in 10,000s)')
xlabel('Time (Days)')
hold off
title('Complication Group');
ylabel('Steps (in 10,000s)')
xlabel('Time (Weeks)')
% xticks(times)
xtickangle(90)
ax = gca;
ax.FontSize = 8;

%CONTROL
subplot(2, 1, 2)
hold on
plot(timepoints, control_smooth_s(:, 1:52), Color = [0,0,0,0.5]);
line(1:52, mean(control_smooth_s, "all")* ones(1, 52), LineWidth = 3)
ylim([0,100000])
xlim([11, 41])
for i = 1:numel(significant_time_points(1:52))
    diff = rate_control(i) - rate_test(i);
    if diff > 0 && significant_time_points(i) > 0 
        col = 'g';
        x = plot([(i) * significant_time_points(i), (i) * significant_time_points(i)], ylim, '-', 'LineWidth', 3, color = col);
        x.Color(4) = 0.2;
    elseif diff < 0 && significant_time_points(i) > 0 
        col = 'b';
        x = plot([(i) * significant_time_points(i), (i) * significant_time_points(i)], ylim, '-', 'LineWidth', 3, color = col);
        x.Color(4) = 0.2;
    elseif diff == 0 && significant_time_points(i) > 0 
        col = 'r';
        x = plot([(i) * significant_time_points(i), (i) * significant_time_points(i)], ylim, '-', 'LineWidth', 3, color = col);
        x.Color(4) = 0.2;  
    end  
end
plot(timepoints, rate_control, color = 'red', LineWidth=3)
hold off
title('Control Group');
ylabel('Steps (in 10,000s)')
xlabel('Time (Weeks)');
% xticks(times)
xtickangle(90)
ax = gca;
ax.FontSize = 8;

annotation('textbox', [0.8, 0.45, 0.1, 0.1], 'String', sprintf('Significant Time Point Highlight\nGreen Highlights: Control Steps > Complication Steps\nBlue Highlights: Control Steps < Complication Steps'), 'HorizontalAlignment', 'center', 'FontSize', 12);


%%
% Loop through each subject
behave_test = cell(size(test_weekly_s, 1), 1);
behave_control = cell(size(control_weekly_s, 1), 1);
for i = 1:37
    % Calculate the autocorrelation for the time series of subject i
    [acf_t, lags_t] = autocorr(test_weekly_s(i, :), 'NumLags', 20);
    if i < 37
        [acf_c, lags_c] = autocorr(control_weekly_s(i, :), 'NumLags', 20);
    end
    % Determine if the subject's time series is cyclical based on the autocorrelation
    if any(acf_t(2:end) > 0.85) % Example threshold, adjust as needed
        behave_test{i} = 'Cyclical';
    else
        behave_test{i} = 'Non-Cyclical';
    end
    if any(acf_c(2:end) > 0.85) && i < 37% Example threshold, adjust as needed
        behave_control{i} = 'Cyclical';
    else
        behave_control{i} = 'Non-Cyclical';
    end
end

% Convert labels to a categorical variable
% Find unique strings and their counts
[uniqueStrings_c, ~, idx] = unique(behave_control);
counts_c = histcounts(idx, 'BinMethod', 'integers', 'BinLimits', [1, numel(uniqueStrings_c)]);
[uniqueStrings_t, ~, idx] = unique(behave_test);
counts_t = histcounts(idx, 'BinMethod', 'integers', 'BinLimits', [1, numel(uniqueStrings_t)]);
% Create the bar plot
counts = [counts_c; counts_t]';
names = {'Control' 'Complication'};
figure;
b = bar(counts);
colors = [
    0, 1, 0; % Green
    0, 1, 0; % Green
    0, 0, 1; % Blue
    0, 0, 1; % Blue
];
% Apply colors to each bar

set(gca, 'XTickLabel', names, 'XTick', 1:2)
xlabel('Categories');
ylabel('Counts');
title('Bar Plot of Step Behavior');
legend('Cyclical', 'Consistent')

%% Time to dropout

transitionIndices_c = cell(size(control_weekly_s, 1), 1);

% Loop through each row
for i = 1:size(control_weekly_s, 1)
    row = control_weekly_s(i, :);
    % Find where the transition from a non-zero number to zero occurs
    indices = find(row(1:end-2) ~= 0 & row(2:end-1) == 0 & row(3:end) == 0) + 1;
    % Store the indices
    if isempty(indices)
        transitionIndices_c{i} = indices;
    else
        transitionIndices_c{i} = indices(1, end);
    end
end

transitionIndices_t = cell(size(control_weekly_s, 1), 1);

% Loop through each row
for i = 1:size(test_weekly_s, 1)
    row = test_weekly_s(i, :);
    % Find where the transition from a non-zero number to zero occurs
    indices = find(row(1:end-2) ~= 0 & row(2:end-1) == 0 & row(3:end) == 0) + 1;
    % Store the indices
    if isempty(indices)
        transitionIndices_t{i} = indices;
    else
        transitionIndices_t{i} = indices(1, end);
    end
end

%% ADF Test

function pValues = check_stationarity(group)
    pValues = zeros(size(group, 1), 1);
    for i = 1:size(group, 1)
        [~, pValue] = adftest(group(i, :));
        pValues(i) = pValue;
    end
end

