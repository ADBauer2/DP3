%% Load Data

data = readtable("/Users/alexbauer/Documents/DP3/DP3_playset.csv");



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
        if sum(row) > 0
            transitionIndices_c{i} = 52;
        else
            transitionIndices_c{i} = NaN;
        end
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
        if sum(row) > 0
            transitionIndices_t{i} = 52;
        else
            transitionIndices_t{i} = NaN;
        end
    else
        transitionIndices_t{i} = indices(1, end);
    end
end

transitionIndices_t = cell2mat(transitionIndices_t);
transitionIndices_c = cell2mat(transitionIndices_c);
[h, p] = ttest2(transitionIndices_c, transitionIndices_t);
disp("Reject Null Hypothesis? 1/0")
disp(h)
disp("P-Val")
disp(p)
disp("Mean Time-to-Drop (weeks) Control Group:")
disp(mean(transitionIndices_c, "omitmissing"))
disp("Difference (Control-Test) in Mean Time-to-Drop (weeks):")
disp(mean(transitionIndices_c, "omitmissing") - mean(transitionIndices_t, "omitmissing"))
disp("Median Time-to-Drop (weeks) Control Group:")
disp(median(transitionIndices_c, "omitmissing"))
disp("Difference (Control-Test) in Median Time-to-Drop (weeks):")
disp(median(transitionIndices_c, "omitmissing") - median(transitionIndices_t, "omitmissing"))

%% Look at averages between points on quarters and trimesters
trimester_points = round(linspace(0, 40, 4));
quarter_points = round(linspace(0, 40, 5));

timesplits = unique(sort([trimester_points, quarter_points]));
% Define smaller matricies - 6 segments
% 1 : 0- 10 weeks
% 2 : 10 - 13 weeks
% 3 : 13 - 20 weeks
% 4 : 20 - 27 weeks
% 5 : 27 - 30 weeks
% 6 : 30 - 40 weeks
test_1 = test_weekly_s(:, (timesplits(1, 1) + 1):timesplits(1, 2));
test_2 = test_weekly_s(:, timesplits(1, 2):timesplits(1, 3));
test_3 = test_weekly_s(:, timesplits(1, 3):timesplits(1, 4));
test_4 = test_weekly_s(:, timesplits(1, 4):timesplits(1, 5));
test_5 = test_weekly_s(:, timesplits(1, 5):timesplits(1, 6));
test_6 = test_weekly_s(:, timesplits(1, 6):timesplits(1, 7));

control_1 = control_weekly_s(:, (timesplits(1, 1) + 1):timesplits(1, 2));
control_2 = control_weekly_s(:, timesplits(1, 2):timesplits(1, 3));
control_3 = control_weekly_s(:, timesplits(1, 3):timesplits(1, 4));
control_4 = control_weekly_s(:, timesplits(1, 4):timesplits(1, 5));
control_5 = control_weekly_s(:, timesplits(1, 5):timesplits(1, 6));
control_6 = control_weekly_s(:, timesplits(1, 6):timesplits(1, 7));

midpoints = [];

for i = 1:6
    value = timesplits(i) + (( timesplits(i+1) - timesplits(i) )/2);
    midpoints = [midpoints, value];
end

%% compare mean trends at each timepoint
varnames_t = {'test_1','test_2','test_3','test_4','test_5','test_6'};
varnames_c = {'control_1','control_2','control_3','control_4','control_5','control_6'};
mean_t = [];
mean_c = [];
median_t = [];
median_c = [];
for i = 1:6
    var_t = varnames_t{i};
    var_c = varnames_c{i};

    mean_t = [mean_t, mean(eval(var_t), "all", "omitmissing")];
    mean_c = [mean_c, mean(eval(var_c), "all", "omitmissing")];
    median_t = [median_t, median(eval(var_t), "all", "omitmissing")];
    median_c = [median_c, median(eval(var_c), "all", "omitmissing")];
end


% Make median lines
med_control = median(control_weekly_s, 1, "omitmissing");
med_test = median(test_weekly_s, 1, "omitmissing");
% Plot
figure;
% Mean
subplot(2, 1, 1);
hold on;
xlim([0, 40])
ylim([-4000, 45000])
title('Mean Steps in Tri/Quarters')
%Trimesters
 area([0, 13], [-4000, -4000])
 area([13, 27], [-4000, -4000])
 area([27, 40], [-4000, -4000])
 newcolors = [252/256 3/256 152/256; 252/256 3/256 219/256;
     186/256 3/256 252/256; 252/256 3/256 152/256; 252/256 3/256 219/256; 
     186/256 3/256 252/256; 115/256 3/256 252/256];
 colororder(newcolors)
%Quarters
 area([0, 10], [-2000, -2000])
 area([10, 20], [-2000, -2000])
 area([20, 30], [-2000, -2000])
 area([30, 40], [-2000, -2000])
plot(midpoints, mean_c, color = 'Green', linewidth = 3);
plot(midpoints, mean_t, color = 'Blue', linewidth = 3);
plot(1:52, rate_control, 'g--', linewidth = 3);
plot(1:52, rate_test, 'b--', linewidth= 3)
% Median
subplot(2, 1, 2);
hold on;
title("Median Steps in Tri/Quarters")
xlim([0, 40])
ylim([-4000, 45000])
%Trimesters
 area([0, 13], [-4000, -4000])
 area([13, 27], [-4000, -4000])
 area([27, 40], [-4000, -4000])
 newcolors = [252/256 3/256 152/256; 252/256 3/256 219/256;
     186/256 3/256 252/256; 252/256 3/256 152/256; 252/256 3/256 219/256; 
     186/256 3/256 252/256; 115/256 3/256 252/256];
 colororder(newcolors)
%Quarters
 area([0, 10], [-2000, -2000])
 area([10, 20], [-2000, -2000])
 area([20, 30], [-2000, -2000])
 area([30, 40], [-2000, -2000])
plot(midpoints, median_c, color = 'Green', linewidth = 3);
plot(midpoints, median_t, color = 'Blue', linewidth = 3);
plot(1:52, med_test, 'b--', linewidth = 3);
plot(1:52, med_control, 'g--', linewidth = 3);

%%
% Binary on drop period 2-30-36
% redo means and medians
%% Length at peak
peak_indicies_c = cell(size(control_weekly_s, 1), 1);

for i = 1:size(control_weekly_s, 1)
    
    subject= control_weekly_s(i, 16:52);
    subject = subject - subject(1, 1);
    % Find where the transition from a non-zero number to zero occurs
    indices = find(subject(1:end-2) >= 0 & subject(2:end-1) < -5000 & subject(3:end) < -5000) +1 - 2;
    % if length(indices) > 1
    %     dist = [];
    %     start = 0;
    %     for j = 1:length(indices)
    %         dist = [dist, start + indices(1,j)];
    %         start = start + indices(1, j);
    %     end
    %     indices = mean(dist);
    % end
    if isempty(indices)
        peak_indicies_c{i} = 52-16;
    else
        peak_indicies_c{i} = indices;
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

