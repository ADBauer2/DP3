%% DP3 weeks view instead of Day view
%% Load Data

data = readtable("C:\Users\alexb\Downloads\DP3_playset.csv");



% Find rows containing NaN values in the numeric column
nanRowsNumeric = isnat(data.fitbit_data_date);


clean = data(~nanRowsNumeric, :);

nanTimePoints = isnan(clean.timepoint);

clean = clean(~nanTimePoints, :);
%% Clean data
timepoints = 1:1:365; %in days

control_data  = clean(clean.compY_n == 0, :);
test_data  = clean(clean.compY_n == 1, :);

uniques_control = unique(control_data.record_id);
uniques_test = unique(test_data.record_id);


test_steps = zeros(length(uniques_test), 365);
control_steps = zeros(length(uniques_control), 365);
test_dist = zeros(length(uniques_test), 365);
control_dist = zeros(length(uniques_control), 365);

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

%% Summate each week into a single value

i = 1;
count = 1;
step_weeky = [];
for n = 7:7:365
    mean(test_steps(:, i:n), 2, 'omitmissing')
    disp(count)
    i = i + n;
    count = count+1;
end

