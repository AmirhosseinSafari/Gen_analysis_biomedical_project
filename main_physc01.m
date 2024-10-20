clear
clc

% Loading dataset
raw_data = load('aaindex_all.mat');
dataset = raw_data.aaind_val;

% Split just special rows (physical/chemical features)
rows_num = [151, 153, 157, 159, 170, 241];
dataset = dataset(rows_num, :);

% Calculate the min and max values for each row
min_values = min(dataset, [], 2);
max_values = max(dataset, [], 2);

% Calculate the range for each row
range = max_values - min_values;

% Divide the range into n parts
n = 5;
part_size = range / n;

% Initialize an array to store the labels
labels = zeros(size(dataset));

% Calculating
for i = 1:size(dataset, 1)
    % Calculate the boundaries for each part
    boundaries = min_values(i) + (1:n) * part_size(i);
    
    % Assign labels
    for j = 1:size(dataset, 2)
        for k = 1:n
            if dataset(i, j) <= boundaries(k)
                labels(i, j) = k;
                % Break when assigned
                break;
            end
        end
    end
end

%===============================================%
% Apply SGT
%===============================================%

% Define the mapping
mapping = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

% Convert each integer in labels to its assci character
[m, n] = size(labels); 
character_sequence = cell(m, 1);

% Map each integer to its character
for i = 1:m
    character_sequence{i} = mapping(labels(i, :));
end

% Applying SGT
k = 0.1;

n = 5;
store_row_size = length(character_sequence);
store_column_size = n ^ 2;
%store_column_size = length(unique(character_sequence{1})) ^ 2;

psi_tot = zeros(store_row_size , store_column_size);
psi_tot_L = zeros(store_row_size , store_column_size);

for i=1:length(character_sequence)
    [psi_tot(i, :),psi_tot_L(i, :)] = sgtfun_v2(character_sequence{i},k, n);

    fprintf('psi and psi_L %d\n', i);
    disp( psi_tot(i,:).' );
    disp( psi_tot_L(i,:).' );
end