clc;
clear;

% Load AAindex data
raw_data = load('aaindex_all.mat');

% Transpose so rows are features
features = raw_data.aaind_val;

% Elbow method to find optimal k
max_k = 30;
sse = zeros(max_k, 1);

for k = 1:max_k
    [~, ~, sumd] = kmeans(features, k, 'Replicates', 5, 'Display', 'off');
    sse(k) = sum(sumd); % Total within-cluster sum of squares
end

% Plot elbow curve
figure;
plot(1:max_k, sse, '-o');
xlabel('Number of clusters (k)');
ylabel('Sum of Squared Errors (SSE)');
title('Elbow Method for Optimal k');

% Optionally, pick k by looking at the plot manually
best_k = input('Enter optimal k from elbow plot: ');

% Define number of clusters
k = best_k;

% Apply kmeans
[idx, C] = kmeans(features, k, 'Distance', 'sqeuclidean', 'Replicates', 10);

% Find the representative feature closest to each cluster centroid
selected_features = zeros(k,1);
for i = 1:k
    cluster_points = features(idx == i, :);
    cluster_indices = find(idx == i);
    
    dists = vecnorm(cluster_points - C(i,:), 2, 2);
    [~, min_idx] = min(dists);
    
    selected_features(i) = cluster_indices(min_idx);
end

% Display selected features
disp('Selected representative feature indices:');
disp(selected_features);
