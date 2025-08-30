clear
clc

load AmyPredData.mat
load aaindex_all.mat

%% Data splitting: Add validation set (20% of original training data)
rng(42); % For reproducibility

% Negative samples
num_neg_tr = length(s_neg_tr);
idx_neg = randperm(num_neg_tr);
val_size_neg = round(0.2 * num_neg_tr);
s_neg_val = s_neg_tr(idx_neg(1:val_size_neg));
s_neg_train = s_neg_tr(idx_neg(val_size_neg+1:end));

% Positive samples
num_pos_tr = length(s_pos_tr);
idx_pos = randperm(num_pos_tr);
val_size_pos = round(0.2 * num_pos_tr);
s_pos_val = s_pos_tr(idx_pos(1:val_size_pos));
s_pos_train = s_pos_tr(idx_pos(val_size_pos+1:end));

% Test sets remain the same
s_neg_test = s_neg_ts;
s_pos_test = s_pos_ts;

%% Main processing
raw_data = load('aaindex_all.mat');
features = raw_data.aaind_val;

rows_num = [58, 90, 144, 401, 111, 30];
features = features(rows_num, :);

% Normalize features (z-score normalization)
features = (features - mean(features, 2)) ./ std(features, 0, 2);

n_list = [2, 5, 10, 15, 20]; % amino acid reduction alphabet
k_list = [0.0001, 0.01, 0.1, 0.5, 1, 2, 2.5, 3]; % kernel parameter

% Initialize best model storage (based on validation accuracy)
bestValAccuracy = 0;
bestParams = struct('n', [], 'k', [], 'modelName', '', 'modelParams', []);

% Store CH indices for each n
ch_indices = zeros(length(n_list), 1);

for ni = 1:length(n_list)
    n = n_list(ni);
    [labels, ch_index] = label_features(features, n);
    ch_indices(ni) = ch_index; % Average CH index for this n
    fprintf('Done the labeling using clustering for n = %d, Calinski-Harabasz Index: %.2f\n', n, ch_index);

    for k = k_list
        % Initialize cell arrays to accumulate psi_L vectors
        psi_L_mat_neg_train = cell(length(s_neg_train), 1);
        psi_L_mat_pos_train = cell(length(s_pos_train), 1);
        psi_L_mat_neg_val = cell(length(s_neg_val), 1);
        psi_L_mat_pos_val = cell(length(s_pos_val), 1);
        psi_L_mat_neg_test = cell(length(s_neg_test), 1);
        psi_L_mat_pos_test = cell(length(s_pos_test), 1);

        for r = 1:length(rows_num)
            % Negative train set
            for i = 1:length(s_neg_train)
                Seq = s_neg_train(i).Sequence;
                [~, psi_L] = sgtfun_v3(Seq, k, labels(r, :), n);
                psi_L_mat_neg_train{i} = [psi_L_mat_neg_train{i}, psi_L];
            end
            fprintf('Done with s_neg_train: %d\n', r);

            % Positive train set
            for i = 1:length(s_pos_train)
                Seq = s_pos_train(i).Sequence;
                [~, psi_L] = sgtfun_v3(Seq, k, labels(r, :), n);
                psi_L_mat_pos_train{i} = [psi_L_mat_pos_train{i}, psi_L];
            end
            fprintf('Done with s_pos_train: %d\n', r);

            % Negative val set
            for i = 1:length(s_neg_val)
                Seq = s_neg_val(i).Sequence;
                [~, psi_L] = sgtfun_v3(Seq, k, labels(r, :), n);
                psi_L_mat_neg_val{i} = [psi_L_mat_neg_val{i}, psi_L];
            end
            fprintf('Done with s_neg_val: %d\n', r);

            % Positive val set
            for i = 1:length(s_pos_val)
                Seq = s_pos_val(i).Sequence;
                [~, psi_L] = sgtfun_v3(Seq, k, labels(r, :), n);
                psi_L_mat_pos_val{i} = [psi_L_mat_pos_val{i}, psi_L];
            end
            fprintf('Done with s_pos_val: %d\n', r);

            % Negative test set
            for i = 1:length(s_neg_test)
                Seq = s_neg_test(i).Sequence;
                [~, psi_L] = sgtfun_v3(Seq, k, labels(r, :), n);
                psi_L_mat_neg_test{i} = [psi_L_mat_neg_test{i}, psi_L];
            end
            fprintf('Done with s_neg_test: %d\n', r);

            % Positive test set
            for i = 1:length(s_pos_test)
                Seq = s_pos_test(i).Sequence;
                [~, psi_L] = sgtfun_v3(Seq, k, labels(r, :), n);
                psi_L_mat_pos_test{i} = [psi_L_mat_pos_test{i}, psi_L];
            end
            fprintf('Done with s_pos_test: %d\n', r);
        end

        disp("Done the calculation of psi of features.");

        % Convert from cell to matrix
        psi_L_mat_neg_train = cell2mat(psi_L_mat_neg_train);
        psi_L_mat_pos_train = cell2mat(psi_L_mat_pos_train);
        psi_L_mat_neg_val = cell2mat(psi_L_mat_neg_val);
        psi_L_mat_pos_val = cell2mat(psi_L_mat_pos_val);
        psi_L_mat_neg_test = cell2mat(psi_L_mat_neg_test);
        psi_L_mat_pos_test = cell2mat(psi_L_mat_pos_test);

        % Create train, val, test sets
        X_train = [psi_L_mat_neg_train; psi_L_mat_pos_train];
        class_train = [ones(length(s_neg_train), 1); 2*ones(length(s_pos_train), 1)];
        X_val = [psi_L_mat_neg_val; psi_L_mat_pos_val];
        class_val = [ones(length(s_neg_val), 1); 2*ones(length(s_pos_val), 1)];
        X_test = [psi_L_mat_neg_test; psi_L_mat_pos_test];
        class_test = [ones(length(s_neg_test), 1); 2*ones(length(s_pos_test), 1)];

        % Initialize table for metrics (add CH Index column)
        model_names = {'DecisionTree', 'SVM_Linear', 'SVM_RBF', 'XGBoost', 'kNN', 'AdaBoost', 'BAG', 'GBDT', 'RF', 'LR'};
        metrics_table = table('Size', [length(model_names), 7], ...
                             'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double', 'double'}, ...
                             'VariableNames', {'Model', 'Accuracy', 'Sensitivity', 'Precision', 'F1_Score', 'AUPR', 'CH_Index'});
        metrics_table.CH_Index(:) = ch_index; % Same for all rows per n

        % === Decision Tree ===
        tree_params = {'MaxNumSplits', [5, 10, 20]};
        best_tree_acc = 0;
        best_tree_param = [];
        for max_splits = tree_params{2}
            treeModel = fitctree(X_train, class_train, 'MaxNumSplits', max_splits);
            [predTree, scoresTree] = predict(treeModel, X_val);
            scores_pos = scoresTree(:, 2);
            [accuracy_tree, ~, ~, ~, ~] = compute_metrics(class_val, predTree, scores_pos);
            if accuracy_tree > best_tree_acc
                best_tree_acc = accuracy_tree;
                best_tree_param = max_splits;
            end
        end
        treeModel = fitctree(X_train, class_train, 'MaxNumSplits', best_tree_param);
        [predTree, scoresTree] = predict(treeModel, X_test);
        scores_pos = scoresTree(:, 2);
        [accuracy_tree, sens_tree, prec_tree, f1_tree, aupr_tree] = compute_metrics(class_test, predTree, scores_pos);
        metrics_table(1, 1:6) = {model_names{1}, accuracy_tree, sens_tree, prec_tree, f1_tree, aupr_tree};
        fprintf('Decision Tree Test Accuracy: %.2f%% (MaxNumSplits=%d)\n', accuracy_tree * 100, best_tree_param);
        
        if best_tree_acc > bestValAccuracy
            bestValAccuracy = best_tree_acc;
            bestParams.n = n;
            bestParams.k = k;
            bestParams.modelName = 'DecisionTree';
            bestParams.modelParams = struct('MaxNumSplits', best_tree_param);
        end

        % === Linear SVM ===
        svm_linear_params = {'BoxConstraint', [0.1, 1, 10]};
        best_svm_linear_acc = 0;
        best_svm_linear_param = [];
        for box = svm_linear_params{2}
            svmLinear = fitcsvm(X_train, class_train, 'KernelFunction', 'linear', 'BoxConstraint', box);
            [predSVMLinear, scoresSVMLinear] = predict(svmLinear, X_val);
            scores_pos = scoresSVMLinear(:, 2);
            [accuracy_svm_linear, ~, ~, ~, ~] = compute_metrics(class_val, predSVMLinear, scores_pos);
            if accuracy_svm_linear > best_svm_linear_acc
                best_svm_linear_acc = accuracy_svm_linear;
                best_svm_linear_param = box;
            end
        end
        svmLinear = fitcsvm(X_train, class_train, 'KernelFunction', 'linear', 'BoxConstraint', best_svm_linear_param);
        [predSVMLinear, scoresSVMLinear] = predict(svmLinear, X_test);
        scores_pos = scoresSVMLinear(:, 2);
        [accuracy_svm_linear, sens_svm_linear, prec_svm_linear, f1_svm_linear, aupr_svm_linear] = compute_metrics(class_test, predSVMLinear, scores_pos);
        metrics_table(2, 1:6) = {model_names{2}, accuracy_svm_linear, sens_svm_linear, prec_svm_linear, f1_svm_linear, aupr_svm_linear};
        fprintf('SVM Linear Test Accuracy: %.2f%% (BoxConstraint=%.1f)\n', accuracy_svm_linear * 100, best_svm_linear_param);
        
        if best_svm_linear_acc > bestValAccuracy
            bestValAccuracy = best_svm_linear_acc;
            bestParams.n = n;
            bestParams.k = k;
            bestParams.modelName = 'SVM_Linear';
            bestParams.modelParams = struct('BoxConstraint', best_svm_linear_param);
        end

        % === RBF SVM ===
        svm_rbf_params = {'BoxConstraint', [0.1, 1, 10], 'KernelScale', [0.1, 1, 10]};
        best_svm_rbf_acc = 0;
        best_svm_rbf_param = struct('BoxConstraint', [], 'KernelScale', []);
        for box = svm_rbf_params{2}
            for scale = svm_rbf_params{4}
                svmRBF = fitcsvm(X_train, class_train, 'KernelFunction', 'rbf', 'BoxConstraint', box, 'KernelScale', scale);
                [predSVMRBF, scoresSVMRBF] = predict(svmRBF, X_val);
                scores_pos = scoresSVMRBF(:, 2);
                [accuracy_svm_rbf, ~, ~, ~, ~] = compute_metrics(class_val, predSVMRBF, scores_pos);
                if accuracy_svm_rbf > best_svm_rbf_acc
                    best_svm_rbf_acc = accuracy_svm_rbf;
                    best_svm_rbf_param.BoxConstraint = box;
                    best_svm_rbf_param.KernelScale = scale;
                end
            end
        end
        svmRBF = fitcsvm(X_train, class_train, 'KernelFunction', 'rbf', 'BoxConstraint', best_svm_rbf_param.BoxConstraint, 'KernelScale', best_svm_rbf_param.KernelScale);
        [predSVMRBF, scoresSVMRBF] = predict(svmRBF, X_test);
        scores_pos = scoresSVMRBF(:, 2);
        [accuracy_svm_rbf, sens_svm_rbf, prec_svm_rbf, f1_svm_rbf, aupr_svm_rbf] = compute_metrics(class_test, predSVMRBF, scores_pos);
        metrics_table(3, 1:6) = {model_names{3}, accuracy_svm_rbf, sens_svm_rbf, prec_svm_rbf, f1_svm_rbf, aupr_svm_rbf};
        fprintf('SVM RBF Test Accuracy: %.2f%% (BoxConstraint=%.1f, KernelScale=%.1f)\n', accuracy_svm_rbf * 100, best_svm_rbf_param.BoxConstraint, best_svm_rbf_param.KernelScale);
        
        if best_svm_rbf_acc > bestValAccuracy
            bestValAccuracy = best_svm_rbf_acc;
            bestParams.n = n;
            bestParams.k = k;
            bestParams.modelName = 'SVM_RBF';
            bestParams.modelParams = best_svm_rbf_param;
        end

        % === XGBoost ===
        xgb_params = {'NumLearningCycles', [50, 100, 200]};
        best_xgb_acc = 0;
        best_xgb_param = [];
        for cycles = xgb_params{2}
            xgbModel = fitcensemble(X_train, class_train, 'Method', 'AdaBoostM1', 'NumLearningCycles', cycles, 'Learners', templateTree());
            [predXGB, scoresXGB] = predict(xgbModel, X_val);
            scores_pos = scoresXGB(:, 2);
            [accuracy_xgb, ~, ~, ~, ~] = compute_metrics(class_val, predXGB, scores_pos);
            if accuracy_xgb > best_xgb_acc
                best_xgb_acc = accuracy_xgb;
                best_xgb_param = cycles;
            end
        end
        xgbModel = fitcensemble(X_train, class_train, 'Method', 'AdaBoostM1', 'NumLearningCycles', best_xgb_param, 'Learners', templateTree());
        [predXGB, scoresXGB] = predict(xgbModel, X_test);
        scores_pos = scoresXGB(:, 2);
        [accuracy_xgb, sens_xgb, prec_xgb, f1_xgb, aupr_xgb] = compute_metrics(class_test, predXGB, scores_pos);
        metrics_table(4, 1:6) = {model_names{4}, accuracy_xgb, sens_xgb, prec_xgb, f1_xgb, aupr_xgb};
        fprintf('XGBoost Test Accuracy: %.2f%% (NumLearningCycles=%d)\n', accuracy_xgb * 100, best_xgb_param);
        
        if best_xgb_acc > bestValAccuracy
            bestValAccuracy = best_xgb_acc;
            bestParams.n = n;
            bestParams.k = k;
            bestParams.modelName = 'XGBoost';
            bestParams.modelParams = struct('NumLearningCycles', best_xgb_param);
        end

        % === k-Nearest Neighbors ===
        knn_params = {'NumNeighbors', [3, 5, 7, 10]};
        best_knn_acc = 0;
        best_knn_param = [];
        for neighbors = knn_params{2}
            knnModel = fitcknn(X_train, class_train, 'NumNeighbors', neighbors);
            [predKNN, scoresKNN] = predict(knnModel, X_val);
            scores_pos = scoresKNN(:, 2);
            [accuracy_knn, ~, ~, ~, ~] = compute_metrics(class_val, predKNN, scores_pos);
            if accuracy_knn > best_knn_acc
                best_knn_acc = accuracy_knn;
                best_knn_param = neighbors;
            end
        end
        knnModel = fitcknn(X_train, class_train, 'NumNeighbors', best_knn_param);
        [predKNN, scoresKNN] = predict(knnModel, X_test);
        scores_pos = scoresKNN(:, 2);
        [accuracy_knn, sens_knn, prec_knn, f1_knn, aupr_knn] = compute_metrics(class_test, predKNN, scores_pos);
        metrics_table(5, 1:6) = {model_names{5}, accuracy_knn, sens_knn, prec_knn, f1_knn, aupr_knn};
        fprintf('kNN Test Accuracy: %.2f%% (NumNeighbors=%d)\n', accuracy_knn * 100, best_knn_param);
        
        if best_knn_acc > bestValAccuracy
            bestValAccuracy = best_knn_acc;
            bestParams.n = n;
            bestParams.k = k;
            bestParams.modelName = 'kNN';
            bestParams.modelParams = struct('NumNeighbors', best_knn_param);
        end

        % === AdaBoost ===
        adaboost_params = {'NumLearningCycles', [50, 100, 200], 'MaxNumSplits', [5, 10]};
        best_ada_acc = 0;
        best_ada_param = struct('NumLearningCycles', [], 'MaxNumSplits', []);
        for cycles = adaboost_params{2}
            for splits = adaboost_params{4}
                adaModel = fitcensemble(X_train, class_train, 'Method', 'AdaBoostM1', 'NumLearningCycles', cycles, 'Learners', templateTree('MaxNumSplits', splits));
                [predAda, scoresAda] = predict(adaModel, X_val);
                scores_pos = scoresAda(:, 2);
                [accuracy_ada, ~, ~, ~, ~] = compute_metrics(class_val, predAda, scores_pos);
                if accuracy_ada > best_ada_acc
                    best_ada_acc = accuracy_ada;
                    best_ada_param.NumLearningCycles = cycles;
                    best_ada_param.MaxNumSplits = splits;
                end
            end
        end
        adaModel = fitcensemble(X_train, class_train, 'Method', 'AdaBoostM1', 'NumLearningCycles', best_ada_param.NumLearningCycles, 'Learners', templateTree('MaxNumSplits', best_ada_param.MaxNumSplits));
        [predAda, scoresAda] = predict(adaModel, X_test);
        scores_pos = scoresAda(:, 2);
        [accuracy_ada, sens_ada, prec_ada, f1_ada, aupr_ada] = compute_metrics(class_test, predAda, scores_pos);
        metrics_table(6, 1:6) = {model_names{6}, accuracy_ada, sens_ada, prec_ada, f1_ada, aupr_ada};
        fprintf('AdaBoost Test Accuracy: %.2f%% (NumLearningCycles=%d, MaxNumSplits=%d)\n', accuracy_ada * 100, best_ada_param.NumLearningCycles, best_ada_param.MaxNumSplits);
        
        if best_ada_acc > bestValAccuracy
            bestValAccuracy = best_ada_acc;
            bestParams.n = n;
            bestParams.k = k;
            bestParams.modelName = 'AdaBoost';
            bestParams.modelParams = best_ada_param;
        end

        % === BAG ===
        bag_params = {'NumLearningCycles', [50, 100, 200]};
        best_bag_acc = 0;
        best_bag_param = [];
        for cycles = bag_params{2}
            bagModel = fitcensemble(X_train, class_train, 'Method', 'Bag', 'NumLearningCycles', cycles, 'Learners', templateTree());
            [predBag, scoresBag] = predict(bagModel, X_val);
            scores_pos = scoresBag(:, 2);
            [accuracy_bag, ~, ~, ~, ~] = compute_metrics(class_val, predBag, scores_pos);
            if accuracy_bag > best_bag_acc
                best_bag_acc = accuracy_bag;
                best_bag_param = cycles;
            end
        end
        bagModel = fitcensemble(X_train, class_train, 'Method', 'Bag', 'NumLearningCycles', best_bag_param, 'Learners', templateTree());
        [predBag, scoresBag] = predict(bagModel, X_test);
        scores_pos = scoresBag(:, 2);
        [accuracy_bag, sens_bag, prec_bag, f1_bag, aupr_bag] = compute_metrics(class_test, predBag, scores_pos);
        metrics_table(7, 1:6) = {model_names{7}, accuracy_bag, sens_bag, prec_bag, f1_bag, aupr_bag};
        fprintf('BAG Test Accuracy: %.2f%% (NumLearningCycles=%d)\n', accuracy_bag * 100, best_bag_param);
        
        if best_bag_acc > bestValAccuracy
            bestValAccuracy = best_bag_acc;
            bestParams.n = n;
            bestParams.k = k;
            bestParams.modelName = 'BAG';
            bestParams.modelParams = struct('NumLearningCycles', best_bag_param);
        end

        % === GBDT ===
        gbdt_params = {'NumLearningCycles', [50, 100, 200], 'MaxNumSplits', [5, 10]};
        best_gbdt_acc = 0;
        best_gbdt_param = struct('NumLearningCycles', [], 'MaxNumSplits', []);
        for cycles = gbdt_params{2}
            for splits = gbdt_params{4}
                gbdtModel = fitcensemble(X_train, class_train, 'Method', 'GentleBoost', 'NumLearningCycles', cycles, 'Learners', templateTree('MaxNumSplits', splits));
                [predGbdt, scoresGbdt] = predict(gbdtModel, X_val);
                scores_pos = scoresGbdt(:, 2);
                [accuracy_gbdt, ~, ~, ~, ~] = compute_metrics(class_val, predGbdt, scores_pos);
                if accuracy_gbdt > best_gbdt_acc
                    best_gbdt_acc = accuracy_gbdt;
                    best_gbdt_param.NumLearningCycles = cycles;
                    best_gbdt_param.MaxNumSplits = splits;
                end
            end
        end
        gbdtModel = fitcensemble(X_train, class_train, 'Method', 'GentleBoost', 'NumLearningCycles', best_gbdt_param.NumLearningCycles, 'Learners', templateTree('MaxNumSplits', best_gbdt_param.MaxNumSplits));
        [predGbdt, scoresGbdt] = predict(gbdtModel, X_test);
        scores_pos = scoresGbdt(:, 2);
        [accuracy_gbdt, sens_gbdt, prec_gbdt, f1_gbdt, aupr_gbdt] = compute_metrics(class_test, predGbdt, scores_pos);
        metrics_table(8, 1:6) = {model_names{8}, accuracy_gbdt, sens_gbdt, prec_gbdt, f1_gbdt, aupr_gbdt};
        fprintf('GBDT Test Accuracy: %.2f%% (NumLearningCycles=%d, MaxNumSplits=%d)\n', accuracy_gbdt * 100, best_gbdt_param.NumLearningCycles, best_gbdt_param.MaxNumSplits);
        
        if best_gbdt_acc > bestValAccuracy
            bestValAccuracy = best_gbdt_acc;
            bestParams.n = n;
            bestParams.k = k;
            bestParams.modelName = 'GBDT';
            bestParams.modelParams = best_gbdt_param;
        end

        % === RF ===
        rf_params = {'NumLearningCycles', [50, 100, 200], 'NumVariablesToSample', [0.5, 0.75, 1.0]};
        best_rf_acc = 0;
        best_rf_param = struct('NumLearningCycles', [], 'NumVariablesToSample', []);
        for cycles = rf_params{2}
            for var_ratio = rf_params{4}
                num_vars = max(1, round(var_ratio * size(X_train, 2)));
                rfModel = fitcensemble(X_train, class_train, 'Method', 'Bag', 'NumLearningCycles', cycles, 'Learners', templateTree('NumVariablesToSample', num_vars));
                [predRf, scoresRf] = predict(rfModel, X_val);
                scores_pos = scoresRf(:, 2);
                [accuracy_rf, ~, ~, ~, ~] = compute_metrics(class_val, predRf, scores_pos);
                if accuracy_rf > best_rf_acc
                    best_rf_acc = accuracy_rf;
                    best_rf_param.NumLearningCycles = cycles;
                    best_rf_param.NumVariablesToSample = num_vars;
                end
            end
        end
        rfModel = fitcensemble(X_train, class_train, 'Method', 'Bag', 'NumLearningCycles', best_rf_param.NumLearningCycles, 'Learners', templateTree('NumVariablesToSample', best_rf_param.NumVariablesToSample));
        [predRf, scoresRf] = predict(rfModel, X_test);
        scores_pos = scoresRf(:, 2);
        [accuracy_rf, sens_rf, prec_rf, f1_rf, aupr_rf] = compute_metrics(class_test, predRf, scores_pos);
        metrics_table(9, 1:6) = {model_names{9}, accuracy_rf, sens_rf, prec_rf, f1_rf, aupr_rf};
        fprintf('RF Test Accuracy: %.2f%% (NumLearningCycles=%d, NumVariablesToSample=%d)\n', accuracy_rf * 100, best_rf_param.NumLearningCycles, best_rf_param.NumVariablesToSample);
        
        if best_rf_acc > bestValAccuracy
            bestValAccuracy = best_rf_acc;
            bestParams.n = n;
            bestParams.k = k;
            bestParams.modelName = 'RF';
            bestParams.modelParams = best_rf_param;
        end

        % === LR (Logistic Regression with LASSO) ===
        lr_params = {'Lambda', [0.001, 0.01, 0.1, 1]};
        best_lr_acc = 0;
        best_lr_param = [];
        y_bin = class_train - 1; % Convert to 0/1 for logistic regression
        for lambda = lr_params{2}
            [B, FitInfo] = lassoglm(X_train, y_bin, 'binomial', 'Lambda', lambda, 'Link', 'logit');
            scoresLr = X_val * B + FitInfo.Intercept;
            predLr = 1 + (scoresLr > 0); % Convert back to 1/2 labels
            scores_pos = scoresLr;
            [accuracy_lr, ~, ~, ~, ~] = compute_metrics(class_val, predLr, scores_pos);
            if accuracy_lr > best_lr_acc
                best_lr_acc = accuracy_lr;
                best_lr_param = lambda;
            end
        end
        [B, FitInfo] = lassoglm(X_train, y_bin, 'binomial', 'Lambda', best_lr_param, 'Link', 'logit');
        scoresLr = X_test * B + FitInfo.Intercept;
        predLr = 1 + (scoresLr > 0);
        scores_pos = scoresLr;
        [accuracy_lr, sens_lr, prec_lr, f1_lr, aupr_lr] = compute_metrics(class_test, predLr, scores_pos);
        metrics_table(10, 1:6) = {model_names{10}, accuracy_lr, sens_lr, prec_lr, f1_lr, aupr_lr};
        fprintf('LR Test Accuracy: %.2f%% (Lambda=%.3f)\n', accuracy_lr * 100, best_lr_param);
        
        if best_lr_acc > bestValAccuracy
            bestValAccuracy = best_lr_acc;
            bestParams.n = n;
            bestParams.k = k;
            bestParams.modelName = 'LR';
            bestParams.modelParams = struct('Lambda', best_lr_param);
        end

        % Display metrics table (test metrics)
        fprintf('\nTest Metrics for n = %d, k = %.4f:\n', n, k);
        disp(metrics_table);
    end
end

%% Retrain best model on train + val and evaluate on test
% Combine train and val
s_neg_full = [s_neg_train; s_neg_val];
s_pos_full = [s_pos_train; s_pos_val];

% Recompute labels for best n
[best_labels, ~] = label_features(features, bestParams.n);

% Recompute psi for full and test with best k
psi_L_mat_neg_full = cell(length(s_neg_full), 1);
psi_L_mat_pos_full = cell(length(s_pos_full), 1);
psi_L_mat_neg_test = cell(length(s_neg_test), 1);
psi_L_mat_pos_test = cell(length(s_pos_test), 1);

for r = 1:length(rows_num)
    % Negative full
    for i = 1:length(s_neg_full)
        Seq = s_neg_full(i).Sequence;
        [~, psi_L] = sgtfun_v3(Seq, bestParams.k, best_labels(r, :), bestParams.n);
        psi_L_mat_neg_full{i} = [psi_L_mat_neg_full{i}, psi_L];
    end

    % Positive full
    for i = 1:length(s_pos_full)
        Seq = s_pos_full(i).Sequence;
        [~, psi_L] = sgtfun_v3(Seq, bestParams.k, best_labels(r, :), bestParams.n);
        psi_L_mat_pos_full{i} = [psi_L_mat_pos_full{i}, psi_L];
    end

    % Negative test
    for i = 1:length(s_neg_test)
        Seq = s_neg_test(i).Sequence;
        [~, psi_L] = sgtfun_v3(Seq, bestParams.k, best_labels(r, :), bestParams.n);
        psi_L_mat_neg_test{i} = [psi_L_mat_neg_test{i}, psi_L];
    end

    % Positive test
    for i = 1:length(s_pos_test)
        Seq = s_pos_test(i).Sequence;
        [~, psi_L] = sgtfun_v3(Seq, bestParams.k, best_labels(r, :), bestParams.n);
        psi_L_mat_pos_test{i} = [psi_L_mat_pos_test{i}, psi_L];
    end
end

X_full = [cell2mat(psi_L_mat_neg_full); cell2mat(psi_L_mat_pos_full)];
class_full = [ones(length(s_neg_full), 1); 2*ones(length(s_pos_full), 1)];
X_test = [cell2mat(psi_L_mat_neg_test); cell2mat(psi_L_mat_pos_test)];
class_test = [ones(length(s_neg_test), 1); 2*ones(length(s_pos_test), 1)];

% Train best model with best hyperparameters
switch bestParams.modelName
    case 'DecisionTree'
        bestModel = fitctree(X_full, class_full, 'MaxNumSplits', bestParams.modelParams.MaxNumSplits);
    case 'SVM_Linear'
        bestModel = fitcsvm(X_full, class_full, 'KernelFunction', 'linear', 'BoxConstraint', bestParams.modelParams.BoxConstraint);
    case 'SVM_RBF'
        bestModel = fitcsvm(X_full, class_full, 'KernelFunction', 'rbf', 'BoxConstraint', bestParams.modelParams.BoxConstraint, 'KernelScale', bestParams.modelParams.KernelScale);
    case 'XGBoost'
        bestModel = fitcensemble(X_full, class_full, 'Method', 'AdaBoostM1', 'NumLearningCycles', bestParams.modelParams.NumLearningCycles, 'Learners', templateTree());
    case 'kNN'
        bestModel = fitcknn(X_full, class_full, 'NumNeighbors', bestParams.modelParams.NumNeighbors);
    case 'AdaBoost'
        bestModel = fitcensemble(X_full, class_full, 'Method', 'AdaBoostM1', 'NumLearningCycles', bestParams.modelParams.NumLearningCycles, 'Learners', templateTree('MaxNumSplits', bestParams.modelParams.MaxNumSplits));
    case 'BAG'
        bestModel = fitcensemble(X_full, class_full, 'Method', 'Bag', 'NumLearningCycles', bestParams.modelParams.NumLearningCycles, 'Learners', templateTree());
    case 'GBDT'
        bestModel = fitcensemble(X_full, class_full, 'Method', 'GentleBoost', 'NumLearningCycles', bestParams.modelParams.NumLearningCycles, 'Learners', templateTree('MaxNumSplits', bestParams.modelParams.MaxNumSplits));
    case 'RF'
        bestModel = fitcensemble(X_full, class_full, 'Method', 'Bag', 'NumLearningCycles', bestParams.modelParams.NumLearningCycles, 'Learners', templateTree('NumVariablesToSample', bestParams.modelParams.NumVariablesToSample));
    case 'LR'
        y_bin = class_full - 1;
        [B, FitInfo] = lassoglm(X_full, y_bin, 'binomial', 'Lambda', bestParams.modelParams.Lambda, 'Link', 'logit');
        bestModel = struct('B', B, 'Intercept', FitInfo.Intercept); % Store coefficients and intercept
end

% Predict on test
if strcmp(bestParams.modelName, 'LR')
    scores_test = X_test * bestModel.B + bestModel.Intercept;
    pred_test = 1 + (scores_test > 0);
    scores_pos = scores_test;
else
    [pred_test, scores_test_mat] = predict(bestModel, X_test);
    scores_pos = scores_test_mat(:, 2);
end

% Compute final metrics for best model
[final_accuracy, final_sens, final_prec, final_f1, final_aupr] = compute_metrics(class_test, pred_test, scores_pos);

% Display final results
fprintf('\nBest model: %s with n = %d, k = %.4f\n', bestParams.modelName, bestParams.n, bestParams.k);
fprintf('Test Accuracy: %.2f%%, Sensitivity: %.2f, Precision: %.2f, F1: %.2f, AUPR: %.2f\n', ...
        final_accuracy * 100, final_sens, final_prec, final_f1, final_aupr);
fprintf('\nTest Metrics for All Models:\n');
disp(metrics_table);

% Save best model and its metadata
save('bestModel.mat', 'bestModel', 'bestParams', 'final_accuracy', 'metrics_table');

%% Function to label features
function [labels, avg_ch_index] = label_features(dataset, n)
    labels = zeros(size(dataset));
    ch_indices = zeros(size(dataset, 1), 1);
    
    for i = 1:size(dataset, 1)
        row_data = dataset(i, :)';
        
        % Run k-means
        [idx, ~] = kmeans(row_data, n, 'Replicates', 5);
        labels(i, :) = idx';
        
        % Compute Calinski-Harabasz Index
        eva = evalclusters(row_data, idx, 'CalinskiHarabasz');
        ch_indices(i) = eva.CriterionValues;
    end
    
    avg_ch_index = mean(ch_indices);
end

%% Function to compute metrics
function [accuracy, sensitivity, precision, f1_score, aupr] = compute_metrics(true_labels, pred_labels, scores_pos)
    % Compute confusion matrix
    cm = confusionmat(true_labels, pred_labels);
    if size(cm, 1) < 2 || size(cm, 2) < 2
        cm = [sum(true_labels == 1 & pred_labels == 1), sum(true_labels == 1 & pred_labels == 2); ...
              sum(true_labels == 2 & pred_labels == 1), sum(true_labels == 2 & pred_labels == 2)];
    end
    
    TP = cm(2, 2); % True Positives (class 2 predicted as 2)
    TN = cm(1, 1); % True Negatives (class 1 predicted as 1)
    FP = cm(1, 2); % False Positives (class 1 predicted as 2)
    FN = cm(2, 1); % False Negatives (class 2 predicted as 1)
    
    % Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN);
    
    % Sensitivity (Recall)
    sensitivity = TP / (TP + FN);
    
    % Precision
    precision = TP / (TP + FP);
    if isnan(precision)
        precision = 0; % Handle case where TP + FP = 0
    end
    
    % F1-score
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity);
    if isnan(f1_score)
        f1_score = 0; % Handle case where precision + sensitivity = 0
    end
    
    % AUPR (Area Under Precision-Recall Curve)
    [~, ~, ~, aupr] = perfcurve(true_labels, scores_pos, 2, 'XCrit', 'reca', 'YCrit', 'prec');
    if isnan(aupr)
        aupr = 0; % Handle case where curve cannot be computed
    end
end
