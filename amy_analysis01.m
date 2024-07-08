clear
clc

load AmyPredData.mat
load aaindex_all.mat

%% Calculating reduced alphabets

raw_data = load('aaindex_all.mat');
dataset = raw_data.aaind_val;

% Split just special rows (physical/chemical features)
% rows_num = [151, 153, 157, 159, 170, 241];
rows_num = 157;

dataset = dataset(rows_num, :);

% Calculate the min and max values for each row
min_values = min(dataset, [], 2);
max_values = max(dataset, [], 2);

% Calculate the range for each row
range = max_values - min_values;

% Divide the range into n parts

% initializing minimum p values
min_p_values = 1;

disp('Loading negative set')
for i=1:length(s_neg_tr)
    Seq=s_neg_tr(i).Sequence;
    [psi_mat_neg(i,:),psi_L_mat_neg(i,:)]=sgtfun_v3(Seq,k,labels(1,:),n);
end
   
disp('Loading positive set')
for i=1:length(s_pos_tr)        
    Seq=s_pos_tr(i).Sequence;
    [psi_mat_pos(i,:),psi_L_mat_pos(i,:)]=sgtfun_v3(Seq,k,labels(1,:),n);    
end

disp('Compeleted')

for n = 1:20

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
    
    %% SGT
    % disp(['Amino acid alphabets are reduced to ',num2str(n),' based on ',aaind_name{rows_num}])

    % k=0.0001;
    k_list = 0.0001:0.005:1;
    for k=k_list

        X_tr = [psi_L_mat_neg;psi_L_mat_pos];
        class_tr=[ones(1,length(s_neg_tr)),2*ones(1,length(s_pos_tr))];
        
        [h,p,ci,stats]=ttest2(X_tr(class_tr==1,:),X_tr(class_tr==2,:),'Vartype','unequal','Alpha',0.001);
        
        if h == 1
            if p < min_p_values
                % Find the maximum value and its index
                [maxVal, linearIdx] = max(X_tr(:));

                % Convert linear index to row and column indices
                [row, col] = ind2sub(size(X_tr), linearIdx);
                
                % Storing values X_tr, psi_mat_pos, psi_mat_neg, n, k, 
                %   max values of X_tr row and column
                data = {X_tr, psi_mat_pos, psi_mat_neg, n, k, row, col};
            end
        end
        
        disp(data)
    end
end

% Writing values into a text file
writeCellArrayToFile(data, "best_sepratable_psi_data");

% mn=mean(X_tr);
% sd=std(X_tr);

% X_tr(:,sd==0)=[];
% mn(sd==0)=[];
% sd(sd==0)=[];
% X_tr_sc=(X_tr-mn)./sd;

% X_tr_sc=(X_tr-mean(X_tr))./(max(X_tr)-min(X_tr));
% 
%  [coeff,score,latent,tsquared,explained,mu] = pca(X_tr_sc);
% %  gscatter(score(:,1),score(:,2),class_tr)
% figure()
% for i=1:size(X_tr,1)   
%     hold on
%     if class_tr(i)==1
%     plot3(score(i,1),score(i,2),score(i,3),'.b')
%     else
%     plot3(score(i,1),score(i,2),score(i,3),'.r')
%     end
% %  text(score(:,1),score(:,2),score(:,3),num2str(class_tr'))
% end
% hold off
