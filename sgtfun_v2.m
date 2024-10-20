function [psi_vec,psi_vec_L]=sgtfun_v2(seq,k,n)

% Calculating number of groups
% unique_groups_elements = unique(seq);
% last_unique_group = unique_groups_elements(end);
% n = double(upper(last_unique_group)) - double('A') + 1;

phi_mat=zeros(n, n);
cnt_mat=zeros(n, n);

for i=1:length(seq)
    for j=i+1:length(seq)
        phi=exp(-k*(j-i));
        
        % Letter to numerical number equvalent
        num_1 = double(upper(seq(i))) - double('A') + 1;
        num_2 = double(upper(seq(j))) - double('A') + 1;

        phi_mat(num_1, num_2) = phi_mat(num_1, num_2) + phi;
        cnt_mat(num_1, num_2)= cnt_mat(num_1, num_2) + 1;
    end
end

psi_mat=phi_mat./cnt_mat;
psi_mat(isnan(psi_mat))=0;
psi_vec=psi_mat(:);
psi_vec_L = psi_vec/length(seq);

psi_vec = psi_vec.';
psi_vec_L = psi_vec_L .';