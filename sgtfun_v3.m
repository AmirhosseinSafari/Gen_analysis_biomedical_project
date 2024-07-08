function [psi_vec,psi_vec_L]=sgtfun_v3(seq,k,label,n)

numseq = label(1,aa2int(seq)); % Convert sequence to integer format

% numseq=aa2int(seq);
phi_mat=zeros(n,n);
cnt_mat=zeros(n,n);

for i=1:length(numseq)
    for j=i+1:length(numseq)
        phi=exp(-k*(j-i));
        if numseq(i)<=20 && numseq(j)<=20
            phi_mat(numseq(i),numseq(j))=phi_mat(numseq(i),numseq(j))+phi;
            cnt_mat(numseq(i),numseq(j))=cnt_mat(numseq(i),numseq(j))+1;
        end
    end
end

psi_mat=phi_mat./cnt_mat;
psi_mat(isnan(psi_mat))=0;
psi_vec=psi_mat(:)';
psi_vec_L = psi_vec/length(numseq);