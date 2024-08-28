function Z=pkr(A,B);
% khatri-rao 

F=size(A,2);
Z=zeros(size(A,1)*size(B,1),F);
for f=1:F
    Z(:,f)=kron(A(:,f),B(:,f));
end