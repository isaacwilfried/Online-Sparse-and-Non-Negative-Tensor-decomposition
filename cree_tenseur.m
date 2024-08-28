function [ X ] = cree_tenseur( A,B,C )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
I = size (A,1);
J = size (B,1);
K = size (C,1);
Xd = A*(transpose(kr(C,B)));
c = size(kr(C,B));
X = reshape (Xd, I ,J, K);

end

