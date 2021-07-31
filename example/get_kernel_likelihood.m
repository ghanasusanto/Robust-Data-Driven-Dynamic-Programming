function [likelihood] = get_kernel_likelihood(x,X,H)
N = length(X);
likelihood = 0;
for i=1:N
    likelihood = likelihood + exp(-0.5*(x-X(i,:)')'/H*(x-X(i,:)'));
end
end

