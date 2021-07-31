function [H] = get_kernel_bandwidth(X,PC_size)
N = size(X,1);
H = (diag(std(X)) * (4/(PC_size+2)/N)^(1/(PC_size+4))) .^ 2;
end

