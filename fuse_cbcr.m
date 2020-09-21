%Multi-Exposure and Multi-Focus Image Fusion in Gradient Domain
function [fused_C] = fuse_cbcr(C_set)

[W, H, N] = size(C_set);
weight_each = zeros([W, H, N]);
weight_total = sum(abs(C_set(:,:,:)-128),3)+eps;
for n = 1:N
    weight_each(:,:,n) = abs(C_set(:,:,n)-128)./weight_total;
end

fused_C = zeros([W, H]);
for n = 1:N
    fused_C(:,:) = fused_C(:,:)+weight_each(:,:,n).*(C_set(:,:,n)-128);
end
fused_C = fused_C+128;

end

