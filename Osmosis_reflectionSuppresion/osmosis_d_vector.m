%% Authors: Simone Parisotto, Marco Caliari
% A on u: div(mask.*(grad u-((grad f)/f).*u))
% L on u: div(mask.*grad u)
% WW: input matrix field for anisotropic diffusion (w*h*4)
function [d1ij, d2ij] = osmosis_d_vector(v)
% Input:
%   umat  = shadowed image
%   umask = shadow boundary indicator function

[mx, my, ~] = size(v);

x  = linspace(1,mx,mx)';
y  = linspace(1,my,my)';
hx = (max(x)-min(x))/(mx-1);
hy = (max(y)-min(y))/(my-1);

% STANDARD DRIFT VECTOR FIELD d
d1ij = zeros(mx+1,my); 
d2ij = zeros(mx,my+1);

d1ij(2:mx,:) = (diff(v,1,1)./(v(2:mx,:)+v(1:mx-1,:))*2/hx);%.*(mask(2:mx,:)+mask(1:mx-1,:))/2;
d2ij(:,2:my) = (diff(v,1,2)./(v(:,2:my)+v(:,1:my-1))*2/hy);%.*(mask(:,2:my)+mask(:,1:my-1))/2;

%  STANDARD OSMOSIS FILTER
%{
Ax = Dxx_old + 1/hx*( ...
    spdiags(reshape(d1ij(2:mx+1,:),mx*my,1),0,mx*my,mx*my)*M1xup - ...
    spdiags(reshape(d1ij(1:mx,:),mx*my,1),0,mx*my,mx*my)*M1xlow);

Ay = Dyy_old + 1/hy*( ...
    spdiags(reshape(d2ij(:,2:my+1),mx*my,1),0,mx*my,mx*my)*M1yup - ...
    spdiags(reshape(d2ij(:,1:my),mx*my,1),0,mx*my,mx*my)*M1ylow);

A = Ax + Ay;
%}

end