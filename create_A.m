%% Authors: Simone Parisotto, Marco Caliari
% A on u: div(mask.*(grad u-((grad f)/f).*u))
% L on u: div(mask.*grad u)
% WW: input matrix field for anisotropic diffusion (w*h*4)
function [A, L] = create_A(v, d1ij, d2ij)
% Input:
%   umat  = shadowed image
%   umask = shadow boundary indicator function

[mx, my, ~] = size(v);

x  = linspace(1,mx,mx)';
y  = linspace(1,my,my)';
hx = (max(x)-min(x))/(mx-1);
hy = (max(y)-min(y))/(my-1);

[~,D1forward,D2forward] = grad_forward(ones(mx,my));

% MATRIX FOR -div(du)
% average upper (u_{i+1,j} + u_{ij})/2  and lower (u_{ij}+u_{i-1,j})/2
m1xup  = spdiags(ones(mx,2)/2,[0,1],mx,mx);
m1xlow = spdiags(ones(mx,2)/2,[-1,0],mx,mx);
% average upper (u_{i,j+1} + u_{ij})/2  and lower (u_{ij}+u_{i,j-1})/2
m1yup  = spdiags(ones(my,2)/2,[0,1],my,my);
m1ylow = spdiags(ones(my,2)/2,[-1,0],my,my);

M1xup  = kron(speye(my),m1xup);
M1xlow = kron(speye(my),m1xlow);
M1yup  = kron(m1yup,speye(mx));
M1ylow = kron(m1ylow,speye(mx));

%%
% LAPLACIAN
Dxx_old = D1forward.'*D1forward;
Dyy_old = D2forward.'*D2forward;

%  STANDARD OSMOSIS FILTER

Ax = Dxx_old + 1/hx*( ...
    spdiags(reshape(d1ij(2:mx+1,:),mx*my,1),0,mx*my,mx*my)*M1xup - ...
    spdiags(reshape(d1ij(1:mx,:),mx*my,1),0,mx*my,mx*my)*M1xlow);

Ay = Dyy_old + 1/hy*( ...
    spdiags(reshape(d2ij(:,2:my+1),mx*my,1),0,mx*my,mx*my)*M1yup - ...
    spdiags(reshape(d2ij(:,1:my),mx*my,1),0,mx*my,mx*my)*M1ylow);

A = Ax + Ay;

L = Dxx_old+Dyy_old;

end
