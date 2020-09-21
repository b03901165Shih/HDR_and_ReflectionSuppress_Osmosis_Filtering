clear;

image_path_root = 'figures\'; % 'MFF_source_images\processed\'; %'source_images\'; %
foregrounds = dir([image_path_root,'*.*g']);
%directories = directories(~ismember({directories.name}, {'.', '..'}));

addpath(genpath('.'));

result_path = 'results\';
if ~exist(result_path, 'dir')
   mkdir(result_path)
end

h = 0.05;
offset = 1;
adam_it = 1751;
plot_img = false;

for index = 1:size(foregrounds, 1)
    
    foreground_filename = foregrounds(index).name;
    Im = im2double(imread([image_path_root,foreground_filename]))+offset;
    %Im = rgb2gray(Im);
    Im = imresize(Im,0.5);
    [W, H, C] = size(Im);
    
    % final all d1ij, d2ij and fusion by gradient mag
    d1ij = zeros([W+1, H, C]);
    d2ij = zeros([W, H+1, C]);
    for c = 1:C
        [d1ij(:,:,c), d2ij(:,:,c)] = osmosis_d_vector(Im(:,:,c)+eps);
    end
    %d1ij = d1ij*0.5; d2ij = d2ij*0.5;

    grad_mag_set = zeros([W, H, C]);
    for c = 1:C
        [gx, gy] = gradient(Im(:,:,c));
        grad_mag_set(:,:,c) = abs(imfilter(Im(:,:,c),[0,1,0;1,-4,1;0,1,0],'circular'));%sqrt(gx.^2+gy.^2); 
    end
    
   
    grad_mag_set = padarray(grad_mag_set, [1 1 0 0], 'replicate', 'both');  %'pre' 
    grad_mag_set = (grad_mag_set(1:end-1,1:end-1,:)+grad_mag_set(1:end-1,2:end,:)+grad_mag_set(2:end,1:end-1,:)+grad_mag_set(2:end,2:end,:))/4;
    grad_mag_set = wthresh(grad_mag_set, 'h', h);                     % gradient thresholding
    ind = double(grad_mag_set == 0);
    ind = imgaussfilt(double(ind),2);
    
    d1ij = (1-ind(:,2:end,:)).*d1ij;%d1ij(ind(:,2:end,:))*0.1;
    d2ij = (1-ind(2:end,:,:)).*d2ij;%d2ij(ind(2:end,:,:))*0.1;
    %%

    [~,D1,D2]      = grad_forward(Im(:,:,1));
    GRAD   = @(u) cat(3,reshape(D1*u(:),W,H),reshape(D2*u(:),W,H));

    grad_duO = @(A,u) reshape( A*u(:) ,W,H);
    grad_duD = @(u, v)  (u-v);

    % default parameters for adam
    alpha = 0.01;
    beta_1 = 0.9;
    beta_2 = 0.999;
    epsilon = 10^(-8);

    %result = zeros([W, H, C]);
    %[ WW ] = get_anisotropic_matrix( Im, rgb2gray(ind) );

    for c = 1:C
        fprintf('Processing channel %d\n', c);
        u = Im(:,:,c)+eps;
        v = u;
        [A, L] = create_A(u, d1ij(:,:,c), d2ij(:,:,c));
        %[A, ~] = osmosis_discretization_masked(u, 1-ind(:,:,c));
        %[~, L] = osmosis_discretization_masked(u, ind(:,:,c)); %WW
        %A = A+L;

        mt = 0;
        vt = 0;
        rhoADMM = 0;
        u_old = u;
        
        for t = 1:adam_it
                grad = grad_duO(A,u)+0.001*grad_duD(u,v);
                mt = beta_1 * mt + (1 - beta_1) * grad;
                vt = beta_2 * vt + (1 - beta_2) * (grad .* grad);
                mth = mt / (1 - beta_1^t);
                vth = vt / (1 - beta_2^t);
                u = u - alpha * mth ./ (sqrt(vth) + epsilon);
                if(mod(t,500)==1)
                    fprintf(' Adam iter #%d \n', t-1);
                    if(plot_img)
                        imshow(u-offset);
                    end
                end
        end
        result(:,:,c) =u;
    end
    
    
    if(plot_img)
        figure, imshow([Im-offset, result-offset]);
    end
    
    imwrite(result-offset, [result_path,foreground_filename]);
end
