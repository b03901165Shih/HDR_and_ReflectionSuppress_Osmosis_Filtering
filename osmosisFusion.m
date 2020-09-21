clear;

image_path_root = 'source_images\'; % 'MFF_source_images\processed\'; %'source_images\'; %
directories = dir(image_path_root);
directories = directories(~ismember({directories.name}, {'.', '..'}));

result_path = 'results_tengaussdouble\';
if ~exist(result_path, 'dir')
   mkdir(result_path)
end

adam_it = 4000;
plot_img = false;

for d = 1:size(directories, 1)
    d_name = directories(d).name;
    %if(~strcmp(d_name, 'Studio'))
        %continue;
    %end
    image_path = [image_path_root, d_name, '\'];
    
    fprintf('Image name: %s\n', d_name);
    
    multi_images = dir ([image_path,'*.*g']);
    multi_images = multi_images(:,:);

    for index = 1:size(multi_images,1) % which data are you using?
        foreground_filename = multi_images(index).name;
        tmp = im2double(imread([image_path,foreground_filename]));
        tmp = imresize(tmp, 1);
        tmp= rgb2ycbcr(tmp);
        if(index==1)
            [W, H, C] = size(tmp);
            image_set = zeros([W, H, C, size(multi_images,1)]);
        end
        % load images
        image_set(:,:,:,index) = tmp;
    end

    % final all d1ij, d2ij and fusion by gradient mag
    C = 1;
    d1ij_set = zeros([W+1, H, C, size(multi_images,1)]);
    d2ij_set = zeros([W, H+1, C, size(multi_images,1)]);

    for c = 1:C
        for  k = 1:size(multi_images,1)
            [d1ij_set(:,:,c,k), d2ij_set(:,:,c,k)] = osmosis_discretization_masked(image_set(:,:,c,k)+5/255);
        end
    end

    f3 = [0,1,0;1,-4,1;0,1,0];
    LAPLACE = @(u) imfilter(u,f3,'circular');
    grad_mag_set = zeros([W, H, C, size(multi_images,1)]);
    for c = 1:C
        for  k = 1:size(multi_images,1)
            [gx, gy] = gradient(image_set(:,:,c,k));
            grad_mag_set(:,:,c,k) = sqrt(gx.^2+gy.^2); %abs(LAPLACE(image_set(:,:,c,k)));%sqrt(gx.^2+gy.^2);
            %grad_mag_set(:,:,c,k)  = imgaussfilt(grad_mag_set(:,:,c,k), 10); %imgaussfilt
            %grad_mag_set(:,:,c,k) = grad_mag_set(:,:,c,k)./((image_set(:,:,c,k)+1e-3));
        end
    end

    max_grad_mag_map = zeros([W, H, C]);
    for  c = 1:C
       max_grad_mag_map(:,:,c) = max(squeeze(grad_mag_set(:,:,c,:)), [], 3);
    end


    fusion_map_set = zeros([W, H, C, size(multi_images,1)]);
    edge_map = zeros([W, H, C]);
    for c = 1:C
        for  k = 1:size(multi_images,1)
            fusion_map_set(:,:,c,k) = double(grad_mag_set(:,:,c,k)==max_grad_mag_map(:,:,c));%(abs(grad_mag_set(:,:,c,k))+eps)./(abs(max_grad_mag_map(:,:,c))+eps);%
            %grad_mag_set(:,:,c,k) = imerode(imdilate(grad_mag_set(:,:,c,k),  strel('disk',10,0)), strel('disk',10,0));
            edge_map(:,:,c)  = edge_map(:,:,c)+(fusion_map_set(:,:,c,k)-imerode(fusion_map_set(:,:,c,k), strel('disk', 1, 0)));
            fusion_map_set(:,:,c,k)  = imgaussfilt(fusion_map_set(:,:,c,k), 10);
        end
    end
    fusion_map_set = padarray(fusion_map_set(:,:,:,:), [1 1 0 0], 'replicate', 'pre');
    weight_map = sum(fusion_map_set(:,:,:,:),  4)+1e-3;

    u_init = zeros([W, H, C]);
    d1ij_fuse = zeros([W+1, H, C]); %zero pad left, zero out right (W+1)
    d2ij_fuse = zeros([W, H+1, C]); %zero pad top, zero out bottom (H+1)
    for c = 1:C
        for  k = 1:size(multi_images,1)
            tmp = image_set(:,:,c,k).*double(fusion_map_set(2:end,2:end,c,k));
            u_init(:,:,c) = u_init(:,:,c)+tmp;%(tmp/mean2(tmp))/size(multi_images,1);
            %figure, imshow(tmp)
            d1ij_fuse(:,:,c) =d1ij_fuse(:,:,c)+d1ij_set(:,:,c,k).*double(fusion_map_set(:,2:end,c,k));
            d2ij_fuse(:,:,c) =d2ij_fuse(:,:,c)+d2ij_set(:,:,c,k).*double(fusion_map_set(2:end,:,c,k));
        end
        u_init(:,:,c) = u_init(:,:,c)./ weight_map(2:end,2:end,c);
        d1ij_fuse(:,:,c) = d1ij_fuse(:,:,c)./weight_map(:,2:end,c);
        d2ij_fuse(:,:,c) = d2ij_fuse(:,:,c)./weight_map(2:end,:,c);
    end
    
    if(plot_img)
        figure, imshow(u_init);
    end

    [~,D1,D2]      = grad_forward(image_set(:,:,1,1));
    GRAD   = @(u) cat(3,reshape(D1*u(:),W,H),reshape(D2*u(:),W,H));

    %weight = 1./sum(image_set(:,:,:,:), 4).^5;
    grad_duO = @(A,u) reshape( A*u(:) ,W,H);
    grad_duD = @(u, v, mask)  (mask).*(u-v);

    sum_grad = @(u) sum(sum(sqrt(reshape(D1*u(:),W,H).^2+reshape(D2*u(:),W,H).^2)));

    % default parameters for adam
    alpha = 0.01;
    beta_1 = 0.9;
    beta_2 = 0.999;
    epsilon = 10^(-8);

    result = zeros([W, H, C]);

    for c = 1:C
        u = u_init(:,:,c);%mean2(u_init(:,:,c))*ones([W, H]);%%image_set(:,:,c,2);%u_init(:,:,c);
        u0 = u;
        [A, L] = create_A(u, d1ij_fuse(:,:,c), d2ij_fuse(:,:,c));
        %A = create_A(u, d1ij_set(:,:,c, 3), d2ij_set(:,:,c, 3));

        mt = 0;
        vt = 0;
        rhoADMM = 0;
        u_old = u;
        %figure, imshow(u);
        for t = 1:adam_it
                grad = grad_duO(A,u);%+0.1*grad_duD(u, u0, grad_mag_set(2:end,2:end,c,2));
                mt = beta_1 * mt + (1 - beta_1) * grad;
                vt = beta_2 * vt + (1 - beta_2) * (grad .* grad);
                mth = mt / (1 - beta_1^t);
                vth = vt / (1 - beta_2^t);
                u = u - alpha * mth ./ (sqrt(vth) + epsilon);
                %u(grad_mag_set(:,:,c,2)==1) = u0(grad_mag_set(:,:,c,2)==1);n);
                if(mod(t,1000)==1)
                    fprintf(' Adam iter #%d | Grad Mag: %2.2f\n', t-1, sum_grad(u));
                    if(plot_img)
                        imshow(u)
                    end
                end
        end
        result(:,:,c) =u;
    end

    for c= 2:3
        result(:,:,c) = fuse_cbcr(squeeze(image_set(:,:,c,:))*255);
    end

    %% Post-Processing
    post_result = result;
    
    %Tone-mapping
    for c = 1:1
        Lm = post_result(:,:,c)./mean2(post_result(:,:,c));
        white = max(Lm(:));
         post_result(:,:,c) = (Lm.*(1+Lm./(white^2))./(1+Lm));    
    end  
    
    
    % Luminance and Chrominance comhination for color images 
    post_result(:,:,1)=ChannelNorm((post_result(:,:,1)*255),[16 235]);   %Gamma Correction       
    post_result = double(post_result);

    %Local histogram equalization
    range_max = max(max(post_result(:,:,1)));
    range_min = min(min(post_result(:,:,1)));
    O = (post_result(:,:,1)-range_min)/(range_max-range_min); 
    O = adapthisteq(O,'NumTiles',[8 8],'ClipLimit',0.003)*219+16; %, 'Distribution', 'rayleigh', 'Alpha', 0.55
    post_result(:,:,1) = uint8(O);
    post_result = ycbcr2rgb(uint8(post_result));
    
    if(plot_img)
        figure, imshow(post_result)
    end
    imwrite(post_result, [result_path, d_name, '_osmosis_fusion.png'])

end
