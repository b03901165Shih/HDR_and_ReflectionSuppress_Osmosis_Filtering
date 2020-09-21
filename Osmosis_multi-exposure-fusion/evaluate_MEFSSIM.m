clear;
addpath(genpath('MEF-SSIM\')); %addpath('MEF-SSIM\supportfunctions\');

image_path_root = 'source_images\';
testbench_path = 'results_tengauss\';%'..\..\MultiExposureFusionPapers\gradient-domain-imagefusion-master\results\';%
directories = dir(image_path_root);
directories = directories(~ismember({directories.name}, {'.', '..'}));
Q_vec = zeros(1, size(directories, 1));

for d = 1:size(directories, 1) 
    d_name = directories(d).name;
    image_path = [image_path_root, d_name, '\'];

    multi_images = dir ([image_path,'*.*g']);
    multi_images = multi_images(:,:);

    for index = 1:size(multi_images,1) % which data are you using?
        foreground_filename = multi_images(index).name;
        tmp = (imread([image_path,foreground_filename]));
        if(index==1)
            [W, H, C] = size(tmp);
            image_set = uint8(zeros([W, H, size(multi_images,1)]));
        end
        % load images
        image_set(:,:,index) = rgb2gray(tmp);
    end
    I_test = imread([testbench_path, d_name, '_osmosis_fusion_mine.png']);%_osmosis_fusion_mine %_poisson_fusion_18
    %figure, imshow(sum(image_set(:,:,:), 3)/size(multi_images,1)/255);
    %figure, imshow(I_test);
    [Q_vec(d), ~, ~] = mef_ms_ssim(image_set, I_test);
    fprintf('MEF-SSIM | %s: %2.4f\n', d_name, Q_vec(d))
end
fprintf('MEF-SSIM | Average: %2.4f\n', mean(Q_vec))


        