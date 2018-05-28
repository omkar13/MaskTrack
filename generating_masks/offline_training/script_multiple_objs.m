% Script to generate 2 deformed masks for each ground truth object in DAVIS
% 2017
% dataset
% Name: Omkar Damle
% Date: April 2018

clear all; close all;
base_path = '/home/omkar/Documents/Omkar/VOS/DAVIS17/';

annotation_path = [base_path 'Annotations/480p/'];
deformation_path = [base_path 'Deformations/480p/'];
mkdir(deformation_path)
videos = dir(annotation_path);
numVideos = numel(videos)-2;

for i = 1:numVideos
   
    mask_gt_path = [annotation_path videos(i+2).name '/']
    deform_folder_path = [deformation_path videos(i+2).name '/'] ;
    
    if 7==exist(deform_folder_path,'dir')
        continue
    end
    
    mkdir(deform_folder_path);
    
        
    frames = dir([mask_gt_path '*.png']);
    numFrames = numel(frames)
    
    no_objects = -1;
    
    for j = 1:numFrames
        frame_path = [mask_gt_path frames(j).name]
        frame_gt_image = imread(frame_path);

        if j == 1
            no_objects = max(frame_gt_image(:));
        end
        frameIndex = frames(j).name(1:end-4);

        for object_id = 1:(no_objects)
            temp = frame_gt_image;
            save_image=0;
            temp(temp~=object_id) = 0;
            temp(temp==object_id) = 1;
           
            affine_transformation_path = [deform_folder_path frameIndex '_' num2str(object_id) '_d1.png'];
            non_rigid_deform_path = [deform_folder_path frameIndex '_' num2str(object_id) '_d2.png'];
            augment_image_and_mask(temp, affine_transformation_path, non_rigid_deform_path);
        end
    end 
end