% Script to generate 2 deformed masks for each ground truth in DAVIS 2016
% dataset
% Name: Omkar Damle
% Date: 20th Feb 2018
%clear all; close all;
base_path = '/home/omkar/VOS/DAVIS-2016/';

annotation_path = [base_path 'Annotations/480p/']
deformation_path = [base_path 'Deformations/480p/']
mkdir(deformation_path)
videos = dir(annotation_path);
numVideos = numel(videos)-2;

for i = 3:numVideos
    
    mask_gt_path = [annotation_path videos(i+2).name '/'];
    deform_folder_path = [deformation_path videos(i+2).name '/'] ;
    mkdir(deform_folder_path);
    frames = dir([mask_gt_path '*.png']);
    numFrames = numel(frames);
    
    for j = 1:numFrames
        frame_path = [mask_gt_path frames(j).name]
        frame_gt_image = imread(frame_path);
        frameIndex = frames(j).name(1:end-4);
        affine_transformation_path = [deform_folder_path frameIndex '_d1.png'];
        non_rigid_deform_path = [deform_folder_path frameIndex '_d2.png'];
        augment_image_and_mask(frame_gt_image, affine_transformation_path, non_rigid_deform_path);
    end
    
end
 
%image = imread('image.jpg');
%label = imread('label.png');
%augment_image_and_mask(image, label, 'augmentations', 'bear', 'myFile');
