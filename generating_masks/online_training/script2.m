function script2(seq_name, no_of_iterations)	
	base_path = '/home/omkar/Documents/Omkar/VOS/DAVIS17/';

	image_path  = [base_path 'JPEGImages/480p/'];
	annotation_path = [base_path 'Annotations/480p/'];
	deformation_path = [base_path 'Deformations/480p/'];
	mkdir(deformation_path)
	videos = dir(annotation_path);
	numVideos = numel(videos)-2;

	mask_gt_path = [annotation_path seq_name '/']
	image_path1 = [image_path seq_name '/']
	%deform_folder_path = [deformation_path seq_name '/'] 
	online_deform_path = [deformation_path seq_name '_online/'] ;
	%mkdir(deform_folder_path);
	mkdir(online_deform_path);
	frames = dir([mask_gt_path '*.png']);
	numFrames = numel(frames);

	for j = 1:1
		frame_path = [mask_gt_path frames(j).name];
		[frame_gt_image,map] = imread(frame_path);
        no_objects = max(frame_gt_image(:));
        
        frame_image_path = [image_path1 frames(j).name(1:end-3) 'jpg'];
		frame_image = imread(frame_image_path);
		frameIndex = frames(j).name(1:end-4);
		%affine_transformation_path = [deform_folder_path frameIndex '_d1.png'];
		%non_rigid_deform_path = [deform_folder_path frameIndex '_d2.png'];
        common_path = [online_deform_path frameIndex];

        for object_id = 1:(no_objects)
        temp = frame_gt_image;
            if object_id == 1
                save_image = 1;
            else
                save_image = 0;
            end
            
            temp(temp~=object_id) = 0;
            temp(temp==object_id) = 1;
            augment_image_and_mask_online_multiple(frame_image, temp, common_path, no_of_iterations, object_id, save_image);
        end
	end

	%image = imread('image.jpg');
	%label = imread('label.png');
	%augment_image_and_mask(image, label, 'augmentations', 'bear', 'myFile');

end
