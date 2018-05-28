% Modified by: Omkar Damle, Feb 2018. 
% Modifications include translation code
% This is the reference implementation of the data augmentation described
% in the paper:
% 
%   Learning Video Object Segmentation from Static Images 
%   A. Khoreva, F. Perazzi,  R. Benenson, B. Schiele and A. Sorkine-Hornung
%   arXiv preprint arXiv:1612.02646, 2016. 
% 
% Please refer to the publication above if you use this software. For an
% up-to-date version go to:
% 
%            http://www.mpi-inf.mpg.de/masktrack
% 
% 
% THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY EXPRESSED OR IMPLIED WARRANTIES
% OF ANY KIND, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THIS SOFTWARE OR THE USE OR OTHER DEALINGS IN
% THIS SOFTWARE.

%For each image, 80 image pairs (input for masktrack) are generated in one iteration.
%For around 1000 images, we need approx. 12 iterations.

function augment_image_and_mask_online(im0, gt0, common_path, no_of_iterations)

    shift=[-0.1:0.01:0.1];

    gt0=uint8(gt0>0);

    im_dim1 = size(gt0,1);
    im_dim2 = size(gt0,2);

    resize=1;
    flip=1;
    num_angles = 10;
    angle_step = 2;
    angles = angle_step:angle_step:num_angles*angle_step;
    angles = [angles, -angles];
    
    for iters=1:no_of_iterations
        im1=im0;
        gt1=double(gt0>0);
        %filename=[file_name '_jit_' num2str(jit)];
        seg=gt1;
        [M,N]=find(gt1>0);

        topM=min(M);
        bottomM=max(M);
        leftN=min(N);
        rightN=max(N);
        w=rightN-leftN;
        h=bottomM-topM; 

        %se - structuring element    
        se = strel('disk',1);       
        % https://en.wikipedia.org/wiki/Dilation_(morphology)#/media/File:Grayscale_Morphological_Dilation.gif
        bound=imdilate(seg,se)-seg;
        [x,y]=find(bound);
        if ~isempty(x)
            if numel(x)>4
                num_points=5;
                rand_p=randsample(numel(x),num_points);
                movingPoints=zeros(num_points,2);fixedPoints=zeros(num_points,2);

                for l=1:numel(rand_p)
                    fixedPoints(l,1)=x(rand_p(l))+ h*shift(randsample(numel(shift),1));
                 fixedPoints(l,2)=y(rand_p(l))+ w*shift(randsample(numel(shift),1));
                movingPoints(l,1)=x(rand_p(l));
                movingPoints(l,2)=y(rand_p(l));
                end
                st = tpaps(movingPoints',fixedPoints');
                [x,y]=find(seg);
                xy=[x,y];
                avals = fnval(st,xy');
                seg2=zeros(size(seg));

                %find the new points and make sure they are bounded

                for k=1:numel(avals)
                    try
                        seg2(min(max(1,floor(avals(1,k))),size(seg,1)),min(max(1,floor(avals(2,k))),size(seg,2)))=1;        
                        seg2(min(max(1,ceil(avals(1,k))),size(seg,1)),min(max(1,ceil(avals(2,k))),size(seg,2)))=1;
                         seg2(min(max(1,floor(avals(1,k))),size(seg,1)),min(max(1,ceil(avals(2,k))),size(seg,2)))=1;
                          seg2(min(max(1,ceil(avals(1,k))),size(seg,1)),min(max(1,floor(avals(2,k))),size(seg,2)))=1;
                    end
                end

                se = strel('disk',5);
                seg_new = imdilate(seg2,se);
                  seg2=uint8(255*(seg_new>0));
            else
                seg2=uint8(255*(seg>0));
            end
        else
          seg2=uint8(255*(seg>0));
        end


        
        %Now let us do the affine transformations
        if isempty(M)
            %imwrite(zeros(im_dim1, im_dim2), affine_transformation_path);
            seg1 = zeros(im_dim1, im_dim2);
        else    

            label = gt0;
            bb = label(topM:bottomM, leftN:rightN);
            scaleRatio = 5;
            randNo = randsample([1 -1],1);

            hd = scaleRatio*h/100.0;
            hw = scaleRatio*w/100.0;

            new_h = h + randNo*round(hd);
            new_w = w + randNo*round(hw);

            new_bb = imresize(bb, [new_h, new_w]);

            new_mask = zeros(im_dim1, im_dim2);

            left_top1 = topM; left_top2 = leftN;

            if randNo == 1
                %enlarge
                left_top1 = left_top1 - round(hd/2.0);
                left_top2 = left_top2 - round(hw/2.0);    
            else
                %shrink
                left_top1 = left_top1 + round(hd/2.0);
                left_top2 = left_top2 + round(hw/2.0);    
            end

            right_bottom1 = left_top1 + new_h;
            right_bottom2 = left_top2 + new_w;

            left_top1_bb = max(1,-left_top1);
            left_top2_bb = max(1, -left_top2);

            left_top1 = max(left_top1, 1);
            left_top2 = max(left_top2, 1);

            right_bottom1 = min(right_bottom1, im_dim1);
            right_bottom2 = min(right_bottom2, im_dim2);

            final_h = right_bottom1 -left_top1;
            final_w = right_bottom2 - left_top2;

            new_mask(left_top1: left_top1 + final_h - 1, left_top2: left_top2 + final_w - 1) = new_bb(left_top1_bb:left_top1_bb + final_h - 1, left_top2_bb:left_top2_bb + final_w - 1);

            translate = [-0.1:0.01:0.1];

            t1 = h*randsample(translate,1);
            t2 = w*randsample(translate,1);

            new_mask = imtranslate(new_mask, [t1 t2]);
            %imwrite(new_mask, affine_transformation_path);
            seg1 = new_mask;
        end
        
        %gt1=uint8(gt1>0);
        seg1=uint8(255*(seg1>0));
        seg2=uint8(255*(seg2>0));
        imwrite(im1, [common_path '_' num2str(iters) '_i.png']);
        imwrite(gt1, [common_path '_' num2str(iters) '_gt.png']);
        imwrite(seg1, [common_path '_' num2str(iters) '_d1.png']);
        imwrite(seg2, [common_path '_' num2str(iters) '_d2.png']);

        %Let us perform rotation and flipping on the deformed image
        parfor a = 1:length(angles)
            angle = angles(a);
            try
                im1_rot_crop = rotate_image(im1, angle, 'bicubic'); 
                gt_rot_crop = rotate_image(gt1, angle, 'nearest');
                seg1_rot_crop = rotate_image(seg1, angle, 'nearest');
                seg2_rot_crop = rotate_image(seg2, angle, 'nearest');
            catch
                im1_rot_crop =0; 
                %gt_rot_crop=0;
                %xxx='hell'
            end

            if resize,
                im1_rot_crop = imresize(im1_rot_crop, [im_dim1, im_dim2], 'bicubic');
                gt_rot_crop = imresize(gt_rot_crop, [im_dim1, im_dim2], 'nearest');
                seg1_rot_crop = imresize(seg1_rot_crop, [im_dim1, im_dim2], 'nearest');
                seg2_rot_crop = imresize(seg2_rot_crop, [im_dim1, im_dim2], 'nearest');    
            end
            gt_rot_crop=uint8(gt_rot_crop>0);
            
         
            seg1_rot_crop=uint8(255*(seg1_rot_crop>0));
            seg2_rot_crop=uint8(255*(seg2_rot_crop>0));
            imwrite(im1_rot_crop, [common_path '_' num2str(iters) '_angle' int2str(angle) '_i.png']);
            imwrite(gt_rot_crop, [common_path '_' num2str(iters) '_angle' int2str(angle) '_gt.png']);
            imwrite(seg1_rot_crop, [common_path '_' num2str(iters) '_angle' int2str(angle) '_d1.png']);
            imwrite(seg2_rot_crop, [common_path '_' num2str(iters) '_angle' int2str(angle) '_d2.png']);
         
            if flip
                gt_rot_crop=uint8(gt_rot_crop>0);
                seg1_rot_crop=uint8(255*(seg1_rot_crop>0));
                seg2_rot_crop=uint8(255*(seg2_rot_crop>0));
    
                imwrite(fliplr(im1_rot_crop), [common_path '_' num2str(iters) '_angle' int2str(angle) '_flipped_i.png']);
                imwrite(fliplr(gt_rot_crop), [common_path '_' num2str(iters) '_angle' int2str(angle) '_flipped_gt.png']);
                imwrite(fliplr(seg1_rot_crop), [common_path '_' num2str(iters) '_angle' int2str(angle) '_flipped_d1.png']);
                imwrite(fliplr(seg2_rot_crop), [common_path '_' num2str(iters) '_angle' int2str(angle) '_flipped_d2.png']);

            end
        end
        
        if flip,
            gt1=uint8(gt1>0);
            seg1=uint8(255*(seg1>0));            
            seg2=uint8(255*(seg2>0));
            imwrite(fliplr(im1), [common_path '_' num2str(iters) '_flipped_i.png']);
            imwrite(fliplr(gt1), [common_path '_' num2str(iters) '_flipped_gt.png']);
            imwrite(fliplr(seg1), [common_path '_' num2str(iters) '_flipped_d1.png']);
            imwrite(fliplr(seg2), [common_path '_' num2str(iters) '_flipped_d2.png']);
        end
    end
end
