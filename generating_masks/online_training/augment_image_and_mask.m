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

function augment_image_and_mask(im0,gt0,folder_path,folder_name,file_name)

shift=[-0.1:0.01:0.1];

im_path=[folder_path,'/images_',folder_name,'/']; % path to image folder
bbs_path=[folder_path,'/masks_',folder_name,'/']; % path to generated mask folder
gt_path=[folder_path,'/gt_',folder_name,'/']; % path to groundtruth maksk folder

mkdir(im_path); mkdir(bbs_path); mkdir(gt_path);

gt0=uint8(gt0>0);

resize=1;
flip=1;
im_dim1 = size(im0,1);
im_dim2 = size(im0,2);
num_angles = 10;
angle_step = 2;
angles = angle_step:angle_step:num_angles*angle_step;
angles = [angles, -angles];
for jit=1:5 
     im1=im0;
     gt1=double(gt0>0);
     filename=[file_name '_jit_' num2str(jit)];
     seg=gt1;
    [M,N]=find(gt1>0);
    topM=min(M);
    bottomM=max(M);
    leftN=min(N);
    rightN=max(N);
    w=rightN-leftN;
    h=bottomM-topM; 

se = strel('disk',1);       
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
seg1=zeros(size(seg));
for k=1:numel(avals)
    try
        seg1(min(max(1,floor(avals(1,k))),size(seg,1)),min(max(1,floor(avals(2,k))),size(seg,2)))=1;        
        seg1(min(max(1,ceil(avals(1,k))),size(seg,1)),min(max(1,ceil(avals(2,k))),size(seg,2)))=1;
         seg1(min(max(1,floor(avals(1,k))),size(seg,1)),min(max(1,ceil(avals(2,k))),size(seg,2)))=1;
          seg1(min(max(1,ceil(avals(1,k))),size(seg,1)),min(max(1,floor(avals(2,k))),size(seg,2)))=1;
    end
end
se = strel('disk',5);
seg_new = imdilate(seg1,se);
  seg1=uint8(255*(seg_new>0));
else
  seg1=uint8(255*(seg>0));
    end
else
  seg1=uint8(255*(seg>0));
end
 gt1=uint8(gt1>0);
            seg1=uint8(255*(seg1>0));
            imwrite(im1, [im_path filename  '.png']);
            imwrite(gt1, [gt_path filename  '.png']);
            imwrite(seg1, [bbs_path filename  '.png']);
 
    parfor a = 1:length(angles),
            angle = angles(a);
            try
            im1_rot_crop = rotate_image(im1, angle, 'bicubic'); 
            gt_rot_crop = rotate_image(gt1, angle, 'nearest');
            seg1_rot_crop = rotate_image(seg1, angle, 'nearest');
             catch
                im1_rot_crop =0; 
            end

            if resize,
                im1_rot_crop = imresize(im1_rot_crop, [im_dim1, im_dim2], 'bicubic');
                gt_rot_crop = imresize(gt_rot_crop, [im_dim1, im_dim2], 'nearest');
                seg1_rot_crop = imresize(seg1_rot_crop, [im_dim1, im_dim2], 'nearest');
           
            end
            gt_rot_crop=uint8(gt_rot_crop>0);
            
         
            seg1_rot_crop=uint8(255*(seg1_rot_crop>0));
            imwrite(im1_rot_crop, [im_path filename '_angle' int2str(angle) '.png']);
            imwrite(gt_rot_crop, [gt_path filename '_angle' int2str(angle) '.png']);
            imwrite(seg1_rot_crop, [bbs_path filename '_angle' int2str(angle) '.png']);

            if flip
                gt_rot_crop=uint8(gt_rot_crop>0);
                seg1_rot_crop=uint8(255*(seg1_rot_crop>0));
                imwrite(fliplr(im1_rot_crop), [im_path filename '_angle' int2str(angle) '_flipped.png']);
                imwrite(fliplr(gt_rot_crop), [gt_path filename '_angle' int2str(angle) '_flipped.png']);
                imwrite(fliplr(seg1_rot_crop), [bbs_path filename '_angle' int2str(angle) '_flipped.png']);
            end
   end
       if flip,
            gt1=uint8(gt1>0);
            seg1=uint8(255*(seg1>0));
            imwrite(fliplr(im1), [im_path filename  '_flipped.png']);
            imwrite(fliplr(gt1), [gt_path filename  '_flipped.png']);
            imwrite(fliplr(seg1), [bbs_path filename  '_flipped.png']);
       end
 end
end
