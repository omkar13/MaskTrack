% Implementaion modified to include translation (imtranslate) and scaling
% (imresize) of image (affine transformations)
% Modified by: Omkar Damle, Feb 2018.
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

function augment_image_and_mask(gt0, affine_transformation_path, non_rigid_deform_path)

    shift=[-0.1:0.01:0.1];

    gt0=uint8(gt0>0);

    im_dim1 = size(gt0,1);
    im_dim2 = size(gt0,2);

  
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
            seg1=zeros(size(seg));

            %find the new points and make sure they are bounded

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


    %gt1=uint8(gt1>0);
    seg1=uint8(255*(seg1>0));
    %imwrite(im1, [im_path filename  '.png']);
    %imwrite(gt1, [gt_path filename  '.png']);
    imwrite(seg1, non_rigid_deform_path);
    
    %Now let us do the affine transformations

    if isempty(M)
        imwrite(zeros(im_dim1, im_dim2), affine_transformation_path);
        return;
    end
    
    label = gt0;
    bb = label(topM:bottomM, leftN:rightN);
    scaleRatio = 5;
    randNo = randsample([1 -1],1);

    hd = scaleRatio*h/100.0;
    hw = scaleRatio*w/100.0;
        
    new_h = h + randNo*round(hd);
    new_w = w + randNo*round(hw);

    if new_h == 0
        new_h=1
    end
    
    if new_w ==0
        new_w = 1
    end
    
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
    imwrite(new_mask, affine_transformation_path);

end
