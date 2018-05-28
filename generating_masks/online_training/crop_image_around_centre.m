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

function [image,y1,y2,x1,x2] = crop_image_around_centre(image, width, height)
    % Given an image, crops it to the given width and height, around it's centre point
    image_size = [size(image, 2), size(image, 1)];
    image_center = [round(image_size(1) * 0.5), round(image_size(2) * 0.5)];
    
    if(width > image_size(1))
        width = image_size(1);
    end
    if(height > image_size(2))
        height = image_size(2);
    end
    
    x1 = round(image_center(1) - width * 0.5)+1;
    x2 = round(image_center(1) + width * 0.5);
    y1 = round(image_center(2) - height * 0.5)+1;
    y2 = round(image_center(2) + height * 0.5);
    
    image = image(y1:y2-1, x1:x2-1,:);
end


