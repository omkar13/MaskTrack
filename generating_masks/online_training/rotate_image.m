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

function [img_crop,y1,y2,x1,x2] = rotate_image(img, angle, interp)
    width = size(img,2); height = size(img, 1);
    img_rot = imrotate(img, angle, interp, 'loose');
    [wr, hr] = get_rectangle_with_max_area(width, height, pi*(angle/180));
    [img_crop,y1,y2,x1,x2] = crop_image_around_centre(img_rot, wr, hr);
    return
end
