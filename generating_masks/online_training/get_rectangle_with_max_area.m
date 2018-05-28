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

function [wr, hr] = get_rectangle_with_max_area(w, h, angle)
% Given a rectangle of size wxh that has been rotated by 'angle' (in
% radians), computes the width and height of the largest possible
% axis-aligned rectangle (maximal area) within the rotated rectangle.
% source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders

    if (w <= 0) || (h <= 0)
        wr = 0; hr = 0;
        return;
    end
    
    width_is_longer = (w >= h);
    if width_is_longer
        side_long = w;
        side_short = h;
    else
        side_long = h;
        side_short = w;
    end
    
    sin_a = abs(sin(angle));
    cos_a = abs(cos(angle));
    
    if side_short <= 2.0*sin_a*cos_a*side_long
        % half constrained case: two crop corners touch the longer side,
        % the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short;
        if width_is_longer,
            wr = x/sin_a;
            hr = x/cos_a;
        else
            wr = x/cos_a;
            hr = x/sin_a;
        end
    else
        cos_2a = cos_a*cos_a - sin_a*sin_a;
        wr = (w*cos_a - h*sin_a)/cos_2a;
        hr = (h*cos_a - w*sin_a)/cos_2a;
    end

    return;
end

