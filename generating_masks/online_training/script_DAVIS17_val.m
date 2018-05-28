function script_DAVIS17_val(num_iterations)
    fid1=fopen('/home/omkar/Documents/Omkar/VOS/DAVIS17/ImageSets/2017/val.txt','r')
    %fid1=fopen('/home/omkar/Documents/Omkar/VOS/DAVIS17_test/ImageSets/2017/test-dev.txt','r')
    while ~feof(fid1)
        line=fgets(fid1);
        line=line(1:end-1)
        script2(line,num_iterations);
    end
    fclose(fid1);
end