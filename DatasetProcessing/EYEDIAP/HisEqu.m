path = 'D:\EYEDIAP\15Frame\Non-enhanced';
save_path = 'D:\EYEDIAP\15Frame\Enhanced';
sub = dir(path);
for i = 3:length(sub)
    clear data gaze z img
    mat_file = fullfile(path,sub(i).name);
    save_name = strcat(save_path,'\',sub(i).name(1:end-8),'Enhanceddata.mat');
    f = load(mat_file);
    img = f.data.img;
    gaze = f.data.gaze;
    m = length(img);
    z = zeros(m,36,224,3);
    for j = 1:m
        a = img(j,:,:,:);
        a = squeeze(a);
        b = histeq(a);
        z(j,:,:,:) = b;
        z = uint8(z);
    end
    data.img = z;
    data.gaze = gaze;
    save(save_name,'data');
end