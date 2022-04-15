% addpath('E:\zyh\mexopencv-2.4\');
root_path2 = 'D:\UT Multi-view\s39\raw';
folder1 = dir(root_path2);
save_name = strcat('D:\UT_改标准化图像\',root_path2(18:20),'.mat');
data.img = zeros(1280,36,224,3);
data.gaze = zeros(1280,2);
for i = 4:length(folder1)-1
    path1 = fullfile(root_path2, folder1(i).name);
    gaze_path = fullfile(root_path2, folder1(3).name);
    Monitor_path = fullfile(root_path2, folder1(end).name);
    floder2 = dir(path1);
    data_path = fullfile(path1, floder2(4).name);
    gaze_3D_point = Extractgaze(gaze_path, Monitor_path);
    Cparams_path00 = strcat(path1, '\', floder2(3).name, '\', '00000000.txt');
    [head_center, hR, fx, fy, u, v] = ExtractHeadpoint(data_path, Cparams_path00);
    path3 = fullfile(path1,floder2(3).name);
    path4 = fullfile(path1,floder2(5).name);
    floder_caprams = dir(path3);
    floder_img = dir(path4);
    for j = 3:length(floder_img)
        Cparams_path = fullfile(path3,floder_caprams(j).name);
        img_path = fullfile(path4,floder_img(j).name);
        [NormImg, hrnew, gcnew, gcold] = ImageNormalization(Cparams_path, img_path, head_center, gaze_3D_point, hR, fx, fy, u, v);
        gaze_theta = asin((-1)*gcnew(2)); % vertical gaze angle
        gaze_phi = atan2((-1)*gcnew(1), (-1)*gcnew(3)); % horizontal gaze angle
        angle = [gaze_theta,gaze_phi];
        index = (j-2)+(i-4)*8;
        data.img(index,:,:,:) = NormImg;
        data.img = uint8(data.img);
        data.gaze(index,:) = angle;
    end
end
save(save_name,'data');


 