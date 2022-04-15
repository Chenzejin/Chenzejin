clear;
clc;

%读取mat和jpg路径
addpath('mexopencv-2.4/');
path_1 = 'D:\脸部特征提取存放';
%标准化图片用到的参数
Matrix = [500 0 320;0 500 240;0 0 1];
eye_image_width  = 224;
eye_image_height = 36;
image_path1 = 'D:\视频数据视频帧';
save_path1 = 'D:\标准化双眼图像h5文件\TMS';
dir_1 = dir(path_1);
dir_name1 = dir_1(3).name;
fileExt1 = '*.mat';
fileExt2 = '*.jpg';
path_2 = fullfile(path_1,dir_name1);
files = fullfile(path_2,fileExt1);
dir_2 = dir(files);
disp(['当前文件夹名为',path_2])
for j = 1:length(dir_2)

    clear data normdata
    filesname = fullfile(path_2,dir_2(j).name);%mat文件保存路径
    image_path2 = strcat(image_path1,filesname(12:end-4));
    save_name = strcat(save_path1,'\',filesname(24:end-4),'.mat');
    image_format = fullfile(image_path2,fileExt2);
    dir_3 = dir(image_format);
    data = load(filesname);
    data = data.data;

%循环读取图片
    FrameLength = length(dir_3);
    [m,n] = size(data);
     if FrameLength == m
         loop_start = 1;
     else
         loop_start = FrameLength-m+1;
     end
         for i = loop_start:m
    %          if data(i,3) < 0.9
    %              continue
    %          end
             imagefile = fullfile(image_path2,dir_3(i).name);%读取帧保存路径
             image = imread(imagefile);
             headpose_hr = data(i,296:298);
             headpose_ht = data(i,293:295);
             hR = rodrigues(headpose_hr);
             right_eye_center = 0.5*([data(i,161),data(i,217),data(i,273)]'+[data(i,167),data(i,223),data(i,279)]');
             left_eye_center = 0.5*([data(i,133),data(i,189),data(i,245)]'+[data(i,139),data(i,195),data(i,251)]');
             head_center = 0.5*(right_eye_center+left_eye_center);
             if all(head_center == 0)
                 normdata.pose(i,:) = [0;0];
                 continue
             end
             [norm_img,headpose] = normalizeImg(image, head_center, hR, [eye_image_width, eye_image_height], Matrix);
             normdata.img(i,:,:,:) = norm_img;
             M = rodrigues(headpose);
             Zv = M(:,3);
             headpose_theta = asin(Zv(2)); % vertical head pose angle
             headpose_phi = atan2(Zv(1), Zv(3)); % horizontal head pose angle
             headpose = [headpose_theta,headpose_phi];
             normdata.pose(i,:) = headpose';
         end
         normdata.img = unit8(normdata.img);
%          hdf5write(save_name,'/img', normdata.img, '/pose',normdata.pose); 
         save(save_name,'normdata');
         disp([image_path2,'已完成转换'])
end
