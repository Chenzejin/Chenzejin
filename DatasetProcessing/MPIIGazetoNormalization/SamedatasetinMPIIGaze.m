clear;
clc;
faceModel = load('E:\Documents\图片和视频数据库\MPIIGaze\Data\6 points-based face model.mat');
faceModel = faceModel.model;
sample_path = 'E:\Documents\图片和视频数据库\MPIIGaze\Evaluation Subset\sample list for eye image\p00.txt';
img_path1 = 'E:\Documents\图片和视频数据库\MPIIGaze\Data\Original\p00';
save_path1 = 'E:\Documents\图片和视频数据库\MPIIGaze\Evaluation Subset';
save_path2 = strcat(save_path1,'\',img_path1(46:end));
save_name = fullfile(save_path1,'p00.mat');
dir_img1 = dir(img_path1);
cameraCalib_path = fullfile(img_path1,dir_img1(3).name);
dir_camera = dir(cameraCalib_path);
cameraCalib_path = fullfile(cameraCalib_path,dir_camera(3).name);
cameraCalib = load(cameraCalib_path);
list_sample = importdata(sample_path);
list_sample = char(list_sample);
data.img = zeros(3000,36,224,3);
data.gaze = zeros(3000,2);
for i = 1:3000   
    index = list_sample(i,7:10);
    index = str2num(index);
    img_path = fullfile(img_path1,list_sample(i,1:14));
    img = imread(img_path);
    annotation_path = strcat(img_path1,'\',list_sample(i,1:5),'\','annotation.txt');
    annotation = load(annotation_path);
    headpose_hr = annotation(index, 30:32);
    headpose_ht = annotation(index, 33:35);
    hR = rodrigues(headpose_hr); 
    Fc = hR* faceModel; % rotate the face model, which is calcluated from facial landmakr detection
    Fc = bsxfun(@plus, Fc, headpose_ht');
        % get the eye center in the original camera cooridnate system.
    right_eye_center = 0.5*(Fc(:,1)+Fc(:,2));
    left_eye_center = 0.5*(Fc(:,3)+Fc(:,4));
    gaze_center = 0.5*(right_eye_center+left_eye_center);
        %right_eye_cetter = 0.5*([-59.9,8.9,562.5]'+[-86,18.1,573.8]');
        % get the gaze target
    gaze_target = annotation(index, 27:29);
    gaze_target = gaze_target';
        % set the size of normalized eye image
    eye_image_width  = 224;
    eye_image_height = 36;
    [center_img, headpose, gaze] = normalizeImg(img, gaze_center, hR, gaze_target, [eye_image_width, eye_image_height], cameraCalib.cameraMatrix);
    gaze_theta = asin((-1)*gaze(2)); % vertical gaze angle
    gaze_phi = atan2((-1)*gaze(1), (-1)*gaze(3)); % horizontal gaze angle
    center_img = histeq(center_img);
    if i <= 1500
        % convert the gaze direction in the camera cooridnate system to the angle
        % in the polar coordinate system
        final_img = center_img;
        gaze1 = [gaze_theta,gaze_phi];   
    else
        final_img = flip(center_img,2);
        gaze1 = [gaze_theta,-gaze_phi];
    end
    data.img(i,:,:,:) = final_img;
    data.gaze(i,:) = gaze1';
end
save(save_name,'data');