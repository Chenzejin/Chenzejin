clear;
clc;

addpath('mexopencv-2.4/'); % we need to use some function from OpenCV

% load the face model
faceModel = load('../MPIIGaze/Data/6 points-based face model.mat');
faceModel = faceModel.model;
img_path1 = 'E:\Documents\图片和视频数据库\MPIIGaze\Data\Original\p00';
save_path1 = 'E:\Documents\图片和视频数据库\MPIIGaze\Data\Normalized图像';
save_path2 = strcat(save_path1,'\',img_path1(46:end));
mkdir(save_path2);
dir_img1 = dir(img_path1);
cameraCalib_path = fullfile(img_path1,dir_img1(3).name);
dir_camera = dir(cameraCalib_path);
cameraCalib_path = fullfile(cameraCalib_path,dir_camera(3).name);
cameraCalib = load(cameraCalib_path);
for i = 4:length(dir_img1)
    clear data
    img_path2 = fullfile(img_path1,dir_img1(i).name);
    save_path3 = fullfile(save_path2,img_path2(end-4:end));
%     mkdir(save_path3);
    save_data = strcat(save_path2,'\',img_path2(end-4:end),'.h5');
    dir_img2 = dir(img_path2);

    for j = 3:(length(dir_img2)-1)
        % load the image, annotation and camera parameters.
        img_path3 = fullfile(img_path2,dir_img2(j).name);
        annotation_path = fullfile(img_path2,dir_img2(length(dir_img2)).name);
        save_img = strcat(save_path3,'\img_',img_path3(end-7:end));
        img = imread(img_path3);
        %img = imread('D:\视频数据\TMS视频帧\1\00044.jpeg');
        annotation = load(annotation_path);
        [short,~] = size(annotation);
        if j == short+1
            break;
        end
        % get head pose
        headpose_hr = annotation(j-2, 30:32);
        headpose_ht = annotation(j-2, 33:35);
        hR = rodrigues(headpose_hr); 
        Fc = hR* faceModel; % rotate the face model, which is calcluated from facial landmakr detection
        Fc = bsxfun(@plus, Fc, headpose_ht');

        % get the eye center in the original camera cooridnate system.
        right_eye_center = 0.5*(Fc(:,1)+Fc(:,2));
        left_eye_center = 0.5*(Fc(:,3)+Fc(:,4));
        gaze_center = 0.5*(right_eye_center+left_eye_center);
        %right_eye_cetter = 0.5*([-59.9,8.9,562.5]'+[-86,18.1,573.8]');
        % get the gaze target
        gaze_target = annotation(j-2, 27:29);
        gaze_target = gaze_target';

        % set the size of normalized eye image
        eye_image_width  = 224;
        eye_image_height = 36;

        % normalization for the right eye, you can do it for left eye by replacing
        % "right_eye_cetter" to "left_eye_center"
        [center_img, headpose, gaze] = normalizeImg(img, gaze_center, hR, gaze_target, [eye_image_width, eye_image_height], cameraCalib.cameraMatrix);
        data.img(j-2,:,:,:) = center_img;
    %     imwrite(center_img,save_img);
        % convert the gaze direction in the camera cooridnate system to the angle
        % in the polar coordinate system
        gaze_theta = asin((-1)*gaze(2)); % vertical gaze angle
        gaze_phi = atan2((-1)*gaze(1), (-1)*gaze(3)); % horizontal gaze angle
        gaze = [gaze_theta,gaze_phi];
        data.gaze(j-2,:) = gaze';
         % save as above, conver head pose to the polar coordinate system
        left_M = rodrigues(headpose);
        left_Zv = left_M(:,3);
        headpose_theta = asin(left_Zv(2)); % vertical head pose angle
        headpose_phi = atan2(left_Zv(1), left_Zv(3)); % horizontal head pose angle
        pose = [headpose_theta,headpose_phi];
        data.pose(j-2,:) = pose';
    end
    data.img = data.img/255;
    data.img = single(data.img);
    data.gaze = single(data.gaze);
    data.pose = single(data.pose);
    hdf5write(save_data,'/img', data.img, '/gaze',data.gaze,'/pose',data.pose); 
end


 
 
 
 
 
