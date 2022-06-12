clc
close all
clear all
h5path = 'G:\�Ա�֢����\˫�����߹��ƴ��';
filenames = dir(h5path);

zhaodao=0;
mei=0;
for j = 3:length(filenames)
    predict_file = filenames(j).name;
    predict_path = fullfile(h5path,predict_file);
    predict_gaze = load(predict_path);
    predict_gaze = predict_gaze.gaze';
    for k = length(predict_file):-1:1
        if predict_file(k) == '_'
            break
        end
    end
    original_path ='G:\�Ա�֢����\����������ȡ���';
    subfolder = predict_file(1:k-1);
    original_path = fullfile(original_path,subfolder);
    
    files = predict_file(k+1:end-3);
    files = strcat(files,'mat');
    original_path = fullfile(original_path,files);
     nameList = strsplit(predict_path,'\');
    nameTmp = char(nameList(length(nameList)));%�ָ�֮����cell���ͣ���Ҫת��Ϊchar����
    fileName = nameTmp(1:length(nameTmp)-4); % fileName ���Ϊ   test
    b='.mat';
    c=[fileName,b];
    if (exist(original_path) ==0)
    % �����bai���Ƿ���ڣ�du���������zhiִ��load����dao
    continue
    end
    rgt=0;
    %disp(files)
    phns=['G:\�Ա�֢����\����\res1.txt'];%Ҫ��ȡ���ĵ����ڵ�·��??
    fpn=fopen(phns,'rt');%���ĵ�?
    while feof(fpn)~=1%�����ж��ļ�ָ��p������ָ���ļ��е�λ�ã�������ļ�ĩ����������1�����򷵻�0
    filexxx=fgetl(fpn);%��ȡ�ĵ���һ��?
    %disp(filexxx)
    %disp('ddd')
    if strfind(filexxx,c)
        disp(filexxx)
        disp(c)
        
        rgt=1;
        
        zhaodao=zhaodao+1;
       
        %disp('�ҵ�')
        break;
        %disp(files)
    end
    end
    fclose(fpn);
    
    if(rgt==0)
        %disp('û��')
        %disp(predict_file)
        mei=mei+1;
        continue
          
    end
    gaze_original1 = load(original_path);
    index = 1;
    [m,~] = size(gaze_original1.data);
    clear gaze_original screen_coordinate
%     for i = 1:m
%         if gaze_original1.data(i,1) ~= 0
%             gaze_original(index,:) = gaze_original1.data(i,:);
%             index = index+1;
%         end
%     end
    gaze_original = gaze_original1.data;
    
    theta = predict_gaze(1,:);
    phi = predict_gaze(2,:);
    gaze_x = -cos(theta).*sin(phi);
    gaze_y = -sin(theta);
    gaze_z = -cos(theta).*cos(phi);
    gaze_predict = [gaze_x;gaze_y;gaze_z];%ע������

    count = 0;
    for i = 1:length(theta)
        focal_new=960;
        distance_new=600;
        headpose = gaze_original(:,296:298);
        headpose_hr = headpose(i,:);
        pitch = headpose_hr(1)/pi*180-20;
        yaw = headpose_hr(2)/pi*180;
        roll = headpose_hr(3)/pi*180;
        hR = rodrigues(headpose_hr);
        if gaze_original(i,3) >= 0.88
            Left_3D_center = [(gaze_original(i,133)+gaze_original(i,139))/2,(gaze_original(i,189)+gaze_original(i,195))/2,(gaze_original(i,245)+gaze_original(i,251))/2]';
            Right_3D_center = [(gaze_original(i,161)+gaze_original(i,167))/2,(gaze_original(i,217)+gaze_original(i,223))/2,(gaze_original(i,273)+gaze_original(i,279))/2]';
            Head_center = (Left_3D_center+Right_3D_center)./2;%����������ͷ���ĵ�3D����
            distance = norm(Head_center);
            z_scale = distance_new/distance;
            scaleMat = [1.0, 0.0, 0.0; 0.0, 1.0, 0.0; 0.0, 0.0, z_scale];
            hRx = hR(:,1);
            forward = (Head_center/distance);
            down = cross(forward, hRx);
            down = down / norm(down);
            right = cross(down, forward);
            right = right / norm(right);
            rotMat = [right, down, forward]';
            cnvMat = scaleMat * rotMat;
            gaze = inv(cnvMat)*gaze_predict(:,i);
            %gaze = 300*gaze+Head_center;
            %��ͷ����̬�����ߵļнǣ�����ĳһֵ����Ϊͷ����̬Ϊ����
            hR_trans = hR';
            rot3 = hR_trans(:,3);
            cos1 = gaze'*rot3/norm(gaze)/norm(rot3);
            alpha = acos(-cos1)*180/pi;
            
            if alpha >= 30 
                gaze = rot3;
                count = count+1;
            end
            rot_theta = -20/180*pi;
            %��ע��������Ȩ���ԽӴ������������Z=0��ƽ��
            zoom_co = -((Head_center(2)*sin(rot_theta)+Head_center(3)*cos(rot_theta))/(gaze(2)*sin(rot_theta)+gaze(3)*cos(rot_theta)));
            gaze_pro = zoom_co*gaze+Head_center;%�������ϵ��Z=0ƽ����ע�������Ľ���
            gaze_pro1 = [gaze_pro;1];
            rot2screen = [1 0 0 0;0 cos(rot_theta) -sin(rot_theta) 0;0 sin(rot_theta) cos(rot_theta) 0;0 0 0 1];
            gaze_screen = rot2screen*gaze_pro1;
            screen_x = -gaze_screen(1)+238;
            screen_y = -(gaze_screen(2)-20)+268;
            screen = [screen_x;screen_y;pitch;yaw;roll];
            screen_coordinate(:,i) = screen;
           
        else
            screen_coordinate(:,i) = [0;0;0;0;0];
        end
    end
   for q = 1:length(screen_coordinate)
       if screen_coordinate(3,q) == 0
           if q == 1 || screen_coordinate(3,q-1) < 0
               screen_coordinate(3,q) = -90;
           else
               screen_coordinate(3,q) = 90;
           end
           if q == 1 || screen_coordinate(4,q-1) < 0
               screen_coordinate(4,q) = -90;
           else
               screen_coordinate(4,q) = 90;
           end
           if q == 1 || screen_coordinate(5,q-1) < 0
               screen_coordinate(5,q) = -90;
           else
               screen_coordinate(5,q) = 90;
           end
       end
   end
           
%     scatter(x,y,'.');
%     axis([0 476 0 268]);
%     x = 0:476/4:476;
%     y = 0:268/3:268;
%     M = meshgrid(x,y); %��������
%     N = meshgrid(y,x); 
%     hold on
%     plot(x,N,'b'); %����ˮƽ����
%     plot(M,y,'b');
    savepath = 'G:\�Ա�֢����\˫��Ԥ����Ļ�����+ͷ����̬';
    name = strcat(subfolder,'_',files);
    savename = fullfile(savepath,name);
    save(savename,'screen_coordinate');
    %disp(strcat(savename,'...saved'))
    disp(zhaodao)
    disp(mei)
end

