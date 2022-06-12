path = './眼预测屏幕坐标点+头部姿态';
file_name = dir(path)
dimension_for_coordinate = 13;
dimension_for_headpose = 3;
total_frame = 1000;
frames_in_onevolume = 40;
x = 0:476/4:476;
y = 0:268/3:268;
dimension_for_total = total_frame/frames_in_onevolume*13+total_frame/frames_in_onevolume*3*3+1;
total_histogram = zeros(length(file_name)-2,dimension_for_total);
b=length(file_name)
for i = 1:length(file_name)-2
    file_path = fullfile(path,file_name(i+2).name);
    if contains(file_path,'小学')
        label = 1;
    elseif contains(file_path,'幼儿园')
        label = 2;
    else
        label = 0;
    end
    feature = load(file_path);
    feature = feature.screen_coordinate;
    
    pitch_histogram = Caculate_headpose_histogram(feature(3,:),total_frame,frames_in_onevolume,dimension_for_headpose);
    yaw_histogram = Caculate_headpose_histogram(feature(4,:),total_frame,frames_in_onevolume,dimension_for_headpose);
    roll_histogram = Caculate_headpose_histogram(feature(5,:),total_frame,frames_in_onevolume,dimension_for_headpose);
    coordinate_histogram = Caculate_coordinate_histogram(feature,total_frame,frames_in_onevolume,dimension_for_coordinate,x,y);
    
    pitch_histogram = Caculate_accumulate(pitch_histogram,dimension_for_headpose);
    yaw_histogram = Caculate_accumulate(yaw_histogram,dimension_for_headpose);
    roll_histogram = Caculate_accumulate(roll_histogram,dimension_for_headpose);
    coordinate_histogram = Caculate_accumulate(coordinate_histogram,dimension_for_coordinate);
    coordinate_histogram = coordinate_histogram/total_frame;
    pitch_histogram = pitch_histogram/total_frame;
    yaw_histogram = yaw_histogram/total_frame;
    roll_histogram = roll_histogram/total_frame;
    total_histogram(i,:) = [label,coordinate_histogram,pitch_histogram,yaw_histogram,roll_histogram];
end
%total_histogram(136:143,:) = [];
%total_histogram(136:260,:) = [];
% x1 = screen_coordinate(1,:);   
% y1 = screen_coordinate(2,:);
% x = 0:476/4:476;
% y = 0:268/3:268;

