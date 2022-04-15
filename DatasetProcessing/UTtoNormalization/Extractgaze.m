function gaze_3D_point = Extractgaze(gaze_path, Monitor_path)
gaze_2D_point = importdata(gaze_path);
gaze_2D_point = gaze_2D_point.data(1,2:3);
gaze_2D_point = [gaze_2D_point,0];
data = importdata(Monitor_path);
z = zeros(5,3);
for i = 1:length(data)
    temp = zeros(1,3);
    if i == 1 || i == 6
        continue
    else
        num = data{i};
        num = regexp(num, '-?\d*\.?\d*', 'match');         
        num = str2double(num);
        if length(num) > 3
%             disp('Dimension Warning!');
            temp(1) = num(1)*10^num(2);
            temp(2) = num(3)*10^num(4);
            temp(3) = num(5)*10^num(6);
        else 
            temp = num;
        end
        z(i,:) = temp;
    end
end
Rm = z(3:end,:);
Tm = z(2,:);
gaze_3D_point = Rm*gaze_2D_point'+Tm';
gaze_3D_point = gaze_3D_point';