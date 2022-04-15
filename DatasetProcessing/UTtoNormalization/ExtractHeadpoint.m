function [head_center_3D, hR, fx, fy, u, v] = ExtractHeadpoint(data_path, Cparams_path00)
data = importdata(data_path);
Cparams = importdata(Cparams_path00);
cparams = Cparams.data;
z = zeros(12,3);
%提取旋转矩阵，头中心坐标。并计算出像素坐标
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
hR = z(3:5,:);
Face = z(7:12,:);
right = 0.5*(Face(1,:)+Face(2,:));
left = 0.5*(Face(3,:)+Face(4,:));
head_center_3D = 0.5*(left+right);
% head_center_z = [head_center_3D';1];
% head_cor = cparams*head_center_z;
% head_cor = head_cor./head_cor(3);
% head_center_2D = round(head_cor(1:2));
fx = cparams(1,1);fy = cparams(2,2);u = cparams(1,3);v = cparams(2,3);