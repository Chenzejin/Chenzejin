%计算头部姿态三个角度的直方图
function [Histogram_angle] = Caculate_headpose_histogram(angle,total_frame,frames_in_onevolume,dimension)
dimensionforhistogram = total_frame/frames_in_onevolume*dimension;
Histogram_angle = zeros(1, dimensionforhistogram);
for i = 1:total_frame/frames_in_onevolume
    for j = 1:frames_in_onevolume
        index = (i-1)*frames_in_onevolume+j;
        if angle(index)<=-5
            Histogram_angle(3*i-2) = Histogram_angle(3*i-2)+1;
        elseif angle(index)>=5
            Histogram_angle(3*i) = Histogram_angle(3*i)+1;
        else
            Histogram_angle(3*i-1) = Histogram_angle(3*i-1)+1;
        end
    end
end
end
            