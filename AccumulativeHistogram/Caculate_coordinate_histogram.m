%计算屏幕注视坐标点直方图
function [Histogram_coordinate] = Caculate_coordinate_histogram(coordinate,total_frame,frames_in_onevolume,dimension,x,y)
x1 = coordinate(1,:);
y1 = coordinate(2,:);
dimensionforhistogram = total_frame/frames_in_onevolume*dimension;
Histogram_coordinate = zeros(1, dimensionforhistogram);
for i = 1:total_frame/frames_in_onevolume
    for j = 1:frames_in_onevolume
        index = (i-1)*frames_in_onevolume+j;
        if x1(index) == 0 && y1(index) == 0
            Histogram_coordinate(13*i-12) = Histogram_coordinate(13*i-12)+1;
        elseif x1(index)>x(5) || y1(index)>y(4)
            Histogram_coordinate(13*i-12) = Histogram_coordinate(13*i-12)+1;
        elseif x1(index)>0 && x1(index)<x(2)
            if y1(index)<y(2)
                Histogram_coordinate(13*i-11) = Histogram_coordinate(13*i-11)+1;
            elseif y1(index)>=y(2) && y1(index)<y(3)
                Histogram_coordinate(13*i-7) = Histogram_coordinate(13*i-7)+1;
            elseif y1(index)>=y(3) && y1(index)<y(4)
                Histogram_coordinate(13*i-3) = Histogram_coordinate(13*i-3)+1;
            end
        elseif x1(index)>=x(2) && x1(index)<x(3)
            if y1(index)<y(2)
                Histogram_coordinate(13*i-10) = Histogram_coordinate(13*i-10)+1;
            elseif y1(index)>=y(2) && y1(index)<y(3)
                Histogram_coordinate(13*i-6) = Histogram_coordinate(13*i-6)+1;
            elseif y1(index)>=y(3) && y1(index)<y(4)
                Histogram_coordinate(13*i-2) = Histogram_coordinate(13*i-2)+1;
            end
        elseif x1(index)>=x(3) && x1(index)<x(4)
            if y1(index)<y(2)
                Histogram_coordinate(13*i-9) = Histogram_coordinate(13*i-9)+1;
            elseif y1(index)>=y(2) && y1(index)<y(3)
                Histogram_coordinate(13*i-5) = Histogram_coordinate(13*i-5)+1;
            elseif y1(index)>=y(3) && y1(index)<y(4)
                Histogram_coordinate(13*i-1) = Histogram_coordinate(13*i-1)+1;
            end
        elseif x1(index)>=x(4) && x1(index)<x(5)
            if y1(index)<y(2)
                Histogram_coordinate(13*i-8) = Histogram_coordinate(13*i-8)+1;
            elseif y1(index)>=y(2) && y1(index)<y(3)
                Histogram_coordinate(13*i-4) = Histogram_coordinate(13*i-4)+1;
            elseif y1(index)>=y(3) && y1(index)<y(4)
                Histogram_coordinate(13*i) = Histogram_coordinate(13*i)+1;
            end
        end
    end
end