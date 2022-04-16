%计算累积直方图
function [accumulated] = Caculate_accumulate(non_accumulated,dimension)

len = length(non_accumulated);
index = dimension+1;
while index<len
    non_accumulated(index:index+dimension-1) = non_accumulated(index-dimension:index-1)+non_accumulated(index:index+dimension-1);
    index = index+dimension;
end

accumulated = non_accumulated;