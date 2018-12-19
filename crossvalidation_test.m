function accuracy=crossvalidation_test(part_1x,part_2x,part_3x,part_4x,part_5x,
                                       part_1y,part_2y,part_3y,part_4y,part_5y,
                                       splits)
                                       
%%%train cross validation on all data sets
fprintf('testing data on cross validation\n')
accuracy=zeros(splits,1);
%train part1 and test on all other parts
remaining_data_x1=part_2x;part_3x;part_4x;part_5x;
remaining_data_x2=part_1x;part_3x;part_4x;part_5x;
remaining_data_x3=part_1x;part_2x;part_4x;part_5x;
remaining_data_x4=part_1x;part_2x;part_3x;part_5x;
remaining_data_x5=part_1x;part_2x;part_3x;part_4x;
remaining_data_y1=part_2y;part_3y;part_4y;part_5y;
remaining_data_y2=part_1y;part_3y;part_4y;part_5y;
remaining_data_y3=part_1y;part_2y;part_4y;part_5y;
remaining_data_y4=part_1y;part_2y;part_3y;part_5y;
remaining_data_y5=part_1y;part_2y;part_3y;part_4y;
%train part1 and test on all other parts
accuracy(1)=nn_model(part_1x,part_1y,remaining_data_x1,remaining_data_y1,0);

%train part2 and test on all other parts
accuracy(2)=nn_model(part_2x,part_2y,remaining_data_x2,remaining_data_y2,0);

%train part3 and test on all other parts
accuracy(3)=nn_model(part_3x,part_3y,remaining_data_x3,remaining_data_y3,0); 

%train part4 and test on all other parts
accuracy(4)=nn_model(part_4x,part_4y,remaining_data_x4,remaining_data_y4,0);

%train part5 and test on all other parts
accuracy(5)=nn_model(part_5x,part_5y,remaining_data_x5,remaining_data_y5,0);

fprintf('\n');
fprintf('Train Accuracy: %f\n',accuracy);
fprintf('\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
end