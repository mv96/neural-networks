function [part_1,part_2,part_3,part_4,part_5]=cross_validation_split(X,y,splits)
%shuffle randomly all row
dimensions=size(X,2);
X=[X y];
X= X(randperm(size(X,1)),:);
y= X(:,dimensions+1);
X=X(:,1:dimensions);

%deciding the division number
division_ratio=(100/splits);
fprintf('divison ratio: \n');
fprintf(' %f \n', division_ratio);
division_number=(division_ratio/100);%computes the total number of rows in train data 

%%%% 1 to division row data in train and division row to end data in test
[rows_1 ,coloumns_1] = find(y==1);%store the index of matrix of 1 in pos
[rows_2 ,coloumns_2] = find(y == 0);%store the index of matrix of 0 in ne
outlier_number=size(rows_1,1);
nonoutlier_number=size(rows_2,1);

initial_1=1;
initial_2=1;
for i=1:splits
count_o=round(division_number*i*outlier_number);%count of outier number in train data
fprintf('The split divison number you set for outlier: \n');
fprintf(' %f \n', count_o);
count_d=round(division_number*i*nonoutlier_number);%count of data number in train
fprintf('The split divison number you set for data: \n');
fprintf(' %f \n', count_d);
ma=X((rows_1(initial_1:count_o)),:);%70 of the coutlier data 
pa=X((rows_2(initial_2:count_d)),:);%70 of the normal data
X_train=[ma;pa];
y_train=[y(rows_1(initial_1:count_o),:);y(rows_2(initial_2:count_d))];
initial_1=count_o+1;
initial_2=count_d+1;
if(i==1)
part_1=[X_train y_train];
end
if(i==2)
part_2=[X_train y_train];
end
if(i==3)
part_3=[X_train y_train];
end
if(i==4)
part_4=[X_train y_train];
end
if(i==5)
part_5=[X_train y_train];
end
end

end