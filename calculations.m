function [hit_rate outlier_accuracy]=calculations(y,check,m,outliers_number,actual_outlier_number)
m=size(y,1);
difference_matrix=(y-check);%identifies the number of non zero terms 
false_negatives_number=size(find(difference_matrix==1),1);
false_positives_number=size(find(difference_matrix==-1),1);
miss=nnz(abs(difference_matrix));
miss_rate=(miss/m)*100;
hit_rate=100-miss_rate;
fprintf('false_positives_number computed: \n');
fprintf(' %f \n', false_positives_number);
fprintf('false negatives computed from the gradient descent: \n');
fprintf(' %f \n', false_negatives_number);
fprintf('miss rate computed: \n');
fprintf(' %f \n', miss_rate);
outlier_accuracy=(outliers_number/actual_outlier_number)*100;
fprintf('hit_rate computed: \n');
fprintf(' %f \n', hit_rate);
fprintf('outliers detection accuracy computed: \n');
fprintf(' %f \n', outlier_accuracy);

end