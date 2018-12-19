function metalevel_cv(varargin)

num_folds = 5;

if (nargin < 3) || (nargin > 4),
    disp('Usage: metalevel_cv(label_vector, instance_matrix, size_of_groups [, ''libsvm_options'']);');
    return;
else
    train_y = varargin{1};
    train_x = varargin{2};
    logindex = varargin{3};
    libsvm_options = '';
    if nargin == 4
        libsvm_options = varargin{4};
        cv_str = regexp(libsvm_options, '-v \w', 'match');
        if ~isempty(cv_str)
            cv_str = regexp(cv_str{1}, ' ', 'split');
            num_folds = str2num(cv_str{2});
            libsvm_options = regexprep(libsvm_options, '-v \w', '');
        end
    end
end

% ensure same folds accross all parameters
rand('seed', 1);

if size(logindex(logindex < 0),1) > 0
    error('Error: input size_of_groups contains negative values');
end

if sum(logindex) ~= size(train_x,1)
   error('Error: number of training instances in train_x is not consistent with the total number of instances in size_of_groups');
end

logindex(logindex == 0) = [];

l = size(train_y, 1);
num_logs = size(logindex,1);
perm = randperm(num_logs)';

start_log = 1;
for i = 1:num_folds
    end_log = floor(num_logs*i/num_folds);
    selected_log = sort(perm(start_log:end_log));
    start = [1; cumsum(logindex)+1];
    subset = [];
    for j=1:size(selected_log,1)
        subset = [subset; [start(selected_log(j)):start(selected_log(j)+1)-1]'];
    end
    train_folds{i} = subset;
    start_log = end_log + 1;
end

cv = 0;
for i=1:num_folds
    y = train_y(train_folds{i},1);
    x = train_x(train_folds{i},:);
    test_subset = setdiff((1:l)', train_folds{i});
    m = svmtrain(y, x, ['-q ' libsvm_options]);
    yt = train_y(test_subset,1);
    xt = train_x(test_subset,:);
    [~, acc, ~] = svmpredict(yt, xt, m, '-q');
    cv = cv + acc(1);
end

fprintf(1, 'Cross Validation Accuracy = %g%%\n',cv/num_folds);


