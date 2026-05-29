% VFDPC: Vertical Federated Density Peaks Clustering Under Nonlinear Mapping
clear;
clc;

k = 31;
K = 1:1:10;
% dataname = {'100leaves','3sources','BBC','BBCSport','HW','HW2sources','NGs','WebKB','Hdigit','Mfeat'};
dataname = {'Hdigit'};
numdata = length(dataname);


for cdata = 1:numdata
    % read data
    disp(char(dataname(cdata)));
    datadir = '';
    dataf = [datadir, cell2mat(dataname(cdata))];
    load(dataf);

    X = data;
    y0 = truelabel{1};
    if size(y0,2) > size(y0,1)
        y0 = y0';
    end
    c = length(unique(y0));

    tic;
    num = size(X{1}, 2);
    m = length(X);
    % Normalization: Z-score
    for i = 1:m
        for j = 1:num
            normItem = std(X{i}(:,j));
            if (0 == normItem)
                normItem = eps;
            end
            X{i}(:,j) = (X{i}(:,j)-mean(X{i}(:,j)))/normItem;
        end
    end
    S0 = cell(1, m);
    for i = 1:m
        S0{i} = pdist2(X{i}', X{i}', "euclidean"); % defualt: 'euclidean'. 'euclidean' or 'cosine'
        S0{i} = (S0{i}-min(min(S0{i})))./(max(max(S0{i}))-min(min(S0{i})));
    end

    [U0, ~, Memo] = rsa_logistic_MVC(S0); % Construct distance matrix by encryption technique


    for j = 1:num
        d_sum = sum(U0(j,:));
        if d_sum == 0
            d_sum = eps;
        end
        U0(j,:) = U0(j,:)/d_sum;
    end
    U = (U0 + U0')/2;

    y = myFDPAS_SC(U, c, k, 0);
    info = whos('y'); % The server sends labels to clients
    Memo = Memo + m * info.bytes / (1024^2);
    toc;

    ACC = accuracy(y0, y);
    NMI = nmi(y0, y);
    ARI = valid_RandIndex(y0, y);
    fprintf('...Runtime %d> ACC:%.4f\tARI:%.4f\tNMI:%.4f\tMemory:%.4f\n',1, ACC, ARI, NMI, Memo);
end