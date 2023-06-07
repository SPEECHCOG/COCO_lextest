function recall = COCO_lextest(audiodir,embdir,opmode,runparallel,outputdir)
% function COCO_lextest(audiodir,embdir,opmode,runparallel,outputdirdir)
%
% Inputs:
%
%   audiodir        : location of the CDI audiofiles (.wav)
%   embdir          : location of corresponding embeddings (.txt)
%   opmode          : 'single' (default) for one embedding per .wav
%                     'full' for frame-level embeddings
%   runparallel     : 0 (default) / 1. Use MATLAB parallel pool? Only
%                     applies to opmode = 'full'.
%   
%   outputdir         : folder to save the results
%   
% Outputs:
%   recall          : overall separability score [0,1]. Larger is better.   
%   lextest_type_scores.csv : results for individual word types
%   lextest_confusion_matrix.csv: confusion matrix of word types


if nargin <5
    outputdir = [fileparts(which('CDI_lextest.m')) '/'];
    fprintf('No output directory specified for LexTest. Saving to current dir.\n');
end

if nargin <4
    runparallel = 0;
end

if nargin <3
    opmode = 'single';
end


% Check and parse audio directory
tmp = dir([audiodir '/*.wav']);

if(length(tmp) ~= 1600)
    error('Number of original audio files does not match expectation (incorrect path?)');
end

filenames = cell(length(tmp),1);
word = cell(length(tmp),1);
spkr = cell(length(tmp),1);
for k = 1:length(tmp)
    filenames{k} = [audiodir '/' tmp(k).name];
    kek = strfind(tmp(k).name,'_');
    word{k} = tmp(k).name(1:kek(1)-1); % word type
    spkr{k} = tmp(k).name(end-5); % speaker ID
end


uq_words = unique(word);

labels = zeros(length(word),1);

for k = 1:length(word)
    labels(k) = find(strcmp(uq_words,word{k}));
end


N_tokens = sum(labels == 1);


if(strcmp(opmode,'single'))

    % Load embeddings

    tmp = dir([embdir '/*.txt']);
    emb_ID = cell(length(tmp),1);
    for k = 1:length(tmp)
        emb_ID{k} = tmp(k).name(1:end-4);
    end

    % Check dimension by loading one
    fid = fopen([embdir '/' tmp(1).name]);
    line = str2num(fgetl(fid));
    fclose(fid);
    dim = length(line);

    X = zeros(length(filenames),dim);

    for k = 1:length(filenames)
        [a,b,c] = fileparts(filenames{k});

        i = find(strcmp(emb_ID,b));
        if(~isempty(i))
            fid = fopen([embdir '/' tmp(i).name]);
            X(k,:) = str2num(fgetl(fid));
            fclose(fid);
        else
            error('cannot find embedding for %s',b);
        end
    end





    % Run a diagnostic classifier

    k_nearest = N_tokens-1;

    recall = zeros(length(uq_words),N_tokens);

    CC = zeros(length(uq_words),length(uq_words));
    for fold = 1:N_tokens
        i_test = fold:N_tokens:length(labels);
        i_train = setxor(i_test,1:length(labels));

        labels_train = labels(i_train);
        labels_test = labels(i_test);

        D = pdist2(X(i_test,:),X(i_train,:),'cosine');

        [D_sort,D_ind] = sort(D,2,'ascend');

        hypos = labels_train(D_ind(:,1:k_nearest));

        for k = 1:size(hypos,1)
            recall(k,fold) = sum(hypos(k,:) == labels_test(k))./(N_tokens-1);

            for j = 1:size(hypos,2)
                CC(labels_test(k),hypos(k,j)) = CC(labels_test(k),hypos(k,j))+1;
            end
        end
    end
   


elseif(strcmp(opmode,'full'))


    % Load embeddings

    tmp = dir([embdir '/*.txt']);
    emb_ID = cell(length(tmp),1);
    for k = 1:length(tmp)
        emb_ID{k} = tmp(k).name(1:end-4);
    end

    % Check dimension by loading one
    fid = fopen([embdir '/' tmp(1).name]);
    line = str2num(fgetl(fid));
    fclose(fid);
    dim = length(line);

    F = cell(length(filenames),1);

    for k = 1:length(filenames)
        F{k} = zeros(1000,dim);
        [a,b,c] = fileparts(filenames{k});

        i = find(strcmp(emb_ID,b));
        if(~isempty(i))
            c = 1;
            fid = fopen([embdir '/' tmp(i).name]);
            line = 1;
            while(line ~= -1)
                line = fgetl(fid);
                if(line ~= -1)
                    F{k}(c,:) = str2num(line);
                    c = c+1;
                end
            end
            F{k}(c:end,:) = [];
            fclose(fid);
        else
            error('cannot find embedding for %s',b);
        end
    end



    % Version 2: use dtw

    k_nearest = N_tokens-1;

    recall = cell(N_tokens,1);
    CC = cell(N_tokens,1);

    D = cell(N_tokens,1);

    if(runparallel)

        parfor fold = 1:N_tokens
            i_test = fold:N_tokens:length(labels);
            i_train = setxor(i_test,1:length(labels));

            CC{fold} = zeros(length(uq_words),length(uq_words));

            labels_train = labels(i_train);
            labels_test = labels(i_test);

            D{fold} = zeros(length(i_test),length(i_train));

            for k = 1:length(i_test)

                Y = F{i_test(k)};
                [row,col] = find(isnan(Y));
                Y(row,:) = [];
                for j = 1:length(i_train)
                    YY = F{i_train(j)};
                    [row,col] = find(isnan(YY));
                    YY(row,:) = [];
                    D{fold}(k,j) = dtw(Y',YY');
                    %[p,q,~,sc] = dpfast(pdist2(Y,YY,'euclidean'));
                    %D{fold}(k,j) = sum(sc);
                end
            end

            [D_sort,D_ind] = sort(D{fold},2,'ascend');

            hypos = labels_train(D_ind(:,1:k_nearest));

            for k = 1:size(hypos,1)
                recall{fold}(k) = sum(hypos(k,:) == labels_test(k))./(N_tokens-1);
                for j = 1:size(hypos,2)
                    CC{fold}(labels_test(k),hypos(k,j)) = CC{fold}(labels_test(k),hypos(k,j))+1;
                end
            end
        end

    else
        for fold = 1:N_tokens
            i_test = fold:N_tokens:length(labels);
            i_train = setxor(i_test,1:length(labels));

            CC{fold} = zeros(length(uq_words),length(uq_words));

            labels_train = labels(i_train);
            labels_test = labels(i_test);

            D{fold} = zeros(length(i_test),length(i_train));

            for k = 1:length(i_test)

                Y = F{i_test(k)};
                [row,col] = find(isnan(Y));
                Y(row,:) = [];
                for j = 1:length(i_train)
                    YY = F{i_train(j)};
                    [row,col] = find(isnan(YY));
                    YY(row,:) = [];
                    D{fold}(k,j) = dtw(Y',YY');
                    %[p,q,~,sc] = dpfast(pdist2(Y,YY,'euclidean'));
                    %D{fold}(k,j) = sum(sc);
                end
            end

            [D_sort,D_ind] = sort(D{fold},2,'ascend');

            hypos = labels_train(D_ind(:,1:k_nearest));

            for k = 1:size(hypos,1)
                recall{fold}(k) = sum(hypos(k,:) == labels_test(k))./(N_tokens-1);
                for j = 1:size(hypos,2)
                    CC{fold}(labels_test(k),hypos(k,j)) = CC{fold}(labels_test(k),hypos(k,j))+1;
                end
            end
        end
    end

    recall_formatted = zeros(length(uq_words),N_tokens);
    for k = 1:N_tokens
        recall_formatted(:,k) = recall{k};
    end    
    recall = recall_formatted;

    CC_formatted = zeros(size(CC{1}));
    for k = 1:N_tokens
        CC_formatted = CC_formatted+CC{k};
    end

    CC = CC_formatted;

end

recall_word = mean(recall,2);
recall_total = mean(mean(recall,2));

%fprintf('Overall recall: %0.2f%%\n',recall_total.*100);

fid = fopen([outputdir '/lextest_overall_score.txt'],'w');
fprintf(fid,'Overall recall: %0.3f\n',recall_total*100);
fclose(fid);


fid = fopen([outputdir '/lextest_type_scores.csv'],'w');
%fprintf(fid,'Overall recall: %0.2f%%\n',recall_total.*100);
%fprintf(fid,'Word-by-word recalls\n');
for k = 1:length(recall_word)
    fprintf(fid,'%s, %0.3f%%\n',uq_words{k},recall_word(k)*100);
end
fclose(fid);

fid = fopen([outputdir '/lextest_confusion_matrix.csv'],'w');
fprintf(fid,'word/word,');
for j = 1:size(CC,2)
    fprintf(fid,'%s,',uq_words{j});
end
fprintf(fid,'\n');
for k = 1:size(CC,1)
    fprintf(fid,'%s,',uq_words{k});
    for j = 1:size(CC,2)
        fprintf(fid,'%d,',CC(k,j));
    end
    fprintf(fid,'\n');
end





