% Generate testing data
clc, clear;
addpath(genpath('./XxUtils'));

%% 路径及参数设置
DataPath = 'D:\code\fluorescence\dataset\BioSR\BioSR\F-actin\F-actin';
SavePath = 'D:\LT\项目\5.Bayes-SIM\2.空域不确定性\Nature版本\Demo_BayesDL-SIM\DatasetGenerate\Trainingdata\F-actin';
DataFilter = 'RawSIMData_level_*.mrc';
GtRawFilter = 'RawSIMData_gt.mrc';
GtFilter = 'SIM_gt.mrc';

cell_list = 1:51;
snr_list = 1:12;
if contains(SavePath,'Non-Linear'), n_perSIM = 25; else, n_perSIM = 9; end
bg_cut_prct = 0;  % percentile value of gt background, used for some specific data, e.g. CCPs

save_raw_path = [SavePath '/Raw/'];            % 保存SIM raw images
save_WF_path = [SavePath '/WF/'];              % 保存WF
save_gt_path = [SavePath '/GT/'];              % 保存GT
if exist(save_raw_path,'dir'), rmdir(save_raw_path,'s'); end
mkdir(save_raw_path);
if exist(save_WF_path,'dir'), rmdir(save_WF_path,'s'); end
mkdir(save_WF_path);
if exist(save_gt_path,'dir'), rmdir(save_gt_path,'s'); end
mkdir(save_gt_path);

%% 
CellList = XxSort(XxDir(DataPath, 'Cell*'));  % 各cell文件夹路径
ImgCount = 1;
for i = cell_list
    ith_cell_gt_path = [save_gt_path 'cell_' int2str(i)];
    ith_cell_WF_path = [save_WF_path 'cell_' int2str(i)];
    ith_cell_raw_path = [save_raw_path 'cell_' int2str(i)];
    mkdir(ith_cell_gt_path);
    mkdir(ith_cell_WF_path);
    mkdir(ith_cell_raw_path);
    
    % ith GT
    files_gt = XxDir(CellList{i}, GtFilter); 
    [header_gt, data_gt] = XxReadMRC(files_gt{1});                        % 读取gt~[0,65535]
    data_gt = reshape(data_gt, header_gt(1), header_gt(2), header_gt(3)); % reshape
    data_gt = XxNorm(data_gt,0,100);                                      % Norm to range~[0,1]
    data_gt = imadjust(data_gt,[bg_cut_prct,1],[]);                       % 灰度变换,[bg_cut_prct,1] --> [0,1]
    data_gt = uint16(data_gt * 65535);                                    % to range~[0,65535]
    imwrite(data_gt,[ith_cell_gt_path '/gt.tif'])
    
    % ith GT Raw
    files_raw_gt = XxSort(XxDir(CellList{i}, GtRawFilter)); % 第i个cell下的RawSIMData_gt.mrc
    disp(['Generate RawSIMData_gt of cell ' int2str(i)])
    mkdir([ith_cell_raw_path '/level_gt']);
    [header_raw_gt, data_raw_gt] = XxReadMRC(files_raw_gt{1});
    data_raw_gt = data_raw_gt-98;
    data_raw_gt = reshape(data_raw_gt, header_raw_gt(1), header_raw_gt(2), header_raw_gt(3));
    data_raw_gt = XxNorm(data_raw_gt,0,100);
    data_raw_gt = uint16(data_raw_gt*65535);
    for k = 1:size(data_raw_gt,3)
        k_img = data_raw_gt(:,:,k);
        imwrite(k_img, [ith_cell_raw_path '/level_gt' '/' int2str(k) '.tif'])
    end
    
    % ith gt wf
    data_wf_gt = sum(data_raw_gt, 3);
    data_wf_gt = uint16(XxNorm(data_wf_gt, 0, 100) * 65535);
    imwrite(data_wf_gt,[ith_cell_WF_path '/level_gt' '.tif'])
    
    % ith raw
    files_input = XxSort(XxDir(CellList{i}, DataFilter));  % 第i个cell下的所有RawSIMData_level_*.mrc文件
    for j = snr_list      
        disp(['Generate data of cell ' int2str(i)...
            ' and SNR level ' int2str(snr_list(j))])
        mkdir([ith_cell_raw_path '/level_' int2str(j)]);        % ith_cell_jth_level需要一个文件夹存9张图
        
        [header, data] = XxReadMRC(files_input{j});            % 读取Raw data
        data = data - 98;                                      % Remove background
        data = reshape(data, header(1), header(2), header(3)); % 502*502*9  range~[62,231]
        data = XxNorm(data,0,100);                             % range~[0,1]
        data = uint16(data * 65535);                           % range~[0,65535]
        
        for k = 1:size(data,3)
            k_img = data(:,:,k); 
            imwrite(k_img,[ith_cell_raw_path '/level_' int2str(j) '/' int2str(k) '.tif'])
        end
        
        % wf
        wf_data = sum(data, 3);
        wf_data = uint16(XxNorm(wf_data, 0, 100) * 65535);
        imwrite(wf_data,[ith_cell_WF_path '/level_' int2str(j) '.tif'])
    end
end