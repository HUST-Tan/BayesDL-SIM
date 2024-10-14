function DirListSelected = XxDir(path, DirType, OutType)

% ------------------------------------------------------------------------
% XxDir: return subdirs of the input path
%
% usage:  DirListSelected = XxDir(path, DirType, OutType)
% where,
%    path        -- root path
%    DirType     -- type of file to glob, eg: '*.mrc'
%    OutType     -- exclude the dirs containing OutType, eg: '.txt'
%
% Author: Chang Qiao
% Email: qc17@mails.tsinghua.edu.cn
% Version: 2019/4/4
% ------------------------------------------------------------------------

if nargin < 3, OutType = '$'; end
if nargin < 2 || isempty(DirType) , DirType = '*'; end          % 输入参数小于2个，则glob的Dirtype设为通配符*
if strcmp(path(end), filesep) == 0, path = [path filesep]; end  % path不以filesep结尾，则在后面加上filesep
DirList = dir([path DirType]);
if strcmp(DirType, '*')
    DirList = {DirList(3:end).name}';
else
    DirList = {DirList.name}';
end
nList = 0;
DirListSelected = cell(1);
for i = 1:size(DirList,1)
    if isempty(strfind(DirList{i},OutType))  % DirList{i}中不含有OutType，才会添加到 DirListSelected
        nList = nList + 1;
        DirListSelected{nList} = [path DirList{i}];
    end
end
DirListSelected = DirListSelected';

end