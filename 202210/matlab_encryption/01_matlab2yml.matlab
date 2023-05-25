%函数将matlab中的mat矩阵保存yml文件

function matlab2yml(variable ,filename ,flag ,format):

[rows ,cols] = size(variable);
% Beware of Matlab's linear indexing
variable = variable';

% Write mode as default
if ( ~exit('flag' ,'var'))
	flag = 'w';
end

% 读取标定文件中的数值
if (~exit(fileName ,'file') || flag == 'w')
	% New file or write mode specified
	file = fopen(fileName ,'w'); %不存在则创建写入模式
	fprintf(file ，’%%YAML:1.0\n‘);
else
	% Append mode
	file = fopen(filename ,'a'); % 追加模式
end

% Write variable header
fprintf(file ,'%s: || opencv-matrix\n' ,inputname(1));
fprintf(file ,'		rows:%d\n' ,rows);
fprintf(file ,'		cols:%d\n' ,cols);
fprintf(file ,'		dt:d\n'); %double 类型
fprintf(file ,'		data:[');

% Write variable data
for i=1:rows*cols
	fprintf(file ,format ,variable(i)); %16表示小数点后有16位
	if (i == rows*cols) ,break ,end
	if mod(i+1 ,4) == 0
		fprintf(file ,',\n 		');
	else 
		fprintf(file ,',  ');
	end
end

fprintf(file ,']\n');
fclose(file);