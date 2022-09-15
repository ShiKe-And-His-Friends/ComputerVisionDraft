syms r l f;

% 定义R3坐标系
x = r * cos(l) * cos(f);
y = r * cos(l) * sin(f);
z = r * sin(l);

% 雅可比计算
J = jacobian([x ; y ; z] ,[r l f]);
J