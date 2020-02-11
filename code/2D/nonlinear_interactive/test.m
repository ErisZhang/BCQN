clear all;
warning('off','all');
addpath(genpath('../'));
addpath(genpath('/Users/zhangjiayi/Documents/utils/gptoolbox'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(0);
[V,F] = readOBJ('../data/ginger.obj');

vec = @(X) X(:);

V = V(:,1:2);
% units are meters
V = V-(min(V)+max(V))*0.5;
V = V/max(V(:));

%%%%%%%%%%%%%%%%%%%%% precomputation %%%%%%%%%%%%%%%%%%%%
% no boundary condition in our case
dirichlet = zeros(2*size(V,1),1);

tri_num = size(F,1);
ver_num = size(V,1);
is_uv_mesh = 0;

[X_g_inv_m, tri_areas_m, F_dot_m, rows, cols, vals, x2u, J_u, J_ui, JT_u, JT_ui, perimeter] = precompute_mex(tri_num, F, ver_num, vec(V'), is_uv_mesh, dirichlet);

global c1 c2 d1 en_type amips_s;
en_type = 'arap';
% en_type = 'mips';
c1 = 8; % shear
d1 = 2;  % bulk
c2 = 0; % meaningless in 2D
amips_s = 1;
clamp = 0;

% check = norm(perimeter)
perimeter_norm = norm(perimeter) * get_energy_characteristic();

X_g_inv = reshape(X_g_inv_m, tri_num, 2, 2);
tri_areas = reshape(tri_areas_m, tri_num, 1);
F_dot = reshape(F_dot_m, tri_num, 6, 2, 2);
total_volume = sum(tri_areas);

JTJ_in = [J_ui(1), J_ui(2), JT_ui(1), JT_ui(2)]';
%%%%%%%%%%%%%%%%%%%%% precomputation %%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%
R = [cos(pi/6) -sin(pi/6); sin(pi/6) cos(pi/6)];
U = V * R;
q_x = vec(U');


[J_q, JV_u, JTV_u, wu, bu, rows, cols, vals] = energy_hessian_mex(tri_num, X_g_inv, tri_areas, F, q_x, get_energy_type(), amips_s, F_dot, ver_num, c1, c2, d1, clamp, J_u, JT_u, JTJ_in, x2u);

grad = J_q;

rows = rows + 1;
cols = cols + 1;

H = sparse(rows, cols, vals, 2 * ver_num, 2 * ver_num);