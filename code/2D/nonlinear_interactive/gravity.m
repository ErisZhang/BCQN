clear all;
warning('off','all');
addpath(genpath('../'));
addpath(genpath('/Users/zhangjiayi/Documents/utils/gptoolbox'));

rng(0);
[V,F] = readOBJ('../data/bar.obj');
V = V(:,1:2);

% normalize the mesh [0 1 0 1]: units are meters
V = V-(max(V)+min(V))/2;
V = V/max(V(:));

vec = @(X) reshape(X',size(X,1)*size(X,2),1);

% precomputed values
E = edges(F);
M = massmatrix(V,F);
Meles = diag(M);
M2eles = zeros(2*size(M,1),1);
M2eles(2*(1:size(M,1))) = Meles;
M2eles(2*(1:size(M,1))-1) = Meles;
M = sparse((1:2*size(M,1)),(1:2*size(M,1)),M2eles,2*size(M,1),2*size(M,1));
g = 0.005*vec(repmat([0 -9.8],size(V,1),1));
dt = 0.03;


fixed = find(V(:,1) < ( min(V(:,1)) + 0.2*(max(V(:,1))-min(V(:,1)))));
fixed = union(2*fixed,2*fixed-1);


%%%%%%%%%%%%%%%%%%%%% precomputation %%%%%%%%%%%%%%%%%%%%
% no boundary condition in our case
dirichlet = zeros(2*size(V,1),1);

tri_num = size(F,1);
ver_num = size(V,1);
is_uv_mesh = 0;

[X_g_inv_m, tri_areas_m, F_dot_m, rows, cols, vals, x2u, J_u, J_ui, JT_u, JT_ui, perimeter] = precompute_mex(tri_num, F, ver_num, vec(V), is_uv_mesh, dirichlet);

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


clf;
hold on;
s = tsurf(F,V,'FaceColor',blue,'EdgeColor','none','FaceAlpha',0.5);
t = tsurf(F,V,'FaceAlpha',0.8,'EdgeAlpha',0.8);
hold off;
axis equal;
% expand_axis(8);
axis([-2 2 -2 2])
axis manual;
drawnow;


% Don't move
U = vec(zeros(size(V)));
Ud = vec(zeros(size(V)));
mqwf = [];


while true

  % save previous steps
  Ud0 = Ud;
  U0 = U;

  max_iter = 10;
  U = U0; % initial guess for U
  for i = 1 : max_iter
    alpha = 1;
    p = 0.5;
    c = 1e-8;
  
    % total energy = gravitational potential energy + kinetic energy +
    % elastic potential energy
    q_x = vec(V)+U;

    [J_q, JV_u, JTV_u, wu, bu, rows, cols, vals] = energy_hessian_mex(tri_num, X_g_inv, tri_areas, F, q_x, get_energy_type(), amips_s, F_dot, ver_num, c1, c2, d1, clamp, J_u, JT_u, JTJ_in, x2u);
    G = J_q;
    rows = rows + 1;
    cols = cols + 1;
    K = sparse(rows, cols, vals, 2 * ver_num, 2 * ver_num);

    f = @(U) -(M*g)'*(vec(V)+U) + 0.5*(U-U0-dt*Ud0)'*M/(dt*dt)*(U-U0-dt*Ud0) + ...
        energy_value_mex(tri_num, X_g_inv, tri_areas, F, vec(V)+U, get_energy_type(), amips_s, c1, c2, d1);
        

    tmp_H = M/(dt^2) + K;
    tmp_g = M/(dt^2) * U - M*(g+U0/dt^2+Ud0/dt) + G;
    

    mqwf = [];
    % solve for complimentary displacemnt
    [dU,mqwf] = min_quad_with_fixed( ...
      0.5 * tmp_H, ...
      tmp_g, ...
      fixed,zeros(size(fixed)),[],[],mqwf);

    % check for newton convergence criterian
    if abs(tmp_g' * dU) < c
      break
    end
    
    disp(i);
    
    % perform backtracking line search
    f0 = f(U);
    s = f0 + c * tmp_g' * dU; % to ensure sufficient decrease

    while alpha > c
      U_tmp = U + alpha * dU;
      if f(U_tmp) <= s
        break
      end
      alpha = alpha * p;
    end

    U = U + alpha*dU;
  end

  Ud = (U-U0)/dt;
  
  t.Vertices = V+reshape(U,size(V,2),size(V,1))';
  drawnow;

end



