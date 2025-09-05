 % problem1_occlusion.m
    % 题目A-问题1：FY1 投放 1 枚烟幕干扰弹对 M1 的遮蔽时长（精确计算+可视化）
    % 期望数值（便于自检）：t1 ≈ 8.03789076 s，t2 ≈ 9.44808816 s，时长 ≈ 1.41019740 s
    
    clear; clc;
    
    % 声明全局变量
    global M_pos C_pos T_center T_radius T_height r
    
    %% ---------- 常量与初始条件 ----------
    O  = [0;   0;   0  ];       % 假目标（坐标原点）
    T_center = [0; 200; 0];     % 真目标下底面圆心
    T_radius = 7.0;             % 真目标圆柱体半径 (m)
    T_height = 10.0;            % 真目标圆柱体高度 (m)
    M0 = [20000; 0; 2000];      % 导弹 M1 初始位置
    vM = 300.0;                 % 导弹速度 (m/s)
    
    vU = 120.0;                 % 无人机 FY1 速度 (m/s)
    g  = 9.8;                   % 重力加速度 (m/s^2)
    r  = 10.0;                  % 烟幕有效半径 (m)
    vc = 3.0;                   % 云团下沉速度 (m/s)
    
    t_drop = 1.5;               % 受领任务后投放时间 (s)
    dt_fuse = 3.6;              % 投放到起爆的时延 (s)
    t_e = t_drop + dt_fuse;     % 起爆时刻 (s)
    t_window = [t_e, t_e + 20]; % 云团有效时窗 [5.1, 25.1] s
    
    % 导弹指向原点的单位向量
    uM = (O - M0) / norm(O - M0);
    
    % 投放点（不参与后续计算，仅作记录）
    P_drop = [17800 - vU * t_drop; 0; 1800];
    
    % 起爆中心 C0（离机后水平方向随无人机速度，竖直方向自由落体）
    x_e = 17800 - vU * (t_drop + dt_fuse);
    z_e = 1800  - 0.5 * g * (dt_fuse^2);
    C0  = [x_e; 0; z_e];
    
    fprintf('Explosion center C0 = (%.3f, %.3f, %.3f)\n', C0(1), C0(2), C0(3));
    
    % 轨迹函数（时间 t 单位：秒）
    M_pos = @(t) M0 + vM * t .* uM;                      % 导弹位置
    C_pos = @(t) C0 + [0;0;-vc] .* (t - t_e);            % 云团中心（仅在 [t_e, t_e+20] 有效）
    
    % 距离函数：f(t) = d^2(t) - r^2，d 为云团中心到圆柱体[T, M(t)]的最短距离
    f_scalar = @(t) point_to_cylinder_dist2(T_center, T_radius, T_height, M_pos(t), C_pos(t)) - r^2;
    
    %% ---------- 数值求根：找到遮蔽的进入/退出时刻 ----------
    tmin = t_window(1); tmax = t_window(2);
    dt_scan = 1e-3;                                  % 粗扫步长（1 ms）
    N = floor((tmax - tmin) / dt_scan);
    ts = tmin + (0:N) * dt_scan;
    
    fs = arrayfun(f_scalar, ts);
    
    % 搜索变号区间并二分求根
    roots = [];
    for i = 1:numel(ts)-1
        f1 = fs(i); f2 = fs(i+1);
        if f1 == 0
            roots(end+1) = ts(i); %#ok<AGROW>
        elseif f1 * f2 < 0
            a = ts(i); b = ts(i+1);
            root = bisect(@(x) f_scalar(x), a, b, 1e-10, 100); % 高精度二分
            roots(end+1) = root; %#ok<AGROW>
        end
    end
    
    % 去重并排序
    if ~isempty(roots)
        roots = unique(roots, 'sorted');
    end
    
    % 由根构造遮蔽区间（f<=0 为遮蔽）
    intervals = [];
    if ~isempty(roots)
        f_at_min = f_scalar(tmin);
        active = (f_at_min <= 0);
        prev = tmin;
        events = [roots(:).' tmax];
        for k = 1:numel(events)
            ev = events(k);
            if active
                intervals = [intervals; prev, ev]; %#ok<AGROW>
            end
            if ev ~= tmax
                active = ~active;
                prev = ev;
            end
        end
    else
        if f_scalar(tmin) <= 0
            intervals = [tmin, tmax];
        end
    end
    
    % 合并数值噪声导致的极短区间（可选）
    intervals = mergeTinyIntervals(intervals, 1e-7);
    
    % 计算总时长与打印
    total_dur = sum(intervals(:,2) - intervals(:,1));
    fprintf('Occlusion intervals within [%.3f, %.3f] s:\n', tmin, tmax);
    if isempty(intervals)
        fprintf('  (none)\n');
    else
        for i = 1:size(intervals,1)
            fprintf('  [%0.9f, %0.9f]  (%.9f s)\n', intervals(i,1), intervals(i,2), intervals(i,2)-intervals(i,1));
        end
    end
    fprintf('Total occlusion duration = %.9f s\n', total_dur);
    
    %% ---------- 可视化（3D 理解图） ----------
    do_plot = true;
    if do_plot
        figure('Color','w','Position',[80 80 1000 720]); hold on; grid on; view(45,25);
        % 导弹路径（t: 0~30 s）
        t_show = linspace(0, 30, 600);
        M_path = M_pos(t_show);
        plot3(M_path(1,:), M_path(2,:), M_path(3,:), 'LineWidth', 1.2, 'DisplayName','Missile M1 path');
    
        % 云团中心有效路径（t: t_e ~ t_e+20）
        t_smoke = linspace(tmin, tmax, 200);
        C_path = C_pos(t_smoke);
        plot3(C_path(1,:), C_path(2,:), C_path(3,:), 'LineWidth', 1.2, 'DisplayName','Smoke center path (effective)');
    
        % 关键点
        scatter3(O(1),O(2),O(3),60,'o','filled','DisplayName','Fake target (origin)');
        % 绘制圆柱体真目标
        drawCylinder(T_center, T_radius, T_height, 'True target cylinder');
        scatter3(C0(1),C0(2),C0(3),60,'s','filled','DisplayName','Explosion center C0');
    
        % 画进入/中点/退出关键帧（若存在）
        if ~isempty(intervals)
            t1 = intervals(1,1); t2 = intervals(1,2);
            tm = 0.5*(t1 + t2);
            add_snapshot(t1, 'Entry');
            add_snapshot(tm, 'Mid');
            add_snapshot(t2, 'Exit');
        end
    
        xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
        title('Occlusion Visualization: Missile–Target Line vs. Smoke Sphere');
        axis vis3d; axis equal;
        legend('Location','northeastoutside');
    
    end
    
    %% ---------- 函数区（存于脚本尾部） ----------
    function add_snapshot(tval, lbl)
        global M_pos C_pos T_center T_radius T_height r
        B = M_pos(tval); 
        Cc = C_pos(tval);
        % 绘制从圆柱体中心到导弹的视线
        plot3([T_center(1) B(1)], [T_center(2) B(2)], [T_center(3) B(3)], 'LineWidth', 1.0, 'DisplayName',[lbl ' line-of-sight']);
        scatter3(B(1),B(2),B(3),40,'x','DisplayName',['Missile @ ' lbl]);
        drawSphereWire(Cc, r, 18); % 球面线框
        text(Cc(1), Cc(2), Cc(3), sprintf('%s\nt=%.3f s', lbl, tval), 'FontSize', 9, 'HorizontalAlignment','center');
    end
    
    function d2 = point_to_cylinder_dist2(center, radius, height, missile_pos, cloud_pos)
    % 返回云团中心到圆柱体表面上所有点与导弹连线的最大平方距离 d^2
    % 遮蔽条件：云团中心到圆柱体表面上所有点与导弹连线的最大距离 < 10m
    % center: 圆柱体下底面圆心
    % radius: 圆柱体半径  
    % height: 圆柱体高度
    % missile_pos: 导弹位置
    % cloud_pos: 云团中心位置
        
        % 计算云团中心到圆柱体表面上所有点与导弹连线的最大距离
        % 遮蔽条件：云团中心到圆柱体表面上所有点与导弹连线的最大距离 < 10m
        
        % 圆柱体上底面圆心
        top_center = center + [0; 0; height];
        
        % 圆柱体轴线方向向量（从下底面到上底面）
        axis_vec = [0; 0; height];
        axis_length = height;
        
        % 云团中心到轴线的投影
        to_center = cloud_pos - center;
        proj_length = dot(to_center, [0; 0; 1]); % Z方向投影
        
        % 计算投影点在轴线上的位置
        if proj_length <= 0
            % 投影在圆柱体下底面以下
            closest_axis_point = center;
        elseif proj_length >= axis_length
            % 投影在圆柱体上底面以上
            closest_axis_point = top_center;
        else
            % 投影在圆柱体内部
            closest_axis_point = center + proj_length * [0; 0; 1];
        end
        
        % 计算云团中心到轴线的距离
        to_axis_dist = norm(cloud_pos - closest_axis_point);
        
        % 到圆柱体表面的距离
        surface_dist = max(0, to_axis_dist - radius);
        
        % 现在需要计算云团中心到圆柱体表面上所有点与导弹连线的最大距离
        % 这等价于计算云团中心到圆柱体表面上所有点与导弹连线的最大距离
        
        % 圆柱体表面上的点可以表示为：
        % P = center + radius * [cos(theta), sin(theta), 0] + h * [0, 0, 1]
        % 其中 theta ∈ [0, 2π], h ∈ [0, height]
        
        % 对于圆柱体表面上的任意一点P，云团中心到线段[P, missile_pos]的距离为：
        % 点到线段的距离公式
        
        % 我们需要找到使这个距离最大的点P
        % 这可以通过分析几何关系来求解
        
        % 简化计算：考虑圆柱体表面的关键点
        % 1. 下底面圆周上的点
        % 2. 上底面圆周上的点  
        % 3. 侧面上的点
        
        max_dist_sq = 0;
        
        % 检查下底面圆周上的点
        for theta = 0:pi/4:2*pi
            P = center + radius * [cos(theta); sin(theta); 0];
            dist_sq = point_to_line_segment_dist_sq(cloud_pos, P, missile_pos);
            max_dist_sq = max(max_dist_sq, dist_sq);
        end
        
        % 检查上底面圆周上的点
        for theta = 0:pi/4:2*pi
            P = top_center + radius * [cos(theta); sin(theta); 0];
            dist_sq = point_to_line_segment_dist_sq(cloud_pos, P, missile_pos);
            max_dist_sq = max(max_dist_sq, dist_sq);
        end
        
        % 检查侧面上的点（在高度方向上采样）
        for h = 0:height/4:height
            for theta = 0:pi/8:2*pi
                P = center + radius * [cos(theta); sin(theta); 0] + h * [0; 0; 1];
                dist_sq = point_to_line_segment_dist_sq(cloud_pos, P, missile_pos);
                max_dist_sq = max(max_dist_sq, dist_sq);
            end
        end
        
        d2 = max_dist_sq;
    end
    
    function dist_sq = point_to_line_segment_dist_sq(point, line_start, line_end)
    % 计算点到线段的最小距离的平方
        % 线段方向向量
        line_vec = line_end - line_start;
        line_length_sq = sum(line_vec.^2);
        
        if line_length_sq == 0
            % 线段退化为点
            dist_sq = sum((point - line_start).^2);
            return;
        end
        
        % 从线段起点到点的向量
        to_point = point - line_start;
        
        % 计算投影参数t
        t = dot(to_point, line_vec) / line_length_sq;
        
        % 将t限制在[0,1]范围内
        t = max(0, min(1, t));
        
        % 线段上最近的点
        closest_point = line_start + t * line_vec;
        
        % 距离的平方
        dist_sq = sum((point - closest_point).^2);
    end
    
    function root = bisect(fun, a, b, tol, itmax)
    % 二分法（要求 fun(a)*fun(b) <= 0），返回 [a,b] 内根
        fa = fun(a); fb = fun(b);
        if fa == 0, root = a; return; end
        if fb == 0, root = b; return; end
        if fa * fb > 0
            error('bisect: interval does not bracket a root.');
        end
        for k = 1:itmax
            m = 0.5*(a+b);
            fm = fun(m);
            if fm == 0 || 0.5*(b-a) < tol
                root = m; return;
            end
            if fa * fm <= 0
                b = m; fb = fm;
            else
                a = m; fa = fm;
            end
        end
        root = 0.5*(a+b);
    end
    
    function intervals_out = mergeTinyIntervals(intervals_in, tol)
    % 合并极短/相邻间隔，避免数值噪声
        if isempty(intervals_in); intervals_out = intervals_in; return; end
        intervals_in = sortrows(intervals_in, 1);
        cur = intervals_in(1,:);
        out = [];
        for i = 2:size(intervals_in,1)
            if intervals_in(i,1) - cur(2) <= tol
                cur(2) = max(cur(2), intervals_in(i,2));
            else
                if (cur(2)-cur(1)) > tol
                    out = [out; cur]; %#ok<AGROW>
                end
                cur = intervals_in(i,:);
            end
        end
        if (cur(2)-cur(1)) > tol
            out = [out; cur];
        end
        intervals_out = out;
    end
    
    function drawSphereWire(center, radius, n)
    % 画线框球体（轻量）
        if nargin < 3, n = 18; end
        [xs, ys, zs] = sphere(n);
        xs = center(1) + radius * xs;
        ys = center(2) + radius * ys;
        zs = center(3) + radius * zs;
        mesh(xs, ys, zs, 'EdgeAlpha', 0.3, 'FaceColor', 'none');
    end
    
    function drawCylinder(center, radius, height, displayName)
    % 绘制圆柱体
        if nargin < 4, displayName = 'Cylinder'; end
        
        % 圆柱体参数
        n = 20; % 圆周分段数
        theta = linspace(0, 2*pi, n+1);
        
        % 下底面
        x_bottom = center(1) + radius * cos(theta);
        y_bottom = center(2) + radius * sin(theta);
        z_bottom = center(3) * ones(size(theta));
        
        % 上底面
        x_top = x_bottom;
        y_top = y_bottom;
        z_top = (center(3) + height) * ones(size(theta));
        
        % 绘制圆柱体表面
        % 侧面
        for i = 1:n
            x_side = [x_bottom(i), x_bottom(i+1), x_top(i+1), x_top(i)];
            y_side = [y_bottom(i), y_bottom(i+1), y_top(i+1), y_top(i)];
            z_side = [z_bottom(i), z_bottom(i+1), z_top(i+1), z_top(i)];
            fill3(x_side, y_side, z_side, [0.8 0.8 0.8], 'EdgeAlpha', 0.3, 'FaceAlpha', 0.3);
        end
        
        % 下底面
        fill3(x_bottom, y_bottom, z_bottom, [0.8 0.8 0.8], 'EdgeAlpha', 0.3, 'FaceAlpha', 0.3, 'DisplayName', displayName);
        
        % 上底面
        fill3(x_top, y_top, z_top, [0.8 0.8 0.8], 'EdgeAlpha', 0.3, 'FaceAlpha', 0.3);
    end
