%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Parameters (differential equation specified in flow_map.c)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Number of grid points to use
numx = 100;
numy = 100;
% Bounding box [x0, x1, y0, y1]
box = [0, 2, 0, 1];
time_span = [0, 7];
% Number of intermediate flow maps to compute
n = 3;
tic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Will produce a sequence of flow maps flow_maps(k, :, :, L) where k = 1..n
% and L = 1, 2. It will also produce the FTLE field exps(k, :, :) indexed
% by k = 1..n and with size [numx, numy].
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[X, Y] = meshgrid(linspace(box(1), box(2), numx), ...
         linspace(box(3), box(4), numy));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Double Gyre Flow
% A = 0.1;
% ep = 0.1;
% w = pi/5;
% 
% g = @(t, x) ep*sin(w*t)*x^2 + x*(1 - 2*ep*sin(w*t));
% dgdx = @(t, x) 1 + (2*x - ep)*ep*sin(w*t); %2*ep*sin(w*t)*x + 1 - ep*ep*sin(w*t);
% 
% %g = @(t, x) x;
% %dgdx = @(t, x) 1;
% 
% f = @(t, x) pi*A*[-sin(pi*g(t, x(1)))*cos(pi*x(2));  
% cos(pi*g(t, x(1)))*sin(pi*x(2))*dgdx(t, x(1))
% ];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t0 = time_span(1);
tend = time_span(2);

h = (tend - t0)/n;

flow_maps = zeros(n, size(X, 1), size(X, 2), 2);

exps = zeros(n, size(X, 1), size(X, 2));

flows = zeros([2*numx, numy]);

for k = 1:n

    [t0+(k-1)*h, t0+k*h]

    [status, result] = system(sprintf('./flow %f %f %f %f %f %f %d %d', box(1), box(2), box(3), box(4), t0+(k-1)*h, t0+k*h, numx, numy));
    flows = eval(result);
    flow_maps(k, :, :, 1) = flows(1:numx, :)';
    flow_maps(k, :, :, 2) = flows(1+numx:end, :)';

end

Dphi = zeros(n, size(X, 1), size(X, 2), 4);

% Linear spacing
dx = X(1, 2) - X(1, 1);
dy = Y(2, 1) - Y(1, 1);

% Iterate over maps
for k = 1:n
    % Compute interpolated map
    if k > 1
        phi_x = griddedInterpolant(X', Y', reshape(flow_maps(k, :, :, 1), [size(X, 1), size(X,2)])');
        phi_y = griddedInterpolant(X', Y', reshape(flow_maps(k, :, :, 2), [size(X, 1), size(X,2)])');

        % Evaluate at previous flow map's image
        flow(:, :, 1) = phi_x(flow_maps(k-1, :, :, 1), flow_maps(k-1, :, :, 2));
        flow(:, :, 2) = phi_y(flow_maps(k-1, :, :, 1), flow_maps(k-1, :, :, 2));
        flow_maps(k, :, :, :) = flow;
        [fx, fy, ~] = gradient(reshape(flow, size(X, 1), size(X, 2), 2), dx, dy, 1);

        % For each grid point, find exponent
        for i = 1:size(X, 1)
            for j = 1:size(X, 2)

                Dphi(k, i, j, :) = reshape([ fx(i, j, :) ; fy(i, j, :) ], [4, 1]);

                Yk = reshape(Dphi(k, i, j, :), [2, 2])*reshape(Dphi(k-1, i, j, :), [2, 2]);

                Dphi(k, i, j, :) = reshape(Yk, [4, 1]);

                exps(k, i , j) = norm(reshape(Dphi(k, i, j, :), [4, 1]));
            end
        end

        exps(k, : , :) = log(exps(k, :, :))/(k*(tend - t0)/n-t0);

    else
        flow = reshape(flow_maps(1, :, :, :), [size(X, 1), size(X, 2), 2]);

        [fx, fy, ~] = gradient(flow, dx, dy, 1);
        % Iterate over grid
        for i = 1:size(X, 1)
            for j = 1:size(X, 2)

                Dphi(k, i, j, :) = reshape([ fx(i, j, :) ; fy(i, j, :) ], [4, 1]);
                exps(k, i , j) = norm(reshape(Dphi(k, i, j, :), [4, 1]));
            end
        end

        exps(k, : , :) = log(exps(k, :, :))/(k*(tend - t0)/n-t0);
    end


end

toc

view_angles = [0, 90];

surf(X, Y, reshape(exps(end, :, :), [numy, numx]));
shading interp
view([0, 90])

