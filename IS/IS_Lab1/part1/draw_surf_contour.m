data = load('surf_data.txt');
theta0_vals = data(:, 1);
theta1_vals = data(:, 2);
J_vals = load('surf_values.txt');
theta = load('theta_vals.txt');
%J_vals = data(:, :);

% производится транспонирование из-за особенности работы программы 
% surf
J_vals = J_vals';
% Отображение поверхности
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Контурное представление
figure;
% Отображение J_vals в виде 15 контуров распределенных в 
% логарифмическом масштабе от 0.01 до 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
