function [J, grad] = costFunctionReg(theta, X, y, lambda)
% COSTFUNCTION ���������� ������� ��������� � �������� ���������(��)���
% ������ ������������� ��������� � ��������������
% J = COSTFUNCTION(theta, X, y, lambda) ��������� ������� ���������, ���������
% theta � �������� ��������� ������������� ���������, � ����� ��������(�)
% ����������

% ������������ �������� �������
m = length(y); % ���������� ��������� ���������

% � �������� ���������� �������, ��������� ���������� ������ ���� ���������
% ��������� 
J = 0;
grad = zeros(size(theta));

% ====================== ��� ��� ����� ======================
%
[rows_num, cols_num] = size(X);

reg_sum = 0;
for j = 2:cols_num
    reg_sum = reg_sum + theta(j)^2;
end
reg_sum = reg_sum * lambda / 2;

for i = 1:m
    J = J - y(i)*log(sigmoid(X(i, :)*theta))-(1-y(i))*log(1-sigmoid(X(i, :)*theta)) + reg_sum;
end
J = J ./ m;



for j = 1:rows_num
    grad(1) = grad(1) + (sigmoid(X(j, :)*theta) - y(j))*X(j, 1);
end
grad(1) = grad(1) / m;
    
for i = 2:cols_num
    for j = 1:rows_num
        grad(i) = grad(i) + (sigmoid(X(j, :)*theta) - y(j))*X(j, i);
    end
    grad(i) = (grad(i) + lambda*theta(i)) / m;
end

%
% =============================================================

end