import casadi.*
x = MX.sym('x')
disp(jacobian(sin(x),x))