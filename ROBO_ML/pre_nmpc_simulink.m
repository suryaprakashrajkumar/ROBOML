clc;clear all;

ts = 1/60;
r = 0.04445;
d = 0.393;
wrwlmax = 10.0;


eta = 0.7;
alpha = 2.5;
N = 20;
Q_cost = diag([100, 100, 10]);
R_cost = diag([0.2, 0.1]);

