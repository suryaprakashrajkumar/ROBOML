clc;clear all;

ts = 1/60;
r = 0.04445;
d = 0.393;
wrwlmax = 10.0;


eta = 1.8;
alpha = 6.1;
N = 20;
Q_cost = diag([100, 100, 10]);
R_cost = diag([1.7, 0.1]);

