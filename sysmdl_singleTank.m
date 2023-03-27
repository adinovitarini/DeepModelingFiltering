function singleTank = sysmdl_singleTank(N,u)
close all;clc
z = tf('z');
Gz = tf([0 0.08123],[1 -0.9895],0.01);
% [AA,BB,CC,D] = tf2ss([0 0.08123],[1 -0.9895]);
AA = [1 0.07599;-0.01648 -0.074];
BB = [0.02094;-0.283];
CC = [-8.022 0.02691];
D = 0;
df = .01; %discount factor 
A = sqrt(df)*AA;
B = sqrt(df)*BB;
C = sqrt(df)*CC;
sys = ss(A,B,C,D,0.01);
%%  Apply Disturbance 
w = wgn(N,1,1);
v = w;
%%  Open Loop System
x = .1*ones(size(A,1),1);
for i = 1:N
    x(:,i+1) = A*x(:,i)+B*u(i)+w(i);
    y(i) = C*x(:,i)+v(i);
    x(:,i) = x(:,i+1);
end
singleTank.sys = sys;
singleTank.x = x;
singleTank.y = y;