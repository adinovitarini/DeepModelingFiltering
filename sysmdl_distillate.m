function distillate = sysmdl_distillate(N,u)
close all;clc
z = tf('z');
AA = [1.1367 -0.77978 -0.41183 -0.93463 0;
    1.0468 1.0221 0.51514 0.55115 0;
    -0.77322 0.73872 -0.83021 0.0026816 0;
    1.1816 0.95094 -0.65203 -0.78764 0;
    0 0 0 0 1];
BB = [-1.3696;0.45253;1.0801;-0.37804;1];
CC = [0.72552 -0.78382 0.97289 -0.3413 1];
D = 0;
df = .01; %discount factor 
A = sqrt(df)*AA;
B = sqrt(df)*BB;
C = sqrt(df)*CC;
sys = ss(A,B,C,D,0.01);
%%  Apply Disturbance 
w = wgn(N,100,1);
v = w;
%%  Open Loop System
x = .1*ones(size(A,1),1);
for i = 1:N
    x(:,i+1) = A*x(:,i)+B*u(i)+w(i);
    y(i) = C*x(:,i)+v(i);
    x(:,i) = x(:,i+1);
end
distillate = struct();
distillate.sys = sys;
distillate.x = x;
distillate.y = y;