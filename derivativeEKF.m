%% Calculate \hat{F} and \hat{H}
function [F_hat,H_hat] = derivativeEKF(net,N,TEST)
nh = net.Layers(2).NumHiddenUnits;
Wi = net.Layers(2).InputWeights(1:nh,1);
Wf = net.Layers(2).InputWeights(nh+1:2*nh,1);
Wc = net.Layers(2).InputWeights(2*nh+1:3*nh,1);
Wo = net.Layers(2).InputWeights(3*nh+1:4*nh,1);
Ri = net.Layers(2).RecurrentWeights(1:nh,:);
Rf = net.Layers(2).RecurrentWeights(nh+1:2*nh,:);
Rc = net.Layers(2).RecurrentWeights(2*nh+1:3*nh,:);
Ro = net.Layers(2).RecurrentWeights(3*nh+1:4*nh,:);
% N = 100;
hh = zeros(nh,N); %state 
% y_hat = zeros(1,N); %out 
y = TEST.input;
% y = dataset_cp_test.y;
inputGate = zeros(nh,1);
forgetGate = inputGate;
outGate = inputGate;
cellGate = inputGate;
cellState = net.Layers(2).CellState(:,1);
hh = cellState(:,1);
for i = 1:N
    inputGate(:,i) = 1./(1+exp((Wi*y(i)+Ri*hh(:,i))));
    forgetGate(:,i) = 1./(1+exp((Wf*y(i)+Rf*hh(:,i))));
    cellGate(:,i) = tanh(Wc*y(i)+Rc*hh(:,i));
    outGate(:,i) = 1./(1+exp((Wo*y(i)+Ro*hh(:,i))));
    cellGate(:,i+1) = inputGate(:,i).*forgetGate(:,i).*cellGate(:,i).*outGate(:,i);
    hh(:,i+1) = outGate(:,i).*tanh(cellGate(:,i));
end
for i = 1:N
F_hat = diag(sigmoidFun(forgetGate(:,i)));
a_hat = diag(sigmoidFun(outGate));
temp = diag(sigmoidFun(cellGate));
b_hat = diag(DsigmoidFun(outGate.*temp));
c_hat = diag(outGate).*diag(DsigmoidFun(outGate));
H_hat = a_hat.*b_hat.*c_hat;
end