function [KG,net] = KalmanNet(delta_x,target,A,C,N)
%%  Normalize state trajectories
% delta_x = [x_next_nw;y];
% delta_x = x_next_nw;
%%  Propagate 
x_hat_kf = A*delta_x(1:end-1,:);
y_hat_kf = C*x_hat_kf;
X = delta_x;
Y = target(1:end-1,:);
tic
numFeatures = size(X,1);
numResponses = size(Y,1);
% numHiddenUnits = 1;
numHiddenUnits = 3;
layers = [ ...
    sequenceInputLayer(numFeatures)
%     fullyConnectedLayer(numFeatures)
    lstmLayer(numHiddenUnits,'StateActivationFunction','tanh','GateActivationFunction','sigmoid')
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',100, 'MiniBatchSize',8,...
    'GradientThreshold',.1, ...
    'InitialLearnRate',0.015, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
%% Train LSTM Net
net = trainNetwork(X,Y,layers,options);
%% Predict 
net = predictAndUpdateState(net,X);
for i = 1:N
    nets(i) = predictAndUpdateState(net,X(:,i));
    KGG(:,i) = nets(i).Layers(2).HiddenState;
end
[net,KG] = predictAndUpdateState(net,target);
time_elapsed = toc;
% x_hat_next = (KG*Y')+x_hat_kf;
% y_hat = C*x_hat_next;
fprintf('MSE : %f\n',mse(KG,target(1,:)));
% end