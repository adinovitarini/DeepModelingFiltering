function [net,ypred]=DeepModel(TRAIN,TEST,DelayElement,nh)
%% Time Delay Unit(TDL) 
na = DelayElement.Input;
nb = DelayElement.Output;
input_train = TRAIN.input;
target_train = TRAIN.target;
input_test = TEST.input;
target_test = TEST.target;
n = size(target_test,2);
varphi_train = TimeDelayBlock(n,na,nb,input_train,target_train);
varphi_test = TimeDelayBlock(n,na,nb,input_test,target_test);
%%  Traing-Phase
inp_train = arrayDatastore(varphi_train);
target_train = arrayDatastore(target_train);
train = combine(inp_train,target_train);
inp_test = arrayDatastore(varphi_test);
target_test = arrayDatastore(target_test);
test = combine(inp_test,target_test);
dsTrain = train;
 opts = trainingOptions("adam",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.001,...
    "MaxEpochs",1000,...
    "MiniBatchSize",8,...
    "Shuffle","every-epoch",...
    "Plots","none",...
    "ValidationData",test,...
    "Verbose",0);
layers = [
    sequenceInputLayer(1,"Name","sequence")
    %scalingLayer("Name","scaling")
    lstmLayer(nh,"Name","lstm","OutputMode","last")
    %dropoutLayer(0.5,"Name","dropout")
    fullyConnectedLayer(n,"Name","fc")
    regressionLayer("Name","regressionoutput")];
[net, ~] = trainNetwork(dsTrain,layers,opts);
[net,ypred] = predictAndUpdateState(net,inp_test);
