function y = DsigmoidFun(x)
y = 1./(1+exp(-x))-(1./((1+exp(-x)).^2));
end