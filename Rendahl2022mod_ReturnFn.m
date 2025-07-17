function F=Rendahl2022mod_ReturnFn(aprime, a, z,r,alpha,delta,tau,gamma)

F=-Inf;
w=(1-alpha)*((r+delta)/alpha)^(alpha/(alpha-1));
c=(1-tau)*w*z+(1+r)*a-aprime; % Budget Constraint

if c>0
    F=(c^(1-gamma) -1)/(1-gamma);
end

end