function F=Rendahl2022_ReturnFn(aprime, a, z,r,alpha,delta,mu,tau,gamma)

F=-Inf;
w=(1-alpha)*((r+delta)/alpha)^(alpha/(alpha-1));
c=(1-tau)*w*(mu*(1-z)+z)+(1+r)*a-aprime; % Budget Constraint

if c>0
    F=(c^(1-gamma) -1)/(1-gamma);
end

end