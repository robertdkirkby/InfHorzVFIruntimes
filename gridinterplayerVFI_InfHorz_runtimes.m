% Do grid-interpolation layer for discretized VFI for simpe problem
% If n_z=2, this is the model of Rendahl (2022)
% Otherwise, is roughly the same value fn problem as Aiyagari (1994) [minor differences]
%
% Purpose of this code is to understand how VFI Toolkit should set defaults
% for Howards iteration vs Howards Greedy (a.k.a., modified-Policy fn
% iteration vs Policy fn iteration). And for multigrid.
% For Howards iteration, there are two versions coded, the first doing the
% expections based on indexing a matrix, the second uses a sparse matrix
% multiplication.
% I also did a 'hardcode' of the interals of VFI Toolkit commands, to see
% if overhead was an issue.
%
% Result of all this is:
% Howards-greedy is fastest for very small problems, VFI Toolkit defaults has been set to use greedy when N_a<400 || N_z<20
% For anything but very small problems, VFI Toolkit uses Howards iteration
% Howards iteration is faster with indexing that spare matrix, for expectations.
% Overhead costs of ValueFnIter command versus the hardcode is negligible.
% [Not covered here, but by default VFI Toolkit would 'refine' away a 'd' (decision) variable. 
% Then based on N_a and N_z will use Howards greedy or iteration on what is left]
%
% Note: I report runtimes, and comparison to those of Rendahl (2022) for
% the model with n_z=2, and n_a=500,1000,1500 (which are what is in the
% paper). Rendahl (2022) uses MCA, so there is no expectation that codes
% here get equally fast, is just used out of interest.
%
% For discussion of VFI, Howards-iteration, Howards-greedy, etc., see
% http://discourse.vfitoolkit.com/t/vfi-in-infhorz-howards-pfi-and-relation-to-implicit-finite-differences/408

%% Set some basic variables

% Size of the grids
n_d=0;
n_a=5;
n_z=2;

% Parameters
Params.gamma=3; % CRRA coeff in preferences
Params.beta=1.03^(-1/4); % discount rate, is over 0.99
Params.delta=0.025; % separation rate (paper says 0.1, but Rendahl's matlab says 0.025; guess the 0.1 was annual)
Params.phi=0.9; % job finding rate
Params.alpha=0.33; % capital share of output
Params.mu=0.4; % replacement rate
Params.r=0.0073; % interest rate [what Rendahl has in eqm]

%% Set up the exogenous shock process
z_gridvals=[0;1];
pi_z=[1-Params.phi, Params.phi; Params.delta, 1-Params.delta];
% exogenous labor model, so we know that L is
Params.L=Params.phi/(Params.phi+Params.delta); % Rendahl calls this 'n', eqn comes from top of pg 4
% and since tax is on earnings, we can balance the gov budget without solving model, just setting
Params.tau=Params.mu*(1-Params.L)/(Params.L+Params.mu*(1-Params.L)); % Rendahl has this eqn on pg 10, nearish bottom


%% Grids
d_grid=[]; %There is no d variable
% Set grid for asset holdings
Params.amax=400; % took this from Rendahl codes, he has evenly spaced points from 0 to amax, and amax=400
a_grid=Params.amax*linspace(0,1,n_a)'; % evenly spaced, not a good idea

%%
DiscountFactorParamNames={'beta'};

if n_z==2
    ReturnFn=@(aprime, a, z,r,alpha,delta,mu,tau,gamma) Rendahl2022_ReturnFn(aprime, a, z,r,alpha,delta,mu,tau,gamma);
else
    ReturnFn=@(aprime, a, z,r,alpha,delta,tau,gamma) Rendahl2022mod_ReturnFn(aprime, a, z,r,alpha,delta,tau,gamma);
end
% The first inputs must be: next period endogenous state, endogenous state, exogenous state. Followed by any parameters

vfoptions=struct();

Tolerance=10^(-9);
maxiter=Inf;
maxhowards=500; % just a safety valve on max number of times to do Howards, not sure it is needed for anything?

%% Loop to get average runtimes
% Loop over some N_a sizes
% Loop over some N_z sizes
% Loop over some H (number of howards iterations to perform
N_a_vec=[100,200,500];
N_z_vec=[2,5,10,20,25]; % at 2 greedy-Howards is better, at 25 Howards-iter is better, looking for cross-over (and if cross-over differs by n_a)
H_vec=[60,100,150,200];
multigridswitch_vec=[100,1000,10000,1/Tolerance]; % multi-grid, which to the interp when at this accuracy, note, last one is essentially turning multigrid off
% orignally I included multigridswitch=10 but became clear this was not worthwhile
ngridinterp=10;

setuptimes=zeros(length(N_a_vec),length(N_z_vec),length(H_vec));
runtimes=zeros(length(N_a_vec),length(N_z_vec),length(H_vec),3);
counter=zeros(length(N_a_vec),length(N_z_vec),length(H_vec),3);
counterA=counter; % multi-grid, rough grid
counterB=counter; % multi-grid, fine grid
checkzero=zeros(length(N_a_vec),length(N_z_vec),length(H_vec),2); % should be zeros as all three Howards give same solution: 1st is the two different iter, 2 is the first iter vs greedy

for a_c=1:length(N_a_vec)
    n_a=N_a_vec(a_c);
    N_a=prod(n_a);
    for z_c=1:length(N_z_vec)
        n_z=N_z_vec(z_c);
        N_z=prod(n_z);

        % a_c=length(N_a_vec)
        % z_c=length(N_z_vec)

        if n_z==2
            ReturnFn=@(aprime, a, z,r,alpha,delta,mu,tau,gamma) Rendahl2022_ReturnFn(aprime, a, z,r,alpha,delta,mu,tau,gamma);
        else
            ReturnFn=@(aprime, a, z,r,alpha,delta,tau,gamma) Rendahl2022mod_ReturnFn(aprime, a, z,r,alpha,delta,tau,gamma);
        end

        if n_z==2
            z_gridvals=[0;1];
            pi_z=[1-Params.phi, Params.phi; Params.delta, 1-Params.delta];
        else
            z_gridvals=linspace(0.5,1.5,n_z)';
            pi_z=rand(n_z,n_z);
            pi_z=pi_z+eye(n_z,n_z);
            pi_z=pi_z./sum(pi_z,2); % normalize rows to one
        end
        a_grid=Params.amax*linspace(0,1,n_a)'; % evenly spaced, not a good idea

        for h_c=1:length(H_vec)
            H=H_vec(h_c); % number of Howards iterations

            for m_c=1:length(multigridswitch_vec)
                multigridswitch=multigridswitch_vec(m_c);

                fprintf('Currently doing a_c=%i, z_c=%i, h_c=%i, m_c=%i \n',a_c,z_c,h_c,m_c)



                %% Some copy-paste of toolkit internals to get things setup
                tic;

                a_grid=gpuArray(a_grid);
                z_gridvals=gpuArray(z_gridvals);
                pi_z=gpuArray(pi_z);

                DiscountFactorParamsVec=Params.beta;
                ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Params);
                ReturnFnParamsVec=CreateVectorFromParams(Params, ReturnFnParamNames);

                % Grid interpolation
                % ngridinterp=9;
                n2short=ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)

                n_aprime=n_a+(n_a-1)*ngridinterp;
                N_aprime=prod(n_aprime);
                aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*ngridinterp))';
                ReturnMatrixfine=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,n_aprime, n_a, n_z, aprime_grid, a_grid, z_gridvals, ReturnFnParamsVec);
                ReturnMatrix=ReturnMatrixfine(1:ngridinterp+1:n_aprime,:,:);

                pi_z_alt=shiftdim(pi_z',-1);

                addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));
                addindexforazfine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1));

                V0=zeros(N_a,N_z,'gpuArray');

                setuptime=toc;

                %% First, Howards iteration, with H iterations, using index
                VKron=V0;

                tic;
                % Setup specific to Howard iterations
                % H=80; % number of Howards iterations
                pi_z_howards=repelem(pi_z,N_a,1);

                tempcounter1=0;
                currdist=1;
                % First, just consider a_grid for next period
                while currdist>(multigridswitch*Tolerance) && tempcounter1<=maxiter
                    VKronold=VKron;

                    % Calc the condl expectation term (except beta), which depends on z but not on control variables
                    EV=VKronold.*pi_z_alt;
                    EV(isnan(EV))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    EV=sum(EV,2); % sum over z', leaving a singular second dimension

                    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; %aprime by a by z

                    %Calc the max and it's index
                    [VKron,Policy]=max(entireRHS,[],1);
                    VKron=shiftdim(VKron,1); % a by z

                    VKrondist=VKron(:)-VKronold(:);
                    VKrondist(isnan(VKrondist))=0;
                    currdist=max(abs(VKrondist));

                    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
                    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter1<maxhowards
                        tempmaxindex=shiftdim(Policy,1)+addindexforaz; % aprime index, add the index for a and z
                        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a,N_z]); % keep return function of optimal policy for using in Howards
                        Policy=Policy(:); % a by z (this shape is just convenient for Howards)

                        for Howards_counter=1:H
                            EVKrontemp=VKron(Policy,:);
                            EVKrontemp=EVKrontemp.*pi_z_howards;
                            EVKrontemp(isnan(EVKrontemp))=0;
                            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
                            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
                        end
                    end

                    tempcounter1=tempcounter1+1;

                end
                
                tempcounter1a=tempcounter1;


                % Now switch to considering the fine/interpolated aprime_grid
                currdist=1; % force going into the next while loop at least one iteration
                while currdist>Tolerance && tempcounter1<=maxiter
                    VKronold=VKron;

                    % Calc the condl expectation term (except beta), which depends on z but not on control variables
                    EV=VKronold.*pi_z_alt;
                    EV(isnan(EV))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    EV=sum(EV,2); % sum over z', leaving a singular second dimension

                    % Interpolate EV over aprime_grid
                    EVinterp=interp1(a_grid,EV,aprime_grid);

                    entireRHS=ReturnMatrixfine+DiscountFactorParamsVec*EVinterp; % aprime by a by z

                    %Calc the max and it's index
                    [VKron,Policy]=max(entireRHS,[],1);

                    VKron=shiftdim(VKron,1); % a by z

                    VKrondist=VKron(:)-VKronold(:);
                    VKrondist(isnan(VKrondist))=0;
                    currdist=max(abs(VKrondist));

                    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
                    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter1<maxhowards
                        tempmaxindex=shiftdim(Policy,1)+addindexforazfine; % aprime index, add the index for a and z
                        Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a,N_z]); % keep return function of optimal policy for using in Howards
                        Policy=Policy(:); % a by z (this shape is just convenient for Howards)
                        for Howards_counter=1:H
                            EVKrontemp=interp1(a_grid,VKron,aprime_grid); % interpolate V as Policy points to the interpolated indexes
                            EVKrontemp=EVKrontemp(Policy,:);
                            EVKrontemp=EVKrontemp.*pi_z_howards;
                            EVKrontemp(isnan(EVKrontemp))=0;
                            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
                            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
                        end
                    end

                    tempcounter1=tempcounter1+1;

                end

                Policy=reshape(Policy,[N_a,N_z]);

                Rendahltest1=toc;

                clear pi_z_howards

                % [setuptime,Rendahltest1]
                % tempcounter1

                VKron_Hiter=VKron;

                %% Second, Howards iteration, with H iterations, using sparse matrix
                VKron=V0;

                tic;
                % Setup specific to Howard iterations
                % H=80; % number of Howards iterations
                pi_z_howards2=shiftdim(pi_z',-1);

                aind=gpuArray((1:1:N_a)');
                N_a_times_zind=N_a*gpuArray(0:1:N_z-1); % already contains -1
                azind=gpuArray(aind+N_a_times_zind);

                tempcounter2=0;
                currdist=1;
                % First, just consider a_grid for next period
                while currdist>(multigridswitch*Tolerance) && tempcounter2<=maxiter
                    VKronold=VKron;

                    % Calc the condl expectation term (except beta), which depends on z but not on control variables
                    EV=VKronold.*pi_z_alt;
                    EV(isnan(EV))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    EV=sum(EV,2); % sum over z', leaving a singular second dimension

                    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % aprime by a by z

                    %Calc the max and it's index
                    [VKron,Policy]=max(entireRHS,[],1);
                    VKron=shiftdim(VKron,1); % a by z

                    VKrondist=VKron(:)-VKronold(:);
                    VKrondist(isnan(VKrondist))=0;
                    currdist=max(abs(VKrondist));

                    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
                    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter2<maxhowards
                        % Get the return matrix F() for the current policy
                        tempmaxindex=shiftdim(Policy,1)+addindexforaz; % aprime index, add the index for a and z
                        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a*N_z,1]); % keep return function of optimal policy for using in Howards
                        indp = shiftdim(Policy,1)+N_a_times_zind;
                        Q = sparse(azind(:),indp(:),1,N_a*N_z,N_a*N_z); % policy as mapping from (a,z) to (a',z)
                        for Howards_counter=1:H
                            EVKrontemp=VKron.*pi_z_howards2; % switch from V on (a',z') to EV on (a',z)
                            EVKrontemp(isnan(EVKrontemp))=0;
                            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a*N_z,1]);
                            VKron=Ftemp+DiscountFactorParamsVec*Q*EVKrontemp; % Q*EV, moves EV from (a',z) to (a,z)
                            VKron=reshape(VKron,[N_a,N_z]); % a by z
                        end
                    end

                    tempcounter2=tempcounter2+1;

                end

                tempcounter2a=tempcounter2;

                % Now switch to considering the fine/interpolated aprime_grid
                currdist=1; % force going into the next while loop at least one iteration
                while currdist>Tolerance && tempcounter2<=maxiter
                    VKronold=VKron;

                    % Calc the condl expectation term (except beta), which depends on z but not on control variables
                    EV=VKronold.*pi_z_alt;
                    EV(isnan(EV))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    EV=sum(EV,2); % sum over z', leaving a singular second dimension

                    % Interpolate EV over aprime_grid
                    EVinterp=interp1(a_grid,EV,aprime_grid);

                    entireRHS=ReturnMatrixfine+DiscountFactorParamsVec*EVinterp; % aprime by a by z

                    %Calc the max and it's index
                    [VKron,Policy]=max(entireRHS,[],1);
                    VKron=shiftdim(VKron,1); % a by z

                    VKrondist=VKron(:)-VKronold(:);
                    VKrondist(isnan(VKrondist))=0;
                    currdist=max(abs(VKrondist));

                    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
                    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter2<maxhowards
                        % Get the return matrix F() for the current policy
                        tempmaxindex=shiftdim(Policy,1)+addindexforazfine; % aprime index, add the index for a and z
                        Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a*N_z,1]); % keep return function of optimal policy for using in Howards

                        Policy_lowerind=max(ceil((Policy-1)/(n2short+1))-1,0)+1;  % lower grid point index
                        Policy_lowerprob=1- ((Policy-(Policy_lowerind-1)*(n2short+1))-1)/(n2short+1);
                        indp = shiftdim(Policy_lowerind,1)+N_a_times_zind;
                        Q = sparse([azind(:);azind(:)],[indp(:);indp(:)+1],[Policy_lowerprob(:),1-Policy_lowerprob(:)],N_a*N_z,N_a*N_z); % policy as mapping from (a,z) to (a',z)
                        for Howards_counter=1:H
                            EVKrontemp=VKron.*pi_z_howards2; % switch from V on (a',z') to EV on (a',z)
                            EVKrontemp(isnan(EVKrontemp))=0;
                            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a*N_z,1]);
                            VKron=Ftemp+DiscountFactorParamsVec*Q*EVKrontemp; % Q*EV, moves EV from (a',z) to (a,z)
                            VKron=reshape(VKron,[N_a,N_z]); % a by z
                        end
                    end

                    tempcounter2=tempcounter2+1;

                end


                Policy=reshape(Policy,[N_a,N_z]);

                Rendahltest2=toc;

                clear pi_z_howards2

                % [setuptime,Rendahltest2]
                % tempcounter2

                VKron_Hiter2=VKron;

                %% Third, greedy Howards, so as a linear system of equations
                VKron=V0;

                tic;
                % Setup specific to greedy Howards
                spI = gpuArray.speye(N_a*N_z);
                T_pi_z=sparse(gpuArray(repelem(pi_z,N_a,N_a))); % row is this period, column is next period: (a,z) to (a',z')
                N_a_times_zind=N_a*gpuArray(0:1:N_z-1); % already contains -1
                azind1=repmat(gpuArray(1:1:N_a*N_z)',1,N_z); % (a-z,zprime)
                azind2=repmat(gpuArray(1:1:N_a*N_z)',2,N_z); % (a-z-2,zprime)
                pi_z_big1=gpuArray(repelem(pi_z,N_a,1)); % (a-z,zprime)
                pi_z_big2=gpuArray(repmat(pi_z_big1,2,1)); % (a-z-2,zprime)

                tempcounter3=0;
                currdist=1;
                % First, just consider a_grid for next period
                while currdist>(multigridswitch*Tolerance) && tempcounter3<=maxiter

                    VKronold=VKron;

                    % Calc the condl expectation term (except beta), which depends on z but not on control variables
                    EV=VKronold.*pi_z_alt;
                    EV(isnan(EV))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    EV=sum(EV,2); % sum over z', leaving a singular second dimension

                    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; %aprime by a by z

                    %Calc the max and it's index
                    [VKron,Policy]=max(entireRHS,[],1);
                    VKron=shiftdim(VKron,1); % a by z

                    VKrondist=VKron(:)-VKronold(:);
                    VKrondist(isnan(VKrondist))=0;
                    currdist=max(abs(VKrondist));

                    % Use greedy-Howards Improvement (except for first few and last few iterations, as it is not a good idea there)
                    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter3<maxhowards
                        tempmaxindex=shiftdim(Policy,1)+addindexforaz; % aprime index, add the index for a and z
                        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a*N_z,1]); % keep return function of optimal policy for using in Howards

                        T_E=sparse(azind1,Policy(:)+N_a_times_zind,pi_z_big1,N_a*N_z,N_a*N_z);

                        VKron=(spI-DiscountFactorParamsVec*T_E)\Ftemp;
                        VKron=reshape(VKron,[N_a,N_z]);
                    end

                    tempcounter3=tempcounter3+1;
                end

                tempcounter3a=tempcounter3;

                % Now switch to considering the fine/interpolated aprime_grid
                currdist=1; % force going into the next while loop at least one iteration
                while currdist>Tolerance && tempcounter2<=maxiter
                    VKronold=VKron;

                    % Calc the condl expectation term (except beta), which depends on z but not on control variables
                    EV=VKronold.*pi_z_alt;
                    EV(isnan(EV))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    EV=sum(EV,2); % sum over z', leaving a singular second dimension

                    % Interpolate EV over aprime_grid
                    EVinterp=interp1(a_grid,EV,aprime_grid);

                    entireRHS=ReturnMatrixfine+DiscountFactorParamsVec*EVinterp; %aprime by a by z

                    %Calc the max and it's index
                    [VKron,Policy]=max(entireRHS,[],1);
                    VKron=shiftdim(VKron,1); % a by z

                    VKrondist=VKron(:)-VKronold(:);
                    VKrondist(isnan(VKrondist))=0;
                    currdist=max(abs(VKrondist));

                    % Use greedy-Howards Improvement (except for first few and last few iterations, as it is not a good idea there)
                    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter3<maxhowards
                        tempmaxindex=shiftdim(Policy,1)+addindexforazfine; % aprime index, add the index for a and z
                        Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a*N_z,1]); % keep return function of optimal policy for using in Howards

                        Policy_lowerind=max(ceil((Policy(:)-1)/(n2short+1))-1,0)+1;  % lower grid point index
                        Policy_lowerprob=1- ((Policy(:)-(Policy_lowerind-1)*(n2short+1))-1)/(n2short+1); % Policy-(Policy_lowerind-1)*(n2short+1) is 2nd layer index
                        indp = Policy_lowerind+N_a_times_zind; % with all tomorrows z (a-z,zprime)

                        T_E=sparse(azind2,[indp;indp+1],[Policy_lowerprob;1-Policy_lowerprob].*pi_z_big2,N_a*N_z,N_a*N_z);

                        VKron=(spI-DiscountFactorParamsVec*T_E)\Ftemp;
                        VKron=reshape(VKron,[N_a,N_z]);
                    end

                    tempcounter3=tempcounter3+1;

                end

                Policy=reshape(Policy,[N_a,N_z]);

                Rendahltest3=toc;


                % [setuptime,Rendahltest3]
                % tempcounter3

                VKron_Hgreedy=VKron;

                %% Check same solution
                checkzero(a_c,z_c,h_c,m_c,1)=max(abs(VKron_Hiter(:)-VKron_Hiter2(:)));
                checkzero(a_c,z_c,h_c,m_c,2)=max(abs(VKron_Hiter(:)-VKron_Hgreedy(:)));

                % Store runtimes
                setuptimes(a_c,z_c,h_c,m_c)=setuptime;
                runtimes(a_c,z_c,h_c,m_c,1)=Rendahltest1;
                runtimes(a_c,z_c,h_c,m_c,2)=Rendahltest2;
                runtimes(a_c,z_c,h_c,m_c,3)=Rendahltest3;

                % Store counters
                counter(a_c,z_c,h_c,m_c,1)=tempcounter1;
                counter(a_c,z_c,h_c,m_c,2)=tempcounter2;
                counter(a_c,z_c,h_c,m_c,3)=tempcounter3;

                counterA(a_c,z_c,h_c,m_c,1)=tempcounter1a;
                counterA(a_c,z_c,h_c,m_c,2)=tempcounter2a;
                counterA(a_c,z_c,h_c,m_c,3)=tempcounter3a;
                counterB(a_c,z_c,h_c,m_c,1)=tempcounter1-tempcounter1a;
                counterB(a_c,z_c,h_c,m_c,2)=tempcounter2-tempcounter2a;
                counterB(a_c,z_c,h_c,m_c,3)=tempcounter3-tempcounter3a;

            end
        end
    end
end

%% Look at results

% First, just check we get same solutions
max(max(max(abs(checkzero(:,:,:,:,1))))) % zero. Good
max(max(max(abs(checkzero(:,:,:,:,2))))) % close to zero, good enough

% Average runtimes across everything
max(max(max(max(abs(setuptimes))))) % this is a tiny fraction of the time
max(max(max(max(abs(runtimes(:,:,:,:,1))))))
max(max(max(max(abs(runtimes(:,:,:,:,2))))))
max(max(max(max(abs(runtimes(:,:,:,:,3)))))) % Greedy is faster most of the time

% Is one of the two iteration implementations always better than the other?
min(min(min(min(abs(runtimes(:,:,:,:,1)./runtimes(:,:,:,:,2)))))) % 0.83
max(max(max(max(abs(runtimes(:,:,:,:,1)./runtimes(:,:,:,:,2)))))) % 0.97
% So runtime1/runtime2 ranges from 0.83 to 0.97, which is always less than
% one. So first implementation (using indexes) is faster than the second
% (using sparse matrix and multiplication).

% Which multigrid is fastest?
[min(min(min(min(abs(runtimes(:,:,:,1,:)))))),mean(mean(mean(mean(abs(runtimes(:,:,:,1,:)))))),max(max(max(max(abs(runtimes(:,:,:,1,:))))))]
[min(min(min(min(abs(runtimes(:,:,:,2,:)))))),mean(mean(mean(mean(abs(runtimes(:,:,:,2,:)))))),max(max(max(max(abs(runtimes(:,:,:,2,:))))))]
[min(min(min(min(abs(runtimes(:,:,:,3,:)))))),mean(mean(mean(mean(abs(runtimes(:,:,:,3,:)))))),max(max(max(max(abs(runtimes(:,:,:,3,:))))))] % fastest, but not by much
[min(min(min(min(abs(runtimes(:,:,:,4,:)))))),mean(mean(mean(mean(abs(runtimes(:,:,:,4,:)))))),max(max(max(max(abs(runtimes(:,:,:,4,:))))))]
% 10000 is fastest, but not by much
% Note: 4 is effectively turning off multi-grid


% What about the best H? (for multigrid 3)
[~,Hoptindex]=min(runtimes(:,:,:,3,1),[],3); % third dimension indexes H
[min(Hoptindex(:)),max(Hoptindex(:))]
% always 4 to 7,
% which corresponds to H=80 to 150
Hoptindex
% no obvious rule.
% What is average using H=150 vs 80 or 100
temp4=runtimes(:,:,1,3,1);
temp5=runtimes(:,:,2,3,1);
temp6=runtimes(:,:,3,3,1);
temp7=runtimes(:,:,4,3,1);
[mean(temp4(:)),mean(temp5(:)),mean(temp6(:)),mean(temp7(:))] % minor differences at most
[min(temp4(:)./temp6(:)), max(temp4(:)./temp6(:))] % 4 is 7% better to 11% worse (than 6, H=150)
[min(temp5(:)./temp6(:)), max(temp5(:)./temp6(:))] % 5 is 7% better to 14% worse (than 6, H=150)
[min(temp6(:)./temp6(:)), max(temp6(:)./temp6(:))] % 
[min(temp7(:)./temp6(:)), max(temp7(:)./temp6(:))] % 7 is 8% better to 23% worse (than 6, H=150)
% So nothing ever beat H=150 by more than ten percent
% And H=150 on average

% Which multigrid is fastest? Given I said H=150
[min(min(min(min(abs(runtimes(:,:,:,1,:)))))),mean(mean(mean(mean(abs(runtimes(:,:,:,1,:)))))),max(max(max(max(abs(runtimes(:,:,:,1,:))))))]
[min(min(min(min(abs(runtimes(:,:,:,2,:)))))),mean(mean(mean(mean(abs(runtimes(:,:,:,2,:)))))),max(max(max(max(abs(runtimes(:,:,:,2,:))))))]
[min(min(min(min(abs(runtimes(:,:,:,3,:)))))),mean(mean(mean(mean(abs(runtimes(:,:,:,3,:)))))),max(max(max(max(abs(runtimes(:,:,:,3,:))))))] % fastest, but not by much
% Still the multigridswitch=1000


%% When is iter with index better than iter with sparse
index=((runtimes(:,:,:,:,1)./runtimes(:,:,:,:,2))<1)
% Mostly but not always (when it was not, was mostly the not using multi-grid)

%% When is Greedy best?
greedy=((runtimes(:,:,:,:,3)./runtimes(:,:,:,:,1))<1)
% Greedy is better for essentially all of them

%% How many iterations are they taking?
% First compare on the rough grid 
squeeze(counterA(:,:,2,3,:))
% Then the fine grid
squeeze(counterB(:,:,2,3,:)) % This is why the iterations take much longer, because they need heaps of VFI steps once they switch to fine grid


% First compare on the rough grid 
squeeze(counterA(:,:,2,4,:)) % without multigrid this is 1 as it should be
% Then the fine grid
squeeze(counterB(:,:,2,4,:)) % This is why the iterations take much longer, because they need heaps of VFI steps once they switch to fine grid


%% Decision: default to always use Howards greedy
% And always use Howards iteration with indexing, not with sparse matrix


%% For everything from here on
multigridswitch=1000;
H=150;


%% Compare to get Rendahl (2022) times
Rendahl_N_a_vec=[500,1000,1500];
% Roughly, takes me 0.14, 0.22, 0.33s, respectively
% Rendahl reports 0.02, 0.04, 0.04 for discrete time
% Rendahl reports 0.02, 0.07, 0.13 for continuous time
% Disabling the 'safety' checks has essentially zero impact on run times
% Of course Rendahl is just doing 'local policy search/MCA', so that should be much faster as checking way fewer points.

n_z=2;
N_z=prod(n_z);
for a_c=1:length(Rendahl_N_a_vec)
    n_a=Rendahl_N_a_vec(a_c);
    N_a=prod(n_a);

    ReturnFn=@(aprime, a, z,r,alpha,delta,mu,tau,gamma) Rendahl2022_ReturnFn(aprime, a, z,r,alpha,delta,mu,tau,gamma);

    z_gridvals=[0;1];
    pi_z=[1-Params.phi, Params.phi; Params.delta, 1-Params.delta];
    a_grid=Params.amax*linspace(0,1,n_a)'; % evenly spaced, not a good idea


    %% First, just some copy-paste of toolkit internals to get things setup
    tic;

    a_grid=gpuArray(a_grid);
    z_gridvals=gpuArray(z_gridvals);
    pi_z=gpuArray(pi_z);

    DiscountFactorParamsVec=Params.beta;
    ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Params);
    ReturnFnParamsVec=CreateVectorFromParams(Params, ReturnFnParamNames);

    % Grid interpolation
    % ngridinterp=9;
    n2short=ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)

    n_aprime=n_a+(n_a-1)*ngridinterp;
    N_aprime=prod(n_aprime);
    aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*ngridinterp))';
    ReturnMatrixfine=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,n_aprime, n_a, n_z, aprime_grid, a_grid, z_gridvals, ReturnFnParamsVec);
    ReturnMatrix=ReturnMatrixfine(1:ngridinterp+1:n_aprime,:,:);

    pi_z_alt=shiftdim(pi_z',-1);

    addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));
    addindexforazfine=gpuArray(N_aprime*(0:1:N_a-1)'+N_aprime*N_a*(0:1:N_z-1));

    V0=zeros(N_a,N_z,'gpuArray');

    setuptime=toc;

    %% Third, greedy Howards, so as a linear system of equations
    VKron=V0;

    tic;
    % Setup specific to greedy Howards
    spI = gpuArray.speye(N_a*N_z);
    T_pi_z=sparse(gpuArray(repelem(pi_z,N_a,N_a))); % row is this period, column is next period: (a,z) to (a',z')
    N_a_times_zind=N_a*gpuArray(0:1:N_z-1); % already contains -1
    azind1=repmat(gpuArray(1:1:N_a*N_z)',1,N_z); % (a-z,zprime)
    azind2=repmat(gpuArray(1:1:N_a*N_z)',2,N_z); % (a-z-2,zprime)
    pi_z_big1=gpuArray(repelem(pi_z,N_a,1)); % (a-z,zprime)
    pi_z_big2=gpuArray(repmat(pi_z_big1,2,1)); % (a-z-2,zprime)

    tempcounter3=1;
    currdist=Inf;
    % First, just consider a_grid for next period
    while currdist>(multigridswitch*Tolerance) && tempcounter3<=maxiter

        VKronold=VKron;

        % Calc the condl expectation term (except beta), which depends on z but not on control variables
        EV=VKronold.*pi_z_alt;
        EV(isnan(EV))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; %aprime by a by z

        %Calc the max and it's index
        [VKron,Policy]=max(entireRHS,[],1);
        VKron=shiftdim(VKron,1); % a by z

        VKrondist=VKron(:)-VKronold(:);
        VKrondist(isnan(VKrondist))=0;
        currdist=max(abs(VKrondist));

        % Use greedy-Howards Improvement (except for first few and last few iterations, as it is not a good idea there)
        if isfinite(currdist) && currdist/Tolerance>10 && tempcounter3<maxhowards
            tempmaxindex=shiftdim(Policy,1)+addindexforaz; % aprime index, add the index for a and z
            Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a*N_z,1]); % keep return function of optimal policy for using in Howards

            T_E=sparse(azind1,Policy(:)+N_a_times_zind,pi_z_big1,N_a*N_z,N_a*N_z);

            VKron=(spI-DiscountFactorParamsVec*T_E)\Ftemp;
            VKron=reshape(VKron,[N_a,N_z]);
        end

        tempcounter3=tempcounter3+1;
    end

    tempcounter3a=tempcounter3;

    % Now switch to considering the fine/interpolated aprime_grid
    currdist=1; % force going into the next while loop at least one iteration
    while currdist>Tolerance && tempcounter3<=maxiter
        VKronold=VKron;

        % Calc the condl expectation term (except beta), which depends on z but not on control variables
        EV=VKronold.*pi_z_alt;
        EV(isnan(EV))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        % Interpolate EV over aprime_grid
        EVinterp=interp1(a_grid,EV,aprime_grid);

        entireRHS=ReturnMatrixfine+DiscountFactorParamsVec*EVinterp; %aprime by a by z

        %Calc the max and it's index
        [VKron,Policy]=max(entireRHS,[],1);
        VKron=shiftdim(VKron,1); % a by z

        VKrondist=VKron(:)-VKronold(:);
        VKrondist(isnan(VKrondist))=0;
        currdist=max(abs(VKrondist));

        % Use greedy-Howards Improvement (except for first few and last few iterations, as it is not a good idea there)
        if isfinite(currdist) && currdist/Tolerance>10 && tempcounter3<maxhowards
            tempmaxindex=shiftdim(Policy,1)+addindexforazfine; % aprime index, add the index for a and z
            Ftemp=reshape(ReturnMatrixfine(tempmaxindex),[N_a*N_z,1]); % keep return function of optimal policy for using in Howards

            Policy_lowerind=max(ceil((Policy(:)-1)/(n2short+1))-1,0)+1;  % lower grid point index
            Policy_lowerprob=1- ((Policy(:)-(Policy_lowerind-1)*(n2short+1))-1)/(n2short+1); % Policy-(Policy_lowerind-1)*(n2short+1) is 2nd layer index
            indp = Policy_lowerind+N_a_times_zind; % with all tomorrows z (a-z,zprime)

            T_E=sparse(azind2,[indp;indp+1],[Policy_lowerprob;1-Policy_lowerprob].*pi_z_big2,N_a*N_z,N_a*N_z);

            VKron=(spI-DiscountFactorParamsVec*T_E)\Ftemp;
            VKron=reshape(VKron,[N_a,N_z]);
        end

        tempcounter3=tempcounter3+1;

    end

    Policy=reshape(Policy,[N_a,N_z]);

    Rendahltest3=toc;

    [n_a,tempcounter3]
    [setuptime+Rendahltest3,setuptime,Rendahltest3]


end
% takes 0.25, 0.38 and 0.64s


%% Lastly, redo Rendahl (2022), but now with full VFI Toolkit overhead, see how that changes the times
n_z=2;
vfoptions.gridinterplayer=1;
vfoptions.ngridinterp=ngridinterp;
for a_c=1:length(Rendahl_N_a_vec)
    n_a=Rendahl_N_a_vec(a_c);

    ReturnFn=@(aprime, a, z,r,alpha,delta,mu,tau,gamma) Rendahl2022_ReturnFn(aprime, a, z,r,alpha,delta,mu,tau,gamma);

    z_gridvals=[0;1];
    pi_z=[1-Params.phi, Params.phi; Params.delta, 1-Params.delta];
    a_grid=Params.amax*linspace(0,1,n_a)'; % evenly spaced, not a good idea

    vfoptions.howardsgreedy=1; % Note: I tried =0 (howards iteration) and =2 (iter for a_grid, then greedy for aprime_grid). But both are slower so default is going to be set to always use greedy.
    tic;
    [V,Policy]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_gridvals,pi_z,ReturnFn,Params,DiscountFactorParamNames,[],vfoptions);
    Rendahltest4=toc
end
% with greedy, takes 0.23, 0.38, 0.65s
% So overhead is trivial








