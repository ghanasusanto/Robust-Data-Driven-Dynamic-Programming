clear all

d = 5;
xi0 = 1*ones(d,1);

%% joint moments
MU_ = 1*ones(d*2,1)';
corr_mat = eye(d*2,d*2);
corr_mat(1:d,d+1:2*d) = 0.3*eye(d,d);
corr_mat(d+1:2*d,1:d) = corr_mat(1:d,d+1:2*d);
SIGMA_ = corr2cov(1*ones(2*d,1)',corr_mat);

%% conditional moments
MU_1 = MU_(1:d)';
MU_2 = MU_(d+1:2*d)';
SIGMA_11 = SIGMA_(1:d,1:d);
SIGMA_12 = SIGMA_(1:d,d+1:2*d);
SIGMA_21 = SIGMA_(d+1:2*d,1:d);
SIGMA_22 = SIGMA_(d+1:2*d,d+1:2*d);
COND_MU = MU_2 + SIGMA_21 * inv(SIGMA_11) * (xi0 - MU_1);
COND_SIGMA = SIGMA_22 - SIGMA_21 * inv(SIGMA_11) * SIGMA_12;
COND_OMEGA = COND_SIGMA + COND_MU*COND_MU';

N = [1:20];
mult = 25;
nN = length(N);
nIter = 200;
exp_true = zeros(nN,nIter);
exp_robust = zeros(nN,nIter);
exp_model = zeros(nN,nIter);
exp = zeros(nN,nIter);
u_robust = cell(nN,nIter);
u = cell(nN,nIter);
rho = 0.1;

n = 100000;
samples = mvnrnd(COND_MU,COND_SIGMA,n);

%% true cost-to-go
[obj_true,u_true] = solve_direct(n,d,ones(n,1)/n,samples,0,rho);
u_true = ones(d,1)/d;
obj_true = -u_true'*COND_MU + rho*u_true'*(COND_SIGMA + COND_MU*COND_MU')*u_true;
gamma = 10;

parfor idx=1:nN
    n = N(idx) * mult;
    fprintf ('number of samples %d \n', n);
    for iter=1:nIter
        samples = mvnrnd(MU_,SIGMA_,n);
        H = get_kernel_bandwidth(samples(:,1:d),d);
        weight = zeros(n,1);
        for i=1:n
            weight(i) = get_kernel_likelihood(xi0,samples(i,1:d)',H);
        end
        weight = weight / sum(weight);
        mu = (weight'*samples(:,d+1:d*2))';
        Omega = samples(:,d+1:d*2)'*diag(weight)*samples(:,d+1:d*2);
        
        %% estimate with DDP decision
        [~,u_] = solve_direct_moments(d,mu,Omega,rho);
        u{idx,iter} = u_;
        exp(idx,iter) = -u_'*mu + rho*u_'*Omega*u_;

        MU_new = mean(samples);
        SIGMA_new = cov(samples);
        MU_1 = MU_new(1:d)';
        MU_2 = MU_new(d+1:2*d)';
        SIGMA_11 = SIGMA_new(1:d,1:d);
        SIGMA_12 = SIGMA_new(1:d,d+1:2*d);
        SIGMA_21 = SIGMA_new(d+1:2*d,1:d);
        SIGMA_22 = SIGMA_new(d+1:2*d,d+1:2*d);
        COND_MU_new = MU_2 + SIGMA_21 * inv(SIGMA_11) * (xi0 - MU_1);
        COND_SIGMA_new = SIGMA_22 - SIGMA_21 * inv(SIGMA_11) * SIGMA_12;
        COND_OMEGA_new = COND_SIGMA_new + COND_MU_new*COND_MU_new';

        %% estimate with model-based decision
        [~,u_model_] = solve_direct_moments(d,COND_MU_new,COND_OMEGA_new,rho);
        exp_model(idx,iter) = -u_model_'*mu + rho*u_model_'*Omega*u_model_;
        
        %% estimate with RDDP decision
        [~,u_robust_] = solve_robust_direct(n,d,weight,samples(:,d+1:d*2),gamma,rho);
        u_robust{idx,iter} = u_robust_;
        exp_robust(idx,iter) = -u_robust_'*mu + rho*u_robust_'*Omega*u_robust_;
        
        %% estimate with true optimal decision 
        exp_true(idx,iter) = -u_true'*mu + rho*u_true'*Omega*u_true;
    end
end

figure(1)
hold on;
h0 = plot(N*mult,repmat(obj_true,nN,1),'k','LineWidth',3);
hhh0 = plot(N*mult,prctile(exp_true',10),'r--','LineWidth',3);
plot(N*mult,prctile(exp_true',90),'g--','LineWidth',3);
hhh1 = plot(N*mult,median(exp_true'),'b--','LineWidth',3);
legend([h0 hhh0 hhh1],'True value', 'Fixed 10th & 90th percentiles', 'Fixed median') 

figure(2)
hold on
h0 = plot(N*mult,repmat(obj_true,nN,1),'k','LineWidth',3);
hh0 = plot(N*mult,prctile(exp',10),'r','LineWidth',3);
plot(N*mult,prctile(exp',90),'g','LineWidth',3);
hh1 = plot(N*mult,median(exp'),'b','LineWidth',3);
legend([h0 hh0 hh1],'True value', 'Nadaraya-Watson 10th & 90th percentiles', 'Nadaraya-Watson median') 

figure(3)
hold on
h0 = plot(N*mult,repmat(obj_true,nN,1),'k','LineWidth',3);
hhhh0 = plot(N*mult,prctile(exp_robust',10),'r:','LineWidth',3);
plot(N*mult,prctile(exp_robust',90),'g:','LineWidth',3);
hhhh1 = plot(N*mult,median(exp_robust'),'b:','LineWidth',3);
legend([h0 hhhh0 hhhh1],'True value', 'Robust 10th & 90th percentiles', 'Robust median') 

figure(4)
hold on
h0 = plot(N*mult,repmat(obj_true,nN,1),'k','LineWidth',3);
hh0 = plot(N*mult,prctile(exp_model',10),'r','LineWidth',3);
plot(N*mult,prctile(exp_model',90),'g','LineWidth',3);
hh1 = plot(N*mult,median(exp_model'),'b','LineWidth',3);
legend([h0 hh0 hh1],'True value', 'Model-based 10th & 90th percentiles', 'Model-based median') 

return


