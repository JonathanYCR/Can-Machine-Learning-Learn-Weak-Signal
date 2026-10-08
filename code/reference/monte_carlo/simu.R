library(pracma)
library(foreach)
library(xtable)
library(doParallel)
library(expm)

###exam2  

main<-function(R2,sparsity,n,p){
tau_n=R2/(1-R2)
sigmaep=1
sigmabeta=1
rho_1=1/2
theta_1=1
theta_2=13/12

set.seed(1000)


Sigma_1=matrix(0,n,n)
for(i in 1:n) 
  for(j in 1:n) 
    Sigma_1[i,j]=rho_1^(abs(i-j))
sqrtSigma_1=sqrtm(Sigma_1)
U2=randortho(p, type = c("orthonormal", "unitary"))
eigen_Sigma_ep=0.5+runif(n)
sqrtSigma_ep=diag(sqrt(eigen_Sigma_ep))
eigen_Sigma_2=0.5+runif(p)
Sigma_2=U2%*%diag(eigen_Sigma_2)%*%t(U2)
sqrtSigma_2=U2%*%diag(sqrt(eigen_Sigma_2))%*%t(U2)

Nslots <- as.numeric(Sys.getenv("SLURM_JOB_CPUS_PER_NODE"))
cl<-makeCluster(Nslots)
registerDoParallel(cl)

final=foreach(ii=1:1000,.combine="rbind",.packages = "glmnet")%dopar%
  {
    set.seed(1000+ii)
    zero_loc=(runif(p)>1-sparsity)+0
    beta=matrix(sqrt(1/sparsity)*sigmabeta*rnorm(p)/sqrt( p*tau_n^(-1) )*zero_loc,nrow=p,ncol=1)
    ep=sqrtSigma_ep%*%matrix(rnorm(n,sd=sigmaep),nrow=n,ncol=1)
    X=sqrtSigma_1%*%matrix(rnorm(p*n),nrow=n,ncol=p)%*%sqrtSigma_2
    y=X%*%beta+ep
    n_oos=10000
    ep_oos=matrix(rnorm(n_oos,sd=sigmaep),nrow=n_oos,ncol=1)
    X_oos=matrix(rnorm(p*n_oos),nrow=n_oos,ncol=p)%*%sqrtSigma_2
    y_oos=X_oos%*%beta+ep_oos
    fit_lasso=cv.glmnet(X,y,alpha=1,intercept=FALSE)
    fit_ridge=cv.glmnet(X,y,alpha=0,intercept=FALSE)
    betahat_lasso=coef(fit_lasso,s = "lambda.min")[-1]
    betahat_ridge=coef(fit_ridge,s = "lambda.min")[-1]
    DeltaLasso=p*n^(-1)*tau_n^(-2)*(sum((sqrtSigma_2%*%(betahat_lasso-beta))^2)-sum((sqrtSigma_2%*%(beta))^2))
    DeltaRidge=p*n^(-1)*tau_n^(-2)*(sum((sqrtSigma_2%*%(betahat_ridge-beta))^2)-sum((sqrtSigma_2%*%(beta))^2))
    R2oos_ridge=1-mean((y_oos-predict(fit_ridge,X_oos,s = "lambda.min"))^2)/mean((y_oos^2))
    R2oos_lasso=1-mean((y_oos-predict(fit_lasso,X_oos,s = "lambda.min"))^2)/mean((y_oos^2))
    temp=c(R2,sparsity,R2oos_lasso,R2oos_ridge,DeltaLasso,DeltaRidge)
    write.table(t(temp), "weaksimu_revised/exam_main.txt", append = TRUE, col.names = FALSE, row.names = FALSE)
}
stopImplicitCluster()
}

main(R2=0.05,sparsity=0.05,n=500,p=300)
main(R2=0.05,sparsity=0.2,n=500,p=300)
main(R2=0.05,sparsity=0.8,n=500,p=300)
main(R2=0.5,sparsity=0.05,n=500,p=300)
main(R2=0.5,sparsity=0.2,n=500,p=300)
main(R2=0.5,sparsity=0.8,n=500,p=300)

