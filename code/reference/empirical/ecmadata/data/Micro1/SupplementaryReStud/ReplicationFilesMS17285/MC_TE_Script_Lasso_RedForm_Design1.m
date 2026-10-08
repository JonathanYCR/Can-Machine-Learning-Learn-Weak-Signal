%(1) LASSO
%(2) post-LASSO
%(3) sqrt LASSO
%(4) post-sqrt LASSO
%(5) indirect post-LASSO (1 aggregated reg)
%(6) second indirect post-LASSO (adding reg. selected from d on X)
%(7) new proposal (adding regressors in (6) and (2) using sqrt LASSO)
%(8) use (7) for estimation but the standard error from (6)
%(9) double selection
%(10) double selection with undersmoothing
%(11) double selection method with I3
%(12) oracle for second stage
%(13) oracle double selection 
%(14) Union of (9) and (11)
%(15) Union of (10) and (11)
%(16) DS with Split Sample
% design
% 1 linear 
% 1001 constant coefficients
% 2 quadratic
% 22 all quadratic
% 3 linear heteroskedastic
% 4 quadratic heteroskedastic
% 44 all quadratic heteroskedastic
% 5 binary treatment (as 1 otherwise)
% 6 random coef  each N(0,1)
% 7 random coef given by 1
% 72 random coef given by 2
% 722 random coef given by 22
% 8 random coef mixed ~ 0.95 N(0,0.1) + 0.05 N(0,5)

clear;

NUM_SIM = 1000;

p = 200; 
n = 100; 

alpha0 = 1/2;

design = 1;

rho = 0.5;

R2vec = [ 0 0.2 0.4 0.6 0.8 ];
% R21vec = 0.2;
% R22vec = 0.0;

NumValC1 = max(size(R2vec));
NumValC2 = max(size(R2vec));
ALL_COVERAGE = zeros(NumValC1,NumValC2,16);
ALL_BIAS = zeros(NumValC1,NumValC2,16);
ALL_SD = zeros(NumValC1,NumValC2,16);

for i1 = 1 : 1 : NumValC1
    for i2 = 1 : 1 : NumValC2

%c1 = 2 + DELTA*i1;
%c2 = 2 + DELTA*i2;
R21 = R2vec(i1);
R22 = R2vec(i2);
% R21 = R21vec(i1);
% R22 = R22vec(i2);

fprintf('R-square First Stage %f\n', R21);                
fprintf('R-square Second Stage %f\n', R22);                
        
[ ALL_StdErr, ALL_ALPHA ] = MC_TE_FixedDesign_Heteroskedastic_Lasso_RedForm ( NUM_SIM, rho, alpha0, R21, R22, design, p, n );

%%% (1) LASSO 
AbsZvalue = abs(ALL_ALPHA(:,1) - alpha0)./ALL_StdErr(:,1);
COVERAGE_LASSO = sum(AbsZvalue > 1.96) / NUM_SIM;
%Bias_LASSO = median(ALL_ALPHA(:,1) - alpha0);
Bias_LASSO = mean(ALL_ALPHA(:,1) - alpha0);
SD_LASSO = sqrt(var(ALL_ALPHA(:,1)));

%%% (2) Post-LASSO 
AbsZvalue = abs(ALL_ALPHA(:,2) - alpha0)./ALL_StdErr(:,2);
COVERAGE_postLASSO = sum(AbsZvalue > 1.96) / NUM_SIM;
%Bias_postLASSO = median(ALL_ALPHA(:,2) - alpha0);
Bias_postLASSO = mean(ALL_ALPHA(:,2) - alpha0);
SD_postLASSO = sqrt(var(ALL_ALPHA(:,2)));


%% (3) SQRT LASSO 
AbsZvalue = abs(ALL_ALPHA(:,3) - alpha0)./ALL_StdErr(:,3);
COVERAGE_SQRTLASSO = sum(AbsZvalue > 1.96) / NUM_SIM;
Bias_SQRTLASSO = median(ALL_ALPHA(:,3) - alpha0);
%Bias_postSQRTLASSO = mean(ALL_ALPHA(:,4) - alpha0);
SD_SQRTLASSO = sqrt(var(ALL_ALPHA(:,3)));

%% (4) POST SQRT LASSO 
AbsZvalue = abs(ALL_ALPHA(:,4) - alpha0)./ALL_StdErr(:,4);
COVERAGE_postSQRTLASSO = sum(AbsZvalue > 1.96) / NUM_SIM;
%Bias_postSQRTLASSO = median(ALL_ALPHA(:,4) - alpha0);
Bias_postSQRTLASSO = mean(ALL_ALPHA(:,4) - alpha0);
SD_postSQRTLASSO = sqrt(var(ALL_ALPHA(:,4)));

%%% (5) AGG (Indirect Post Lasso)
AbsZvalue = abs(ALL_ALPHA(:,5) - alpha0)./ALL_StdErr(:,5);
COVERAGE_AGG = sum(AbsZvalue > 1.96) / NUM_SIM;
%Bias_AGG = median(ALL_ALPHA(:,5) - alpha0);
Bias_AGG = mean(ALL_ALPHA(:,5) - alpha0);
SD_AGG = sqrt(var(ALL_ALPHA(:,5)));

%%% (6) AGG SECOND
AbsZvalue = abs(ALL_ALPHA(:,6) - alpha0)./ALL_StdErr(:,6);
COVERAGE_AGGSECOND = sum(AbsZvalue > 1.96) / NUM_SIM;
%Bias_AGGSECOND = median(ALL_ALPHA(:,6) - alpha0);
Bias_AGGSECOND = mean(ALL_ALPHA(:,6) - alpha0);
SD_AGGSECOND = sqrt(var(ALL_ALPHA(:,6)));

%%% (7) NEW REVISION
AbsZvalue = abs(ALL_ALPHA(:,7) - alpha0)./ALL_StdErr(:,7);
COVERAGE_NEW = sum(AbsZvalue > 1.96) / NUM_SIM;
%Bias_NEW = median(ALL_ALPHA(:,7) - alpha0);
Bias_NEW = mean(ALL_ALPHA(:,7) - alpha0);
SD_NEW = sqrt(var(ALL_ALPHA(:,7)));

%%% (8) New that should give smaller RP
AbsZvalue = abs(ALL_ALPHA(:,8) - alpha0)./ALL_StdErr(:,8);
COVERAGE_BELOW = sum(AbsZvalue > 1.96) / NUM_SIM;
%Bias_BELOW = median(ALL_ALPHA(:,8) - alpha0);
Bias_BELOW = mean(ALL_ALPHA(:,8) - alpha0);
SD_BELOW = sqrt(var(ALL_ALPHA(:,8)));


%%% (9) double selection
AbsZvalue = abs(ALL_ALPHA(:,9) - alpha0)./ALL_StdErr(:,9);
COVERAGE_9 = sum(AbsZvalue > 1.96) / NUM_SIM;
%Bias_9 = median(ALL_ALPHA(:,9) - alpha0);
Bias_9 = mean(ALL_ALPHA(:,9) - alpha0);
SD_9 = sqrt(var(ALL_ALPHA(:,9)));

%%% (10) double selection undersmoothing
AbsZvalue = abs(ALL_ALPHA(:,10) - alpha0)./ALL_StdErr(:,10);
COVERAGE_10 = sum(AbsZvalue > 1.96) / NUM_SIM;
%Bias_10 = median(ALL_ALPHA(:,10) - alpha0);
Bias_10 = mean(ALL_ALPHA(:,10) - alpha0);
SD_10 = sqrt(var(ALL_ALPHA(:,10)));

%%% (11) double selection method with I3
AbsZvalue = abs(ALL_ALPHA(:,11) - alpha0)./ALL_StdErr(:,11);
COVERAGE_11 = sum(AbsZvalue > 1.96) / NUM_SIM;
%Bias_11 = median(ALL_ALPHA(:,11) - alpha0);
Bias_11 = mean(ALL_ALPHA(:,11) - alpha0);
SD_11 = sqrt(var(ALL_ALPHA(:,11)));

%%% (12) Oracle second stage
AbsZvalue = abs(ALL_ALPHA(:,12) - alpha0)./ALL_StdErr(:,12);
COVERAGE_ORACLE = sum(AbsZvalue > 1.96) / NUM_SIM;
%Bias_ORACLE = median(ALL_ALPHA(:,12) - alpha0);
Bias_ORACLE = mean(ALL_ALPHA(:,12) - alpha0);
SD_ORACLE = sqrt(var(ALL_ALPHA(:,12)));

%%% (13) Oracle double selection
AbsZvalue = abs(ALL_ALPHA(:,13) - alpha0)./ALL_StdErr(:,13);
COVERAGE_ORACLEds = sum(AbsZvalue > 1.96) / NUM_SIM;
%Bias_ORACLEds = median(ALL_ALPHA(:,13) - alpha0);
Bias_ORACLEds = mean(ALL_ALPHA(:,13) - alpha0);
SD_ORACLEds = sqrt(var(ALL_ALPHA(:,13)));

%%% (14) Union DS and ADS
AbsZvalue = min( abs(ALL_ALPHA(:,9) - alpha0)./ALL_StdErr(:,9), abs(ALL_ALPHA(:,11) - alpha0)./ALL_StdErr(:,11) );
UpperBound = max( ALL_ALPHA(:,9) + 1.96*ALL_StdErr(:,9) , ALL_ALPHA(:,11) + 1.96*ALL_StdErr(:,11) );
LowerBound = max( ALL_ALPHA(:,9) - 1.96*ALL_StdErr(:,9) , ALL_ALPHA(:,11) - 1.96*ALL_StdErr(:,11) );

COVERAGE_Union = sum(AbsZvalue > 1.96) / NUM_SIM;
%Bias_Union = median(ALL_ALPHA(:,12) - alpha0);
Bias_Union = mean( (UpperBound+LowerBound)/2 - alpha0);
SD_Union = sqrt(var((UpperBound+LowerBound)/2));

%%% (15) Union DS undersm.  and ADS
AbsZvalue = min( abs(ALL_ALPHA(:,10) - alpha0)./ALL_StdErr(:,10), abs(ALL_ALPHA(:,11) - alpha0)./ALL_StdErr(:,11) );
UpperBound = max( ALL_ALPHA(:,10) + 1.96*ALL_StdErr(:,10) , ALL_ALPHA(:,11) + 1.96*ALL_StdErr(:,11) );
LowerBound = max( ALL_ALPHA(:,10) - 1.96*ALL_StdErr(:,10) , ALL_ALPHA(:,11) - 1.96*ALL_StdErr(:,11) );
COVERAGE_Union_DSU_ADS = sum(AbsZvalue > 1.96) / NUM_SIM;
%Bias_Union = median(ALL_ALPHA(:,12) - alpha0);
Bias_Union_DSU_ADS = mean( (UpperBound+LowerBound)/2 - alpha0);
SD_Union_DSU_ADS = sqrt(var((UpperBound+LowerBound)/2));


%%% (16) Split Sample
AbsZvalue = abs(ALL_ALPHA(:,14) - alpha0)./ALL_StdErr(:,14);
COVERAGE_SS = sum(AbsZvalue > 1.96) / NUM_SIM;
%Bias_ORACLE = median(ALL_ALPHA(:,14) - alpha0);
Bias_SS = mean(ALL_ALPHA(:,14) - alpha0);
SD_SS = sqrt(var(ALL_ALPHA(:,14)));

aux = (1:1:16)';

COVERAGE = [ COVERAGE_LASSO COVERAGE_postLASSO COVERAGE_SQRTLASSO ...
             COVERAGE_postSQRTLASSO  COVERAGE_AGG COVERAGE_AGGSECOND ...
             COVERAGE_NEW COVERAGE_BELOW COVERAGE_9 COVERAGE_10 COVERAGE_11 ...
             COVERAGE_ORACLE COVERAGE_ORACLEds ...
             COVERAGE_Union COVERAGE_Union_DSU_ADS ...
             COVERAGE_SS]'; 
     

BIAS = [ Bias_LASSO Bias_postLASSO Bias_SQRTLASSO ...
             Bias_postSQRTLASSO  Bias_AGG Bias_AGGSECOND ...
             Bias_NEW Bias_BELOW Bias_9 Bias_10 Bias_11 ...
             Bias_ORACLE Bias_ORACLEds ...
             Bias_Union Bias_Union_DSU_ADS ...
             Bias_SS]'; 
     

SD = [ SD_LASSO SD_postLASSO SD_SQRTLASSO ...
             SD_postSQRTLASSO  SD_AGG SD_AGGSECOND ...
             SD_NEW SD_BELOW SD_9 SD_10 SD_11 ...
             SD_ORACLE SD_ORACLEds ...
             SD_Union SD_Union_DSU_ADS ...
             SD_SS]';

[aux COVERAGE  BIAS SD ] %#ok<NOPTS>

ALL_COVERAGE(i1,i2,:) = COVERAGE;
ALL_BIAS(i1,i2,:) = BIAS;
ALL_SD(i1,i2,:) = SD;

    end
end
% C1 = R2vec(1:NumValC1);
% C2 = R2vec(1:NumValC2);
% 
% figure; hold on;
% subplot(3,2,1); surf(C1,C2,ALL_COVERAGE(:,:,1)); title('Lasso RP(0.05)','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex'); 
% axis([ min(C1) max(C1) min(C2) max(C2) min(0,min(min(ALL_COVERAGE(:,:,1)))) max(0.5,max(max(ALL_COVERAGE(:,:,1))))]);
% subplot(3,2,3); surf(C1,C2,ALL_BIAS(:,:,1)); title('Lasso Mean Bias','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% subplot(3,2,5); surf(C1,C2,ALL_SD(:,:,1)); title('Lasso Std Dev.','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% subplot(3,2,2); surf(C1,C2,ALL_COVERAGE(:,:,2)); title('Post-Lasso RP(0.05)','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% axis([ min(C1) max(C1) min(C2) max(C2) min(0,min(min(ALL_COVERAGE(:,:,2)))) max(0.5,max(max(ALL_COVERAGE(:,:,2))))]);
% subplot(3,2,4); surf(C1,C2,ALL_BIAS(:,:,2)); title('Post-Lasso Mean Bias','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% subplot(3,2,6); surf(C1,C2,ALL_SD(:,:,2)); title('Post-Lasso Std Dev.','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% 
% figure; hold on;
% subplot(3,2,1); surf(C1,C2,ALL_COVERAGE(:,:,9)); title('DS RP(0.05)','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% axis([ min(C1) max(C1) min(C2) max(C2) min(0,min(min(ALL_COVERAGE(:,:,9)))) max(0.5,max(max(ALL_COVERAGE(:,:,9))))]);
% subplot(3,2,3); surf(C1,C2,ALL_BIAS(:,:,9)); title('DS Mean Bias','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% subplot(3,2,5); surf(C1,C2,ALL_SD(:,:,9)); title('DS Std Dev.','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% subplot(3,2,2); surf(C1,C2,ALL_COVERAGE(:,:,14)); title('DS Union RP(0.05)','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% axis([ min(C1) max(C1) min(C2) max(C2) min(0,min(min(ALL_COVERAGE(:,:,14)))) max(0.5,max(max(ALL_COVERAGE(:,:,14))))]);
% subplot(3,2,4); surf(C1,C2,ALL_BIAS(:,:,14)); title('DS Union Mean Bias','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% subplot(3,2,6); surf(C1,C2,ALL_SD(:,:,14)); title('DS Union Std Dev.','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% %subplot(3,3,2); surf(C1,C2,ALL_COVERAGE(:,:,10)); title('DS undersm RP(0.05)','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% %axis([ min(C1) max(C1) min(C2) max(C2) min(0,min(min(ALL_COVERAGE(:,:,10)))) max(0.5,max(max(ALL_COVERAGE(:,:,10))))]);
% %subplot(3,3,5); surf(C1,C2,ALL_BIAS(:,:,10)); title('DS undersm Mean Bias','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% %subplot(3,3,8); surf(C1,C2,ALL_SD(:,:,10)); title('DS undersm Std Dev.','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% 
% figure; hold on;
% subplot(3,2,1); surf(C1,C2,ALL_COVERAGE(:,:,11)); title('DS I3 RP(0.05)','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% axis([ min(C1) max(C1) min(C2) max(C2) min(0,min(min(ALL_COVERAGE(:,:,11)))) max(0.5,max(max(ALL_COVERAGE(:,:,11))))]);
% subplot(3,2,3); surf(C1,C2,ALL_BIAS(:,:,11)); title('DS I3 Mean Bias','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% subplot(3,2,5); surf(C1,C2,ALL_SD(:,:,11)); title('DS I3 Std Dev.','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% subplot(3,2,2); surf(C1,C2,ALL_COVERAGE(:,:,16)); title('DS Split Sample RP(0.05)','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% axis([ min(C1) max(C1) min(C2) max(C2) min(0,min(min(ALL_COVERAGE(:,:,16)))) max(0.5,max(max(ALL_COVERAGE(:,:,16))))]);
% subplot(3,2,4); surf(C1,C2,ALL_BIAS(:,:,16)); title('DS Split Sample Mean Bias','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% subplot(3,2,6); surf(C1,C2,ALL_SD(:,:,16)); title('DS Split Sample Std Dev.','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% 
% %subplot(3,3,3); surf(C1,C2,ALL_COVERAGE(:,:,15)); title('DS Union Second RP(0.05)','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% %axis([ min(C1) max(C1) min(C2) max(C2) min(0,min(min(ALL_COVERAGE(:,:,15)))) max(0.5,max(max(ALL_COVERAGE(:,:,15))))]);
% %subplot(3,3,6); surf(C1,C2,ALL_BIAS(:,:,15)); title('DS Union Second Mean Bias','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% %subplot(3,3,9); surf(C1,C2,ALL_SD(:,:,15)); title('DS Union Second Std Dev.','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% 
% figure; hold on;
% subplot(3,2,1); surf(C1,C2,ALL_COVERAGE(:,:,12)); title('Oracle RP(0.05)','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% axis([ min(C1) max(C1) min(C2) max(C2) min(0,min(min(ALL_COVERAGE(:,:,12)))) max(0.5,max(max(ALL_COVERAGE(:,:,12))))]);
% subplot(3,2,3); surf(C1,C2,ALL_BIAS(:,:,12)); title('Oracle Mean Bias','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% subplot(3,2,5); surf(C1,C2,ALL_SD(:,:,12)); title('Oracle Std Dev.','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% subplot(3,2,2); surf(C1,C2,ALL_COVERAGE(:,:,13)); title('Oracle DS RP(0.05)','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% axis([ min(C1) max(C1) min(C2) max(C2) min(0, min(min(ALL_COVERAGE(:,:,13)))) max(0.5,max(max(ALL_COVERAGE(:,:,13))))]);
% subplot(3,2,4); surf(C1,C2,ALL_BIAS(:,:,13)); title('Oracle DS Mean Bias','Interpreter','Latex'); xlabel('First Stage $R^2$','Interpreter','Latex'); ylabel('Second Stage $R^2$','Interpreter','Latex');
% subplot(3,2,6); surf(C1,C2,ALL_SD(:,:,13)); title('Oracle DS Std Dev.','Interpreter','Latex'); 

save Design1PerformanceNewJK_RedForm;



