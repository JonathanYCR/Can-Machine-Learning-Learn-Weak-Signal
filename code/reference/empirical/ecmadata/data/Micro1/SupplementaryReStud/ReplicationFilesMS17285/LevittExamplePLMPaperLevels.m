diary('levittPLMLevels.txt')

clear;

%% Import data 

data = dlmread('levitt_ex.dat','\t',1,0);

% Remove DC
state = data(:,1);
ind = (state ~= 9);
data = data(ind,:); 
% Remove Alaska
state = data(:,1);
ind = (state ~= 2);
data = data(ind,:);
% Remove Hawaii
state = data(:,1);
ind = (state ~= 12);
data = data(ind,:);

year   = data(:,2);

% Remove data for years not used in Levitt analysis
I         = find(year < 85 | year > 97);
year(I,:) = [];
data(I,:) = [];
year      = year - 84;

pop = log(data(:,3));
weight = sqrt(data(:,3)/sum(data(:,3)));
state  = recode(data(:,1));
y_viol = data(:,4);
y_prop = data(:,5);
y_murd = data(:,6);
xx     = data(:,10:size(data,2));
sdum   = dummy(state);
tdum   = dummyvar(year);
x_viol = data(:,8);
x_prop = data(:,9);
x_murd = data(:,7);

T      = max(year);
N      = max(state);

% Census divisions
% 1. Washington, Alaska, Hawaii, Oregon, California
% 2. Montana, Idaho, Wyoming, Nevada, Utah, Colorado, Arizona, New Mexico
% 3. North Dakota, South Dakota, Nebraska, Kansas, Minnesota, Iowa,
% Missouri
% 4. Oklahoma, Arkansas, Texas, Louisiana
% 5. Wisconsin, Illinois, Michigan, Indiana, Ohio
% 6. New York, Pennsylvania, New Jersey
% 7. Vermont, New Hampshire, Maine, Massachussetts, Connecticut, Rhode
% Island
% 8. Kentucky, Tennessee, Mississippi, Alabama
% 9. West Virginia, DC, Maryland, Delaware, Virginia, North Carolina, South
% Carolina, Georgia, Florida

district = zeros(size(state));
district(state == 1) = 8; district(state == 2) = 2; district(state == 3) = 4;
district(state == 4) = 1; district(state == 5) = 2; district(state == 6) = 7;
district(state == 7) = 9; district(state == 8) = 9; district(state == 9) = 9;
district(state == 10) = 2; district(state == 11) = 5; district(state == 12) = 5;
district(state == 13) = 3; district(state == 14) = 3; district(state == 15) = 8;
district(state == 16) = 4; district(state == 17) = 7; district(state == 18) = 9;
district(state == 19) = 7; district(state == 20) = 5; district(state == 21) = 3;
district(state == 22) = 8; district(state == 23) = 3; district(state == 24) = 2;
district(state == 25) = 3; district(state == 26) = 2; district(state == 27) = 7;
district(state == 28) = 6; district(state == 29) = 2; district(state == 30) = 6;
district(state == 31) = 9; district(state == 32) = 3; district(state == 33) = 5;
district(state == 34) = 4; district(state == 35) = 1; district(state == 36) = 6;
district(state == 37) = 7; district(state == 38) = 9; district(state == 39) = 3;
district(state == 40) = 8; district(state == 41) = 4; district(state == 42) = 2;
district(state == 43) = 7; district(state == 44) = 9; district(state == 45) = 1;
district(state == 46) = 9; district(state == 47) = 5; district(state == 48) = 2;

clear data I;

%% Estimate linear models in levels with state effects, time effects
XL = [xx sdum tdum ];

kL = size(XL,2);

%Partial out control variables
XLMX = XL*pinv(XL'*XL);
xvL = x_viol - XLMX*(XL'*x_viol);
yvL = y_viol - XLMX*(XL'*y_viol);
xpL = x_prop - XLMX*(XL'*x_prop);
ypL = y_prop - XLMX*(XL'*y_prop);
xmL = x_murd - XLMX*(XL'*x_murd);
ymL = y_murd - XLMX*(XL'*y_murd);

% Estimate regression coefficients and residuals
bv  = (xvL'*xvL)\(xvL'*yvL);
ev    = yvL - xvL*bv;

bp  = (xpL'*xpL)\(xpL'*ypL);
ep    = ypL - xpL*bp;

bm  = (xmL'*xmL)\(xmL'*ymL);
em    = ymL - xmL*bm;

%% Estimate standard errors
% Clustered standard errors by state
sv = cluster_se(xvL,ev,inv(xvL'*xvL),state,kL+1);
sp = cluster_se(xpL,ep,inv(xpL'*xpL),state,kL+1);
sm = cluster_se(xmL,em,inv(xmL'*xmL),state,kL+1);

% Display results
disp('Number of variables');
disp(kL);
disp('Violence - Level')
disp([bv(1,1) sv(1,:)])
disp(' ')
disp('Property - Level')
disp([bp(1,1) sp(1,:)])
disp(' ')
disp('Murder - Level')
disp([bm(1,1) sm(1,:)])
disp(' ')


%% Estimate linear models in levels with state effects, state trends, and time effects
XL = [xx sdum tdum year sdum.*(year*ones(1,N-1))];

kL = rank(XL);

%Partial out control variables
XLMX = XL*pinv(XL'*XL);
xvL = x_viol - XLMX*(XL'*x_viol);
yvL = y_viol - XLMX*(XL'*y_viol);
xpL = x_prop - XLMX*(XL'*x_prop);
ypL = y_prop - XLMX*(XL'*y_prop);
xmL = x_murd - XLMX*(XL'*x_murd);
ymL = y_murd - XLMX*(XL'*y_murd);

% Estimate regression coefficients and residuals
bvt  = (xvL'*xvL)\(xvL'*yvL);
evt    = yvL - xvL*bvt;

bpt  = (xpL'*xpL)\(xpL'*ypL);
ept    = ypL - xpL*bpt;

bmt  = (xmL'*xmL)\(xmL'*ymL);
emt    = ymL - xmL*bmt;

%% Estimate standard errors
% Clustered standard errors by state
svt = cluster_se(xvL,evt,inv(xvL'*xvL),state,kL+1);
spt = cluster_se(xpL,ept,inv(xpL'*xpL),state,kL+1);
smt = cluster_se(xmL,emt,inv(xmL'*xmL),state,kL+1);

% Display results
disp('Number of variables');
disp(kL);
disp('Violence - Level')
disp([bvt(1,1) svt(1,:)])
disp(' ')
disp('Property - Level')
disp([bpt(1,1) spt(1,:)])
disp(' ')
disp('Murder - Level')
disp([bmt(1,1) smt(1,:)])
disp(' ')


%% (1) Require models to be larger than DL baseline
kxx = size(xx,2);
xxS = zscore(xx);
InitZ = [xxS sdum tdum];  % These will be forced to be in every model
kInit = size(InitZ,2);
NameInitZ = {
    'prison' , 'police' , 'ur' , 'inc' , 'pov' , 'afdc' , 'gun' , 'beer' , ...
    's1' , 's2' , 's3' , 's4' , 's5' , 's6' , 's7' , 's8' , 's9' , 's10' , ...
    's11' , 's12' , 's13' , 's14' , 's15' , 's16' , 's17' , 's18' , 's19' , 's20' , ...
    's21' , 's22' , 's23' , 's24' , 's25' , 's26' , 's27' , 's28' , 's29' , 's30' , ...
    's31' , 's32' , 's33' , 's34' , 's35' , 's36' , 's37' , 's38' , 's39' , 's40' , ...
    's41' , 's42' , 's43' , 's44' , 's45' , 's46' , 's47' , ...
    't1' , 't2' , 't3' , 't4' , 't5' , 't6' , 't7' , 't8' , 't9' , 't10' , ...
    't11' , 't12' , 't13' ...
    }';

%% Generate nonlinear trends for controls
trend = year/T;  % Normalize to unit interval

for jj = 2:3
    trend = [trend trend(:,1).^jj]; %#ok<AGROW>
end

for jj = 1:3
    trend = [trend sin((pi*jj)*trend(:,1))]; %#ok<AGROW>
end

for jj = 1:3
    trend = [trend cos((pi*jj)*trend(:,1))]; %#ok<AGROW>
end

NameTrend = {'P1' , 'P2' , 'P3' , ...
    'sin1' , 'sin2' , 'sin3' , ...
    'cos1' , 'cos2' , 'cos3' , ...
    }';


%% Get various "initial" variables for interacting with trends
xxprop = zeros(N*T,19);
xxviol = zeros(N*T,19);
xxmurd = zeros(N*T,19);
for ii = 1:N
    I1 = state == ii;
    xxprop(I1,1) = mean(x_prop(I1));
    xxviol(I1,1) = mean(x_viol(I1));
    xxmurd(I1,1) = mean(x_murd(I1));
    xxprop(I1,2) = x_prop(state == ii & year == 1);
    xxviol(I1,2) = x_viol(state == ii & year == 1);
    xxmurd(I1,2) = x_murd(state == ii & year == 1);
    xxprop(I1,3) = y_prop(state == ii & year == 1);
    xxviol(I1,3) = y_viol(state == ii & year == 1);
    xxmurd(I1,3) = y_murd(state == ii & year == 1);
    xxprop(I1,4:11) = ones(sum(I1),1)*mean(xx(I1,:));
    xxviol(I1,4:11) = ones(sum(I1),1)*mean(xx(I1,:));
    xxmurd(I1,4:11) = ones(sum(I1),1)*mean(xx(I1,:));
    xxprop(I1,12:19) = ones(sum(I1),1)*xx(state == ii & year == 1,:);
    xxviol(I1,12:19) = ones(sum(I1),1)*xx(state == ii & year == 1,:);
    xxmurd(I1,12:19) = ones(sum(I1),1)*xx(state == ii & year == 1,:);    
end
xxprop = [xxprop xxS];
xxviol = [xxviol xxS];
xxmurd = [xxmurd xxS];

NameInit = {'abar' , 'a85' , 'y85' , ...
    'prisonbar' , 'policebar' , 'urbar' , 'incbar' , 'povbar' , 'afdcbar' , 'gunbar' , 'beerbar' , ...
    'prison85' , 'police85' , 'ur85' , 'inc85' , 'pov85' , 'afdc85' , 'gun85' , 'beer85' , ...
    'prison' , 'police' , 'ur' , 'inc' , 'pov' , 'afdc' , 'gun' , 'beer' ...
    }';


%% Generate interactions
k0 = size(xxprop,2);
kT = size(trend,2);
BigP = zeros(N*T,k0*kT);
BigV = zeros(N*T,k0*kT);
BigM = zeros(N*T,k0*kT);
BigName = cell(k0*kT,1);
for ii = 1:k0
    for jj = 1:kT
        CurrCol = kT*(ii-1)+jj;
        BigP(:,CurrCol) = xxprop(:,ii).*trend(:,jj);
        BigV(:,CurrCol) = xxviol(:,ii).*trend(:,jj);
        BigM(:,CurrCol) = xxmurd(:,ii).*trend(:,jj);
        CurrName = strcat(NameInit{ii},NameTrend{jj});
        BigName{CurrCol} = CurrName;
    end
end

%% Partial out variables included in all models
zV = zscore(BigV-InitZ*((InitZ'*InitZ)\(InitZ'*BigV)));
zP = zscore(BigP-InitZ*((InitZ'*InitZ)\(InitZ'*BigP)));
zM = zscore(BigM-InitZ*((InitZ'*InitZ)\(InitZ'*BigM)));
xV = x_viol-InitZ*((InitZ'*InitZ)\(InitZ'*x_viol));
xP = x_prop-InitZ*((InitZ'*InitZ)\(InitZ'*x_prop));
xM = x_murd-InitZ*((InitZ'*InitZ)\(InitZ'*x_murd));
yV = y_viol-InitZ*((InitZ'*InitZ)\(InitZ'*y_viol));
yP = y_prop-InitZ*((InitZ'*InitZ)\(InitZ'*y_prop));
yM = y_murd-InitZ*((InitZ'*InitZ)\(InitZ'*y_murd));

%% Regression with "all" controls
kzV = size(zV,2);
kzP = size(zP,2);
kzM = size(zM,2);

%Partial out control variables
PzV = zV/(zV'*zV);
PzP = zP/(zP'*zP);
PzM = zM/(zM'*zM);

MxV = xV - PzV*(zV'*xV);
MyV = yV - PzV*(zV'*yV);
MxP = xP - PzP*(zP'*xP);
MyP = yP - PzP*(zP'*yP);
MxM = xM - PzM*(zM'*xM);
MyM = yM - PzM*(zM'*yM);

% Estimate regression coefficients and residuals
bvA  = (MxV'*MxV)\(MxV'*MyV);
evA    = MyV - MxV*bvA;

bpA  = (MxP'*MxP)\(MxP'*MyP);
epA    = MyP - MxP*bpA;

bmA  = (MxM'*MxM)\(MxM'*MyM);
emA    = MyM - MxM*bmA;

%% Estimate standard errors
% Clustered standard errors by state
svA = cluster_se(MxV,evA,inv(MxV'*MxV),state,kzV+kInit+1);
spA = cluster_se(MxP,epA,inv(MxP'*MxP),state,kzP+kInit+1);
smA = cluster_se(MxM,emA,inv(MxM'*MxM),state,kzM+kInit+1);

% Display results
disp('Number of variables used');
disp([kzV kzP kzM]+kInit)
disp('Violence - Level')
disp([bvA(1,1) svA(1,:)])
disp(' ')
disp('Property - Level')
disp([bpA(1,1) spA(1,:)])
disp(' ')
disp('Murder - Level')
disp([bmA(1,1) smA(1,:)])
disp(' ')

%% Initialize a couple things for LASSO
kV = kzV + kInit;
kP = kzP + kInit;
kM = kzM + kInit;

InitResidV = xV;
InitResidP = xP;
InitResidM = xM;

InitResidyV = yV;
InitResidyP = yP;
InitResidyM = yM;

maxIter = 100;

%% LASSO - Violence
% Prediction of variable of interest
StV = zV.*(InitResidV*ones(1,kzV));
UpsV = sqrt(.1*sum(StV.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kV/.05))));
PInitV = LassoShooting(zV./(ones(T*N,1)*UpsV), xV , lambda, 'Verbose', 0);
IndInitV = abs(PInitV) > 0;
ZV = zV(:,IndInitV);
RefResidV = xV-ZV*((ZV'*ZV)\(ZV'*xV));

kk = 1;
StRefV = zV.*(RefResidV*ones(1,kzV));
UpsRefV = sqrt(sum(StRefV.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kV/.05))));
lastNorm = Inf;
while norm(UpsRefV-UpsV) > .02 && kk < maxIter && norm(UpsRefV-UpsV) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefV-UpsV);
    disp(lastNorm);
    PRefV = LassoShooting(zV./(ones(T*N,1)*UpsRefV), xV , lambda, 'Verbose', 0);
    IndRefV = abs(PRefV) > 0;
    ZV = zV(:,IndRefV);
    bfsV = ((ZV'*ZV)\(ZV'*xV));
    efsV = xV-ZV*bfsV;
    UpsV = UpsRefV;
    StRefV = zV.*(efsV*ones(1,kzV));
    UpsRefV = sqrt(sum(StRefV.^2)/(T*N));    
    kk = kk+1;
end
sfsV = cluster_se(ZV,efsV,inv(ZV'*ZV),state);

% Structural equation
StyV = zV.*(InitResidyV*ones(1,kzV));
UpsyV = sqrt(.1*sum(StyV.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kV/.05))));
PInityV = LassoShooting(zV./(ones(T*N,1)*UpsyV), yV , lambda, 'Verbose', 0);
IndInityV = abs(PInityV) > 0;
ZyV = zV(:,IndInityV);
RefResidyV = yV-ZyV*((ZyV'*ZyV)\(ZyV'*yV));

kk = 1;
StRefyV = zV.*(RefResidyV*ones(1,kzV));
UpsRefyV = sqrt(sum(StRefyV.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kV/.05))));
lastNorm = Inf;
while norm(UpsRefyV-UpsyV) > 1e-4 && kk < maxIter && norm(UpsRefyV-UpsyV) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefyV-UpsyV);
    disp(lastNorm);
    PRefyV = LassoShooting(zV./(ones(T*N,1)*UpsRefyV), yV , lambda, 'Verbose', 0);
    IndRefyV = abs(PRefyV) > 0;
    ZyV = zV(:,IndRefyV);
    bfsyV = ((ZyV'*ZyV)\(ZyV'*yV));
    efsyV = yV-ZyV*bfsyV;
    UpsyV = UpsRefyV;
    StRefyV = zV.*(efsyV*ones(1,kzV));
    UpsRefyV = sqrt(sum(StRefyV.^2)/(T*N));    
    kk = kk+1;
end
sfsyV = cluster_se(ZyV,efsyV,inv(ZyV'*ZyV),state);

IndUnionV = max([IndRefyV IndRefV],[],2);
ZUnionV = zV(:,IndUnionV);

R2_1V = sum((ZUnionV*(ZUnionV\xV)).^2)/(sum(xV.^2));
R2_2V = sum((ZUnionV*(ZUnionV\yV)).^2)/(sum(yV.^2)); 

zVLASSO = [xV ZUnionV];
bviolLASSO = (zVLASSO'*zVLASSO)\(zVLASSO'*yV);
eviolLASSO = yV - zVLASSO*bviolLASSO;
ResviolLASSO = cluster_se(zVLASSO,eviolLASSO,inv(zVLASSO'*zVLASSO),state,kInit+sum(IndUnionV));

disp('Violence - Level - LASSO')
disp('Selected Variables - Abortion')
disp(BigName(IndRefV))
disp(' ');
disp('Selected Variables - Crime')
disp(BigName(IndRefyV))
disp(' ');
disp('Abortion equation')
disp([bfsV sfsV])
disp('R^2')
disp(R2_1V)
disp(' ')
disp('Crime equation')
disp([bfsyV sfsyV])
disp('R^2')
disp(R2_2V)
disp(' ')
disp('Number of Selected Variables')
disp(kInit+sum(IndUnionV));
disp(' ')
disp('Treatment Effect')
disp([bviolLASSO(1) ResviolLASSO(1)])
disp(' ')


%% LASSO - Property
% Prediction of variable of interest
StP = zP.*(InitResidP*ones(1,kzP));
UpsP = sqrt(.1*sum(StP.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kP/.05))));
PInitP = LassoShooting(zP./(ones(T*N,1)*UpsP), xP , lambda, 'Verbose', 0);
IndInitP = abs(PInitP) > 0;
ZP = zP(:,IndInitP);
RefResidP = xP-ZP*((ZP'*ZP)\(ZP'*xP));

kk = 1;
StRefP = zP.*(RefResidP*ones(1,kzP));
UpsRefP = sqrt(sum(StRefP.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kP/.05))));
lastNorm = Inf;
while norm(UpsRefP-UpsP) > 1e-4 && kk < maxIter && norm(UpsRefP-UpsP) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefP-UpsP);
    disp(lastNorm);
    PRefP = LassoShooting(zP./(ones(T*N,1)*UpsRefP), xP , lambda, 'Verbose', 0);
    IndRefP = abs(PRefP) > 0;
    ZP = zP(:,IndRefP);
    bfsP = ((ZP'*ZP)\(ZP'*xP));
    efsP = xP-ZP*bfsP;
    UpsP = UpsRefP;
    StRefP = zP.*(efsP*ones(1,kzP));
    UpsRefP = sqrt(sum(StRefP.^2)/(T*N));    
    kk = kk+1;
end
sfsP = cluster_se(ZP,efsP,inv(ZP'*ZP),state);

% Structural equation
StyP = zP.*(InitResidyP*ones(1,kzP));
UpsyP = sqrt(.1*sum(StyP.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kP/.05))));
PInityP = LassoShooting(zP./(ones(T*N,1)*UpsyP), yP , lambda, 'Verbose', 0);
IndInityP = abs(PInityP) > 0;
ZyP = zP(:,IndInityP);
RefResidyP = yP-ZyP*((ZyP'*ZyP)\(ZyP'*yP));

kk = 1;
StRefyP = zP.*(RefResidyP*ones(1,kzP));
UpsRefyP = sqrt(sum(StRefyP.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kP/.05))));
lastNorm = Inf;
while norm(UpsRefyP-UpsyP) > 1e-4 && kk < maxIter && norm(UpsRefyP-UpsyP) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefyP-UpsyP);
    disp(lastNorm);
    PRefyP = LassoShooting(zP./(ones(T*N,1)*UpsRefyP), yP , lambda, 'Verbose', 0);
    IndRefyP = abs(PRefyP) > 0;
    ZyP = zP(:,IndRefyP);
    bfsyP = ((ZyP'*ZyP)\(ZyP'*yP));
    efsyP = yP-ZyP*bfsyP;
    UpsyP = UpsRefyP;
    StRefyP = zP.*(efsyP*ones(1,kzP));
    UpsRefyP = sqrt(sum(StRefyP.^2)/(T*N));    
    kk = kk+1;
end
sfsyP = cluster_se(ZyP,efsyP,inv(ZyP'*ZyP),state);

IndUnionP = max([IndRefyP IndRefP],[],2);
ZUnionP = zP(:,IndUnionP);

R2_1P = sum((ZUnionP*(ZUnionP\xP)).^2)/(sum(xP.^2));
R2_2P = sum((ZUnionP*(ZUnionP\yP)).^2)/(sum(yP.^2)); 

zPLASSO = [xP ZUnionP];
bpropLASSO = (zPLASSO'*zPLASSO)\(zPLASSO'*yP);
epropLASSO = yP - zPLASSO*bpropLASSO;
RespropLASSO = cluster_se(zPLASSO,epropLASSO,inv(zPLASSO'*zPLASSO),state,kInit+sum(IndUnionP));

disp('Property - Level - LASSO')
disp('Selected Variables - Abortion')
disp(BigName(IndRefP))
disp(' ');
disp('Selected Variables - Crime')
disp(BigName(IndRefyP))
disp(' ');
disp('Abortion equation')
disp([bfsP sfsP])
disp('R^2')
disp(R2_1P)
disp(' ')
disp('Crime equation')
disp([bfsyP sfsyP])
disp('R^2')
disp(R2_2P)
disp(' ')
disp('Number of Selected Variables')
disp(kInit+sum(IndUnionP));
disp(' ')
disp('Treatment Effect')
disp([bpropLASSO(1) RespropLASSO(1)])
disp(' ')


%% LASSO - Murder
% Prediction of variable of interest
StM = zM.*(InitResidM*ones(1,kzM));
UpsM = sqrt(.1*sum(StM.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kM/.05))));
PInitM = LassoShooting(zM./(ones(T*N,1)*UpsM), xM , lambda, 'Verbose', 0);
IndInitM = abs(PInitM) > 0;
ZM = zM(:,IndInitM);
RefResidM = xM-ZM*((ZM'*ZM)\(ZM'*xM));

kk = 1;
StRefM = zM.*(RefResidM*ones(1,kzM));
UpsRefM = sqrt(sum(StRefM.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kM/.05))));
lastNorm = Inf;
while norm(UpsRefM-UpsM) > 1e-4 && kk < maxIter && norm(UpsRefM-UpsM) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefM-UpsM);
    disp(lastNorm);
    PRefM = LassoShooting(zM./(ones(T*N,1)*UpsRefM), xM , lambda, 'Verbose', 0);
    IndRefM = abs(PRefM) > 0;
    ZM = zM(:,IndRefM);
    bfsM = ((ZM'*ZM)\(ZM'*xM));
    efsM = xM-ZM*bfsM;
    UpsM = UpsRefM;
    StRefM = zM.*(efsM*ones(1,kzM));
    UpsRefM = sqrt(sum(StRefM.^2)/(T*N));    
    kk = kk+1;
end
sfsM = cluster_se(ZM,efsM,inv(ZM'*ZM),state);

% Structural equation
StyM = zM.*(InitResidyM*ones(1,kzM));
UpsyM = sqrt(.1*sum(StyM.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kM/.05))));
PInityM = LassoShooting(zM./(ones(T*N,1)*UpsyM), yM , lambda, 'Verbose', 0);
IndInityM = abs(PInityM) > 0;
ZyM = zM(:,IndInityM);
RefResidyM = yM-ZyM*((ZyM'*ZyM)\(ZyM'*yM));

kk = 1;
StRefyM = zM.*(RefResidyM*ones(1,kzM));
UpsRefyM = sqrt(sum(StRefyM.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kM/.05))));
lastNorm = Inf;
while norm(UpsRefyM-UpsyM) > 1e-4 && kk < maxIter && norm(UpsRefyM-UpsyM) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefyM-UpsyM);
    disp(lastNorm);
    PRefyM = LassoShooting(zM./(ones(T*N,1)*UpsRefyM), yM , lambda, 'Verbose', 0);
    IndRefyM = abs(PRefyM) > 0;
    ZyM = zM(:,IndRefyM);
    bfsyM = ((ZyM'*ZyM)\(ZyM'*yM));
    efsyM = yM-ZyM*bfsyM;
    UpsyM = UpsRefyM;
    StRefyM = zM.*(efsyM*ones(1,kzM));
    UpsRefyM = sqrt(sum(StRefyM.^2)/(T*N));    
    kk = kk+1;
end
sfsyM = cluster_se(ZyM,efsyM,inv(ZyM'*ZyM),state);

IndUnionM = max([IndRefyM IndRefM],[],2);
ZUnionM = zM(:,IndUnionM);

R2_1M = sum((ZUnionM*(ZUnionM\xM)).^2)/(sum(xM.^2));
R2_2M = sum((ZUnionM*(ZUnionM\yM)).^2)/(sum(yM.^2)); 

zMLASSO = [xM ZUnionM];
bmurdLASSO = (zMLASSO'*zMLASSO)\(zMLASSO'*yM);
emurdLASSO = yM - zMLASSO*bmurdLASSO;
ResmurdLASSO = cluster_se(zMLASSO,emurdLASSO,inv(zMLASSO'*zMLASSO),state,kInit+sum(IndUnionM));

disp('Murder - Level - LASSO')
disp('Selected Variables - Abortion')
disp(BigName(IndRefM))
disp(' ');
disp('Selected Variables - Crime')
disp(BigName(IndRefyM))
disp(' ');
disp('Abortion equation')
disp([bfsM sfsM])
disp('R^2')
disp(R2_1M)
disp(' ')
disp('Crime equation')
disp([bfsyM sfsyM])
disp('R^2')
disp(R2_2M)
disp(' ')
disp('Number of Selected Variables')
disp(kInit+sum(IndUnionM));
disp(' ')
disp('Treatment Effect')
disp([bmurdLASSO(1) ResmurdLASSO(1)])
disp(' ')



%% (2) Selection over all variables including DL original controls, state FE, and time FE
zV = zscore([BigV InitZ(:,1:end-1)]);
zP = zscore([BigP InitZ(:,1:end-1)]);
zM = zscore([BigM InitZ(:,1:end-1)]);
xV = x_viol-mean(x_viol);
xP = x_prop-mean(x_prop);
xM = x_murd-mean(x_murd);
yV = y_viol-mean(y_viol);
yP = y_prop-mean(y_prop);
yM = y_murd-mean(y_murd);

kzV = size(zM,2);
kV = size(zM,2);
kzP = size(zM,2);
kP = size(zM,2);
kzM = size(zM,2);
kM = size(zM,2);
kInit = 1;

BigName = [BigName ; NameInitZ(1:end-1)];

%% LASSO - Violence
% Prediction of variable of interest
StV = zV.*(InitResidV*ones(1,kzV));
UpsV = sqrt(sum(StV.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kV/.05))));
PInitV = LassoShooting(zV./(ones(T*N,1)*UpsV), xV , lambda, 'Verbose', 0);
IndInitV = abs(PInitV) > 0;
ZV = zV(:,IndInitV);
RefResidV = xV-ZV*((ZV'*ZV)\(ZV'*xV));

kk = 1;
StRefV = zV.*(RefResidV*ones(1,kzV));
UpsRefV = sqrt(sum(StRefV.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kV/.05))));
lastNorm = Inf;
while norm(UpsRefV-UpsV) > 1e-4 && kk < maxIter && norm(UpsRefV-UpsV) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefV-UpsV);
    disp(lastNorm);
    PRefV = LassoShooting(zV./(ones(T*N,1)*UpsRefV), xV , lambda, 'Verbose', 0);
    IndRefV = abs(PRefV) > 0;
    ZV = zV(:,IndRefV);
    bfsV = ((ZV'*ZV)\(ZV'*xV));
    efsV = xV-ZV*bfsV;
    UpsV = UpsRefV;
    StRefV = zV.*(efsV*ones(1,kzV));
    UpsRefV = sqrt(sum(StRefV.^2)/(T*N));    
    kk = kk+1;
end
sfsV = cluster_se(ZV,efsV,inv(ZV'*ZV),state);

% Structural equation
StyV = zV.*(InitResidyV*ones(1,kzV));
UpsyV = sqrt(sum(StyV.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kV/.05))));
PInityV = LassoShooting(zV./(ones(T*N,1)*UpsyV), yV , lambda, 'Verbose', 0);
IndInityV = abs(PInityV) > 0;
ZyV = zV(:,IndInityV);
RefResidyV = yV-ZyV*((ZyV'*ZyV)\(ZyV'*yV));

kk = 1;
StRefyV = zV.*(RefResidyV*ones(1,kzV));
UpsRefyV = sqrt(sum(StRefyV.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kV/.05))));
lastNorm = Inf;
while norm(UpsRefyV-UpsyV) > 1e-4 && kk < maxIter && norm(UpsRefyV-UpsyV) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefyV-UpsyV);
    disp(lastNorm);
    PRefyV = LassoShooting(zV./(ones(T*N,1)*UpsRefyV), yV , lambda, 'Verbose', 0);
    IndRefyV = abs(PRefyV) > 0;
    ZyV = zV(:,IndRefyV);
    bfsyV = ((ZyV'*ZyV)\(ZyV'*yV));
    efsyV = yV-ZyV*bfsyV;
    UpsyV = UpsRefyV;
    StRefyV = zV.*(efsyV*ones(1,kzV));
    UpsRefyV = sqrt(sum(StRefyV.^2)/(T*N));    
    kk = kk+1;
end
sfsyV = cluster_se(ZyV,efsyV,inv(ZyV'*ZyV),state);

IndUnionV = max([IndRefyV IndRefV],[],2);
ZUnionV = zV(:,IndUnionV);

R2_1V = sum((ZUnionV*(ZUnionV\xV)).^2)/(sum(xV.^2));
R2_2V = sum((ZUnionV*(ZUnionV\yV)).^2)/(sum(yV.^2)); 

zVLASSO = [xV ZUnionV];
bviolLASSO = (zVLASSO'*zVLASSO)\(zVLASSO'*yV);
eviolLASSO = yV - zVLASSO*bviolLASSO;
ResviolLASSO = cluster_se(zVLASSO,eviolLASSO,inv(zVLASSO'*zVLASSO),state,kInit+sum(IndUnionV));

disp('Violence - Level - LASSO')
disp('Selected Variables - Abortion')
disp(BigName(IndRefV))
disp(' ');
disp('Selected Variables - Crime')
disp(BigName(IndRefyV))
disp(' ');
disp('Abortion equation')
disp([bfsV sfsV])
disp('R^2')
disp(R2_1V)
disp(' ')
disp('Crime equation')
disp([bfsyV sfsyV])
disp('R^2')
disp(R2_2V)
disp(' ')
disp('Number of Selected Variables')
disp(kInit+sum(IndUnionV));
disp(' ')
disp('Treatment Effect')
disp([bviolLASSO(1) ResviolLASSO(1)])
disp(' ')


%% LASSO - Property
% Prediction of variable of interest
StP = zP.*(InitResidP*ones(1,kzP));
UpsP = sqrt(sum(StP.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kP/.05))));
PInitP = LassoShooting(zP./(ones(T*N,1)*UpsP), xP , lambda, 'Verbose', 0);
IndInitP = abs(PInitP) > 0;
ZP = zP(:,IndInitP);
RefResidP = xP-ZP*((ZP'*ZP)\(ZP'*xP));

kk = 1;
StRefP = zP.*(RefResidP*ones(1,kzP));
UpsRefP = sqrt(sum(StRefP.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kP/.05))));
lastNorm = Inf;
while norm(UpsRefP-UpsP) > 1e-4 && kk < maxIter && norm(UpsRefP-UpsP) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefP-UpsP);
    disp(lastNorm);
    PRefP = LassoShooting(zP./(ones(T*N,1)*UpsRefP), xP , lambda, 'Verbose', 0);
    IndRefP = abs(PRefP) > 0;
    ZP = zP(:,IndRefP);
    bfsP = ((ZP'*ZP)\(ZP'*xP));
    efsP = xP-ZP*bfsP;
    UpsP = UpsRefP;
    StRefP = zP.*(efsP*ones(1,kzP));
    UpsRefP = sqrt(sum(StRefP.^2)/(T*N));    
    kk = kk+1;
end
sfsP = cluster_se(ZP,efsP,inv(ZP'*ZP),state);

% Structural equation
StyP = zP.*(InitResidyP*ones(1,kzP));
UpsyP = sqrt(sum(StyP.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kP/.05))));
PInityP = LassoShooting(zP./(ones(T*N,1)*UpsyP), yP , lambda, 'Verbose', 0);
IndInityP = abs(PInityP) > 0;
ZyP = zP(:,IndInityP);
RefResidyP = yP-ZyP*((ZyP'*ZyP)\(ZyP'*yP));

kk = 1;
StRefyP = zP.*(RefResidyP*ones(1,kzP));
UpsRefyP = sqrt(sum(StRefyP.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kP/.05))));
lastNorm = Inf;
while norm(UpsRefyP-UpsyP) > 1e-4 && kk < maxIter && norm(UpsRefyP-UpsyP) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefyP-UpsyP);
    disp(lastNorm);
    PRefyP = LassoShooting(zP./(ones(T*N,1)*UpsRefyP), yP , lambda, 'Verbose', 0);
    IndRefyP = abs(PRefyP) > 0;
    ZyP = zP(:,IndRefyP);
    bfsyP = ((ZyP'*ZyP)\(ZyP'*yP));
    efsyP = yP-ZyP*bfsyP;
    UpsyP = UpsRefyP;
    StRefyP = zP.*(efsyP*ones(1,kzP));
    UpsRefyP = sqrt(sum(StRefyP.^2)/(T*N));    
    kk = kk+1;
end
sfsyP = cluster_se(ZyP,efsyP,inv(ZyP'*ZyP),state);

IndUnionP = max([IndRefyP IndRefP],[],2);
ZUnionP = zP(:,IndUnionP);

R2_1P = sum((ZUnionP*(ZUnionP\xP)).^2)/(sum(xP.^2));
R2_2P = sum((ZUnionP*(ZUnionP\yP)).^2)/(sum(yP.^2)); 

zPLASSO = [xP ZUnionP];
bpropLASSO = (zPLASSO'*zPLASSO)\(zPLASSO'*yP);
epropLASSO = yP - zPLASSO*bpropLASSO;
RespropLASSO = cluster_se(zPLASSO,epropLASSO,inv(zPLASSO'*zPLASSO),state,kInit+sum(IndUnionP));

disp('Property - Level - LASSO')
disp('Selected Variables - Abortion')
disp(BigName(IndRefP))
disp(' ');
disp('Selected Variables - Crime')
disp(BigName(IndRefyP))
disp(' ');
disp('Abortion equation')
disp([bfsP sfsP])
disp('R^2')
disp(R2_1P)
disp(' ')
disp('Crime equation')
disp([bfsyP sfsyP])
disp('R^2')
disp(R2_2P)
disp(' ')
disp('Number of Selected Variables')
disp(kInit+sum(IndUnionP));
disp(' ')
disp('Treatment Effect')
disp([bpropLASSO(1) RespropLASSO(1)])
disp(' ')


%% LASSO - Murder
% Prediction of variable of interest
StM = zM.*(InitResidM*ones(1,kzM));
UpsM = sqrt(sum(StM.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kM/.05))));
PInitM = LassoShooting(zM./(ones(T*N,1)*UpsM), xM , lambda, 'Verbose', 0);
IndInitM = abs(PInitM) > 0;
ZM = zM(:,IndInitM);
RefResidM = xM-ZM*((ZM'*ZM)\(ZM'*xM));

kk = 1;
StRefM = zM.*(RefResidM*ones(1,kzM));
UpsRefM = sqrt(sum(StRefM.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kM/.05))));
lastNorm = Inf;
while norm(UpsRefM-UpsM) > 1e-4 && kk < maxIter && norm(UpsRefM-UpsM) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefM-UpsM);
    disp(lastNorm);
    PRefM = LassoShooting(zM./(ones(T*N,1)*UpsRefM), xM , lambda, 'Verbose', 0);
    IndRefM = abs(PRefM) > 0;
    ZM = zM(:,IndRefM);
    bfsM = ((ZM'*ZM)\(ZM'*xM));
    efsM = xM-ZM*bfsM;
    UpsM = UpsRefM;
    StRefM = zM.*(efsM*ones(1,kzM));
    UpsRefM = sqrt(sum(StRefM.^2)/(T*N));    
    kk = kk+1;
end
sfsM = cluster_se(ZM,efsM,inv(ZM'*ZM),state);

% Structural equation
StyM = zM.*(InitResidyM*ones(1,kzM));
UpsyM = sqrt(sum(StyM.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kM/.05))));
PInityM = LassoShooting(zM./(ones(T*N,1)*UpsyM), yM , lambda, 'Verbose', 0);
IndInityM = abs(PInityM) > 0;
ZyM = zM(:,IndInityM);
RefResidyM = yM-ZyM*((ZyM'*ZyM)\(ZyM'*yM));

kk = 1;
StRefyM = zM.*(RefResidyM*ones(1,kzM));
UpsRefyM = sqrt(sum(StRefyM.^2)/(T*N));
lambda = 1.1*2*sqrt(2*T*N*(log(2*(kM/.05))));
lastNorm = Inf;
while norm(UpsRefyM-UpsyM) > 1e-4 && kk < maxIter && norm(UpsRefyM-UpsyM) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefyM-UpsyM);
    disp(lastNorm);
    PRefyM = LassoShooting(zM./(ones(T*N,1)*UpsRefyM), yM , lambda, 'Verbose', 0);
    IndRefyM = abs(PRefyM) > 0;
    ZyM = zM(:,IndRefyM);
    bfsyM = ((ZyM'*ZyM)\(ZyM'*yM));
    efsyM = yM-ZyM*bfsyM;
    UpsyM = UpsRefyM;
    StRefyM = zM.*(efsyM*ones(1,kzM));
    UpsRefyM = sqrt(sum(StRefyM.^2)/(T*N));    
    kk = kk+1;
end
sfsyM = cluster_se(ZyM,efsyM,inv(ZyM'*ZyM),state);

IndUnionM = max([IndRefyM IndRefM],[],2);
ZUnionM = zM(:,IndUnionM);

R2_1M = sum((ZUnionM*(ZUnionM\xM)).^2)/(sum(xM.^2));
R2_2M = sum((ZUnionM*(ZUnionM\yM)).^2)/(sum(yM.^2)); 

zMLASSO = [xM ZUnionM];
bmurdLASSO = (zMLASSO'*zMLASSO)\(zMLASSO'*yM);
emurdLASSO = yM - zMLASSO*bmurdLASSO;
ResmurdLASSO = cluster_se(zMLASSO,emurdLASSO,inv(zMLASSO'*zMLASSO),state,kInit+sum(IndUnionM));

disp('Murder - Level - LASSO')
disp('Selected Variables - Abortion')
disp(BigName(IndRefM))
disp(' ');
disp('Selected Variables - Crime')
disp(BigName(IndRefyM))
disp(' ');
disp('Abortion equation')
disp([bfsM sfsM])
disp('R^2')
disp(R2_1M)
disp(' ')
disp('Crime equation')
disp([bfsyM sfsyM])
disp('R^2')
disp(R2_2M)
disp(' ')
disp('Number of Selected Variables')
disp(kInit+sum(IndUnionM));
disp(' ')
disp('Treatment Effect')
disp([bmurdLASSO(1) ResmurdLASSO(1)])
disp(' ')


%% Import data that replicates Table III of Donohue and Levitt (2008) 

clear ;

data08 = dlmread('levitt2008.txt','\t',1,0);
data08 = data08(data08(:,4) ~= 0,:);

year08 = data08(:,1);
state08 = data08(:,2);
div08 = data08(:,3);
yprop08 = data08(:,4); 
yviol08 = data08(:,5);
ymurd08 = data08(:,6);
efaprop08 = data08(:,7);
efaviol08 = data08(:,8);
efamurd08 = data08(:,9);

%% Re-import original Levitt data to get some covariates

data = dlmread('levitt_ex.dat','\t',1,0);

yearOr = data(:,2);
lpopOr = log(data(:,3));
stateOr = data(:,1);
xxOr = data(:,10:size(data,2));  % prison, police, ur, inc', pov, afdc, gun, beer


%% Get various "initial" variables for interacting with trends
xxP = zeros(size(yprop08,1),20);
xxV = zeros(size(yviol08,1),20);
xxM = zeros(size(ymurd08,1),20);
for ii = 1:max(state08)
    I1 = state08 == ii;
    if sum(I1) > 0
        Tmin = min(year08(state08 == ii));
        xxP(I1,1) = mean(efaprop08(I1));
        xxV(I1,1) = mean(efaviol08(I1));
        xxM(I1,1) = mean(efamurd08(I1));
        xxP(I1,2) = efaprop08(state08 == ii & year08 == 1985);
        xxV(I1,2) = efaviol08(state08 == ii & year08 == 1985);
        xxM(I1,2) = efamurd08(state08 == ii & year08 == 1985);
        xxP(I1,3) = yprop08(state08 == ii & year08 == Tmin);
        xxV(I1,3) = yviol08(state08 == ii & year08 == Tmin);
        xxM(I1,3) = ymurd08(state08 == ii & year08 == Tmin);
        xxP(I1,4) = yprop08(state08 == ii & year08 == Tmin+1);
        xxV(I1,4) = yviol08(state08 == ii & year08 == Tmin+1);
        xxM(I1,4) = ymurd08(state08 == ii & year08 == Tmin+1);
        xxP(I1,5:12) = ones(sum(I1),1)*mean(xxOr(stateOr == ii,:));
        xxV(I1,5:12) = ones(sum(I1),1)*mean(xxOr(stateOr == ii,:));
        xxM(I1,5:12) = ones(sum(I1),1)*mean(xxOr(stateOr == ii,:));
        xxP(I1,13:20) = ones(sum(I1),1)*xxOr(stateOr == ii & yearOr == 85,:);
        xxV(I1,13:20) = ones(sum(I1),1)*xxOr(stateOr == ii & yearOr == 85,:);
        xxM(I1,13:20) = ones(sum(I1),1)*xxOr(stateOr == ii & yearOr == 85,:);
    end
end

xxP = zscore(xxP);
xxM = zscore(xxM);
xxV = zscore(xxV);

NameInit = {'abar' , 'a85' , 'y0' , 'y1' , 'prisonbar' , 'policebar' , ...
    'urbar' , 'incbar' , 'povbar' , 'afdcbar' , 'gunbar' , 'beerbar' , ...
    'prison85' , 'police85' , ...
    'ur85' , 'inc85' , 'pov85' , 'afdc85' , 'gun85' , 'beer85'}';

Time = recode(year08);
state = recode(state08);
div = recode(div08);

%% Generate nonlinear trends for controls
Ltrend = Time/max(Time);  % Normalize to unit interval

trend = Ltrend.^2;
for jj = 3:5
    trend = [trend Ltrend.^jj]; %#ok<AGROW>
end

for jj = 1:4
    trend = [trend sin((pi*jj)*Ltrend)]; %#ok<AGROW>
end

for jj = 1:4
    trend = [trend cos((pi*jj)*Ltrend)]; %#ok<AGROW>
end

NameTrend = {'P2' , 'P3' , 'P4' , 'P5' , ...
    'sin1' , 'sin2' , 'sin3' , 'sin4' , ...
    'cos1' , 'cos2' , 'cos3' , 'cos4' ...
    }';


%% Generate interactions
N08 = size(xxP,1);
k0 = size(xxP,2);
kT = size(trend,2);
BigP = zeros(N08,k0*kT);
BigV = zeros(N08,k0*kT);
BigM = zeros(N08,k0*kT);
BigName = cell(k0*kT,1);
for ii = 1:k0
    for jj = 1:kT
        CurrCol = kT*(ii-1)+jj;
        BigP(:,CurrCol) = xxP(:,ii).*trend(:,jj);
        BigV(:,CurrCol) = xxV(:,ii).*trend(:,jj);
        BigM(:,CurrCol) = xxM(:,ii).*trend(:,jj);
        CurrName = strcat(NameInit{ii},NameTrend{jj});
        BigName{CurrCol} = CurrName;
    end
end

%% All models include state effects, state-specific linear trends, and a full set of district x time trends
[ydum,~,yxddum] = dummy(Time,div);
sdum = dummyvar(state);
% Need to leave one state out from each division when forming
% state-specific trends to avoid perfect collinearity with district x year
% effects
% Also going to leave out one state dummy and put in an intercept to make
% keeping track of indices easier later
sdumUse = ones(48,1);
sdumUse(4) = 0;
strendUse = ones(48,1);
strendUse(4) = 0;
strendUse(6) = 0;   % division 1
strendUse(28) = 0;  % division 2
strendUse(11) = 0;  % division 3
strendUse(13) = 0;  % division 4
strendUse(7) = 0;   % division 5
strendUse(1) = 0;   % division 6
strendUse(3) = 0;   % division 7
strendUse(2) = 0;   % division 8
strendUse(35) = 0;  % division 9
sdumUse = logical(sdumUse);
strendUse = logical(strendUse);
dums = [sdum(:,sdumUse) , sdum(:,strendUse).*(Ltrend*ones(1,sum(strendUse))) , ...
    ydum , yxddum , ones(N08,1)];
kDums = size(dums,2);
NameInitZ = {'s1'};
for ii = 2:48
    if sdumUse(ii) > 0
        NameInitZ{end+1,1} = strcat('s',num2str(ii)); %#ok<SAGROW>
    end
end
for ii = 1:48
    if strendUse(ii) > 0
        NameInitZ{end+1,1} = strcat('s',num2str(ii),'*t'); %#ok<SAGROW>
    end
end
for ii = 1:size(ydum,2)
    NameInitZ{end+1,1} = strcat('y',num2str(ii)); %#ok<SAGROW>
end    
for ii = 1:size(yxddum,2)
    NameInitZ{end+1,1} = strcat('y*d',num2str(ii)); %#ok<SAGROW>
end    

%% Partial these effects out from everything
EyP = yprop08 - dums*((dums'*dums)\(dums'*yprop08));
EyV = yviol08 - dums*((dums'*dums)\(dums'*yviol08));
EyM = ymurd08 - dums*((dums'*dums)\(dums'*ymurd08));
ExP = efaprop08 - dums*((dums'*dums)\(dums'*efaprop08));
ExV = efaviol08 - dums*((dums'*dums)\(dums'*efaviol08));
ExM = efamurd08 - dums*((dums'*dums)\(dums'*efamurd08));
EBigP = BigP - dums*((dums'*dums)\(dums'*BigP));
EBigV = BigV - dums*((dums'*dums)\(dums'*BigV));
EBigM = BigM - dums*((dums'*dums)\(dums'*BigM));

I = (std(EBigP) > 1e-8);
EBigP = EBigP(:,I);
EBigV = EBigV(:,I);
EBigM = EBigM(:,I);
BigName = BigName(I);

EBigP = zscore(EBigP);
EBigV = zscore(EBigV);
EBigM = zscore(EBigM);

kzP = size(EBigP,2);
kzV = size(EBigV,2);
kzM = size(EBigM,2);

%% FE Regression, DL Specification
b08Prop = ExP\EyP;
b08Viol = ExV\EyV;
b08Murd = ExM\EyM;

s08Prop = cluster_se(ExP,EyP-b08Prop*ExP,inv(ExP'*ExP),state,kDums+1);
s08Viol = cluster_se(ExV,EyV-b08Viol*ExV,inv(ExV'*ExV),state,kDums+1);
s08Murd = cluster_se(ExM,EyM-b08Murd*ExM,inv(ExM'*ExM),state,kDums+1);

% Display results
disp('Number of variables used');
disp(kDums)
disp('Violence - Level')
disp([b08Viol(1,1) s08Viol(1,1)])
disp(' ')
disp('Property - Level')
disp([b08Prop(1,1) s08Prop(1,1)])
disp(' ')
disp('Murder - Level')
disp([b08Murd(1,1) s08Murd(1,1)])
disp(' ')


%% Kitchen Sink Regression
EExP = ExP - EBigP*((EBigP'*EBigP)\(EBigP'*ExP));
EExV = ExV - EBigV*((EBigV'*EBigV)\(EBigV'*ExV));
EExM = ExM - EBigM*((EBigM'*EBigM)\(EBigM'*ExM));
EEyP = EyP - EBigP*((EBigP'*EBigP)\(EBigP'*EyP));
EEyV = EyV - EBigV*((EBigV'*EBigV)\(EBigV'*EyV));
EEyM = EyM - EBigM*((EBigM'*EBigM)\(EBigM'*EyM));

b08PropKS = EExP\EEyP;
b08ViolKS = EExV\EEyV;
b08MurdKS = EExM\EEyM;

s08PropKS = cluster_se(EExP,EEyP-b08Prop*EExP,inv(EExP'*EExP),state,kDums+1+kzP);
s08ViolKS = cluster_se(EExV,EEyV-b08Viol*EExV,inv(EExV'*EExV),state,kDums+1+kzV);
s08MurdKS = cluster_se(EExM,EEyM-b08Murd*EExM,inv(EExM'*EExM),state,kDums+1+kzM);

disp('No Selection - 2008')
disp('Number of variables used');
disp(kDums+[kzV kzP kzM])
disp('Violence')
disp([b08ViolKS(1) s08ViolKS(1)])
disp('Property')
disp([b08PropKS(1) s08PropKS(1)])
disp('Murder')
disp([b08MurdKS(1) s08MurdKS(1)])

%% (3) LASSO in 2008 data forcing all DL 2008 variables to enter model
%% Initialize a couple things for LASSO
kV = kzV + kDums;
kP = kzP + kDums;
kM = kzM + kDums;

xV = ExV;
xP = ExP;
xM = ExM;
yV = EyV;
yP = EyP;
yM = EyM;
zV = EBigV;
zP = EBigP;
zM = EBigM;

InitResidV = xV;
InitResidP = xP;
InitResidM = xM;

InitResidyV = yV;
InitResidyP = yP;
InitResidyM = yM;

maxIter = 100;

%% LASSO - Violence
% Prediction of variable of interest
StV = zV.*(InitResidV*ones(1,kzV));
UpsV = sqrt(.1*sum(StV.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kV/.05))));
PInitV = LassoShooting(zV./(ones(N08,1)*UpsV), xV , lambda, 'Verbose', 0);
IndInitV = abs(PInitV) > 0;
ZV = zV(:,IndInitV);
RefResidV = xV-ZV*((ZV'*ZV)\(ZV'*xV));

kk = 1;
StRefV = zV.*(RefResidV*ones(1,kzV));
UpsRefV = sqrt(sum(StRefV.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kV/.05))));
lastNorm = Inf;
while norm(UpsRefV-UpsV) > .02 && kk < maxIter && norm(UpsRefV-UpsV) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefV-UpsV);
    disp(lastNorm);
    PRefV = LassoShooting(zV./(ones(N08,1)*UpsRefV), xV , lambda, 'Verbose', 0);
    IndRefV = abs(PRefV) > 0;
    ZV = zV(:,IndRefV);
    bfsV = ((ZV'*ZV)\(ZV'*xV));
    efsV = xV-ZV*bfsV;
    UpsV = UpsRefV;
    StRefV = zV.*(efsV*ones(1,kzV));
    UpsRefV = sqrt(sum(StRefV.^2)/(N08));    
    kk = kk+1;
end
sfsV = cluster_se(ZV,efsV,inv(ZV'*ZV),state);

% Structural equation
StyV = zV.*(InitResidyV*ones(1,kzV));
UpsyV = sqrt(.1*sum(StyV.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kV/.05))));
PInityV = LassoShooting(zV./(ones(N08,1)*UpsyV), yV , lambda, 'Verbose', 0);
IndInityV = abs(PInityV) > 0;
ZyV = zV(:,IndInityV);
RefResidyV = yV-ZyV*((ZyV'*ZyV)\(ZyV'*yV));

kk = 1;
StRefyV = zV.*(RefResidyV*ones(1,kzV));
UpsRefyV = sqrt(sum(StRefyV.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kV/.05))));
lastNorm = Inf;
while norm(UpsRefyV-UpsyV) > 1e-4 && kk < maxIter && norm(UpsRefyV-UpsyV) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefyV-UpsyV);
    disp(lastNorm);
    PRefyV = LassoShooting(zV./(ones(N08,1)*UpsRefyV), yV , lambda, 'Verbose', 0);
    IndRefyV = abs(PRefyV) > 0;
    ZyV = zV(:,IndRefyV);
    bfsyV = ((ZyV'*ZyV)\(ZyV'*yV));
    efsyV = yV-ZyV*bfsyV;
    UpsyV = UpsRefyV;
    StRefyV = zV.*(efsyV*ones(1,kzV));
    UpsRefyV = sqrt(sum(StRefyV.^2)/(N08));    
    kk = kk+1;
end
sfsyV = cluster_se(ZyV,efsyV,inv(ZyV'*ZyV),state);

IndUnionV = max([IndRefyV IndRefV],[],2);
ZUnionV = zV(:,IndUnionV);

R2_1V = sum((ZUnionV*(ZUnionV\xV)).^2)/(sum(xV.^2));
R2_2V = sum((ZUnionV*(ZUnionV\yV)).^2)/(sum(yV.^2)); 

zVLASSO = [xV ZUnionV];
bviolLASSO = (zVLASSO'*zVLASSO)\(zVLASSO'*yV);
eviolLASSO = yV - zVLASSO*bviolLASSO;
ResviolLASSO = cluster_se(zVLASSO,eviolLASSO,inv(zVLASSO'*zVLASSO),state,kDums+sum(IndUnionV));

disp('Violence - Level - LASSO')
disp('Selected Variables - Abortion')
disp(BigName(IndRefV))
disp(' ');
disp('Selected Variables - Crime')
disp(BigName(IndRefyV))
disp(' ');
disp('Abortion equation')
disp([bfsV sfsV])
disp('R^2')
disp(R2_1V)
disp(' ')
disp('Crime equation')
disp([bfsyV sfsyV])
disp('R^2')
disp(R2_2V)
disp(' ')
disp('Number of Selected Variables')
disp(kDums+sum(IndUnionV));
disp(' ')
disp('Treatment Effect')
disp([bviolLASSO(1) ResviolLASSO(1)])
disp(' ')


%% LASSO - Property
% Prediction of variable of interest
StP = zP.*(InitResidP*ones(1,kzP));
UpsP = sqrt(.1*sum(StP.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kP/.05))));
PInitP = LassoShooting(zP./(ones(N08,1)*UpsP), xP , lambda, 'Verbose', 0);
IndInitP = abs(PInitP) > 0;
ZP = zP(:,IndInitP);
RefResidP = xP-ZP*((ZP'*ZP)\(ZP'*xP));

kk = 1;
StRefP = zP.*(RefResidP*ones(1,kzP));
UpsRefP = sqrt(sum(StRefP.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kP/.05))));
lastNorm = Inf;
while norm(UpsRefP-UpsP) > 1e-4 && kk < maxIter && norm(UpsRefP-UpsP) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefP-UpsP);
    disp(lastNorm);
    PRefP = LassoShooting(zP./(ones(N08,1)*UpsRefP), xP , lambda, 'Verbose', 0);
    IndRefP = abs(PRefP) > 0;
    ZP = zP(:,IndRefP);
    bfsP = ((ZP'*ZP)\(ZP'*xP));
    efsP = xP-ZP*bfsP;
    UpsP = UpsRefP;
    StRefP = zP.*(efsP*ones(1,kzP));
    UpsRefP = sqrt(sum(StRefP.^2)/(N08));    
    kk = kk+1;
end
sfsP = cluster_se(ZP,efsP,inv(ZP'*ZP),state);

% Structural equation
StyP = zP.*(InitResidyP*ones(1,kzP));
UpsyP = sqrt(.1*sum(StyP.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kP/.05))));
PInityP = LassoShooting(zP./(ones(N08,1)*UpsyP), yP , lambda, 'Verbose', 0);
IndInityP = abs(PInityP) > 0;
ZyP = zP(:,IndInityP);
RefResidyP = yP-ZyP*((ZyP'*ZyP)\(ZyP'*yP));

kk = 1;
StRefyP = zP.*(RefResidyP*ones(1,kzP));
UpsRefyP = sqrt(sum(StRefyP.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kP/.05))));
lastNorm = Inf;
while norm(UpsRefyP-UpsyP) > 1e-4 && kk < maxIter && norm(UpsRefyP-UpsyP) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefyP-UpsyP);
    disp(lastNorm);
    PRefyP = LassoShooting(zP./(ones(N08,1)*UpsRefyP), yP , lambda, 'Verbose', 0);
    IndRefyP = abs(PRefyP) > 0;
    ZyP = zP(:,IndRefyP);
    bfsyP = ((ZyP'*ZyP)\(ZyP'*yP));
    efsyP = yP-ZyP*bfsyP;
    UpsyP = UpsRefyP;
    StRefyP = zP.*(efsyP*ones(1,kzP));
    UpsRefyP = sqrt(sum(StRefyP.^2)/(N08));    
    kk = kk+1;
end
sfsyP = cluster_se(ZyP,efsyP,inv(ZyP'*ZyP),state);

IndUnionP = max([IndRefyP IndRefP],[],2);
ZUnionP = zP(:,IndUnionP);

R2_1P = sum((ZUnionP*(ZUnionP\xP)).^2)/(sum(xP.^2));
R2_2P = sum((ZUnionP*(ZUnionP\yP)).^2)/(sum(yP.^2)); 

zPLASSO = [xP ZUnionP];
bpropLASSO = (zPLASSO'*zPLASSO)\(zPLASSO'*yP);
epropLASSO = yP - zPLASSO*bpropLASSO;
RespropLASSO = cluster_se(zPLASSO,epropLASSO,inv(zPLASSO'*zPLASSO),state,kDums+sum(IndUnionP));

disp('Property - Level - LASSO')
disp('Selected Variables - Abortion')
disp(BigName(IndRefP))
disp(' ');
disp('Selected Variables - Crime')
disp(BigName(IndRefyP))
disp(' ');
disp('Abortion equation')
disp([bfsP sfsP])
disp('R^2')
disp(R2_1P)
disp(' ')
disp('Crime equation')
disp([bfsyP sfsyP])
disp('R^2')
disp(R2_2P)
disp(' ')
disp('Number of Selected Variables')
disp(kDums+sum(IndUnionP));
disp(' ')
disp('Treatment Effect')
disp([bpropLASSO(1) RespropLASSO(1)])
disp(' ')


%% LASSO - Murder
% Prediction of variable of interest
StM = zM.*(InitResidM*ones(1,kzM));
UpsM = sqrt(.1*sum(StM.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kM/.05))));
PInitM = LassoShooting(zM./(ones(N08,1)*UpsM), xM , lambda, 'Verbose', 0);
IndInitM = abs(PInitM) > 0;
ZM = zM(:,IndInitM);
RefResidM = xM-ZM*((ZM'*ZM)\(ZM'*xM));

kk = 1;
StRefM = zM.*(RefResidM*ones(1,kzM));
UpsRefM = sqrt(sum(StRefM.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kM/.05))));
lastNorm = Inf;
while norm(UpsRefM-UpsM) > .007 && kk < maxIter && norm(UpsRefM-UpsM) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefM-UpsM);
    disp(lastNorm);
    PRefM = LassoShooting(zM./(ones(N08,1)*UpsRefM), xM , lambda, 'Verbose', 0);
    IndRefM = abs(PRefM) > 0;
    ZM = zM(:,IndRefM);
    bfsM = ((ZM'*ZM)\(ZM'*xM));
    efsM = xM-ZM*bfsM;
    UpsM = UpsRefM;
    StRefM = zM.*(efsM*ones(1,kzM));
    UpsRefM = sqrt(sum(StRefM.^2)/(N08));    
    kk = kk+1;
end
sfsM = cluster_se(ZM,efsM,inv(ZM'*ZM),state);

% Structural equation
StyM = zM.*(InitResidyM*ones(1,kzM));
UpsyM = sqrt(.1*sum(StyM.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kM/.05))));
PInityM = LassoShooting(zM./(ones(N08,1)*UpsyM), yM , lambda, 'Verbose', 0);
IndInityM = abs(PInityM) > 0;
ZyM = zM(:,IndInityM);
RefResidyM = yM-ZyM*((ZyM'*ZyM)\(ZyM'*yM));

kk = 1;
StRefyM = zM.*(RefResidyM*ones(1,kzM));
UpsRefyM = sqrt(sum(StRefyM.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kM/.05))));
lastNorm = Inf;
while norm(UpsRefyM-UpsyM) > 1e-4 && kk < maxIter && norm(UpsRefyM-UpsyM) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefyM-UpsyM);
    disp(lastNorm);
    PRefyM = LassoShooting(zM./(ones(N08,1)*UpsRefyM), yM , lambda, 'Verbose', 0);
    IndRefyM = abs(PRefyM) > 0;
    ZyM = zM(:,IndRefyM);
    bfsyM = ((ZyM'*ZyM)\(ZyM'*yM));
    efsyM = yM-ZyM*bfsyM;
    UpsyM = UpsRefyM;
    StRefyM = zM.*(efsyM*ones(1,kzM));
    UpsRefyM = sqrt(sum(StRefyM.^2)/(N08));    
    kk = kk+1;
end
sfsyM = cluster_se(ZyM,efsyM,inv(ZyM'*ZyM),state);

IndUnionM = max([IndRefyM IndRefM],[],2);
ZUnionM = zM(:,IndUnionM);

R2_1M = sum((ZUnionM*(ZUnionM\xM)).^2)/(sum(xM.^2));
R2_2M = sum((ZUnionM*(ZUnionM\yM)).^2)/(sum(yM.^2)); 

zMLASSO = [xM ZUnionM];
bmurdLASSO = (zMLASSO'*zMLASSO)\(zMLASSO'*yM);
emurdLASSO = yM - zMLASSO*bmurdLASSO;
ResmurdLASSO = cluster_se(zMLASSO,emurdLASSO,inv(zMLASSO'*zMLASSO),state,kDums+sum(IndUnionM));

disp('Murder - Level - LASSO')
disp('Selected Variables - Abortion')
disp(BigName(IndRefM))
disp(' ');
disp('Selected Variables - Crime')
disp(BigName(IndRefyM))
disp(' ');
disp('Abortion equation')
disp([bfsM sfsM])
disp('R^2')
disp(R2_1M)
disp(' ')
disp('Crime equation')
disp([bfsyM sfsyM])
disp('R^2')
disp(R2_2M)
disp(' ')
disp('Number of Selected Variables')
disp(kDums+sum(IndUnionM));
disp(' ')
disp('Treatment Effect')
disp([bmurdLASSO(1) ResmurdLASSO(1)])
disp(' ')


%% (4) LASSO in 2008 data selecting over ALL variables
zV = zscore([BigV dums(:,1:end-1)]);
zP = zscore([BigP dums(:,1:end-1)]);
zM = zscore([BigM dums(:,1:end-1)]);
xV = efaviol08-mean(efaviol08);
xP = efaprop08-mean(efaprop08);
xM = efamurd08-mean(efamurd08);
yV = yviol08-mean(yviol08);
yP = yprop08-mean(yprop08);
yM = ymurd08-mean(ymurd08);

kzV = size(zM,2);
kV = size(zM,2);
kzP = size(zM,2);
kP = size(zM,2);
kzM = size(zM,2);
kM = size(zM,2);
kDums = 1;

BigName = [BigName ; NameInitZ];

%% LASSO - Violence
% Prediction of variable of interest
StV = zV.*(InitResidV*ones(1,kzV));
UpsV = sqrt(.1*sum(StV.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kV/.05))));
PInitV = LassoShooting(zV./(ones(N08,1)*UpsV), xV , lambda, 'Verbose', 0);
IndInitV = abs(PInitV) > 0;
ZV = zV(:,IndInitV);
RefResidV = xV-ZV*((ZV'*ZV)\(ZV'*xV));

kk = 1;
StRefV = zV.*(RefResidV*ones(1,kzV));
UpsRefV = sqrt(sum(StRefV.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kV/.05))));
lastNorm = Inf;
while norm(UpsRefV-UpsV) > .02 && kk < maxIter && norm(UpsRefV-UpsV) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefV-UpsV);
    disp(lastNorm);
    PRefV = LassoShooting(zV./(ones(N08,1)*UpsRefV), xV , lambda, 'Verbose', 0);
    IndRefV = abs(PRefV) > 0;
    ZV = zV(:,IndRefV);
    bfsV = ((ZV'*ZV)\(ZV'*xV));
    efsV = xV-ZV*bfsV;
    UpsV = UpsRefV;
    StRefV = zV.*(efsV*ones(1,kzV));
    UpsRefV = sqrt(sum(StRefV.^2)/(N08));    
    kk = kk+1;
end
sfsV = cluster_se(ZV,efsV,inv(ZV'*ZV),state);

% Structural equation
StyV = zV.*(InitResidyV*ones(1,kzV));
UpsyV = sqrt(.1*sum(StyV.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kV/.05))));
PInityV = LassoShooting(zV./(ones(N08,1)*UpsyV), yV , lambda, 'Verbose', 0);
IndInityV = abs(PInityV) > 0;
ZyV = zV(:,IndInityV);
RefResidyV = yV-ZyV*((ZyV'*ZyV)\(ZyV'*yV));

kk = 1;
StRefyV = zV.*(RefResidyV*ones(1,kzV));
UpsRefyV = sqrt(sum(StRefyV.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kV/.05))));
lastNorm = Inf;
while norm(UpsRefyV-UpsyV) > 1e-4 && kk < maxIter && norm(UpsRefyV-UpsyV) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefyV-UpsyV);
    disp(lastNorm);
    PRefyV = LassoShooting(zV./(ones(N08,1)*UpsRefyV), yV , lambda, 'Verbose', 0);
    IndRefyV = abs(PRefyV) > 0;
    ZyV = zV(:,IndRefyV);
    bfsyV = ((ZyV'*ZyV)\(ZyV'*yV));
    efsyV = yV-ZyV*bfsyV;
    UpsyV = UpsRefyV;
    StRefyV = zV.*(efsyV*ones(1,kzV));
    UpsRefyV = sqrt(sum(StRefyV.^2)/(N08));    
    kk = kk+1;
end
sfsyV = cluster_se(ZyV,efsyV,inv(ZyV'*ZyV),state);

IndUnionV = max([IndRefyV IndRefV],[],2);
ZUnionV = zV(:,IndUnionV);

R2_1V = sum((ZUnionV*(ZUnionV\xV)).^2)/(sum(xV.^2));
R2_2V = sum((ZUnionV*(ZUnionV\yV)).^2)/(sum(yV.^2)); 

zVLASSO = [xV ZUnionV];
bviolLASSO = (zVLASSO'*zVLASSO)\(zVLASSO'*yV);
eviolLASSO = yV - zVLASSO*bviolLASSO;
ResviolLASSO = cluster_se(zVLASSO,eviolLASSO,inv(zVLASSO'*zVLASSO),state,kDums+sum(IndUnionV));

disp('Violence - Level - LASSO')
disp('Selected Variables - Abortion')
disp(BigName(IndRefV))
disp(' ');
disp('Selected Variables - Crime')
disp(BigName(IndRefyV))
disp(' ');
disp('Abortion equation')
disp([bfsV sfsV])
disp('R^2')
disp(R2_1V)
disp(' ')
disp('Crime equation')
disp([bfsyV sfsyV])
disp('R^2')
disp(R2_2V)
disp(' ')
disp('Number of Selected Variables')
disp(kDums+sum(IndUnionV));
disp(' ')
disp('Treatment Effect')
disp([bviolLASSO(1) ResviolLASSO(1)])
disp(' ')


%% LASSO - Property
% Prediction of variable of interest
StP = zP.*(InitResidP*ones(1,kzP));
UpsP = sqrt(.1*sum(StP.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kP/.05))));
PInitP = LassoShooting(zP./(ones(N08,1)*UpsP), xP , lambda, 'Verbose', 0);
IndInitP = abs(PInitP) > 0;
ZP = zP(:,IndInitP);
RefResidP = xP-ZP*((ZP'*ZP)\(ZP'*xP));

kk = 1;
StRefP = zP.*(RefResidP*ones(1,kzP));
UpsRefP = sqrt(sum(StRefP.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kP/.05))));
lastNorm = Inf;
while norm(UpsRefP-UpsP) > 1e-4 && kk < maxIter && norm(UpsRefP-UpsP) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefP-UpsP);
    disp(lastNorm);
    PRefP = LassoShooting(zP./(ones(N08,1)*UpsRefP), xP , lambda, 'Verbose', 0);
    IndRefP = abs(PRefP) > 0;
    ZP = zP(:,IndRefP);
    bfsP = ((ZP'*ZP)\(ZP'*xP));
    efsP = xP-ZP*bfsP;
    UpsP = UpsRefP;
    StRefP = zP.*(efsP*ones(1,kzP));
    UpsRefP = sqrt(sum(StRefP.^2)/(N08));    
    kk = kk+1;
end
sfsP = cluster_se(ZP,efsP,inv(ZP'*ZP),state);

% Structural equation
StyP = zP.*(InitResidyP*ones(1,kzP));
UpsyP = sqrt(.1*sum(StyP.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kP/.05))));
PInityP = LassoShooting(zP./(ones(N08,1)*UpsyP), yP , lambda, 'Verbose', 0);
IndInityP = abs(PInityP) > 0;
ZyP = zP(:,IndInityP);
RefResidyP = yP-ZyP*((ZyP'*ZyP)\(ZyP'*yP));

kk = 1;
StRefyP = zP.*(RefResidyP*ones(1,kzP));
UpsRefyP = sqrt(sum(StRefyP.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kP/.05))));
lastNorm = Inf;
while norm(UpsRefyP-UpsyP) > 1e-4 && kk < maxIter && norm(UpsRefyP-UpsyP) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefyP-UpsyP);
    disp(lastNorm);
    PRefyP = LassoShooting(zP./(ones(N08,1)*UpsRefyP), yP , lambda, 'Verbose', 0);
    IndRefyP = abs(PRefyP) > 0;
    ZyP = zP(:,IndRefyP);
    bfsyP = ((ZyP'*ZyP)\(ZyP'*yP));
    efsyP = yP-ZyP*bfsyP;
    UpsyP = UpsRefyP;
    StRefyP = zP.*(efsyP*ones(1,kzP));
    UpsRefyP = sqrt(sum(StRefyP.^2)/(N08));    
    kk = kk+1;
end
sfsyP = cluster_se(ZyP,efsyP,inv(ZyP'*ZyP),state);

IndUnionP = max([IndRefyP IndRefP],[],2);
ZUnionP = zP(:,IndUnionP);

R2_1P = sum((ZUnionP*(ZUnionP\xP)).^2)/(sum(xP.^2));
R2_2P = sum((ZUnionP*(ZUnionP\yP)).^2)/(sum(yP.^2)); 

zPLASSO = [xP ZUnionP];
bpropLASSO = (zPLASSO'*zPLASSO)\(zPLASSO'*yP);
epropLASSO = yP - zPLASSO*bpropLASSO;
RespropLASSO = cluster_se(zPLASSO,epropLASSO,inv(zPLASSO'*zPLASSO),state,kDums+sum(IndUnionP));

disp('Property - Level - LASSO')
disp('Selected Variables - Abortion')
disp(BigName(IndRefP))
disp(' ');
disp('Selected Variables - Crime')
disp(BigName(IndRefyP))
disp(' ');
disp('Abortion equation')
disp([bfsP sfsP])
disp('R^2')
disp(R2_1P)
disp(' ')
disp('Crime equation')
disp([bfsyP sfsyP])
disp('R^2')
disp(R2_2P)
disp(' ')
disp('Number of Selected Variables')
disp(kDums+sum(IndUnionP));
disp(' ')
disp('Treatment Effect')
disp([bpropLASSO(1) RespropLASSO(1)])
disp(' ')


%% LASSO - Murder
% Prediction of variable of interest
StM = zM.*(InitResidM*ones(1,kzM));
UpsM = sqrt(.1*sum(StM.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kM/.05))));
PInitM = LassoShooting(zM./(ones(N08,1)*UpsM), xM , lambda, 'Verbose', 0);
IndInitM = abs(PInitM) > 0;
ZM = zM(:,IndInitM);
RefResidM = xM-ZM*((ZM'*ZM)\(ZM'*xM));

kk = 1;
StRefM = zM.*(RefResidM*ones(1,kzM));
UpsRefM = sqrt(sum(StRefM.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kM/.05))));
lastNorm = Inf;
while norm(UpsRefM-UpsM) > .007 && kk < maxIter && norm(UpsRefM-UpsM) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefM-UpsM);
    disp(lastNorm);
    PRefM = LassoShooting(zM./(ones(N08,1)*UpsRefM), xM , lambda, 'Verbose', 0);
    IndRefM = abs(PRefM) > 0;
    ZM = zM(:,IndRefM);
    bfsM = ((ZM'*ZM)\(ZM'*xM));
    efsM = xM-ZM*bfsM;
    UpsM = UpsRefM;
    StRefM = zM.*(efsM*ones(1,kzM));
    UpsRefM = sqrt(sum(StRefM.^2)/(N08));    
    kk = kk+1;
end
sfsM = cluster_se(ZM,efsM,inv(ZM'*ZM),state);

% Structural equation
StyM = zM.*(InitResidyM*ones(1,kzM));
UpsyM = sqrt(.1*sum(StyM.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kM/.05))));
PInityM = LassoShooting(zM./(ones(N08,1)*UpsyM), yM , lambda, 'Verbose', 0);
IndInityM = abs(PInityM) > 0;
ZyM = zM(:,IndInityM);
RefResidyM = yM-ZyM*((ZyM'*ZyM)\(ZyM'*yM));

kk = 1;
StRefyM = zM.*(RefResidyM*ones(1,kzM));
UpsRefyM = sqrt(sum(StRefyM.^2)/(N08));
lambda = 1.1*2*sqrt(2*N08*(log(2*(kM/.05))));
lastNorm = Inf;
while norm(UpsRefyM-UpsyM) > 1e-4 && kk < maxIter && norm(UpsRefyM-UpsyM) - lastNorm ~= 0,
    disp(kk);
    lastNorm = norm(UpsRefyM-UpsyM);
    disp(lastNorm);
    PRefyM = LassoShooting(zM./(ones(N08,1)*UpsRefyM), yM , lambda, 'Verbose', 0);
    IndRefyM = abs(PRefyM) > 0;
    ZyM = zM(:,IndRefyM);
    bfsyM = ((ZyM'*ZyM)\(ZyM'*yM));
    efsyM = yM-ZyM*bfsyM;
    UpsyM = UpsRefyM;
    StRefyM = zM.*(efsyM*ones(1,kzM));
    UpsRefyM = sqrt(sum(StRefyM.^2)/(N08));    
    kk = kk+1;
end
sfsyM = cluster_se(ZyM,efsyM,inv(ZyM'*ZyM),state);

IndUnionM = max([IndRefyM IndRefM],[],2);
ZUnionM = zM(:,IndUnionM);

R2_1M = sum((ZUnionM*(ZUnionM\xM)).^2)/(sum(xM.^2));
R2_2M = sum((ZUnionM*(ZUnionM\yM)).^2)/(sum(yM.^2)); 

zMLASSO = [xM ZUnionM];
bmurdLASSO = (zMLASSO'*zMLASSO)\(zMLASSO'*yM);
emurdLASSO = yM - zMLASSO*bmurdLASSO;
ResmurdLASSO = cluster_se(zMLASSO,emurdLASSO,inv(zMLASSO'*zMLASSO),state,kDums+sum(IndUnionM));

disp('Murder - Level - LASSO')
disp('Selected Variables - Abortion')
disp(BigName(IndRefM))
disp(' ');
disp('Selected Variables - Crime')
disp(BigName(IndRefyM))
disp(' ');
disp('Abortion equation')
disp([bfsM sfsM])
disp('R^2')
disp(R2_1M)
disp(' ')
disp('Crime equation')
disp([bfsyM sfsyM])
disp('R^2')
disp(R2_2M)
disp(' ')
disp('Number of Selected Variables')
disp(kDums+sum(IndUnionM));
disp(' ')
disp('Treatment Effect')
disp([bmurdLASSO(1) ResmurdLASSO(1)])
disp(' ')

diary off ;
