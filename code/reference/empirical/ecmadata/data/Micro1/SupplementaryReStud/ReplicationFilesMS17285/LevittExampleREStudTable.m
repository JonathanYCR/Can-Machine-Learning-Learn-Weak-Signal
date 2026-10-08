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
sdum   = dummyvar(state);
tdum   = dummy(year);
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

%% Estimate linear models in first differences with time effects and differenced control variables

% Generate first differences
Dyviol = reshape(diff(reshape(y_viol,T,N)),N*(T-1),1);
Dxviol = reshape(diff(reshape(x_viol,T,N)),N*(T-1),1);
Dyprop = reshape(diff(reshape(y_prop,T,N)),N*(T-1),1);
Dxprop = reshape(diff(reshape(x_prop,T,N)),N*(T-1),1);
Dymurd = reshape(diff(reshape(y_murd,T,N)),N*(T-1),1);
Dxmurd = reshape(diff(reshape(x_murd,T,N)),N*(T-1),1);
Dxx = zeros(N*(T-1),size(xx,2));
for ii = 1:size(xx,2);
    Dxx(:,ii) = reshape(diff(reshape(xx(:,ii),T,N)),N*(T-1),1);
end
t1 = find(year > 1);
Dyear = year(t1);
Dstate = state(t1);
Ddistrict = district(t1);
Dtime = recode(Dyear);
DT = max(Dtime);
xxL = xx(year < T,:);

DX = [Dxx dummyvar(recode(Dyear))];
Dk = size(DX,2) + 1;

%Partial out control variables
DXMX = DX/(DX'*DX);
Dxv = Dxviol - DXMX*(DX'*Dxviol);
Dyv = Dyviol - DXMX*(DX'*Dyviol);
Dxp = Dxprop - DXMX*(DX'*Dxprop);
Dyp = Dyprop - DXMX*(DX'*Dyprop);
Dxm = Dxmurd - DXMX*(DX'*Dxmurd);
Dym = Dymurd - DXMX*(DX'*Dymurd);

% Estimate regression coefficients and residuals
Dbv  = (Dxv'*Dxv)\(Dxv'*Dyv);
Dev    = Dyv - Dxv*Dbv;

Dbp  = (Dxp'*Dxp)\(Dxp'*Dyp);
Dep    = Dyp - Dxp*Dbp;

Dbm  = (Dxm'*Dxm)\(Dxm'*Dym);
Dem    = Dym - Dxm*Dbm;

%% Estimate standard errors
% Clustered standard errors by state
Dsv = cluster_se(Dxv,Dev,inv(Dxv'*Dxv),Dstate,Dk);
Dsp = cluster_se(Dxp,Dep,inv(Dxp'*Dxp),Dstate,Dk);
Dsm = cluster_se(Dxm,Dem,inv(Dxm'*Dxm),Dstate,Dk);

% Display results
disp('Difference Violence - Level')
disp([Dbv(1,1) Dsv(1,:)])
disp(' ')
disp('Difference Property - Level')
disp([Dbp(1,1) Dsp(1,:)])
disp(' ')
disp('Difference Murder - Level')
disp([Dbm(1,1) Dsm(1,:)])
disp(' ')

acfM = acfcomp2(Dem,20);
acfP = acfcomp2(Dep,20);
acfV = acfcomp2(Dev,20);

disp('Autocorrelation 1-3')
disp([acfV(2:4) acfP(2:4) acfM(2:4)]);
disp(' ')


%% Make a big set of potential control variables
PS = ((sdum'*sdum)\sdum');
kxx = size(Dxx,2);
TDums = dummyvar(Dtime);
Dxx(:,6) = Dxx(:,6)/10000;
xxL(:,6) = xxL(:,6)/10000;
xx(:,6) = xx(:,6)/10000;
xx(:,5) = xx(:,5)/100;
xxL(:,5) = xxL(:,5)/100;
Dxx(:,5) = Dxx(:,5)/100;
xx(:,4) = xx(:,4)/100;
xxL(:,4) = xxL(:,4)/100;
Dxx(:,4) = Dxx(:,4)/100;
xx(:,8) = xx(:,8)/100;
xxL(:,8) = xxL(:,8)/100;
Dxx(:,8) = Dxx(:,8)/100;
InitZ = Dxx;
Dtime = Dtime/DT;
Z = [InitZ xxL xxL.^2 (Dtime*ones(1,kxx)).*Dxx ((Dtime.^2)*ones(1,kxx)).*Dxx ...
    Dxx.^2 (Dtime*ones(1,kxx)).*(Dxx.^2) ((Dtime.^2)*ones(1,kxx)).*(Dxx.^2) ...
    (Dxx(:,1)*ones(1,kxx-1)).*Dxx(:,2:8) (Dxx(:,2)*ones(1,kxx-2)).*Dxx(:,3:8) ...
    (Dxx(:,3)*ones(1,kxx-3)).*Dxx(:,4:8) (Dxx(:,4)*ones(1,kxx-4)).*Dxx(:,5:8) ...
    (Dxx(:,5)*ones(1,kxx-5)).*Dxx(:,6:8) (Dxx(:,6)*ones(1,kxx-6)).*Dxx(:,7:8) ...
    (Dxx(:,7)*ones(1,kxx-7)).*Dxx(:,8) ...
    ((Dxx(:,1).*Dtime)*ones(1,kxx-1)).*Dxx(:,2:8) ...
    ((Dxx(:,2).*Dtime)*ones(1,kxx-2)).*Dxx(:,3:8) ...
    ((Dxx(:,3).*Dtime)*ones(1,kxx-3)).*Dxx(:,4:8) ...
    ((Dxx(:,4).*Dtime)*ones(1,kxx-4)).*Dxx(:,5:8) ...
    ((Dxx(:,5).*Dtime)*ones(1,kxx-5)).*Dxx(:,6:8) ...
    ((Dxx(:,6).*Dtime)*ones(1,kxx-6)).*Dxx(:,7:8) ...
    ((Dxx(:,7).*Dtime)*ones(1,kxx-7)).*Dxx(:,8) ...
    ((Dxx(:,1).*(Dtime.^2))*ones(1,kxx-1)).*Dxx(:,2:8) ...
    ((Dxx(:,2).*(Dtime.^2))*ones(1,kxx-2)).*Dxx(:,3:8) ...
    ((Dxx(:,3).*(Dtime.^2))*ones(1,kxx-3)).*Dxx(:,4:8) ...
    ((Dxx(:,4).*(Dtime.^2))*ones(1,kxx-4)).*Dxx(:,5:8) ...
    ((Dxx(:,5).*(Dtime.^2))*ones(1,kxx-5)).*Dxx(:,6:8) ...
    ((Dxx(:,6).*(Dtime.^2))*ones(1,kxx-6)).*Dxx(:,7:8) ...
    ((Dxx(:,7).*(Dtime.^2))*ones(1,kxx-7)).*Dxx(:,8) ...
    kron(Dxx(Dyear-1 == 1,[1:6,8]),ones(DT,1)) ...
    kron(Dxx(Dyear-1 == 1,[1:6,8]),ones(DT,1)).^2 ...
    kron(xx(year == 1,:),ones(DT,1)) ...
    kron(xx(year == 1,:),ones(DT,1)).^2 ...
    kron(Dxx(Dyear-1 == 1,[1:6,8]),ones(DT,1)).*(Dtime*ones(1,kxx-1)) ...
    (kron(Dxx(Dyear-1 == 1,[1:6,8]),ones(DT,1)).^2).*(Dtime*ones(1,kxx-1)) ...
    kron(xx(year == 1,:),ones(DT,1)).*(Dtime*ones(1,kxx)) ...
    (kron(xx(year == 1,:),ones(DT,1)).^2).*(Dtime*ones(1,kxx)) ...
    kron(Dxx(Dyear-1 == 1,[1:6,8]),ones(DT,1)).*((Dtime.^2)*ones(1,kxx-1)) ...
    (kron(Dxx(Dyear-1 == 1,[1:6,8]),ones(DT,1)).^2).*((Dtime.^2)*ones(1,kxx-1)) ...
    kron(xx(year == 1,:),ones(DT,1)).*((Dtime.^2)*ones(1,kxx)) ...
    (kron(xx(year == 1,:),ones(DT,1)).^2).*((Dtime.^2)*ones(1,kxx)) ...
    kron(PS*xx,ones(DT,1)) kron(PS*xx,ones(DT,1)).^2 ...
    (Dtime*ones(1,kxx)).*kron(PS*xx,ones(DT,1)) ((Dtime.^2)*ones(1,kxx)).*kron(PS*xx,ones(DT,1)) ...
    (Dtime*ones(1,kxx)).*(kron(PS*xx,ones(DT,1)).^2) ((Dtime.^2)*ones(1,kxx)).*(kron(PS*xx,ones(DT,1)).^2) ...
    ];
% Z = [Z DDYD dummyvar(Dstate)];
InitZ = InitZ - TDums*((TDums'*TDums)\(TDums'*InitZ));
NameZ = {...
    'Dprison','Dpolice','Dur','Dinc','Dpov','Dafdc','Dgun','Dbeer',...
    'Lprison','Lpolice','Lur','Linc','Lpov','Lafdc','Lgun','Lbeer',...    
    'Lprison^2','Lpolice^2','Lur^2','Linc^2','Lpov^2','Lafdc^2','Lgun^2','Lbeer^2',...    
    'Dprison*t','Dpolice*t','Dur*t','Dinc*t','Dpov*t','Dafdc*t','Dgun*t','Dbeer*t',...
    'Dprison*t^2','Dpolice*t^2','Dur*t^2','Dinc*t^2','Dpov*t^2','Dafdc*t^2','Dgun*t^2','Dbeer*t^2',...
    'Dprison^2','Dpolice^2','Dur^2','Dinc^2','Dpov^2','Dafdc^2','Dgun^2','Dbeer^2',...
    'Dprison^2*t','Dpolice^2*t','Dur^2*t','Dinc^2*t','Dpov^2*t','Dafdc^2*t','Dgun^2*t','Dbeer^2*t',...
    'Dprison^2*t^2','Dpolice^2*t^2','Dur^2*t^2','Dinc^2*t^2','Dpov^2*t^2','Dafdc^2*t^2','Dgun^2*t^2','Dbeer^2*t^2',...
    'Dprison*Dpolice','Dprison*Dur','Dprison*Dinc','Dprison*Dpov','Dprison*Dafdc','Dprison*Dgun','Dprison*Dbeer',...
    'Dpolice*Dur','Dpolice*Dinc','Dpolice*Dpov','Dpolice*Dafdc','Dpolice*Dgun','Dpolice*Dbeer',...
    'Dur*Dinc','Dur*Dpov','Dur*Dafdc','Dur*Dgun','Dur*Dbeer',...
    'Dinc*Dpov','Dinc*Dafdc','Dinc*Dgun','Dinc*Dbeer',...
    'Dpov*Dafdc','Dpov*Dgun','Dpov*Dbeer',...
    'Dafdc*Dgun','Dafdc*Dbeer',...
    'Dgun*Dbeer',...
    'Dprison*Dpolice*t','Dprison*Dur*t','Dprison*Dinc*t','Dprison*Dpov*t','Dprison*Dafdc*t','Dprison*Dgun*t','Dprison*Dbeer*t',...
    'Dpolice*Dur*t','Dpolice*Dinc*t','Dpolice*Dpov*t','Dpolice*Dafdc*t','Dpolice*Dgun*t','Dpolice*Dbeer*t',...
    'Dur*Dinc*t','Dur*Dpov*t','Dur*Dafdc*t','Dur*Dgun*t','Dur*Dbeer*t',...
    'Dinc*Dpov*t','Dinc*Dafdc*t','Dinc*Dgun*t','Dinc*Dbeer*t',...
    'Dpov*Dafdc*t','Dpov*Dgun*t','Dpov*Dbeer*t',...
    'Dafdc*Dgun*t','Dafdc*Dbeer*t',...
    'Dgun*Dbeer*t',...
    'Dprison*Dpolice*t^2','Dprison*Dur*t^2','Dprison*Dinc*t^2','Dprison*Dpov*t^2','Dprison*Dafdc*t^2','Dprison*Dgun*t^2','Dprison*Dbeer*t^2',...
    'Dpolice*Dur*t^2','Dpolice*Dinc*t^2','Dpolice*Dpov*t^2','Dpolice*Dafdc*t^2','Dpolice*Dgun*t^2','Dpolice*Dbeer*t^2',...
    'Dur*Dinc*t^2','Dur*Dpov*t^2','Dur*Dafdc*t^2','Dur*Dgun*t^2','Dur*Dbeer*t^2',...
    'Dinc*Dpov*t^2','Dinc*Dafdc*t^2','Dinc*Dgun*t^2','Dinc*Dbeer*t^2',...
    'Dpov*Dafdc*t^2','Dpov*Dgun*t^2','Dpov*Dbeer*t^2',...
    'Dafdc*Dgun*t^2','Dafdc*Dbeer*t^2',...
    'Dgun*Dbeer*t^2',...
    'Dprison0','Dpolice0','Dur0','Dinc0','Dpov0','Dafdc0','Dbeer0',...
    'Dprison0^2','Dpolice0^2','Dur0^2','Dinc0^2','Dpov0^2','Dafdc0^2','Dbeer0^2',...
    'Lprison0','Lpolice0','Lur0','Linc0','Lpov0','Lafdc0','Lgun0','Lbeer0',...    
    'Lprison0^2','Lpolice0^2','Lur0^2','Linc0^2','Lpov0^2','Lafdc0^2','Lgun0^2','Lbeer0^2',...    
    'Dprison0*t','Dpolice0*t','Dur0*t','Dinc0*t','Dpov0*t','Dafdc0*t','Dbeer0*t',...
    'Dprison0^2*t','Dpolice0^2*t','Dur0^2*t','Dinc0^2*t','Dpov0^2*t','Dafdc0^2*t','Dbeer0^2*t',...
    'Lprison0*t','Lpolice0*t','Lur0*t','Linc0*t','Lpov0*t','Lafdc0*t','Lgun0*t','Lbeer0*t',...    
    'Lprison0^2*t','Lpolice0^2*t','Lur0^2*t','Linc0^2*t','Lpov0^2*t','Lafdc0^2*t','Lgun0^2*t','Lbeer0^2*t',...    
    'Dprison0*t^2','Dpolice0*t^2','Dur0*t^2','Dinc0*t^2','Dpov0*t^2','Dafdc0*t^2','Dbeer0*t^2',...
    'Dprison0^2*t^2','Dpolice0^2*t^2','Dur0^2*t^2','Dinc0^2*t^2','Dpov0^2*t^2','Dafdc0^2*t^2','Dbeer0^2*t^2',...
    'Lprison0*t^2','Lpolice0*t^2','Lur0*t^2','Linc0*t^2','Lpov0*t^2','Lafdc0*t^2','Lgun0*t^2','Lbeer0*t^2',...    
    'Lprison0^2*t^2','Lpolice0^2*t^2','Lur0^2*t^2','Linc0^2*t^2','Lpov0^2*t^2','Lafdc0^2*t^2','Lgun0^2*t^2','Lbeer0^2*t^2',...    
    'prisonBar','policeBar','urBar','incBar','povBar','afdcBar','gunBar','beerBar',...    
    'prisonBar^2','policeBar^2','urBar^2','incBar^2','povBar^2','afdcBar^2','gunBar^2','beerBar^2',...    
    'prisonBar*t','policeBar*t','urBar*t','incBar*t','povBar*t','afdcBar*t','gunBar*t','beerBar*t',...    
    'prisonBar*t^2','policeBar*t^2','urBar*t^2','incBar*t^2','povBar*t^2','afdcBar*t^2','gunBar*t^2','beerBar*t^2',...    
    'prisonBar^2*t','policeBar^2*t','urBar^2*t','incBar^2*t','povBar^2*t','afdcBar^2*t','gunBar^2*t','beerBar^2*t',...    
    'prisonBar^2*t^2','policeBar^2*t^2','urBar^2*t^2','incBar^2*t^2','povBar^2*t^2','afdcBar^2*t^2','gunBar^2*t^2','beerBar^2*t^2',...    
    }';
% for ii = 1:size([DDYD dummyvar(Dstate)],2)
%     NameZ{end+1,1} = num2str(ii); %#ok<SAGROW>
% end

Zviol = [Z kron(Dxviol(Dyear - 1 == 1),ones(DT,1)) kron(Dxviol(Dyear - 1 == 1),ones(DT,1)).^2 ...
    kron(Dxviol(Dyear - 1 == 1),ones(DT,1)).*Dtime (kron(Dxviol(Dyear - 1 == 1),ones(DT,1)).^2).*Dtime ...
    kron(Dxviol(Dyear - 1 == 1),ones(DT,1)).*(Dtime.^2) (kron(Dxviol(Dyear - 1 == 1),ones(DT,1)).^2).*(Dtime.^2) ...
    kron(x_viol(year == 1),ones(DT,1)) kron(x_viol(year == 1),ones(DT,1)).^2 ...
    kron(x_viol(year == 1),ones(DT,1)).*Dtime (kron(x_viol(year == 1),ones(DT,1)).^2).*Dtime ...
    kron(x_viol(year == 1),ones(DT,1)).*(Dtime.^2) (kron(x_viol(year == 1),ones(DT,1)).^2).*(Dtime.^2) ...
    ];
NameZviol = {...
    'DxV0','DxV0^2','DxV0*t','DxV0^2*t','DxV0*t^2','DxV0^2*t^2',...
    'xV0','xV0^2','xV0*t','xV0^2*t','xV0*t^2','xV0^2*t^2',...
    }';
NameZviol = [NameZ;NameZviol];

Zprop = [Z kron(Dxprop(Dyear - 1 == 1),ones(DT,1)) kron(Dxprop(Dyear - 1 == 1),ones(DT,1)).^2 ...
    kron(Dxprop(Dyear - 1 == 1),ones(DT,1)).*Dtime (kron(Dxprop(Dyear - 1 == 1),ones(DT,1)).^2).*Dtime ...
    kron(Dxprop(Dyear - 1 == 1),ones(DT,1)).*(Dtime.^2) (kron(Dxprop(Dyear - 1 == 1),ones(DT,1)).^2).*(Dtime.^2) ...
    kron(x_prop(year == 1),ones(DT,1)) kron(x_prop(year == 1),ones(DT,1)).^2 ...
    kron(x_prop(year == 1),ones(DT,1)).*Dtime (kron(x_prop(year == 1),ones(DT,1)).^2).*Dtime ...
    kron(x_prop(year == 1),ones(DT,1)).*(Dtime.^2) (kron(x_prop(year == 1),ones(DT,1)).^2).*(Dtime.^2) ...
    ];
NameZprop = {...
    'DxP0','DxP0^2','DxP0*t','DxP0^2*t','DxP0*t^2','DxP0^2*t^2',...
    'xP0','xP0^2','xP0*t','xP0^2*t','xP0*t^2','xP0^2*t^2',...
    }';
NameZprop = [NameZ;NameZprop];
    
Zmurd = [Z kron(Dxmurd(Dyear - 1 == 1),ones(DT,1)) kron(Dxmurd(Dyear - 1 == 1),ones(DT,1)).^2 ...
    kron(Dxmurd(Dyear - 1 == 1),ones(DT,1)).*Dtime (kron(Dxmurd(Dyear - 1 == 1),ones(DT,1)).^2).*Dtime ...
    kron(Dxmurd(Dyear - 1 == 1),ones(DT,1)).*(Dtime.^2) (kron(Dxmurd(Dyear - 1 == 1),ones(DT,1)).^2).*(Dtime.^2) ...
    kron(x_murd(year == 1),ones(DT,1)) kron(x_murd(year == 1),ones(DT,1)).^2 ...
    kron(x_murd(year == 1),ones(DT,1)).*Dtime (kron(x_murd(year == 1),ones(DT,1)).^2).*Dtime ...
    kron(x_murd(year == 1),ones(DT,1)).*(Dtime.^2) (kron(x_murd(year == 1),ones(DT,1)).^2).*(Dtime.^2) ...
    ];
NameZmurd = {...
    'DxM0','DxM0^2','DxM0*t','DxM0^2*t','DxM0*t^2','DxM0^2*t^2',...
    'xM0','xM0^2','xM0*t','xM0^2*t','xM0*t^2','xM0^2*t^2',...
    }';
NameZmurd = [NameZ;NameZmurd];


Zviol = Zviol-TDums*((TDums'*TDums)\(TDums'*Zviol));
Zprop = Zprop-TDums*((TDums'*TDums)\(TDums'*Zprop));
Zmurd = Zmurd-TDums*((TDums'*TDums)\(TDums'*Zmurd));
DxV = Dxviol-TDums*((TDums'*TDums)\(TDums'*Dxviol));
DxP = Dxprop-TDums*((TDums'*TDums)\(TDums'*Dxprop));
DxM = Dxmurd-TDums*((TDums'*TDums)\(TDums'*Dxmurd));
DyV = Dyviol-TDums*((TDums'*TDums)\(TDums'*Dyviol));
DyP = Dyprop-TDums*((TDums'*TDums)\(TDums'*Dyprop));
DyM = Dymurd-TDums*((TDums'*TDums)\(TDums'*Dymurd));

maxIter = 100;

keepZviol = findNonCollinear(Zviol);
Zviol = Zviol(:,keepZviol);
NameZviol = NameZviol(keepZviol,:);
keepZprop = findNonCollinear(Zprop);
Zprop = Zprop(:,keepZprop);
NameZprop = NameZprop(keepZprop,:);
keepZmurd = findNonCollinear(Zmurd);
Zmurd = Zmurd(:,keepZmurd);
NameZmurd = NameZmurd(keepZmurd,:);


%% LASSO - Violence
% Prediction of variable of interest
StV = Zviol.*(DxV*ones(1,size(Zviol,2)));
UpsV = sqrt(sum(StV.^2)/(DT*N));
% lambda = 1.1*2*sqrt(2*DT*N*(log(2*(size(Zviol,2)/.05))));
lambda = 1.1*2*sqrt(DT*N)*norminv(1-.05/(2*size(Zviol,2)));
PInitV = LassoShooting2(Zviol, DxV , lambda, UpsV, 'Verbose', 0);
IndInitV = abs(PInitV) > 0;
ZV = Zviol(:,IndInitV);
RefResidV = DxV-Zviol*PInitV;
efsV = DxV-ZV*((ZV'*ZV)\(ZV'*DxV)) ;
IndRefV = IndInitV;

kk = 1;
StRefV = Zviol.*(RefResidV*ones(1,size(Zviol,2)));
UpsRefV = sqrt(sum(StRefV.^2)/(DT*N));
% lambda = 1.1*2*sqrt(2*DT*N*(log(2*(size(Zviol,2)/.05))));
while norm(UpsRefV-UpsV) > 1e-4 && kk < maxIter,
    disp(kk);
    PRefV = LassoShooting2(Zviol, DxV , lambda, UpsRefV , 'Verbose', 0);
    IndRefV = abs(PRefV) > 0;
    ZV = Zviol(:,IndRefV);
    bfsV = ((ZV'*ZV)\(ZV'*DxV));
    efsV = DxV-ZV*bfsV;
    UpsV = UpsRefV;
    StRefV = Zviol.*((DxV-Zviol*PRefV)*ones(1,size(Zviol,2)));
    UpsRefV = sqrt(sum(StRefV.^2)/(DT*N));    
    kk = kk+1;
end
sfsV = cluster_se(ZV,efsV,inv(ZV'*ZV),Dstate);

% Structural equation
StyV = Zviol.*(DyV*ones(1,size(Zviol,2)));
UpsyV = sqrt(sum(StyV.^2)/(DT*N));
% lambda = 1.1*2*sqrt(2*DT*N*(log(2*(size(Zviol,2)/.05))));
PInityV = LassoShooting2(Zviol, DyV , lambda, UpsyV , 'Verbose', 0);
IndInityV = abs(PInityV) > 0;
ZyV = Zviol(:,IndInityV);
RefResidyV = DyV-Zviol*PInityV;
efsyV = DyV-ZyV*((ZyV'*ZyV)\(ZyV'*DyV)) ;
IndRefyV = IndInityV;

kk = 1;
StRefyV = Zviol.*(RefResidyV*ones(1,size(Zviol,2)));
UpsRefyV = sqrt(sum(StRefyV.^2)/(DT*N));
% lambda = 1.1*2*sqrt(2*DT*N*(log(2*(size(Zviol,2)/.05))));
while norm(UpsRefyV-UpsyV) > 1e-4 && kk < maxIter,
    disp(kk);
    PRefyV = LassoShooting2(Zviol, DyV , lambda, UpsRefyV, 'Verbose', 0);
    IndRefyV = abs(PRefyV) > 0;
    ZyV = Zviol(:,IndRefyV);
    bfsyV = ((ZyV'*ZyV)\(ZyV'*DyV));
    efsyV = DyV-ZyV*bfsyV;
    UpsyV = UpsRefyV;
    StRefyV = Zviol.*((DyV-Zviol*PRefyV)*ones(1,size(Zviol,2)));
    UpsRefyV = sqrt(sum(StRefyV.^2)/(DT*N));    
    kk = kk+1;
end
sfsyV = cluster_se(ZyV,efsyV,inv(ZyV'*ZyV),Dstate);

IndUnionV = max([IndRefyV IndRefV],[],2);
ZUnionV = Zviol(:,IndUnionV);

R2_1V = sum((ZUnionV*(ZUnionV\DxV)).^2)/(sum(DxV.^2));
R2_2V = sum((ZUnionV*(ZUnionV\DyV)).^2)/(sum(DyV.^2)); 

ZviolLASSO = [DxV ZUnionV];
bviolLASSO = (ZviolLASSO'*ZviolLASSO)\(ZviolLASSO'*DyV);
eviolLASSO = DyV - ZviolLASSO*bviolLASSO;
ResviolLASSO = cluster_se(ZviolLASSO,eviolLASSO,inv(ZviolLASSO'*ZviolLASSO),Dstate);



%% LASSO - Property
% Prediction of variable of interest
StP = Zprop.*(DxP*ones(1,size(Zprop,2)));
UpsP = sqrt(sum(StP.^2)/(DT*N));
% lambda = 1.1*2*sqrt(2*DT*N*(log(2*(size(Zprop,2)/.05))));
PInitP = LassoShooting2(Zprop, DxP , lambda, UpsP,  'Verbose', 0);
IndInitP = abs(PInitP) > 0;
ZP = Zprop(:,IndInitP);
RefResidP = DxP - Zprop*PInitP;
efsP = DxP-ZP*((ZP'*ZP)\(ZP'*DxP)) ;
IndRefP = IndInitP;

kk = 1;
StRefP = Zprop.*(RefResidP*ones(1,size(Zprop,2)));
UpsRefP = sqrt(sum(StRefP.^2)/(DT*N));
% lambda = 1.1*2*sqrt(2*DT*N*(log(2*(size(Zprop,2)/.05))));
while norm(UpsRefP-UpsP) > 1e-4 && kk < maxIter,
    disp(kk);
    PRefP = LassoShooting2(Zprop, DxP , lambda, UpsRefP, 'Verbose', 0);
    IndRefP = abs(PRefP) > 0;
    ZP = Zprop(:,IndRefP);
    bfsP = ((ZP'*ZP)\(ZP'*DxP));
    efsP = DxP-ZP*bfsP;
    UpsP = UpsRefP;
    StRefP = Zprop.*((DxP-Zprop*PRefP)*ones(1,size(Zprop,2)));
    UpsRefP = sqrt(sum(StRefP.^2)/(DT*N));    
    kk = kk+1;
end
sfsP = cluster_se(ZP,efsP,inv(ZP'*ZP),Dstate);

% Structural equation
StyP = Zprop.*(DyP*ones(1,size(Zprop,2)));
UpsyP = sqrt(sum(StyP.^2)/(DT*N));
% lambda = 1.1*2*sqrt(2*DT*N*(log(2*(size(Zprop,2)/.05))));
PInityP = LassoShooting2(Zprop, DyP , lambda, UpsyP , 'Verbose', 0);
IndInityP = abs(PInityP) > 0;
ZyP = Zprop(:,IndInityP);
RefResidyP = DyP-Zprop*PInityP;
efsyP = DyP-ZyP*((ZyP'*ZyP)\(ZyP'*DyP)) ;
IndRefyP = IndInityP;

kk = 1;
StRefyP = Zprop.*(RefResidyP*ones(1,size(Zprop,2)));
UpsRefyP = sqrt(sum(StRefyP.^2)/(DT*N));
% lambda = 1.1*2*sqrt(2*DT*N*(log(2*(size(Zprop,2)/.05))));
while norm(UpsRefyP-UpsyP) > 5e-4 && kk < maxIter,
    disp(kk);
    PRefyP = LassoShooting2(Zprop, DyP , lambda, UpsRefyP, 'Verbose', 0);
    IndRefyP = abs(PRefyP) > 0;
    ZyP = Zprop(:,IndRefyP);
    bfsyP = ((ZyP'*ZyP)\(ZyP'*DyP));
    efsyP = DyP-ZyP*bfsyP;
    UpsyP = UpsRefyP;
    StRefyP = Zprop.*((DyP-Zprop*PRefyP)*ones(1,size(Zprop,2)));
    UpsRefyP = sqrt(sum(StRefyP.^2)/(DT*N));    
    kk = kk+1;
end
sfsyP = cluster_se(ZyP,efsyP,inv(ZyP'*ZyP),Dstate);

IndUnionP = max([IndRefyP IndRefP],[],2);
ZUnionP = Zprop(:,IndUnionP);

R2_1P = sum((ZUnionP*(ZUnionP\DxP)).^2)/(sum(DxP.^2));
R2_2P = sum((ZUnionP*(ZUnionP\DyP)).^2)/(sum(DyP.^2)); 

ZpropLASSO = [DxP ZUnionP];
bpropLASSO = (ZpropLASSO'*ZpropLASSO)\(ZpropLASSO'*DyP);
epropLASSO = DyP - ZpropLASSO*bpropLASSO;
RespropLASSO = cluster_se(ZpropLASSO,epropLASSO,inv(ZpropLASSO'*ZpropLASSO),Dstate);


%% LASSO - Murder
% Prediction of variable of interest
StM = Zmurd.*(DxM*ones(1,size(Zmurd,2)));
UpsM = sqrt(sum(StM.^2)/(DT*N));
% lambda = 1.1*2*sqrt(2*DT*N*(log(2*(size(Zmurd,2)/.05))));
PInitM = LassoShooting2(Zmurd, DxM , lambda, UpsM', 'Verbose', 0);
IndInitM = abs(PInitM) > 0;
ZM = Zmurd(:,IndInitM);
RefResidM = DxM-Zmurd*PInitM;
efsM = DxM-ZM*((ZM'*ZM)\(ZM'*DxM)) ;
IndRefM = IndInitM;

kk = 1;
StRefM = Zmurd.*(RefResidM*ones(1,size(Zmurd,2)));
UpsRefM = sqrt(sum(StRefM.^2)/(DT*N));
% lambda = 1.1*2*sqrt(2*DT*N*(log(2*(size(Zmurd,2)/.05))));
while norm(UpsRefM-UpsM) > 1e-4 && kk < maxIter,
    disp(kk);
    PRefM = LassoShooting2(Zmurd, DxM , lambda , UpsRefM', 'Verbose', 0);
    IndRefM = abs(PRefM) > 0;
    ZM = Zmurd(:,IndRefM);
    bfsM = ((ZM'*ZM)\(ZM'*DxM));
    efsM = DxM-ZM*bfsM;
    UpsM = UpsRefM;
    StRefM = Zmurd.*((DxM-Zmurd*PRefM)*ones(1,size(Zmurd,2)));
    UpsRefM = sqrt(sum(StRefM.^2)/(DT*N));    
    kk = kk+1;
end
sfsM = cluster_se(ZM,efsM,inv(ZM'*ZM),Dstate);

% Structural equation
StyM = Zmurd.*(DyM*ones(1,size(Zmurd,2)));
UpsyM = sqrt(sum(StyM.^2)/(DT*N));
% lambda = 1.1*2*sqrt(2*DT*N*(log(2*(size(Zmurd,2)/.05))));
PInityM = LassoShooting2(Zmurd, DyM , lambda , UpsyM', 'Verbose', 0);
IndInityM = abs(PInityM) > 0;
ZyM = Zmurd(:,IndInityM);
RefResidyM = DyM-Zmurd*PInityM;
efsyM = DyM-ZyM*((ZyM'*ZyM)\(ZyM'*DyM)) ;
IndRefyM = IndInityM;

kk = 1;
StRefyM = Zmurd.*(RefResidyM*ones(1,size(Zmurd,2)));
UpsRefyM = sqrt(sum(StRefyM.^2)/(DT*N));
% lambda = 1.1*2*sqrt(2*DT*N*(log(2*(size(Zmurd,2)/.05))));
while norm(UpsRefyM-UpsyM) > 1e-4 && kk < maxIter,
    disp(kk);
    PRefyM = LassoShooting2(Zmurd, DyM , lambda , UpsRefyM', 'Verbose', 0);
    IndRefyM = abs(PRefyM) > 0;
    ZyM = Zmurd(:,IndRefyM);
    bfsyM = ((ZyM'*ZyM)\(ZyM'*DyM));
    efsyM = DyM-ZyM*bfsyM;
    UpsyM = UpsRefyM;
    StRefyM = Zmurd.*((DyM-Zmurd*PRefyM)*ones(1,size(Zmurd,2)));
    UpsRefyM = sqrt(sum(StRefyM.^2)/(DT*N));    
    kk = kk+1;
end
sfsyM = cluster_se(ZyM,efsyM,inv(ZyM'*ZyM),Dstate);

IndUnionM = max([IndRefyM IndRefM],[],2);
ZUnionM = Zmurd(:,IndUnionM);

R2_1M = sum((ZUnionM*(ZUnionM\DxM)).^2)/(sum(DxM.^2));
R2_2M = sum((ZUnionM*(ZUnionM\DyM)).^2)/(sum(DyM.^2)); 

ZmurdLASSO = [DxM ZUnionM];
bmurdLASSO = (ZmurdLASSO'*ZmurdLASSO)\(ZmurdLASSO'*DyM);
emurdLASSO = DyM - ZmurdLASSO*bmurdLASSO;
ResmurdLASSO = cluster_se(ZmurdLASSO,emurdLASSO,inv(ZmurdLASSO'*ZmurdLASSO),Dstate);

acfML = acfcomp2(emurdLASSO,20);
acfPL = acfcomp2(epropLASSO,20);
acfVL = acfcomp2(eviolLASSO,20);


%% Display results
disp('Difference Violence - Level - LASSO')
disp('Selected Variables - Abortion')
disp(NameZviol(IndRefV))
disp(' ');
disp('Selected Variables - Crime')
disp(NameZviol(IndRefyV))
disp(' ');
disp('Crime equation')
disp([bviolLASSO(1) ResviolLASSO(1)])
disp(' ')
disp('Difference Property - Level - LASSO')
disp('Selected Variables - Abortion')
disp(NameZprop(IndRefP))
disp(' ');
disp('Selected Variables - Crime')
disp(NameZprop(IndRefyP))
disp(' ');
disp('Crime equation')
disp([bpropLASSO(1) RespropLASSO(1)])
disp(' ')
disp('Difference Murder - Level - LASSO')
disp('Selected Variables - Abortion')
disp(NameZmurd(IndRefM))
disp(' ');
disp('Selected Variables - Crime')
disp(NameZmurd(IndRefyM))
disp(' ');
disp('Crime equation')
disp([bmurdLASSO(1) ResmurdLASSO(1)])
disp(' ')

disp('First Order Autocorrelation')
disp([acfVL(2) acfPL(2) acfML(2)]);
disp(' ')

%% Estimates with Donohue and Levitt variables in amelioration set
IndUnionV(1:8) = 1;
IndUnionP(1:8) = 1;
IndUnionM(1:8) = 1;
ZUnionV = Zviol(:,IndUnionV);
ZUnionP = Zprop(:,IndUnionP);
ZUnionM = Zprop(:,IndUnionM);

ZviolLASSO = [DxV ZUnionV];
bviolLASSO = (ZviolLASSO'*ZviolLASSO)\(ZviolLASSO'*DyV);
eviolLASSO = DyV - ZviolLASSO*bviolLASSO;
ResviolLASSO = cluster_se(ZviolLASSO,eviolLASSO,inv(ZviolLASSO'*ZviolLASSO),Dstate);

ZpropLASSO = [DxP ZUnionP];
bpropLASSO = (ZpropLASSO'*ZpropLASSO)\(ZpropLASSO'*DyP);
epropLASSO = DyP - ZpropLASSO*bpropLASSO;
RespropLASSO = cluster_se(ZpropLASSO,epropLASSO,inv(ZpropLASSO'*ZpropLASSO),Dstate);

ZmurdLASSO = [DxM ZUnionM];
bmurdLASSO = (ZmurdLASSO'*ZmurdLASSO)\(ZmurdLASSO'*DyM);
emurdLASSO = DyM - ZmurdLASSO*bmurdLASSO;
ResmurdLASSO = cluster_se(ZmurdLASSO,emurdLASSO,inv(ZmurdLASSO'*ZmurdLASSO),Dstate);

acfMLp = acfcomp2(emurdLASSO,20);
acfPLp = acfcomp2(epropLASSO,20);
acfVLp = acfcomp2(eviolLASSO,20);


disp('Difference Violence - LASSO + Original')
disp('Crime equation')
disp([bviolLASSO(1) ResviolLASSO(1)])
disp(' ')
disp('Difference Property - LASSO + Original')
disp('Crime equation')
disp([bpropLASSO(1) RespropLASSO(1)])
disp(' ')
disp('Difference Murder - LASSO + Original')
disp('Crime equation')
disp([bmurdLASSO(1) ResmurdLASSO(1)])
disp(' ')

disp('Autocorrelation 1-3')
disp([acfVLp(2:4) acfPLp(2:4) acfMLp(2:4)]);
disp(' ')


%% Estimates without any selection
xVall = [DxV Zviol];
bVall = xVall\DyV;
sVall = cluster_se(xVall,DyV-xVall*bVall,inv(xVall'*xVall),Dstate);

xPall = [DxP Zprop];
bPall = xPall\DyP;
sPall = cluster_se(xPall,DyP-xPall*bPall,inv(xPall'*xPall),Dstate);

xMall = [DxM Zmurd];
bMall = xMall\DyM;
sMall = cluster_se(xMall,DyM-xMall*bMall,inv(xMall'*xMall),Dstate);

disp('No Selection')
disp('Violence')
disp([bVall(1) sVall(1)])
disp('Property')
disp([bPall(1) sPall(1)])
disp('Murder')
disp([bMall(1) sMall(1)])

