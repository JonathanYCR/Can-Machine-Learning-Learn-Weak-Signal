close all
clear all

rng(10)
load Abortion_data_u13.mat  % new data with time dummies as fixed controls
data=[TDums zscore([Dxmurd Zmurd Dymurd],1)]; clearvars -except data dataset Dyear Dstate

Year = Dyear;
State = Dstate;

year = unique(Year);
state = unique(State);
T = length(year);
N = length(state);


T0est = 4;
Nest = floor(N*.5);
Mcv = 1000;

cont = 0;
for jcv = 1:Mcv
    for jt = T0est:T-1
        cont               =  cont+1;
        CrossPerm          =  randperm(N)';
        CrossEst         =  state(CrossPerm(1:Nest));
        CrossEval          =  state(CrossPerm(Nest+1:end));
        TimeEst          =  year(1:jt);
        TimeEval           =  year(jt+1);
        Temp               =  (ismember(Year,TimeEst) | ismember(State,CrossEst));
        EstSmplCv{cont}  =  find(Temp==1);
        Temp               =  (ismember(State,CrossEval)  & ismember(Year,TimeEval));
        EvalSmplCv{cont}   =  find(Temp==1);
    end
end;

% save CvSamplesMicro1 EstSmplCv EvalSmplCv