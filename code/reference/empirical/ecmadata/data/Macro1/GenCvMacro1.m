close all
clear all

rng(10)

load FredMDlargeHor1.mat
data=[X Y]; Time = Time(2:end); clear X Y;

tempTime=datevec(Time);
yrs = tempTime(:,1);
mths = tempTime(:,2);

StartYearOut = 1970;
EndYearOut = 2014;

cont = 0;
for jy = StartYearOut:EndYearOut
    cont               =  cont+1;
    EstSmplCv{cont}    =  find(yrs<=jy-1);
    EvalSmplCv{cont}   =  find(yrs==jy);
end;

% save CvSamplesMacro1 EstSmplCv EvalSmplCv