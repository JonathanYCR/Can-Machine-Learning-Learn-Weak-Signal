close all
clear all

rng(10)

load DataWeberQuintiles.mat

tempTime=datevec(Time);
yrs = tempTime(:,1);
mths = tempTime(:,2);

StartYearOut = 1975;
EndYearOut = 2014;

count = 0;
for jy = StartYearOut:EndYearOut
    count               =  count+1;
    EstSmplCv{count}    =  find(yrs<=jy-1);
    EvalSmplCv{count}   =  find(yrs==jy);
end;

% save CvSamplesFinance2 EstSmplCv EvalSmplCv