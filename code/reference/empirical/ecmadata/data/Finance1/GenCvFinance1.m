close all
clear all

rng(10);

load Goyal.mat

yrs=Time;
StartYearOut = 1965;
EndYearOut = 2015;

count = 0;
for jy = StartYearOut:EndYearOut
    count               =  count+1;
    EstSmplCv{count}    =  find(yrs<=jy-1);
    EvalSmplCv{count}   =  find(yrs==jy);
end;

% save CvSamplesFinance1 EstSmplCv EvalSmplCv
