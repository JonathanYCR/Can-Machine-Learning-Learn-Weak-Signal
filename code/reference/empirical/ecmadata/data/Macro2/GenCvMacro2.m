close all
clear all

rng(10);

load GrowthData.mat

N=size(data,1);
Nin=floor(.5*N);

Mcv = 1000;

for jcv = 1:Mcv
   temp=randperm(N);
   EstSmplCv{jcv}    =  temp(1:Nin);
   EvalSmplCv{jcv}   =  temp(Nin+1:end);
end

% save CvSamplesMacro2 EstSmplCv EvalSmplCv