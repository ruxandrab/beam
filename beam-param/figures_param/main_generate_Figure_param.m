

%% Script to generate Figure parameter-aware
%% for SCEE
clear all
close all

% filename1 = '1data_[6.1]_[3.6]_[1.9]_48valid.dat';  % true values somewhat similar
% filename2 = '1data_[6.1]_[3.6]_[1.73]_91train.dat';
filename1 = '2data_[6.1]_[4.]_[1.9]_284train.dat';  % the prediction not very good
filename2 = '2data_[6.1]_[4.]_[1.73]_45train.dat';
% filename1 = '3data_[6.1]_[4.]_[1.9]_270train.dat';  % 
% filename2 = '3data_[6.1]_[4.]_[1.73]_58valid.dat';
% filename1 = '4data_[6.]_[3.6]_[1.9]_60train.dat';
% filename2 = '4data_[6.]_[3.6]_[1.73]_72test.dat';
% filename1 = '5data_[6.]_[4.]_[1.9]_73valid.dat';
% filename2 = '5data_[6.]_[4.]_[1.73]_43test.dat';

filenamein = 'data45_25_100.dat';

data1 = load(filename1);
data2 = load(filename2);
  
data4525100 = load(filenamein);

% %% Plots
leg3 = {'Air viscosity = 1.9\cdot 10^{−5}','Air viscosity = 1.73\cdot 10^{−5}'};

time = data1(:,1);
% 

input = data4525100(:,2);
true = [data1(:,2) data2(:,2)];
pred = [data1(:,3) data2(:,3)];

figure(10)  
% plot(time,true(:,1),'b','LineWidth',2);
% plot(time,input)
hold on
plot(time,pred(:,1),'bp:','LineWidth',2);
hold on
% plot(time,true(:,2),'b','LineWidth',2);
hold on
plot(time,pred(:,2),'rp:','LineWidth',2);
legend(leg3)
set(gca, 'FontSize', 16);
xlabel('Time [ms]');
ylabel('Minimum gap [um]');
grid on

figure(11)
plot(time,input,'k-', 'LineWidth',2)
