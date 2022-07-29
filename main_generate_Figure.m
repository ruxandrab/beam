

%% Script to generate Figure 

clear all
close all

runs = 10;
epochs = 10000;

folder1 = 'Experiment1';
folder2 = ['RNN', 'LSTM', 'GRU'];
folder3RNN = ['16','64'];
folder3GRU = ['8','32'];
% runsstrings = {'Run1','Run2','Run3'};
runsstrings = {'Run1','Run2','Run3','Run4','Run5',...
  'Run6', 'Run7','Run8','Run9','Run10'};
% runsstrings = {'Run10'};
% runsstring2 = {'Run1'};


% % RNN
rmseRNN16T = zeros(runs,epochs);
rmseRNN64T = zeros(runs,epochs);
rmseRNN16V = zeros(runs,epochs);
rmseRNN64V = zeros(runs,epochs);
for i = 1:runs
  filename16 = ['Experiment1/RNN/16/',runsstrings{i},'/training_history.dat'];
  filename64 = ['Experiment1/RNN/64/',runsstrings{i},'/training_history.dat'];
  mseRNN16 = load(filename16);
  mseRNN64 = load(filename64);
%   
  rmseRNN16T(i,:) = sqrt(mseRNN16(:,1))';
  rmseRNN16V(i,:) = sqrt(mseRNN16(:,2))';
%   
  rmseRNN64T(i,:) = sqrt(mseRNN64(:,1))';
  rmseRNN64V(i,:) = sqrt(mseRNN64(:,2))';
end
% 
% % LSTM
rmseLSTM8T = zeros(runs,epochs);
rmseLSTM32T = zeros(runs,epochs);
rmseLSTM8V = zeros(runs,epochs);
rmseLSTM32V = zeros(runs,epochs);
for i = 1:runs
  filename4 = ['Experiment1/LSTM/4/',runsstrings{i},'/training_history.dat'];
  filename8 = ['Experiment1/LSTM/8/',runsstrings{i},'/training_history.dat'];
  filename32 = ['Experiment1/LSTM/32/',runsstrings{i},'/training_history.dat'];
% 
  mseLSTM4 = load(filename4);
  mseLSTM8 = load(filename8);
  mseLSTM32 = load(filename32);
%
  rmseLSTM4T(i,:) = sqrt(mseLSTM4(:,1))';
  rmseLSTM4V(i,:) = sqrt(mseLSTM4(:,2))';
%   
  rmseLSTM8T(i,:) = sqrt(mseLSTM8(:,1))';
  rmseLSTM8V(i,:) = sqrt(mseLSTM8(:,2))';
%   
  rmseLSTM32T(i,:) = sqrt(mseLSTM32(:,1))';
  rmseLSTM32V(i,:) = sqrt(mseLSTM32(:,2))';
end

% GRU
rmseGRU8T = zeros(runs,epochs);
rmseGRU32T = zeros(runs,epochs);
rmseGRU8V = zeros(runs,epochs);
rmseGRU32V = zeros(runs,epochs);
for i = 1:runs
  filename4 = ['Experiment1/GRU/4/',runsstrings{i},'/training_history.dat'];
  filename8 = ['Experiment1/GRU/8/',runsstrings{i},'/training_history.dat'];
  filename32 = ['Experiment1/GRU/32/',runsstrings{i},'/training_history.dat'];
  
  mseGRU4 = load(filename4);
  mseGRU8 = load(filename8);
  mseGRU32 = load(filename32);
  
  rmseGRU4T(i,:) = sqrt(mseGRU4(:,1))';
  rmseGRU4V(i,:) = sqrt(mseGRU4(:,2))';
  
  rmseGRU8T(i,:) = sqrt(mseGRU8(:,1))';
  rmseGRU8V(i,:) = sqrt(mseGRU8(:,2))';
  
  rmseGRU32T(i,:) = sqrt(mseGRU32(:,1))';
  rmseGRU32V(i,:) = sqrt(mseGRU32(:,2))';
end

%% mean (and min) of each column
rmseRNN16T_avg =  mean(rmseRNN16T);
rmseRNN16V_avg  = mean(rmseRNN16V);
rmseRNN64T_avg  = mean(rmseRNN64T);
rmseRNN64V_avg  = mean(rmseRNN64V);

% [minRNN16V, minRNN16Vidx] = min(rmseRNN16V')
% meanminRNN16V = mean(minRNN16V)
% meanminRNN16T = mean(diag(rmseRNN16T(:,minRNN16Vidx)))
[minRNN16V, minRNN16Vidx] = min(rmseRNN16V_avg)
minRNN16T = rmseRNN16T_avg(minRNN16Vidx)
% [minRNN64V, minRNN64Vidx] = min(rmseRNN64V')
% meanminRNN64V = mean(minRNN64V)
% meanminRNN64T = mean(diag(rmseRNN64T(:,minRNN64Vidx)))
[minRNN64V, minRNN64Vidx] = min(rmseRNN64V_avg)
minRNN64T = rmseRNN64T_avg(minRNN64Vidx)

rmseLSTM4T_avg  = mean(rmseLSTM4T);
rmseLSTM4V_avg  = mean(rmseLSTM4V);

rmseLSTM8T_avg  = mean(rmseLSTM8T);
rmseLSTM8V_avg  = mean(rmseLSTM8V);
rmseLSTM32T_avg = mean(rmseLSTM32T); 
rmseLSTM32V_avg = mean(rmseLSTM32V);

[minLSTM4V, minLSTM4Vidx] = min(rmseLSTM4V_avg)
minLSTM4T = rmseLSTM4T_avg(minLSTM4Vidx)

% [minLSTM8V, minLSTM8Vidx] = min(rmseLSTM8V')
% meanminLSTM8V = mean(minLSTM8V)
% meanminLSTM8T = mean(diag(rmseLSTM8T(:,minLSTM8Vidx)))
[minLSTM8V, minLSTM8Vidx] = min(rmseLSTM8V_avg)
minLSTM8T = rmseLSTM8T_avg(minLSTM8Vidx)
% [minLSTM32V, minLSTM32Vidx] = min(rmseLSTM32V')
% meanminLSTM32V = mean(minLSTM32V)
% meanminLSTM32T = mean(diag(rmseLSTM32T(:,minLSTM32Vidx)))
[minLSTM32V, minLSTM32Vidx] = min(rmseLSTM32V_avg)
minLSTM32T = rmseLSTM32T_avg(minLSTM32Vidx)

rmseGRU4T_avg  = mean(rmseGRU4T); 
rmseGRU4V_avg = mean(rmseGRU4V);

rmseGRU8T_avg  = mean(rmseGRU8T); 
rmseGRU8V_avg = mean(rmseGRU8V);
rmseGRU32T_avg = mean(rmseGRU32T);
rmseGRU32V_avg = mean(rmseGRU32V);

[minGRU4V, minGRU4Vidx] = min(rmseGRU4V_avg)
minGRU4T = rmseGRU4T_avg(minGRU4Vidx)
% [minGRU8V, minGRU8Vidx] = min(rmseGRU8V')
% meanminGRU8V = mean(minGRU8V)
% meanminGRU8T = mean(diag(rmseGRU8T(:,minGRU8Vidx)))
[minGRU8V, minGRU8Vidx] = min(rmseGRU8V_avg)
minGRU8T = rmseGRU8T_avg(minGRU8Vidx)
% [minGRU32V, minGRU32Vidx] = min(rmseGRU32V')
% meanminGRU32V = mean(minGRU32V)
% meanminGRU32T = mean(diag(rmseGRU32T(:,minGRU32Vidx)))
[minGRU32V, minGRU32Vidx] = min(rmseGRU32V_avg)
minGRU32T = rmseGRU32T_avg(minGRU32Vidx)

% %% Plots
leg3a = {'RNN 16 training','RNN 16 validation',...
  'LSTM 8 training','LSTM 8 validation',...
  'GRU 8 training','GRU 8 validation'};
leg3b = {'RNN 64 training','RNN 64 validation',...
  'LSTM 32 training','LSTM 32 validation',...
  'GRU 32 training','GRU 32 validation'};
% leg3a = {'GRU 8 training','GRU 8 validation'};
% leg3b = {'GRU 32 training','GRU 32 validation'};


figure(3)  % Figure 3 left
semilogx(1:epochs,rmseRNN16T_avg,'r','LineWidth',2);
hold on
semilogx(1:epochs,rmseRNN16V_avg,'r-.','LineWidth',2);
hold on
semilogx(1:epochs,rmseLSTM8T_avg,'g','LineWidth',2);
hold on
semilogx(1:epochs,rmseLSTM8V_avg,'g-.','LineWidth',2);
hold on
semilogx(1:epochs,rmseGRU8T_avg,'b','LineWidth',2);
hold on
semilogx(1:epochs,rmseGRU8V_avg,'b-.','LineWidth',2);
hold on
legend(leg3a);
set(gca, 'FontSize', 14);
xlabel('Epochs');
ylabel('RMSE');
grid on

figure(4) % Figure 3 right
semilogx(1:epochs,rmseRNN64T_avg,'r','LineWidth',2);
hold on
semilogx(1:epochs,rmseRNN64V_avg,'r-.','LineWidth',2);
hold on
semilogx(1:epochs,rmseLSTM32T_avg,'g','LineWidth',2);
hold on
semilogx(1:epochs,rmseLSTM32V_avg,'g-.','LineWidth',2);
hold on
semilogx(1:epochs,rmseGRU32T_avg,'b','LineWidth',2);
hold on
semilogx(1:epochs,rmseGRU32V_avg,'b-.','LineWidth',2);
hold on
legend(leg3b);
set(gca, 'FontSize', 14);
xlabel('Epochs');
ylabel('RMSE');
grid on
