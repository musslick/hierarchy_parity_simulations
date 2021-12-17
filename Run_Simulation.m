clc;
clear all;

% set up network parameters
nHidden = 20;              % number of hidden units
learningRate = 0.5;         % learning rate
bias = -3;                      % weight from bias units to hidden & output units
init_scale = 1;              % max. magnitude of initial weights
thresh = 0.0001;            % mean-squared error stopping criterion
decay = 0.0000;             % weight penalization parameter
hiddenPathSize = 1;        % group size of hidden units that receive the same weights from the task layer)
outputPathSize = 1;         % group size of output units that receive the same weights from the task layer)
MSE_thresh = 0.000;

repetitions = 20;

dimensionalities = 1:5;

% generate training data

% RM Phase

parity.input = [0 1   0 1   0 1; ... % I C L    Parity Task
                   0 1   0 1   1 0; ... % I C H
                   0 1   1 0   0 1; ... % I A L
                   0 1   1 0   1 0; ... % I A H
                   1 0   0 1   0 1; ... % O C L
                   1 0   0 1   1 0; ... % O C H
                   1 0   1 0   0 1; ... % O A L
                   1 0   1 0   1 0]; % O A H

hierarchy.input =   [0 1   0 1   0 1; ... % I C L     Hierarchy Task
                               0 1   0 1   1 0; ... % I C H
                               0 1   1 0   0 1; ... % I A L
                               0 1   1 0   1 0; ... % I A H
                               1 0   0 1   0 1; ... % O C L
                               1 0   0 1   1 0; ... % O C H
                               1 0   1 0   0 1; ... % O A L
                               1 0   1 0   1 0];    % O A H  
                   

parity.task = zeros(size(parity.input, 1), 2); 
parity.task(:, 1) = 1;

hierarchy.task = zeros(size(parity.input, 1), 2); 
hierarchy.task(:, 2) = 1;

% all.task(1:size(all.input, 1)/2, 1) = 1;
% all.task((size(all.input, 1)/2+1):end, 2) = 1;

parity.output =   [0 1; ... % I C L    Parity Task
                           1 0; ... % I C H
                           1 0 ; ... % I A L
                           0 1; ... % I A H
                           1 0; ... % O C L
                           0 1; ... % O C H
                           0 1; ... % O A L
                           1 0]; % O A H

hierarchy.output = [0 1; ... % I C L     Hierarchy Task
                               0 1; ... % I C H
                               1 0; ... % I A L
                               0 1; ... % I A H
                               0 1; ... % O C L
                               1 0; ... % O C H
                               1 0; ... % O A L
                               1 0];    % O A H  


all.input = [parity.input; hierarchy.input];
all.task = [parity.task; hierarchy.task];
all.output = [parity.output; hierarchy.output];

% run simulation

dim_log_parity = nan(repetitions, length(dimensionalities));
dim_log_hierarchy = nan(repetitions, length(dimensionalities));

max_dims = min([nHidden, size(parity.input,1)-1]);

pca_log_parity = nan(repetitions, max_dims);
pca_log_hierarchy = nan(repetitions, max_dims);

MSE_log_parity = nan(1, repetitions);
MSE_log_hierarchy = nan(1, repetitions);

for rep = 1:repetitions

%% set up model
taskNet = NNmodel(nHidden, learningRate, bias, init_scale, thresh, decay, hiddenPathSize, outputPathSize);

% set training data
taskNet.setData(all.input, all.task, all.output);

% initialize network
taskNet.configure(); 

%% Task Training

taskNet.thresh = MSE_thresh;

% set training data
% taskNet.setData(hierarchy.input, hierarchy.task, hierarchy.output);
% taskNet.setData(parity.input, parity.task, parity.output);
taskNet.setData(all.input, all.task, all.output);

% train network
taskNet.trainOnline(3000);

% plot learning curve
plot(taskNet.MSE_log);

%% determine dimensionalities

% parity task
[~, hiddenData, MSE] = taskNet.runSet(parity.input, parity.task, parity.output);
MSE_log_parity(rep) = mean(MSE);
[MSE_log, ~] = get_dim_AE(hiddenData, dimensionalities);
dim_log_parity(rep, :)  = MSE_log;

[~, SCORE, LATENT] = pca(hiddenData);
pca_log_parity(rep, :) = LATENT;

% hierarchy task
[~, hiddenData, MSE] = taskNet.runSet(hierarchy.input, hierarchy.task, hierarchy.output);
MSE_log_hierarchy(rep) = mean(MSE);
[MSE_log, ~] = get_dim_AE(hiddenData, dimensionalities);
dim_log_hierarchy(rep, :)  = MSE_log;

[~, ~, LATENT] = pca(hiddenData);
pca_log_hierarchy(rep, :) = LATENT;

disp(['repetition ' num2str(rep) '/' num2str(repetitions)]);
end

%% plots
plotSettings;
lineWidth = 3;

% Autoencoder Dimensionality Plot
parity_dim_mean = mean(dim_log_parity);
parity_dim_sem = std(dim_log_parity)/sqrt(repetitions);

hierarchy_dim_mean = mean(dim_log_hierarchy);
hierarchy_dim_sem = std(dim_log_hierarchy)/sqrt(repetitions);

x = dimensionalities;

fig = figure(1);
set(fig, 'Position', [100, 100, 250, 300]);

errorbar(x, parity_dim_mean, parity_dim_sem, '-', 'LineWidth', lineWidth, 'Color', colors(1,  :)); hold on;
errorbar(x, hierarchy_dim_mean, hierarchy_dim_sem, '-', 'LineWidth', lineWidth, 'Color', colors(2,  :)); hold off;

set(gca, 'XTick', x);
leg = legend('parity', 'hierarchy');
% ylim([30 100]);
xlim([0 max(dimensionalities)+1]);
title(['Learned Dimensionality (Nonlinear)'], 'FontSize', fontSize_title, 'color', 'w');
ylabel('Autoencoder Loss (MSE)', 'FontSize', fontSize_ylabel);
xlabel('Dimensionality', 'FontSize', fontSize_ylabel);
set(gca, 'FontSize', fontSize_gca);
set(leg, 'TextColor', 'w');
set(leg, 'Color', 'none');
set(gcf, 'Color', 'k');
set(gca, 'Color', 'k');
set(gca, 'xColor', 'w');
set(gca, 'yColor', 'w');
set(gca, 'zColor', 'w');


% PCA Dimensionality Plot
parity_pca_mean = mean(pca_log_parity);
parity_pca_sem = std(pca_log_parity)/sqrt(repetitions);

hierarchy_pca_mean = mean(pca_log_hierarchy);
hierarchy_pca_sem = std(pca_log_hierarchy)/sqrt(repetitions);

x = 1:max_dims;

fig = figure(2);
set(fig, 'Position', [100, 100, 250, 300]);

errorbar(x, parity_pca_mean, parity_pca_sem, '-', 'LineWidth', lineWidth, 'Color', colors(1,  :)); hold on;
errorbar(x, hierarchy_pca_mean, hierarchy_pca_sem, '-', 'LineWidth', lineWidth, 'Color', colors(2,  :)); hold off;

set(gca, 'XTick', x);
leg = legend('parity', 'hierarchy');
% ylim([30 100]);
xlim([0 max(dimensionalities)+1]);
title(['Learned Dimensionality (Linear)'], 'FontSize', fontSize_title, 'color', 'w');
ylabel('Eigenvalue', 'FontSize', fontSize_ylabel);
xlabel('Component', 'FontSize', fontSize_ylabel);
set(gca, 'FontSize', fontSize_gca);
set(leg, 'TextColor', 'w');
set(leg, 'Color', 'none');
set(gcf, 'Color', 'k');
set(gca, 'Color', 'k');
set(gca, 'xColor', 'w');
set(gca, 'yColor', 'w');
set(gca, 'zColor', 'w');

% MSE (Performance) plot

fig = figure(3);
set(fig, 'Position', [400, 100, 500, 300]);

bardata_mean = [mean(MSE_log_parity), mean(MSE_log_hierarchy)];
bardata_sem = [std(MSE_log_parity), std(MSE_log_parity)]/sqrt(repetitions);
x = [1, 2];

scatter(x(1), bardata_mean(1), 50, colors(1,  :)); hold on;
errorbar(x(1), bardata_mean(1), bardata_sem(1), '.', 'LineWidth', lineWidth, 'Color', colors(1,  :)); 

scatter(x(2), bardata_mean(2), 50, colors(2,  :)); 
errorbar(x(2), bardata_mean(2), bardata_sem(2), '.', 'LineWidth', lineWidth, 'Color', colors(2,  :));

hold off;

set(gca, 'XTick', [1 2]);
set(gca, 'XTickLabel', {'parity task', 'hierarchy task'});
% ylim([30 100]);
xlim([0.5, 2.5]);
title(['Task Performance'], 'FontSize', fontSize_title, 'color', 'w');
ylabel('Mean Squared Error', 'FontSize', fontSize_ylabel);
xlabel('', 'FontSize', fontSize_ylabel);
set(gca, 'FontSize', fontSize_gca);
% set(leg, 'TextColor', 'w');
% set(leg, 'Color', 'none');
set(gcf, 'Color', 'k');
set(gca, 'Color', 'k');
set(gca, 'xColor', 'w');
set(gca, 'yColor', 'w');
set(gca, 'zColor', 'w');


% bar_handle = errorbar_groups(bardata_mean', bardata_sem','FigID', fig1, 'bar_colors', plotColors,'errorbar_width',0.5,'optional_errorbar_arguments',{'LineStyle','none','Marker','none','LineWidth',1.5});
%     set(gca,'XTickLabel',{' ', ' ',' ',' '},'FontSize', fontSize_gca, 'FontName', fontName);