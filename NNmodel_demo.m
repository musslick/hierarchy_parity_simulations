clc;

% set up network parameters
nHidden = 3;              % number of hidden units
learningRate = 3;         % learning rate
bias = -2;                      % weight from bias units to hidden & output units
init_scale = 0.5;              % max. magnitude of initial weights
thresh = 0.0001;            % mean-squared error stopping criterion
decay = 0.0000;             % weight penalization parameter
hiddenPathSize = 1;        % group size of hidden units that receive the same weights from the task layer)
outputPathSize = 1;         % group size of output units that receive the same weights from the task layer)
trainingIterations = 20;   % training iterations

% generate training data
inputPatterns = [1 1; 0 1; 1 0; 0 0];
outputPatterns = [0; 1; 1; 0];
taskPatterns = [0; 0; 0; 0];

% set up model
taskNet = NNmodel(nHidden, learningRate, bias, init_scale, thresh, decay, hiddenPathSize, outputPathSize);

% set training data
taskNet.setData(inputPatterns, taskPatterns, outputPatterns);

% initialize network
taskNet.configure(); 

taskNet.weights.W_TH = zeros(size(taskNet.weights.W_TH));
taskNet.weights.W_TO = zeros(size(taskNet.weights.W_TO));

taskNet.weights.W_IH = [1, 2; 3, 4; 5, 6];
taskNet.weights.W_HO = [1, 2, 3];

% train network
taskNet.trainOnline(trainingIterations);

% plot learning curve
plot(taskNet.MSE_log);
