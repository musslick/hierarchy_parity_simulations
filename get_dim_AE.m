function [MSE_log, dimensionalities] = get_dim_AE(hiddenData, dimensionalities)

MSE_log = nan(1, length(dimensionalities));

for dim_idx = 1:length(dimensionalities)
    
    disp(['testing dimension ' num2str(dim_idx) '/' num2str(length(dimensionalities))]);
     
    dim = dimensionalities(dim_idx);
    
    iterations = 500;
    repetitions = 10;
    
    nHidden = dim;              % number of hidden units
    learningRate = 0.3;         % learning rate
    bias = -2;                      % weight from bias units to hidden & output units
    init_scale = 0.5;              % max. magnitude of initial weights
    thresh = 0.0001;            % mean-squared error stopping criterion
    decay = 0.0000;             % weight penalization parameter
    hiddenPathSize = 1;        % group size of hidden units that receive the same weights from the task layer)
    outputPathSize = 1;         % group size of output units that receive the same weights from the task layer)
    
    input = hiddenData;
    task = zeros(size(input, 1), 1); 
    output = hiddenData;
    
    % set up model
    taskNet = NNmodel(nHidden, learningRate, bias, init_scale, thresh, decay, hiddenPathSize, outputPathSize);

    % set training data
    taskNet.setData(input, task, output);

    % initialize network
    taskNet.configure(); 
    
    MSE_log_repetitions = nan(1, repetitions);
    
    for rep = 1:repetitions
        
         % train network
        taskNet.trainOnline(iterations);

        % log MSE
        MSE_log_repetitions(rep) = taskNet.MSE_log(end);
        
    end
    
    MSE_log(dim_idx) =  min(MSE_log_repetitions);

end