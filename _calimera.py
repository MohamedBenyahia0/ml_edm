from joblib import Parallel, delayed
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, train_test_split
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch
torch.manual_seed(42)
np.random.seed(42)

torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from ml_edm.alert import DQN
from ._base import BaseTriggerModel
import os
import xgboost as xgb
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.signal import find_peaks
from sklearn.svm import OneClassSVM
from collections import Counter
# Set environment variables to limit the number of threads used by MKL, NumExpr, and OpenMP
# This is important to avoid performance issues when using multiple threads
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['KMP_WARNINGS'] = 'off'
#from alert import DQN


class CALIMERA(BaseTriggerModel):
    """
    CALIMERA: A new early time series classification method
    Inspired by : https://github.com/JakubBilski/CALIMERA 
    """

    """Remark when running ( there are two _fit methods):
    The One reg and The epsilon variants are implemented in the _fit method
    The Ref Point, True Plateau, True Label variants and its epsilon fusion are implemented in the _fit2 method so rename it _fit if you want to use these variants with correct parameters
   """
    def __init__(self,
                 timestamps,regressor ='many',variant='default',eps_method_one_reg=False,num_samples_eps=50,cross_val='kfold',target_swap=False,var_true_plateau=False, var_ref_point = False,var_classificat=False, min_flat_length=1,eval_epochs=10,num_epochs=1000,start_val_epoch=20, epsilon_method=False,
                 one_reg_full=False,n_jobs=1):
        
        super().__init__()
        self.timestamps = timestamps
        self.n_jobs = n_jobs
        self.variant = variant #('true_label','corrected_prior','transpose','prior','p_hat_p_hat','default') the variant to use for generating costs
        self.epsilon_method = epsilon_method # to use the CALIMERA epsilon variant
        self.eps_method_one_reg= eps_method_one_reg # to use the one reg epsilon variant
        
        self.opt_epsilon= None # variable to store the optimal found epsilon value
        self.num_samples_eps=num_samples_eps
        self.cross_val=cross_val # type of cross val (classic, stratified)
        self.target_swap = target_swap # flatten with ground truth target in classic calimera
        self.min_flat_length = min_flat_length # hyperparameter for the ref point and true plateau variant to control the minimum height between succesive plateaus
        self.var_true_plateau = var_true_plateau
        self.y_triggers=[]
        self.regressor = regressor # ('one','many') either to use one regression model or a collection of regressors
        self.start_val_epoch = start_val_epoch # the index of the epoch to start validating
        self.one_reg_full = one_reg_full # to use in the one regressor either full timesteps or not 

        # One regressor
        self.num_epochs = num_epochs # number of training epochs
        self.eval_epochs = eval_epochs # the hop of epochs for evaluating average cost


        # Compare with reference points of flattened costs 
        self.var_ref_point = var_ref_point
        
        self.var_classificat = var_classificat
            
    
    def _generate_features(self, probas, time_idx,y):
        
         

        max_probas = np.max(probas, axis=-1)
        second_max_probas = np.partition(probas, -2)[:,-2]
        diff = max_probas - second_max_probas

        features = np.concatenate(
            (probas, diff[:,None], max_probas[:,None]), axis=-1
        )
    
        #delay_cost = self.alpha * self.models_input_lengths[time_idx] / self.max_length
        #costs = 1 - max_probas + delay_cost
        
        ## Hypothesis : max P(y|X) == proba to be in diag and 1 - max P(y|X) == not in diag 
        #
        #delay = np.mean(self.cost_matrices.delay_cost[time_idx])
        #proba_correct = np.mean(np.diagonal(self.cost_matrices.missclf_cost[time_idx])) * max_probas # weight Cd by prob ?
        #non_diag = self.cost_matrices[time_idx] - \
        #    (np.eye(self.n_classes) * np.diagonal(self.cost_matrices[time_idx]))
        #proba_incorrect = np.sum(non_diag) / (self.n_classes**2 - self.n_classes) * (1-max_probas) # weight Cm one by one ?
        #costs = proba_correct + proba_incorrect + delay 

        delay = np.mean(self.cost_matrices.delay_cost[time_idx])
        misclf_cost = [(prob * [self.cost_matrices.missclf_cost[time_idx][prob.argmax()][yy]
                               for yy in self.classes_]).sum() for prob in probas]
        
        
        costs = misclf_cost + delay
        

        return features, costs
    
    def _generate_features_variant_true_label(self, probas, time_idx,y):
        
        max_probas = np.max(probas, axis=-1)
        second_max_probas = np.partition(probas, -2)[:,-2]
        diff = max_probas - second_max_probas

        features = np.concatenate(
            (probas, diff[:,None], max_probas[:,None]), axis=-1
        )
        
        delay = np.mean(self.cost_matrices.delay_cost[time_idx])
        misclf_cost = [(probas[k] * [self.cost_matrices.missclf_cost[time_idx][yy][y[k]]
                               for yy in self.classes_]).sum() for k in range(len(probas))]
        costs = misclf_cost + delay
        

        return features, costs

    def _generate_features_variant_transpose(self, probas, time_idx,y):
        
        

        max_probas = np.max(probas, axis=-1)
        second_max_probas = np.partition(probas, -2)[:,-2]
        diff = max_probas - second_max_probas

        features = np.concatenate(
            (probas, diff[:,None], max_probas[:,None]), axis=-1
        )
        
        delay = np.mean(self.cost_matrices.delay_cost[time_idx])
        misclf_cost = [(prob * [self.cost_matrices.missclf_cost[time_idx][yy][prob.argmax()]
                               for yy in self.classes_]).sum() for prob in probas]
        costs = misclf_cost + delay
        return features, costs
    
    def _generate_features_variant_corrected_prior(self, probas, time_idx,y):
       
        max_probas = np.max(probas, axis=-1)
        second_max_probas = np.partition(probas, -2)[:,-2]
        diff = max_probas - second_max_probas

        features = np.concatenate(
            (probas, diff[:,None], max_probas[:,None]), axis=-1
        )

        delay = np.mean(self.cost_matrices.delay_cost[time_idx])

        # Estimate the prior probabilities from the training data
        prior = np.zeros((self.n_classes,))
        for i in range(self.n_classes):
            prior[i] = np.mean(y == i)
        # Normalize the prior probabilities
        prior /= np.sum(prior)
        # repeat the prior probabilities for each sample
        priors = np.tile(prior, (probas.shape[0], 1))
        
        misclf_cost = [(prob * [self.cost_matrices.missclf_cost[time_idx][prob.argmax()][yy]
                               for yy in self.classes_]).sum() for prob in probas] 
        # Use prior to weight the predicted probabilities
        for prob in probas:
            for i in range(self.n_classes):
                for j in range(self.n_classes):
                    if j!= prob.argmax():
                        misclf_cost += (self.cost_matrices.missclf_cost[time_idx][i][j] * prior[i])
       
        costs = misclf_cost/2 + delay
        
        return features, costs


    def _generate_features_variant_prior(self, probas, time_idx, y):
        

        max_probas = np.max(probas, axis=-1)
        second_max_probas = np.partition(probas, -2)[:,-2]
        diff = max_probas - second_max_probas

        features = np.concatenate(
            (probas, diff[:,None], max_probas[:,None]), axis=-1
        )
        delay = np.mean(self.cost_matrices.delay_cost[time_idx])

        # Estimate the prior probabilities from the training data
        prior = np.zeros((self.n_classes,))
        for i in range(self.n_classes):
            prior[i] = np.mean(y == i)
        # Normalize the prior probabilities
        prior /= np.sum(prior)
        # repeat the prior probabilities for each sample
        prior = np.tile(prior, (probas.shape[0], 1))
        # Use prior to weight the predicted probabilities
        misclf_cost = (prior @ self.cost_matrices.missclf_cost[time_idx]).sum(axis=-1)

        costs = misclf_cost /2+ delay
        return features, costs
    
    def _generate_features_variant_p_hat_p_hat(self, probas, time_idx, y):
        

        max_probas = np.max(probas, axis=-1)
        second_max_probas = np.partition(probas, -2)[:,-2]
        diff = max_probas - second_max_probas

        features = np.concatenate(
            (probas, diff[:,None], max_probas[:,None]), axis=-1
        )
        delay = np.mean(self.cost_matrices.delay_cost[time_idx])
        
        # Use prior to weight the predicted probabilities
        misclf_cost = (probas @ self.cost_matrices.missclf_cost[time_idx]).sum(axis=-1)    
        
        costs = misclf_cost/2 + delay
        return features, costs
    
    
    
    
    
        
    
   
       
    
    def _fit(self, X_probas, y,X_val=None,X_probas_val=None,y_val=None,X_test=None,X_probas_test=None,y_test=None):
        if self.regressor not in ['one', 'many']:
            raise ValueError("regressor must be 'one' or 'many'")

        self.max_timestamp_idx = len(self.timestamps)
        features , costs = self.get_features_costs(X_probas,y)
        
    

        
        self.pred_cost_diffs=np.zeros((X_probas.shape[0], self.max_timestamp_idx-1))

        self.r2_scores = [None for _ in range(self.max_timestamp_idx-1)]


        
    

        
        if self.regressor=='many':
            self.halters = [None for _ in range(self.max_timestamp_idx-1)]
            
          



            for t in range(self.max_timestamp_idx-2, -1, -1):

                X_trigger = features[t]
                y_trigger = costs[t+1] - costs[t] 
                
                self.y_triggers.append(y_trigger)
                
                   
                model = KernelRidge(kernel='rbf').fit(X_trigger, y_trigger)
                self.halters[t] = model
                predicted_cost_difference = model.predict(X_trigger)  
                self.r2_scores[t]=r2_score(y_trigger,predicted_cost_difference)             
                

                

                
                for j in range(len(X_trigger)):
                    if not self.target_swap :
                        self.pred_cost_diffs[j][t]=predicted_cost_difference[j]
                        
                        if predicted_cost_difference[j] < 0:
                                costs[t, j] = costs[t+1, j]
                    else :
                        if y_trigger[j]<0:
                            costs[t, j] = costs[t+1, j]
            

    
            
        else : # regressor=='one'
            
            orig_costs = copy.deepcopy(costs)
            
            model = DQN(state_dim=features.shape[2]+1, n_actions=1, hidden_dim=32, n_layers=1)
            self.halter= model
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            num_epochs = self.num_epochs
            
            
            model.train()
            train_losses = []
            
            val_losses =[]
            avg_test_costs=[]
            avg_val_costs = []
           

            time_indices_all,X_all, y_all = self.get_full_X_y(features,orig_costs)
            features_val , costs_val = self.get_features_costs(X_probas_val,y_val)
            time_indices_all_val,X_all_val, y_all_val = self.get_full_X_y(features_val, costs_val)
            criterion = torch.nn.MSELoss()
            best_epoch = None
            best_epoch_outputs_train = None
            best_model_state = None

            self.epsilon_method = False
            for epoch in range(num_epochs):
                model.train()
                if self.one_reg_full :
                    time_indices_tr,X_tr,y_tr,train_loader = self.get_train_loader( features, costs,epoch,full=True)  #IF YOU WANT TO USE FLATTENED ALL TIMESTEPS 
                else : 
                    time_indices_tr,X_tr,y_tr,train_loader = self.get_train_loader( features, costs,epoch) # IF YOU WANT TO USE FOR EACH EPOCH A SINGLE DIFFERENT TIMESTEP FOR EACH EXAMPLE
                torch.manual_seed(42)
                
                for batch_idx, train_batch in enumerate(train_loader):
                    optimizer.zero_grad()
                
                    states, targets = train_batch
                    
                    start_idx = batch_idx * train_loader.batch_size
                    # Forward pass
                    outputs = model(states)
                    outputs =outputs.squeeze()

                
                    
                    if not self.one_reg_full : # Use this for flattening when using FOR EACH EPOCH A SINGLE DIFFERENT TIMESTEP FOR EACH EXAMPLE
                        negative_indices = (targets < 0).nonzero(as_tuple=True)[0]
                        if len(negative_indices) > 0  :
                                
                                t_indices = np.array(time_indices_tr)[negative_indices]
                                idxs = start_idx + negative_indices.numpy() if hasattr(negative_indices, 'numpy') else start_idx + negative_indices
                                costs[t_indices, idxs] = costs[t_indices + 1, idxs]
                

                    loss = criterion(outputs,targets)
                    loss.backward()
                    optimizer.step()


                # run evaluation on the val set for model selection
                if X_probas_val is not None and y_val is not None:
                    
                        
                        if  epoch >= self.start_val_epoch : #( default 20)
                            with torch.no_grad():
                                model.eval()
                                
                                X_trigger_with_timestamp_train = self.scaler.transform(X_all)
                                X_trigger_with_timestamp_val = self.scaler.transform(X_all_val) 
                                
                            
                                # Transform to tensor
                                X_trigger_train = torch.tensor(X_trigger_with_timestamp_train, dtype=torch.float32)
                                X_trigger_val = torch.tensor(X_trigger_with_timestamp_val, dtype=torch.float32)

                                outputs_train = model(X_trigger_train)
                                outputs_train = outputs_train.squeeze()
                                outputs_val = model(X_trigger_val)
                                outputs_val = outputs_val.squeeze()

                                # Monitor the loss
                                train_loss = criterion(outputs_train, torch.tensor(y_all, dtype = torch.float32)).item()
                                val_loss=criterion(outputs_val, torch.tensor(y_all_val, dtype=torch.float32)).item()

                                train_losses.append(train_loss)
                                val_losses.append(val_loss)
                                
                                # save model based on min loss so far
                                if len(val_losses)==1 or val_loss < min(val_losses[:-1]) :
                                        best_model_state = copy.deepcopy(model.state_dict())
                                        best_epoch = epoch         
                                        best_epoch_outputs_train = outputs_train 
                                
                        if  (epoch+1) % self.eval_epochs == 0:
                            
                                model.eval()
                                avg_test_cost = self._eval_test(X_test,X_probas_test,y_test)
                                avg_val_cost = self._eval_test(X_val,X_probas_val,y_val)
                                avg_test_costs.append(np.mean(avg_test_cost))
                                avg_val_costs.append(np.mean(avg_val_cost))

                            
                                        
            # convert train and test losses to numpy arrays
            self.train_losses = np.array(train_losses)
            self.val_losses = np.array(val_losses)
            self.avg_test_costs = avg_test_costs
            self.avg_val_costs = avg_val_costs
           
           
            # Set the model to evaluation mode
            if best_model_state is not None :

                model.load_state_dict(best_model_state)
                print("Loaded best model state")
            if best_epoch is not None :
                self.best_epoch = best_epoch

            model.eval()
           
            self.halter= model
            if best_epoch_outputs_train is not None :
                self.pred_cost_diffs = torch.Tensor.numpy(best_epoch_outputs_train)

            self.epsilon_method = self.eps_method_one_reg

        if self.epsilon_method:
             
                           
                    self.opt_epsilon = 0



                    lower_bound = np.percentile(self.pred_cost_diffs,5 )  
                    upper_bound = np.percentile(self.pred_cost_diffs, 95)
                    
                    # Try to adapt to unbalanced setting
                    # Create a kernel density estimate
                    kde = gaussian_kde(self.pred_cost_diffs.flatten()) 
                    # Generate a range of values to evaluate the density
                    x_values = np.linspace(lower_bound, upper_bound, 1000) 
                    density = kde(x_values)
                    # Sample candidate epsilon values based on the density
                    num_samples = self.num_samples_eps
                    np.random.seed(42)
                    candidate_epsilon = np.random.choice(x_values, size=num_samples, p=density/density.sum())
                    candidate_epsilon = np.hstack((candidate_epsilon,[0]))
                    candidate_epsilon = np.sort(candidate_epsilon)
                    # choose decreasing candidate epsilon values
                    self.candidate_epsilon = np.flip(candidate_epsilon)
                    avg_costs = []
                    # Cross-validation setup
            
                    if self.cross_val == 'kfold':
                        kf = KFold(n_splits=5,random_state=42,shuffle=True) 
                        
                        

                        for epsilon in candidate_epsilon:
                            fold_costs = []

                            for train_index, val_index in kf.split(X_probas):
                                
                                X_train, X_val = X_probas[train_index], X_probas[val_index]
                                y_train, y_val = y[train_index], y[val_index]
                                

                                # Calculate the score for the current fold
                                cost = self._get_score_sped_up(X_val, y_val, self.pred_cost_diffs[val_index], epsilon)
                                
                                
                                fold_costs.append(cost)
                            
                            

                            avg_costs.append(np.mean(fold_costs))
                    else :
                        skf = StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=42)

                        for epsilon in candidate_epsilon:
                            fold_costs = []

                            for train_index, val_index in skf.split(X_probas,y):
                                X_train, X_val = X_probas[train_index], X_probas[val_index]
                                y_train, y_val = y[train_index], y[val_index]
                                

                                # Calculate the score for the current fold
                                cost = self._get_score_sped_up(X_val, y_val, self.pred_cost_diffs[val_index], epsilon)
                                
                                fold_costs.append(cost)
                            
                            

                            avg_costs.append(np.mean(fold_costs))
               
                   
                   

                    self.opt_epsilon = candidate_epsilon[np.argmin(avg_costs)]



    
                
               
               

        return self
    
    def _eval_test(self,X,X_probas,y):
        past_trigger = np.zeros((X.shape[0], )).astype(bool)
        trigger_mask = np.zeros((X.shape[0], )).astype(bool)

        all_preds = np.zeros((X.shape[0],))-1
        all_t_star = np.zeros((X.shape[0],))-1
        all_f_star = np.zeros((X.shape[0],))-1

        

        for t, l in enumerate(self.timestamps):
            probas = X_probas[:,t,:]
            classes = probas.argmax(axis=-1)

            trigger_mask = self.predict(X[:, :l], probas, self.cost_matrices)
            
        
            # already made predictions
            if past_trigger.sum() != 0: 
                trigger_mask[np.where(past_trigger==True)] = False

            all_preds[trigger_mask] = classes[trigger_mask]
            all_t_star[trigger_mask] = l
            all_f_star[trigger_mask] = np.array([self.cost_matrices[t][int(p)][y[trigger_mask][i]]
                                                 for i, p in enumerate(classes[trigger_mask])])

            # update past triggers with current triggers
            past_trigger = past_trigger | trigger_mask

            # if all TS have been triggered 
            if past_trigger.sum() == X.shape[0]:
                break

            if l == self.timestamps[-1]:
                all_t_star[np.where(all_t_star < 0)] = l
                # final prediction is the valid prediction
                all_preds[np.where(all_preds < 0)] = classes[np.where(all_preds < 0)]
                # if no prediction so far, output majority class 
                if np.isnan(all_preds).any():
                    all_preds[np.where(np.isnan(all_preds))] = np.unique(y)[np.argmax(self.chronological_classifiers.class_prior)]

                all_f_star[np.where(all_f_star < 0)] = np.array([self.cost_matrices[-1][int(p)][y[np.where(all_f_star < 0)][i]]
                                                                for i, p in enumerate(all_preds[np.where(all_f_star < 0)])])
                break
                
        
        avg_score = np.mean(all_f_star)
        return avg_score
    

   
        
    def get_train_loader(self, features_train, costs_train,epoch,t_star_train=None,full=False):
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)
        if not full : 
            # To sample for each example a different timestep for each epoch
            time_indices_tr,X_tr, y_tr = self.get_X_y(features_train, costs_train,epoch)
        else :
            # To use all timesteps

            time_indices_tr, X_tr, y_tr = self.get_flattened_full_X_y(features_train,costs_train) 
            # To sample timesteps around t_star
            if t_star_train is not None: 
                final_dict = self.get_prob_sampled_flattened_full_X_y_pooled_detailed(features_train,costs_train,epoch,t_star_train)
                X_tr= final_dict['X_all']
                y_tr = final_dict['y_all']
                time_indices_tr = final_dict['time_indices']
                self.results_dict = final_dict['t_star_groups']
            
            shuffled_ids = np.arange(X_tr.shape[0], dtype=int)
            np.random.shuffle(shuffled_ids)
            X_tr = X_tr[shuffled_ids]
            y_tr = y_tr[shuffled_ids]
            time_indices_tr = np.array(time_indices_tr)[shuffled_ids]
            
        
                

            
        
        
        scaler = StandardScaler()
        #X_tr = self.scaler_full.transform(X_tr)
        X_tr = scaler.fit_transform(X_tr)
        
        self.scaler=scaler

        X_tr = torch.tensor(np.array(X_tr), dtype=torch.float32)
        y_tr = torch.tensor(np.array(y_tr), dtype=torch.float32)

    
        # Create a Train TensorDataset 
        train_dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
        # Create a Test TensorDataset
        
        # Transform it to loader 
        batch_size_train=features_train.shape[1]//10
        if batch_size_train % 2 == 1:
            batch_size_train += 1
        
        # Ensure batch sizes are within reasonable bounds
        min_batch_size = 4 
        max_batch_size = 128
        batch_size_train = max(min_batch_size, min(batch_size_train, max_batch_size))

        #batch_size_train = 64

        #batch_size_train = min(32, features_train.shape[1])
        #batch_size_train = (self.max_timestamp_idx-1)*5
        #batch_size_train = (self.max_timestamp_idx-1)*10
        
        #batch_size_train = min(batch_size_train,features_train.shape[1]* (self.max_timestamp_idx-1))
    
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size_train, shuffle=True, drop_last=True)
        
        return time_indices_tr,X_tr,y_tr,train_loader
      

    def get_X_y(self, features, costs,epoch):
        X_all = []
        y_all = []
        time_indices = []   
        
        np.random.seed(epoch)
        
        for j in range(features.shape[1]):  
                # select randomly a timestamp
            
            t = np.random.randint(0,self.max_timestamp_idx-1)
            time_indices.append(t)
            current_timestamp_feature = np.array([t])  
            # Concatenate the current timestamp feature with the features   
            X_with_timestamp = np.concatenate((features[t][j],current_timestamp_feature))   
            X_all.append(X_with_timestamp)
            y_trigger = costs[t + 1,j] - costs[t,j]
            self.y_triggers.append(y_trigger)
            y_all.append(y_trigger)
    
        X_all=np.array(X_all)
        y_all = np.array(y_all)
        return time_indices,X_all,y_all
    
    def get_full_X_y(self, features, costs):
        X_all = []
        y_all = []
        time_indices = []
        
        for j in range(features.shape[1]):
            for t in range(self.max_timestamp_idx-1):
                current_timestamp_feature = np.array([t])
                time_indices.append(t)
                # Concatenate the current timestamp feature with the features
                X_with_timestamp = np.concatenate((features[t][j], current_timestamp_feature))
                X_all.append(X_with_timestamp)
                y_trigger = costs[t + 1, j] - costs[t, j]
                self.y_triggers.append(y_trigger)
                y_all.append(y_trigger)

        X_all = np.array(X_all)
        y_all = np.array(y_all)
        
        return time_indices, X_all, y_all
    def get_t_star(self,X_probas,y):
        all_f = np.zeros((len(self.timestamps), len(y)))
        all_preds = np.zeros((len(self.timestamps), len(y)))

        for t, l in enumerate(self.timestamps):
            probas = X_probas[:,t,:]
            classes = np.argmax(probas, axis=-1)
            all_preds[t] = classes
            all_f[t] = np.array(
                    [self.cost_matrices[t][:][y[i]] * probas[i] for i in range(len(y))]
                ).sum(axis=-1)
        t_star_idx = all_f.argmin(axis=0)
        all_t_star = [self.timestamps[idx]
                      for idx in t_star_idx]
        return all_t_star
    
    
    def get_flattened_full_X_y(self, features, costs):
                X_all = []
                y_all = []

                num_samples = features.shape[1]
                time_indices = []
                for j in range(num_samples):  # Loop over each sample
                    # Temporary lists for this sample
                    X_sample = []
                    y_sample = []

                    # Traverse timestamps in reverse order (as before)
                    for t in range(self.max_timestamp_idx - 2, -1, -1):
                        y_trigger = costs[t + 1, j] - costs[t, j]
                        y_sample.append(y_trigger)

                        # Add timestamp as feature
                        current_timestamp_feature = np.array([t])
                        time_indices.append(t)
                        X_with_timestamp = np.concatenate((features[t, j], current_timestamp_feature))
                        X_sample.append(X_with_timestamp)

                        # Update cost if needed (same logic as before)
                        if y_trigger < 0:
                            costs[t, j] = costs[t + 1, j]
                        
                   

                    # Append all data from this sample
                    X_all.extend(X_sample)
                    y_all.extend(y_sample)

                X_all = np.array(X_all)
                y_all = np.array(y_all)

                return time_indices,X_all, y_all
    
    def get_prob_sampled_flattened_full_X_y_pooled_detailed(self, features, costs, epoch,t_star_train=None, 
                                                       samples_per_group=64, std=2.0):
       
       
        if t_star_train is None:
            # Fall back to original behavior
            return self._original_function_logic(features, costs, t_star_train)
        
        # Group samples by their t_star_train values
        unique_t_stars = np.unique(t_star_train)
        results_dict = {}
        
        for t_star_value in unique_t_stars:
            # Find all samples with this t_star_train value
            sample_indices = np.where(t_star_train == t_star_value)[0]
            
            #print(f"Processing t_star={t_star_value} with {len(sample_indices)} samples")
            
            # Make a copy of costs to avoid modifying the original
            costs_copy = costs.copy()
            
            # Collect ALL timesteps from ALL samples in this group into one big pool
            X_pool = []
            y_pool = []
            time_pool = []
            sample_id_pool = []  # Track which sample each timestep came from
            
            # First pass: collect all timesteps from all samples in this group
            for j in sample_indices:
                # Process all timesteps for this sample
                for t in range(self.max_timestamp_idx - 2, -1, -1):
                    y_trigger = costs_copy[t + 1, j] - costs_copy[t, j]
                    
                    # Add timestamp as feature
                    current_timestamp_feature = np.array([t])
                    X_with_timestamp = np.concatenate((features[t, j], current_timestamp_feature))
                    
                    # Add to the pool
                    X_pool.append(X_with_timestamp)
                    y_pool.append(y_trigger)
                    time_pool.append(t)
                    sample_id_pool.append(j)
                    
                    # Update cost if needed (same logic as before)
                    if y_trigger < 0:
                        costs_copy[t, j] = costs_copy[t + 1, j]
            
            # Convert pools to arrays for easier handling
            X_pool = np.array(X_pool)
            y_pool = np.array(y_pool)
            time_pool = np.array(time_pool)
            sample_id_pool = np.array(sample_id_pool)
            
            total_pool_size = len(X_pool)
            
            # Now sample from this entire pool using probability distribution centered on t_star_value
            mean = t_star_value
            
            time_pool_array = np.array(time_pool)
            squared_distances = ((time_pool_array - mean) / std) ** 2
            
            # Prevent overflow in exp by clipping very large values
            max_exp_arg = 700  # exp(700) is close to the limit before overflow
            squared_distances = np.clip(squared_distances, 0, max_exp_arg)
            
            probs = np.exp(-0.5 * squared_distances)
            
            # Handle the case where all probabilities are zero (shouldn't happen with clipping, but safety check)
            if np.sum(probs) == 0:
                probs = np.ones(len(time_pool)) / len(time_pool)
            else:
                probs = probs / np.sum(probs)  # Normalize to get valid probabilities
            
            # Final check for NaN values
            if np.any(np.isnan(probs)):
                
                probs = np.ones(len(time_pool)) / len(time_pool)
            
            
            # Show some statistics about the sampling distribution
            most_likely_timesteps = time_pool[probs > np.percentile(probs, 90)]
        
            
            # Determine how many samples to draw
            if samples_per_group is None:
                num_samples_to_draw = total_pool_size  # Use all available
            else:
                num_samples_to_draw = min(samples_per_group, total_pool_size)
            
        
            
            np.random.seed(epoch)
            # Sample from the entire pool
            sampled_indices = np.random.choice(
                total_pool_size, 
                size=num_samples_to_draw, 
                p=probs, 
                replace=True
            )
            
            # Extract the sampled data
            X_sampled = X_pool[sampled_indices]
            y_sampled = y_pool[sampled_indices]
            time_sampled = time_pool[sampled_indices]
            sample_id_sampled = sample_id_pool[sampled_indices]
            
            # Store detailed results for this t_star value
            results_dict[f't_star_{t_star_value}'] = {
                'time_indices': time_sampled.tolist(),
                'X_all': X_sampled,
                'y_all': y_sampled,
                'sample_ids': sample_id_sampled.tolist(),  # Which original samples the data came from
                'total_pool_size': total_pool_size,
                'num_samples_drawn': num_samples_to_draw,
                't_star_value': t_star_value,
                'original_sample_indices': sample_indices.tolist(),
                'timestep_distribution': {
                    'unique_timesteps': np.unique(time_sampled).tolist(),
                    'timestep_counts': {str(t): int(np.sum(time_sampled == t)) for t in np.unique(time_sampled)}
                }
            }
        # Concatenate all X_all arrays across different t_star values
        all_X = np.vstack([group['X_all'] for group in results_dict.values()])
        all_y = np.concatenate([group['y_all'] for group in results_dict.values()])
        all_time_indices = np.concatenate([group['time_indices'] for group in results_dict.values()])

        # Create the final results dictionary including the concatenated arrays
        final_results = {
            'X_all': all_X,
            'y_all': all_y,
            'time_indices': all_time_indices,
            't_star_groups': results_dict  # Keep the original grouped data
        }
        return final_results  
    
    def detect_flat_regions(self,signal, threshold=1e-5, min_flat_length=1, min_height_diff=0.0):
       
        signal = np.asarray(signal)
        diff = np.abs(np.diff(signal))
        flat_mask = diff < threshold

        ref_points = []
        last_flat_value = None
        i = 0
        plateaus = [] # to store all indices of flat regions
        plateau_index= 0 # to store the index of the current plateau
        timestamps_plateaus = {}

        while i < len(flat_mask):
            if flat_mask[i]:
                
                timestamps_plateaus[i] = plateau_index

                start = i
                plateau=[start]
                while i < len(flat_mask) and flat_mask[i]:
                    i += 1
                    timestamps_plateaus[i] = plateau_index
                    plateau.append(i)
                end = i + 1  # tentative end of flat region
                if (i - start + 1) >= min_flat_length and i != len(signal) - 1:
                    flat_value = np.mean(signal[start:end])  # average value of flat region

                    if last_flat_value is None or abs(flat_value - last_flat_value) >= min_height_diff:
                        ref_points.append(end)
                        last_flat_value = flat_value
                    
                        plateaus.append(plateau)
                        plateau_index+=1
                
            else:
                i += 1

        return plateaus, timestamps_plateaus,ref_points
    def _fit2(self, X_probas, y):
        if self.regressor not in ['one', 'many']:
            raise ValueError("regressor must be 'one' or 'many'")

        self.max_timestamp_idx = len(self.timestamps)
        features, costs = self.get_features_costs(X_probas, y)
        self.orig_costs = copy.deepcopy(costs)
        self.pred_cost_diffs=np.zeros((X_probas.shape[0], self.max_timestamp_idx-1))
        self.pred_cost_diff_max_probas = np.zeros((X_probas.shape[0], self.max_timestamp_idx-1))
    

        self.halters = [None for _ in range(self.max_timestamp_idx-1)]
        
        
        for t in range(self.max_timestamp_idx-2, -1, -1):

                X_trigger = features[t]

                if not self.var_classificat:
                   
                    y_trigger = costs[t+1] - costs[t] # local cost difference only teaches the model whether waiting one step reduces cost 
                    self.y_triggers.append(y_trigger)   
                    if not self.target_swap :
                        model = KernelRidge(kernel='rbf').fit(X_trigger, y_trigger)
                        self.halters[t] = model
                        predicted_cost_difference = model.predict(X_trigger) 
                
                else :
                    """ Variant  : Converting to binary classification problem ( 0s are counted as positive class)"""
                    
                    y_trigger = (costs[t+1] - costs[t] >= 0).astype(int)
                    unique_classes = np.unique(y_trigger)
                    if unique_classes.size == 1:
                        
                        
                        # Don't learn any model, just trigger 
                        predicted_cost_difference = np.ones(X_probas.shape[0])*y_trigger[0]
                        pred_cost_diff_max_proba = np.ones(X_probas.shape[0])



                    else:
                        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='auc').fit(X_trigger, y_trigger)
                
                        predicted_cost_difference = model.predict(X_trigger)
                        pred_cost_diff_probas = model.predict_proba(X_trigger)
                        pred_cost_diff_max_proba = np.max(pred_cost_diff_probas, axis=1)
                    
                        self.halters[t] = model 
                    
                
                for j in range(len(X_trigger)):
                    
                   
                    if not self.var_classificat :
                            if not self.target_swap:
                                self.pred_cost_diffs[j][t]=predicted_cost_difference[j]
                            if self.target_swap :
                            
                                if y_trigger[j] < 0 : 
                                    costs[t, j] = costs[t+1, j]    
                            else : 
                                if predicted_cost_difference[j] < 0: # future cost is propagated down the time axis if and only if the sample triggers a waiting decision.
                                    costs[t, j] = costs[t+1, j]
                    else : # binary classification problem
                            """ Variant  : Converting to binary classification problem ( 0s are counted as positive class)"""
                            if not unique_classes.size == 1:
                                self.pred_cost_diff_max_probas[j][t]=pred_cost_diff_max_proba[j]
                                
                                self.pred_cost_diffs[j][t]=predicted_cost_difference[j]
                                if self.target_swap :
                                
                                    if y_trigger[j] == 0 : 
                                        costs[t, j] = costs[t+1, j]    
                                    
                                else : 
                                    if predicted_cost_difference[j] == 0 : 
                                        costs[t, j] = costs[t+1, j]
        
        self.flat_costs = copy.deepcopy(costs)  

        

              

        
        """Variant : Reference point"""  
        if self.var_ref_point :
            self.ref_points = [None  for _ in range(X_probas.shape[0])]
            self.pred_cost_diffs=np.zeros((X_probas.shape[0], self.max_timestamp_idx-1))
            self.halters = [None for _ in range(self.max_timestamp_idx-1)]
            for j in range(X_probas.shape[0]):
                min_height_diff = 0.05 * (np.max((self.flat_costs[:,j])) - np.min((self.flat_costs[:,j])) )
                plateaus, timestamps_plateaus,ref_points=self.detect_flat_regions(self.flat_costs[:,j],min_flat_length = self.min_flat_length,min_height_diff=min_height_diff) 
                self.plateaus=plateaus
                "Try later to tune with height of step between plateaus and minimum plateau size"
                self.ref_points[j]=ref_points if len(ref_points)>=1 else np.array([0])

            for t in range(self.max_timestamp_idx-1):
                X_trigger = features[t]
                
                y_trigger = np.zeros(X_probas.shape[0])
                for j in range(X_probas.shape[0]):
                    # select the ref point after the nearest plateau
                    candidates = [ num for num in self.ref_points[j] if num >= t]
                    nearest_ref_point = min(candidates) if candidates else t+1 
                    y_trigger[j]= self.flat_costs[nearest_ref_point,j]- self.orig_costs[t,j]

                model = KernelRidge(kernel='rbf').fit(X_trigger, y_trigger)
                self.halters[t]= model
                self.pred_cost_diffs[:,t]=self.halters[t].predict(X_trigger)

        """Variant  : True Plateau"""
        if self.var_true_plateau :
            self.pred_cost_diffs = np.zeros((X_probas.shape[0], self.max_timestamp_idx-1))
            self.halters = [None for _ in range(self.max_timestamp_idx-1)]
            for t in range(self.max_timestamp_idx-2, -1, -1):
                X_trigger = features[t]
                y_trigger =np.zeros(X_probas.shape[0])
                for j in range(X_probas.shape[0]):
                    min_height_diff = 0.05 * (np.max((self.flat_costs[:,j])) - np.min((self.flat_costs[:,j])) )
                    plateaus, timestamps_plateaus,ref_points=self.detect_flat_regions(self.flat_costs[:,j],min_flat_length = self.min_flat_length, min_height_diff=min_height_diff)
                    
                    true_plateau = any( t in p for p in plateaus)
                    
                    y_trigger[j]= self.flat_costs[t+1,j]-self.orig_costs[t,j]
                    print(y_trigger[j])
                    if y_trigger[j] == 0 and true_plateau :
                        plateau_end = plateaus[timestamps_plateaus[t]][-1]  # Get the end of the plateau
                        y_trigger[j] = self.flat_costs[plateau_end+1,j] - self.orig_costs[plateau_end,j]
                model = KernelRidge(kernel='rbf').fit(X_trigger, y_trigger)
                self.halters[t]= model
                self.pred_cost_diffs[:,t]=self.halters[t].predict(X_trigger)
                

        
        if self.epsilon_method :
             
           
                self.opt_epsilon = 0

                if not self.var_classificat :

                    lower_bound = np.percentile(self.pred_cost_diffs,5 )  
                    upper_bound = np.percentile(self.pred_cost_diffs, 95)
                
                    
                    
                    kde = gaussian_kde(self.pred_cost_diffs.flatten())
                    
                else :
                    lower_bound = np.percentile(self.pred_cost_diff_max_probas,5 )  
                    upper_bound = np.percentile(self.pred_cost_diff_max_probas, 95)
                
                    try :
                        kde = gaussian_kde(self.pred_cost_diff_max_probas.flatten())
                
                    # catch LinAlg      : SIngular Matrix error
                    except np.linalg.LinAlgError:
                        # Fallback: add small noise to avoid singular matrix error
                        noise = np.random.normal(scale=1e-8, size=self.pred_cost_diff_max_probas.flatten().shape)
                        kde = gaussian_kde(self.pred_cost_diff_max_probas.flatten() + noise)
                x_values = np.linspace(lower_bound, upper_bound, 1000) 
                density = kde(x_values)

                # Sample candidate epsilon values based on the density
                num_samples = self.num_samples_eps
                np.random.seed(42) # i just added it at 04/06/2025
                candidate_epsilon = np.random.choice(x_values, size=num_samples, p=density/density.sum())
                if not self.var_classificat: 
                    candidate_epsilon = np.hstack((candidate_epsilon,[0])) 
                   
                candidate_epsilon = np.sort(candidate_epsilon)
                # choose decreasing candidate epsilon values
                self.candidate_epsilon = np.flip(candidate_epsilon)
              
                # Cross-validation setup
                avg_costs = []           
                kf = KFold(n_splits=5,random_state=42,shuffle=True) 

                for epsilon in candidate_epsilon:
                    fold_costs = []

                    for train_index, val_index in kf.split(X_probas):
                        
                        X_train, X_val = X_probas[train_index], X_probas[val_index]
                        y_train, y_val = y[train_index], y[val_index]
                        

                        # Calculate the score for the current fold
                        if not self.var_classificat :
                            cost = self._get_score_sped_up(X_val, y_val, self.pred_cost_diffs[val_index], epsilon)
                        else :
                            cost = self._get_score_sped_up(X_val, y_val, self.pred_cost_diffs[val_index], epsilon,self.pred_cost_diff_max_probas[val_index])
                            
                        
                        
                        fold_costs.append(cost)
                    
                    

                    avg_costs.append(np.mean(fold_costs))
               
                self.opt_epsilon = candidate_epsilon[np.argmin(avg_costs)]
    
        return self





    def get_features_costs(self, X_probas, y):
        if self.variant == 'transpose':
            results  = [self._generate_features_variant_transpose(X_probas[:,t,:], t, y) 
                        for t in range(X_probas.shape[1])]
        elif self.variant == 'prior':
            results  = [self._generate_features_variant_prior(X_probas[:,t,:], t, y) 
                        for t in range(X_probas.shape[1])]
        elif self.variant == 'corrected_prior':
            results  = [self._generate_features_variant_corrected_prior(X_probas[:,t,:], t, y) 
                        for t in range(X_probas.shape[1])]
        elif self.variant == 'p_hat_p_hat':
            results  = [self._generate_features_variant_p_hat_p_hat(X_probas[:,t,:], t, y) 
                        for t in range(X_probas.shape[1])]
        elif self.variant == 'true_label':
            results  = [self._generate_features_variant_true_label(X_probas[:,t,:], t, y) 
                        for t in range(X_probas.shape[1])]
        else:
            results  = [self._generate_features(X_probas[:,t,:], t, y) 
                        for t in range(X_probas.shape[1])]
        
        features, costs = zip(*results)
        features, costs = (np.array(features), np.array(costs))
        return features,costs

    
    def _get_score_sped_up(self, X_probas, y, pred_cost_diffs_mat, epsilon,pred_cost_diffs_max_probas=None):
        costs = []
        
        # Iterate over each sample
        for i, probas in enumerate(X_probas):
            # Determine the indices where the condition is met
            if pred_cost_diffs_max_probas is None : 
                triggers = pred_cost_diffs_mat[i] > epsilon
            else :
                
                triggers = np.equal(pred_cost_diffs_mat[i],1) & (pred_cost_diffs_max_probas[i] > epsilon)
            
            # Find the first trigger or the last timestamp
            if np.any(triggers):
                j = np.argmax(triggers)  # Get the first index where trigger is True
            else:
                j = len(self.timestamps) - 2  # Last timestamp if no triggers
            
            # Get the predicted class
            pred = np.argmax(probas[j])
            c = self.cost_matrices[j][pred][y[i]]
            
            costs.append(c)

        return np.mean(costs)
    
        



        


    
    def _predict(self, X_probas, X_timestamps):

        triggers, self.cost_forecast = [], []
        for i, probas in enumerate(X_probas):
        
            trigger = False
            # if last timestamp is reached
            if X_timestamps[i] == self.timestamps[-1]:
                triggers.append(True)
                self.cost_forecast.append(np.nan)
                continue

            time_idx = np.where(X_timestamps[i] == self.timestamps)[0][0]
            if self.variant=='transpose':
                X_trigger, _ = self._generate_features_variant_transpose(probas[None,:], time_idx, 
                                                   np.zeros(X_probas.shape[0], dtype=int))
            elif self.variant=='prior':
                X_trigger, _ = self._generate_features_variant_prior(probas[None,:], time_idx, 
                                                   np.zeros(X_probas.shape[0], dtype=int))
            elif self.variant=='corrected_prior':
                X_trigger, _ = self._generate_features_variant_corrected_prior(probas[None,:], time_idx, 
                                                   np.zeros(X_probas.shape[0], dtype=int))
            elif self.variant=='p_hat_p_hat':
                X_trigger, _ = self._generate_features_variant_p_hat_p_hat(probas[None,:], time_idx, 
                                                   np.zeros(X_probas.shape[0], dtype=int))
            elif self.variant=='true_label':
                X_trigger, _ = self._generate_features_variant_true_label(probas[None,:], time_idx, 
                                                   np.zeros(X_probas.shape[0], dtype=int))
            else:
                X_trigger, _ = self._generate_features(probas[None,:], time_idx, 
                                                   np.zeros(X_probas.shape[0], dtype=int))
            
            if self.regressor=='many':
                if self.var_classificat:

                    
                    try:
                        test_cost_diff = self.halters[time_idx].predict(X_trigger)
                        max_proba_test_cost_diff = np.max(self.halters[time_idx].predict_proba(X_trigger))
                    except AttributeError: # We didn't learn any model, we always trigger
                       test_cost_diff = 1
                       max_proba_test_cost_diff=1
                else :
                    test_cost_diff = self.halters[time_idx].predict(X_trigger)
        
            else: # regressor=='one'
                current_timestamp_feature = np.array([time_idx])  # Assuming t is a scalar
                
                # Concatenate the current timestamp feature with the features
                X_trigger = X_trigger.squeeze()  # Remove the extra dimension
               
                X_trigger_with_timestamp = np.concatenate((X_trigger, current_timestamp_feature))
                X_trigger_with_timestamp = self.scaler.transform(X_trigger_with_timestamp.reshape(1, -1))  # Reshape to 2D for scaling
                # Transform to tensor
                X_trigger = torch.tensor(X_trigger_with_timestamp, dtype=torch.float32)
                
                if self.var_classificat:

                    
                    try:
                        test_cost_diff = self.halter.predict(X_trigger)
                        max_proba_test_cost_diff = np.max(self.halter.predict_proba(X_trigger))
                    except AttributeError: # We didn't learn any model, we always trigger
                       test_cost_diff = 1
                       max_proba_test_cost_diff=1
                else :
                    # predict the output with trained DQN model
                    # just forward pass
                    test_cost_diff = self.halter(X_trigger)
                    test_cost_diff = test_cost_diff.detach().numpy()
            
            if not self.var_classificat :
                if not self.epsilon_method:
                    if test_cost_diff > 0:
                        trigger = True
               
                else :
                    
                    if test_cost_diff >= self.opt_epsilon :
                    
                        trigger = True
            else : # Binary Classification problem 
                
                if not self.epsilon_method:
                    if test_cost_diff == 1:
                        trigger = True
                
                else :
                
                    if  test_cost_diff == 1 and max_proba_test_cost_diff >= self.opt_epsilon :
                        trigger = True
            triggers.append(trigger)
            self.cost_forecast.append(test_cost_diff if isinstance(test_cost_diff,int) else test_cost_diff[0])
        
        

        return np.array(triggers)