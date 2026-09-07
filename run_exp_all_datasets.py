from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import copy
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['KMP_WARNINGS'] = 'off'
import time
import numpy as np
import sys
import json
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifierCV
from aeon.classification.convolution_based import MiniRocketClassifier
from joblib import Parallel, delayed
os.chdir('C:\\Users\\WTHK0243\\OneDrive - orange.com\\Bureau\\ml_edm\\src\\ml_edm')
from warnings import filterwarnings
# Use the current working directory as the base path
sys.path.append(os.path.abspath(os.path.join('C:\\Users\\WTHK0243\\OneDrive - orange.com\\Bureau\\ml_edm\\src\\ml_edm\\tests', '..','..')))

from ml_edm.classification.classifiers_collection import ClassifiersCollection
from ml_edm.utils import check_timestamps 
from ml_edm.early_classifier import EarlyClassifier
from ml_edm.cost_matrices import CostMatrices
from ml_edm.trigger import *
from ml_edm.alert import Alert

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def plot_kde_triggers_distribution(trigger,alpha,dataset,pred_cost_diffs,y_triggers,opt_epsilon):
    kde = gaussian_kde(pred_cost_diffs.flatten())
    lower_bound = np.percentile(pred_cost_diffs, 5)  # 5th percentile
    upper_bound = np.percentile(pred_cost_diffs, 95)  # 95th percentile
    candidate_epsilon_no_kde=np.linspace(lower_bound,upper_bound,num=50)
    # Generate a range of values to evaluate the density
    x_values = np.linspace(lower_bound, upper_bound, 1000)
    density = kde(x_values)

    # Sample candidate epsilon values based on the density
    num_samples = 50  # Adjust the number of samples as needed
    candidate_epsilon = np.random.choice(x_values, size=num_samples, p=density/density.sum())
    candidate_epsilon = np.sort(candidate_epsilon)
    plt.plot(x_values, density, label='Density Estimate')
    plt.hist(pred_cost_diffs.flatten(),bins=70, density=True, alpha=0.5, label='Pred Cost Diff Histogram')
    plt.hist(np.array(y_triggers).flatten(),bins=70, density=True, alpha=0.4, label='True Cost Diff Histogram')
    plt.scatter(candidate_epsilon, np.zeros_like(candidate_epsilon), color='red', label='Sampled Epsilon Points',alpha=0.8)
    plt.scatter(candidate_epsilon_no_kde, np.zeros_like(candidate_epsilon_no_kde), color='black', label='Sampled Epsilon Points no kde',alpha=0.5)
    plt.axvline(x=lower_bound, color='green', linestyle='--', label='5th Percentile')
    plt.axvline(x=upper_bound, color='blue', linestyle='--', label='95th Percentile')

    plt.axvline(x=opt_epsilon, color='red', linestyle='--', label='Opt epsilon')
    #plt.xlim(lower_bound-2,upper_bound+2)

    plt.legend()
    os.chdir('C:\\Users\\WTHK0243\\OneDrive - orange.com\\Bureau\\ml_edm\\src\\ml_edm\\tests\\plots_unbalanced_exp_delay_global_minima')
    plt.savefig(f'{trigger}_{alpha}_{dataset}_kde_epsilon_trigger_distribution.pdf')
    plt.close('all')
def visualize_trigger_values_grid(trigger,alpha,dataset,y_triggers, pred_cost_diffs, optimal_epsilon, num_lines=5):
    """
    Visualize the distribution of trigger values and predicted cost differences in a grid layout.

    Parameters:
    - y_triggers: List of arrays containing trigger values for each timestamp.
    - pred_cost_diffs: List of arrays containing predicted cost differences for each timestamp.
    - optimal_epsilon: The optimal epsilon value to be marked with a vertical line.
    - num_lines: Number of rows in the grid layout.
    """
    num_timestamps = len(y_triggers)
    num_cols = (num_timestamps + num_lines - 1) // num_lines  # Calculate the number of columns

    # Determine global min and max for x-axis
    all_values = np.concatenate(y_triggers + pred_cost_diffs)
    global_min = np.min(all_values)
    global_max = np.max(all_values)

    # Set a consistent y-axis limit
    max_frequency = 0
    for t in range(num_timestamps):
        max_frequency = max(max_frequency, np.histogram(y_triggers[t], bins=30)[0].max(), 
                            np.histogram(pred_cost_diffs[t], bins=30)[0].max())

    fig, axes = plt.subplots(num_lines, num_cols, figsize=(15, 5 * num_lines))
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    for t in range(num_timestamps):
        # Plot histograms for y_triggers and pred_cost_diffs
        axes[t].hist(y_triggers[t], alpha=0.5, color='blue', label='y_triggers')
        axes[t].hist(pred_cost_diffs[t], alpha=0.8, color='orange', label='pred_cost_diffs')
        
        axes[t].set_xlim(global_min, global_max)  # Set consistent x-axis limits
        axes[t].set_ylim(0, max_frequency)  # Set consistent y-axis limits
        axes[t].set_xlabel('Trigger Value')
        axes[t].set_ylabel('Frequency')
        axes[t].set_title(f'Trigger Values at Timestamp {t}')
        axes[t].grid()

        # Add vertical lines at 0 and optimal_epsilon
        axes[t].axvline(x=0, color='red', linestyle='--', label='0')
        axes[t].axvline(x=optimal_epsilon, color='green', linestyle='--', label='Optimal Epsilon')
        axes[t].legend()

    # Hide any unused subplots
    for t in range(num_timestamps, len(axes)):
        axes[t].axis('off')

    plt.tight_layout()
    os.chdir('C:\\Users\\WTHK0243\\OneDrive - orange.com\\Bureau\\ml_edm\\src\\ml_edm\\tests\\plots_unbalanced_exp_delay_global_minima')
    plt.savefig(f'{trigger}_{alpha}_{dataset}_timestamps_epsilon_histogramms_trigger.pdf')
    plt.close('all')

def plot_plateau_costs(trigger,alpha,dataset,orig_costs,flat_costs,ref_points,n_series=12,num_lines=3):
    num_cols = (n_series+num_lines-1)  // num_lines
    fig, axes = plt.subplots(nrows=num_lines, ncols=num_cols, figsize=(10 * num_cols, 4 * num_lines), sharey=True)
    n=0
    classes_num_series = {c : 0 for c in range(datasets_num_classes[dataset])}
    i=0
    axes = axes.flatten()  # pour simplifier la gestion
    
    
    while n < n_series :
        c=true_labels_dict[dataset][i]
        if min(classes_num_series.values())==0:
        
            n+=1

            ax = axes[i]
            ax.set_xlabel('Time')
            
            ax.plot(orig_costs[:,i], label = 'Original cost')
            ax.plot(flat_costs[:,i], label = 'Flattened cost')
            for t in ref_points[i]:
        
                plt.axvline(x=t, color='red', linestyle='--', label='Ref  point')
            ax.set_title(f'Class {c}')
            ax.legend()

        elif classes_num_series[c]==0:
            classes_num_series[c]=1
            n+=1

            ax = axes[i]
            ax.set_xlabel('Time')
            
            ax.plot(orig_costs[:,i], label = 'Original cost')
            ax.plot(flat_costs[:,i], label = 'Flattened cost')
            for t in ref_points[i]:
        
                plt.axvline(x=t, color='red', linestyle='--', label='Ref  point')
            ax.set_title(f'Class {c}')
            ax.legend()
        i=i+1
    

            
    for i in range(n_series,len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    os.chdir('C:\\Users\\WTHK0243\\OneDrive - orange.com\\Bureau\\ml_edm\\src\\ml_edm\\tests\\plots_costs_plateau_5percent')
    plt.savefig(f'{trigger}_{alpha}_{dataset}_plots_costs_plateau.pdf')
    plt.close('all')



    

def load_dataset(dataset_name, split, z_normalize=False):
    
    data_train = np.loadtxt(f'{dataset_name}\\{dataset_name}_TRAIN.txt')
    data_test = np.loadtxt(f'{dataset_name}\\{dataset_name}_TEST.txt')
    
    X_train, X_test = (data_train[:, 1:], data_test[:, 1:])
    y_train, y_test = (data_train[:, 0], data_test[:, 0])

    if split != 'default':
        X_ = np.concatenate((X_train, X_test), axis=0)
        y_ = np.concatenate((y_train, y_test), axis=0)
        
        X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=split, random_state=44, stratify=y_)
    
    if z_normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    lb = LabelEncoder()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    data_dict  = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }
    
    return data_dict
def train_early_classifier(X_trigger,y_trigger,X_test,y_test,alpha,timestamps, clf, misclf_cost,delay_cost,trigger,return_post=False):
    n_classes= misclf_cost.shape[0]
    

    cost_matrices = CostMatrices(
        timestamps=timestamps, 
        n_classes=n_classes, 
        misclf_cost=misclf_cost,
        alpha=alpha,
        delay_cost=delay_cost

    )
    # define early classifier 
    early_clf = EarlyClassifier(
        chronological_classifiers=clf, 
        trigger_model=trigger, 
        cost_matrices=cost_matrices, 
        prefit_classifiers=True
    )  
   
    if isinstance(trigger, Alert):
        trigger = copy.deepcopy(trigger)
        X_probas = np.stack(clf.predict_past_proba(X_trigger, cost_matrices))
        X_probas_test = np.stack(clf.predict_past_proba(X_test, cost_matrices))

        trigger.fit(X_trigger, X_probas, y_trigger, X_probas_test, y_test,X_test, cost_matrices) # error  8 positional arg were given instead of 5              
        res = early_clf.score(X_test, y_test, return_metrics=True) 

        return copy.deepcopy(res)
    elif isinstance(trigger, CALIMERA) and trigger.regressor=='one':
            X_trigger_train,X_trigger_val,y_trigger_train,y_trigger_val = train_test_split(X_trigger,y_trigger,test_size=0.3,random_state=42)
            unique_classes = np.unique(y_trigger)
            missing_classes = [cls for cls in unique_classes if cls not in np.unique(y_trigger_train)]
            if missing_classes:
                # Add one sample of each missing class from y_trigger_val to y_trigger_train
                for cls in missing_classes:
                    idx = np.where(y_trigger_val == cls)[0][0]
                    # Concatenate the sample to train set and remove from val set
                    X_trigger_train = np.vstack([X_trigger_train, X_trigger_val[idx]])
                    y_trigger_train = np.append(y_trigger_train, y_trigger_val[idx])
                    X_trigger_val = np.delete(X_trigger_val, idx, axis=0)
                    y_trigger_val = np.delete(y_trigger_val, idx)

            X_probas = np.stack(clf.predict_past_proba(X_trigger_train, cost_matrices))
            X_probas_test = np.stack(clf.predict_past_proba(X_test, cost_matrices))
            X_probas_val = np.stack(clf.predict_past_proba(X_trigger_val, cost_matrices))
            trigger.fit(X=X_trigger_train,X_probas=X_probas, y=y_trigger_train,X_val=X_trigger_val,X_probas_val= X_probas_val,y_val=y_trigger_val,X_test=X_test,  X_probas_test=X_probas_test,y_test=y_test,cost_matrices=cost_matrices )           
            return early_clf.score(X_test, y_test, return_metrics=True) 

    

        
 
    else: 
        early_clf.fit(X_trigger, y_trigger)
        if return_post : 
            return early_clf.get_post(X_test, y_test, use_probas=False, return_metrics=True)
        else :

            return early_clf.score(X_test, y_test, return_metrics=True)



def process_dataset_fixed_alpha(metrics,trigger_models,dataset,X_trigger,y_trigger,X_test,y_test,alpha,timestamps, collection_clf,misclf_cost_mat,delay_cost):
    for trigger in trigger_models:
        if trigger == 'Economy':
            trigger_model =  EconomyGamma(timestamps,n_jobs=-1)

        elif trigger == 'CALIMERA':
            trigger_model = CALIMERA(timestamps, n_jobs=-1)
        elif trigger == 'ALERT':
            trigger_model = Alert(timestamps, include_max_proba=True, include_probas=False, include_margin=True, 
                               include_time=True, include_conf=True, include_pred=True, include_serie=False,
                               learning_rate=1e-4, tau=3e-3, n_layer=1, hidden_dim=32, batch_size=64, past_probas=0, 
                               num_epochs=2000, online_finetuning=False, n_resample=4, n_jobs=5, random_state=42,
                               eval_freq=10, cql_loss=False, discount_factor=1)
        elif trigger == 'CALIMERA_no_flatten':
            trigger_model = CALIMERA(timestamps, n_jobs=-1,flatten=False)
        elif trigger == 'CALIMERA_prior':
            trigger_model = CALIMERA(timestamps,n_jobs=-1, variant='corrected_prior')
        elif trigger == 'CALIMERA_one_reg':
            trigger_model = CALIMERA(timestamps, n_jobs=-1,regressor='one',num_epochs=1000,eval_epochs=10,start_val_epoch=20)
        elif trigger == 'CALIMERA_one_reg_eps':
            trigger_model = CALIMERA(timestamps, n_jobs=-1,regressor='one',eps_method=True,num_epochs=1000,eval_epochs=10,start_val_epoch=20)
        elif trigger == 'CALIMERA_one_reg_eps_true_plateau' :
            trigger_model = CALIMERA(timestamps, n_jobs = -1, regressor='one',var_true_plateau = True,epsilon_method =True, num_epochs = 200)
        elif trigger == 'CALIMERA_true_label':
            trigger_model = CALIMERA(timestamps, n_jobs=-1,variant='true_label')
        elif trigger == 'CALIMERA_true_target' :
            trigger_model = CALIMERA(timestamps, n_jobs = -1, target_swap = True)
        elif trigger == 'CALIMERA_eps' :
        
            trigger_model = CALIMERA(timestamps, epsilon_method = True,n_jobs=-1)
        elif trigger == 'CALIMERA_eps_class':
            trigger_model = CALIMERA(timestamps, epsilon_method = True,var_classificat=True,n_jobs=-1)

        elif trigger == 'CALIMERA_eps_true_plateau':
            trigger_model = CALIMERA(timestamps,n_jobs=-1,target_swap=True,epsilon_method=True, var_true_plateau=True)
        
        elif trigger == 'CALIMERA_eps_ref_point' :
            # Tune dynamically min flat length
            #base = 1
            #max_increase = 3
            #min_flat_length = int(round(base+max_increase*alpha))
            trigger_model = CALIMERA(timestamps, epsilon_method = True,n_jobs=-1,target_swap=True,var_ref_point=True,min_flat_length=1)
        elif trigger == 'CALIMERA_eps_calib':
            trigger_model = CALIMERA(timestamps, epsilon_method = True,n_jobs=-1,cal_reg=True,eps_cal=True)
        elif trigger == 'CALIMERA_true_label_eps' :
            
            trigger_model = CALIMERA(timestamps, n_jobs=-1,variant='true_label',epsilon_method=True)
        elif trigger == 'CALIMERA_true_label_eps_true_plateau' :
            
            trigger_model = CALIMERA(timestamps, n_jobs=-1,variant='true_label',epsilon_method=True,var_true_plateau=True,target_swap=True)
        
        elif trigger == 'CALIMERA_true_label_eps_class' :
            
            trigger_model = CALIMERA(timestamps, n_jobs=-1,variant='true_label',var_classificat=True,epsilon_method=True)
        elif trigger == 'CALIMERA_true_label_eps_ref_point' :
            # Tune dynamically min flat length
            #base = 1
            #max_increase = 3
            #min_flat_length = int(round(base+max_increase*alpha))
            trigger_model = CALIMERA(timestamps, n_jobs=-1,variant='true_label',epsilon_method=True,target_swap=True,var_ref_point=True,min_flat_length=1)
        elif trigger == 'CALIMERA_true_label_eps_one_reg':
            trigger_model = CALIMERA(timestamps, n_jobs=-1,variant='true_label',epsilon_method=True,regressor='one')
        elif trigger == 'CALIMERA_true_label_eps_calib':
            trigger_model = CALIMERA(timestamps, n_jobs=-1,variant='true_label',epsilon_method=True,cal_reg=True,eps_cal=True)
       


        else:
            
            trigger_model = ProbabilityThreshold(timestamps, n_jobs=-1)

        metrics[trigger][alpha][dataset] = train_early_classifier(X_trigger, y_trigger, X_test, y_test, alpha,
                                                                            timestamps, collection_clf,
                                                                            misclf_cost=misclf_cost_mat,
                                                                            delay_cost=delay_cost,
                                                                            trigger=trigger_model)
        
        """metrics_post[alpha][dataset] =  train_early_classifier(X_trigger, y_trigger, X_test, y_test, alpha,
                                                                            timestamps, collection_clf,
                                                                            misclf_cost=misclf_cost_mat,
                                                                            delay_cost=delay_cost,
                                                                            trigger= ProbabilityThreshold(timestamps, n_jobs=-1,manual_threshold=0),return_post=True)"""
        if trigger == 'CALIMERA_eps' or trigger =='CALIMERA_true_label_eps' or  trigger == 'CALIMERA_eps_flat' or trigger == 'CALIMERA_true_label_eps_flat':
            y_triggers_dict[trigger][alpha][dataset] = trigger_model.y_triggers
            pred_cost_diffs_dict[trigger][alpha][dataset] = trigger_model.pred_cost_diffs
            opt_epsilon_dict[trigger][alpha][dataset] = trigger_model.opt_epsilon
            orig_costs_dict[trigger][alpha][dataset] = trigger_model.orig_costs
            flat_costs_dict[trigger][alpha][dataset] = trigger_model.flat_costs
        if trigger == 'CALIMERA_eps_ref_point' or trigger == 'CALIMERA_true_label_eps_ref_point' :
            ref_points_dict[trigger][alpha][dataset] = trigger_model.ref_points
        if trigger == 'CALIMERA_one_reg_eps' or trigger == 'CALIMERA_true_label_eps_one_reg' or trigger == 'CALIMERA_one_reg' or trigger == 'CALIMERA_one_reg_eps_true_plateau':
            train_losses[trigger][alpha][dataset] = trigger_model.train_losses
            val_losses[trigger][alpha][dataset] = trigger_model.val_losses
            avg_test_costs[trigger][alpha][dataset] = trigger_model.avg_test_costs
            avg_val_costs[trigger][alpha][dataset] = trigger_model.avg_val_costs
            opt_epsilon_dict[trigger][alpha][dataset] = trigger_model.opt_epsilon
            #num_masked_epochs[alpha][dataset] = trigger_model.num_masked_epochs

            best_epochs[trigger][alpha][dataset] = trigger_model.best_epoch
    print('Processed alpha : ',alpha)
         
def save_plots_for_one_dataset(trigger,dataset):
    #Parallel(n_jobs=1, backend='multiprocessing')(delayed(visualize_trigger_values_grid)(trigger,alpha,dataset,y_triggers_dict[trigger][alpha][dataset],pred_cost_diffs_dict[trigger][alpha][dataset].T, opt_epsilon_dict[trigger][alpha][dataset]  ) for alpha in alphas)
    #Parallel(n_jobs=1, backend='multiprocessing')(delayed( plot_kde_triggers_distribution)(trigger,alpha,dataset,pred_cost_diffs_dict[trigger][alpha][dataset],y_triggers_dict[trigger][alpha][dataset], opt_epsilon_dict[trigger][alpha][dataset]  ) for alpha in alphas)
    Parallel(n_jobs=1, backend='multiprocessing')(delayed( plot_plateau_costs)(trigger,alpha,dataset,orig_costs_dict[trigger][alpha][dataset],flat_costs_dict[trigger][alpha][dataset],ref_points_dict[trigger][alpha][dataset]) for alpha in alphas)
def save_plots_for_all_datasets():
    for dataset in datasets :
    
       
        save_plots_for_one_dataset('CALIMERA_eps_ref_point',dataset)
    
        save_plots_for_one_dataset('CALIMERA_true_label_eps_ref_point',dataset)
def train_for_one_dataset(data_path,dataset,metrics):
    print('Processing dataset:', dataset)
    os.chdir(data_path)
    if data_path == 'C:\\Users\\WTHK0243\\OneDrive - orange.com\\Bureau\\ml_edm\\src\\ml_edm\\wholes_cylinders\\':
        data_dict=load_dataset(dataset, split='default', z_normalize=False)
    else : 
        if dataset != 'GestureMidAirD1' :
            data_dict=load_dataset(dataset, split='default', z_normalize=False)
        else :
            data_dict=load_dataset(dataset, split=0.7, z_normalize=False)
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    X_classif, X_trigger, y_classif, y_trigger = train_test_split(X_train, y_train, test_size=0.4, random_state=44)
    true_labels_dict[dataset]=y_trigger
    sampling_ratio = 0.05
    max_T = X_train.shape[1] 
    timestamps = [int(max_T * (sampling_ratio * i)) for i in range(1, int(1/sampling_ratio)+1)]
    # delete potential duplicates, O's
    timestamps = check_timestamps(timestamps)
    collection_clf = ClassifiersCollection(
        base_classifier=RidgeClassifierCV(),
        feature_extraction={"method": "minirocket", "path": "models/", "params": {"random_state":42, "n_jobs": -1}},
        timestamps=timestamps,
        cv_method='sigmoid' # change it to isotonic after testing
    )
    collection_clf.fit(X_classif, y_classif)
    classes, counts = np.unique(y_train, return_counts=True)
    def delay_cost(t):
        inflexion_point = 0
        return np.exp(((t/X_train.shape[1])-inflexion_point) * np.log(100))
    
    """small_values = 1
    misclf_cost_mat = small_values - np.eye(len(classes)) * small_values

    classes, counts = np.unique(y_train, return_counts=True)
    idx_min_class = classes[counts.argmin()]
    misclf_cost_mat[:, idx_min_class] *= 100"""

    misclf_cost_mat = 1 - np.eye(len(classes))
    # Parallelize for loop over alphas
    Parallel(n_jobs=1, backend='multiprocessing')(delayed(process_dataset_fixed_alpha)(metrics,trigger_models,dataset,X_trigger,y_trigger,X_test,y_test,alpha,timestamps, collection_clf,misclf_cost_mat, None) for alpha in alphas)

    print(f"Finished processing dataset: {dataset}")

def train_for_all_datasets():
    for dataset in datasets:
        train_for_one_dataset(data_path,dataset,metrics)

if __name__=='__main__':
    filterwarnings("ignore")
    trigger_models = ['CALIMERA_one_reg']
    
    
    """datasets = ["AcousticContaminationMadrid_nmv", "AluminiumConcentration", "BitcoinSentiment", "BME","ChilledWaterPredictor",
    "Chinatown", "DhakaHourlyAirQuality", "DodgerLoopDay", "ElectricityPredictor", "EOGVerticalSignal", "FloodModeling3","GunPointAgeSpan", 
    "HouseholdPowerConsumption1", "HouseTwenty", "HotwaterPredictor", "GestureMidAirD1", "MadridPM10Quality_nmv", 
    "MelbournePedestrian", "ParkingBirmingham_eq", "PLAID", "PrecipitationAndalusia_nmv", "Rock", "SemgHandGenderCh2", 
    "SmoothSubspace", "SolarRadiationAndalusia_nmv", "SteamPredictor", "TetuanEnergyConsumption", "UMD", "WindTurbinePower"] """# GestureMidAirD1  needs 0.7 split

    
    """datasets = ["AcousticContaminationMadrid_nmv", "AluminiumConcentration", "BitcoinSentiment", "BME","ChilledWaterPredictor",
    "Chinatown", "HouseTwenty", "DodgerLoopDay","UMD" , "EOGVerticalSignal"]"""

    datasets = ["DhakaHourlyAirQuality","ElectricityPredictor","FloodModeling3","GunPointAgeSpan", 
    "HouseholdPowerConsumption1", "HotwaterPredictor", "GestureMidAirD1", "MadridPM10Quality_nmv", 
    "MelbournePedestrian", "ParkingBirmingham_eq", "PLAID", "PrecipitationAndalusia_nmv", "Rock", "SemgHandGenderCh2", 
    "SmoothSubspace", "SolarRadiationAndalusia_nmv", "SteamPredictor", "TetuanEnergyConsumption", "WindTurbinePower"]
      
    


    #num_classes = [2,2,2,3,2,2,24,2,7,2,12,2,2,2,2,2,3,2,10,2,11,2,4,2,3,2,2,2,3,2]
 
    """datasets = ["BME","Chinatown","Crop","DodgerLoopDay","EOGVerticalSignal","GestureMidAirD1","GunPointAgeSpan","HouseTwenty","MelbournePedestrian","PLAID","Rock","SemgHandGenderCh2","SmoothSubspace","UMD",
                "AcousticContaminationMadrid_nmv", "AluminiumConcentration", "BitcoinSentiment", "ChilledWaterPredictor",
                "DhakaHourlyAirQuality", "ElectricityPredictor", "FloodModeling3","HouseholdPowerConsumption1",
                "HotwaterPredictor", "MadridPM10Quality_nmv", "ParkingBirmingham_eq", "PrecipitationAndalusia_nmv", "SolarRadiationAndalusia_nmv",
                "SteamPredictor", "TetuanEnergyConsumption", "WindTurbinePower"]"""
    
    
    
   
    
    
    
    

    
    
    #num_classes = [3,2,7,12,26,2,2,10,11,4,2,3,3]+[2]*16

    #datasets_num_classes = {datasets[i]: num_classes[i] for i in range(len(datasets)) }
    
    


    data_path='C:\\Users\\WTHK0243\\OneDrive - orange.com\\Bureau\\ml_edm\\src\\ml_edm\\orange_cylinder\\'
    #data_path='C:\\Users\\WTHK0243\\OneDrive - orange.com\\Bureau\\ml_edm\\src\\ml_edm\\wholes_cylinders\\'
    alpha0 = 0
    dict_trigger = dict.fromkeys(trigger_models)
    dict_data = dict.fromkeys(datasets)
    alphas = [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]

    os.chdir(data_path)

    metrics = {trigger: {alpha: {dataset: None for dataset in datasets} for alpha in alphas} for trigger in trigger_models}
    train_losses = {trigger: {alpha: {dataset: None for dataset in datasets} for alpha in alphas} for trigger in trigger_models}
    val_losses = {trigger: {alpha: {dataset: None for dataset in datasets} for alpha in alphas} for trigger in trigger_models}
    avg_test_costs =  {trigger: {alpha: {dataset: None for dataset in datasets} for alpha in alphas} for trigger in trigger_models}
    avg_val_costs =  {trigger: {alpha: {dataset: None for dataset in datasets} for alpha in alphas} for trigger in trigger_models}
    best_epochs = {trigger: {alpha: {dataset : None for dataset in datasets} for alpha in alphas} for trigger in trigger_models}
    num_masked_epochs = {alpha: {dataset : None for dataset in datasets} for alpha in alphas}
    y_triggers_dict = {trigger: {alpha: {dataset: None for dataset in datasets} for alpha in alphas} for trigger in trigger_models}
    pred_cost_diffs_dict = {trigger: {alpha: {dataset: None for dataset in datasets} for alpha in alphas} for trigger in trigger_models}
    opt_epsilon_dict ={trigger: {alpha: {dataset: None for dataset in datasets} for alpha in alphas} for trigger in trigger_models}
    orig_costs_dict ={trigger: {alpha: {dataset: None for dataset in datasets} for alpha in alphas} for trigger in trigger_models}
    flat_costs_dict ={trigger: {alpha: {dataset: None for dataset in datasets} for alpha in alphas} for trigger in trigger_models}
    ref_points_dict = {trigger: {alpha: {dataset: None for dataset in datasets} for alpha in alphas} for trigger in trigger_models}
    true_labels_dict = {dataset : None for dataset in datasets } 
    metrics_post ={alpha: {dataset: None for dataset in datasets} for alpha in alphas}
    start_time = time.time()
    train_for_all_datasets()
    
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")

    os.chdir('C:\\Users\\WTHK0243\\OneDrive - orange.com\\Bureau\\ml_edm\\src\\ml_edm')
    # Save the metrics dictionary to a file
    with open('tests\\metrics_one_reg_flatten_full_random_batch_t_star_2.json', 'w') as tmp_file:
        json.dump(metrics, tmp_file, cls=NpEncoder)

    print("Metrics dictionary has been saved to 'metrics_one_reg_flatten_full_random_batch_t_star_2.json'.")
    with open('tests\\train_losses_one_reg_flatten_full_random_batch_t_star_2.json', 'w') as tmp_file:
        json.dump(train_losses, tmp_file, cls=NpEncoder)
    print("Train losses dictionary has been saved to 'train_losses_one_reg_flatten_full_random_batch_t_star_2.json'.")
    with open('tests\\val_losses_one_reg_flatten_full_random_batch_t_star_2.json', 'w') as tmp_file:
        json.dump(val_losses, tmp_file, cls=NpEncoder)
    print("Val losses dictionary has been saved to 'val_losses_one_reg_flatten_full_random_batch_t_star_2.json'.")
    with open('tests\\avg_val_one_reg_flatten_full_random_batch_t_star_2.json', 'w') as tmp_file:
        json.dump(avg_val_costs, tmp_file, cls=NpEncoder)
    print("Avg cost val dictionary has been saved to 'avg_val_one_reg_flatten_full_random_batch_t_star_2.json'.")
    with open('tests\\avg_test_one_reg_flatten_full_random_batch_t_star_2.json', 'w') as tmp_file:
        json.dump(avg_test_costs, tmp_file, cls=NpEncoder)
    print("Avg cost test dictionary has been saved to 'avg_test_one_reg_flatten_full_random_batch_t_star_2.json'.")
   
    with open('tests\\best_epoch_one_reg_flatten_full_random_batch_t_star_2.json', 'w') as tmp_file:
        json.dump(best_epochs, tmp_file, cls=NpEncoder)
    print("Best epoch dictionary has been saved to 'best_epoch_one_reg_flatten_full_random_batch_t_star_2.json'.")
    
    """with open('tests\\metrics_post_default_setting_unb.json', 'w') as tmp_file:
        json.dump(metrics_post, tmp_file, cls=NpEncoder)

    print("Metrics dictionary has been saved to 'metrics_post_default_setting_unb.json'.")"""
    """with open('tests\\y_triggers_dict_eps_flat_unb_exp_delay.json', 'w') as tmp_file:
        json.dump(y_triggers_dict, tmp_file, cls=NpEncoder)

    print("Metrics dictionary has been saved to 'y_triggers_dict_eps_flat_unb_exp_delay.json'.")
    with open('tests\\pred_cost_diffs_dict_eps_flat_unb_exp_delay.json', 'w') as tmp_file:
        json.dump(pred_cost_diffs_dict, tmp_file, cls=NpEncoder)

    print("Metrics dictionary has been saved to 'pred_cost_diffs_dict_eps_flat_unb_exp_delay.json'.")"""
    """with open('tests\\opt_eps_one_reg_all_datasets_unb_exp_delay.json', 'w') as tmp_file:
        json.dump(opt_epsilon_dict, tmp_file, cls=NpEncoder)

    print("Metrics dictionary has been saved to 'opt_eps_one_reg_all_datasets_unb_exp_delay.json'.")"""

    #save_plots_for_all_datasets()
    #print("Plots have been saved.")