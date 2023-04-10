from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
import sklearn
from ei import EnsembleIntegration

def drop_modes(original_dict, delete_these_modes = ['fdg_pet_roi', 'dti_roi', 'av1451_pet_roi', 'av45_pet_roi']):
    for m in delete_these_modes:
        if m in original_dict.keys():
            del original_dict[m]
    return original_dict

def other_process(other_mode):
    if 'APOE4' in other_mode.keys():
        one_hot = pd.get_dummies(other_mode, columns=['APOE4'])
    else:
        one_hot = other_mode
    encoded = encode_exclude_nan(one_hot)
    return encoded

def sample(X, y, strategy, random_state):
    if strategy is None:
        X_resampled, y_resampled = X, y
    elif strategy == "undersampling":  # define sampler
        sampler = RandomUnderSampler(random_state=random_state)
    elif strategy == "oversampling":
        sampler = RandomOverSampler(random_state=random_state)
    elif strategy == 'hybrid':
        y_pos = float(sum(y==1))
        y_total = y.shape[0]
        if (y_pos/y_total) < 0.5:
            y_min_count = y_pos
            y_maj_count = (y_total - y_pos)
            maj_class = 0
        else:
            y_maj_count = y_pos
            y_min_count = (y_total - y_pos)
            maj_class = 1
        rus = RandomUnderSampler(random_state=random_state, 
                                sampling_strategy=y_min_count/(y_total/2))
        ros = RandomOverSampler(random_state=random_state,   
                                sampling_strategy=(y_total/2)/y_maj_count)
        X_maj, y_maj = rus.fit_resample(X=X, y=y)
        X_maj = X_maj[y_maj==maj_class]
        y_maj = y_maj[y_maj==maj_class]
        X_min, y_min = ros.fit_resample(X=X, y=y)
        X_min = X_min[y_min!=maj_class]
        y_min = y_min[y_min!=maj_class]
        X_resampled = np.concatenate([X_maj, X_min])
        y_resampled = np.concatenate([y_maj, y_min])
    
    if (strategy == "undersampling") or (strategy == "oversampling"):
        X_resampled, y_resampled = sampler.fit_resample(X=X, y=y)
    return X_resampled, y_resampled


def impute_per_mode(train_dict, test_dict=None):
    if test_dict == None:
        imputed_dict = {}
        for mode in train_dict:
            imputer = KNNImputer(missing_values=np.nan)
            #print(original_dict[mode])
            imputer = imputer.fit(train_dict[mode])
            imputed_dict[mode] = imputer.transform(train_dict[mode])
        return imputed_dict
    elif test_dict != None:
        imputed_train_dict = {}
        imputed_test_dict = {}
        
        for mode in train_dict:
            #print(mode)
            imputer = KNNImputer(missing_values=np.nan)
            #print(original_dict[mode])
            imputer = imputer.fit(train_dict[mode])
            imputed_train_dict[mode] = imputer.transform(train_dict[mode])
            imputed_test_dict[mode] = imputer.transform(test_dict[mode])
        return imputed_train_dict, imputed_test_dict

def normalize_per_col(train_dict, test_dict=None):
    if test_dict == None:
        mode_dict_norm= {}
        for m in train_dict:
            mode_dict_norm[m] = pd.DataFrame()
            for col in range(train_dict[m].shape[1]):
                #print(mode_dict_feature_filtered_coldrop_train[m])
                dfmax = train_dict[m][:, col].max()
                dfmin = train_dict[m][:, col].min()
                norm_col_train = pd.DataFrame((train_dict[m][:, col] - 
                                                    dfmin) * 1.0 / (dfmax - dfmin))
                mode_dict_norm[m] = pd.concat([mode_dict_norm[m], pd.DataFrame(norm_col_train)], axis=1)
            mode_dict_norm[m] = np.array(mode_dict_norm[m])
        return(mode_dict_norm)
    elif test_dict != None:
        mode_dict_norm_train = {}
        mode_dict_norm_test = {}
        for m in train_dict:
            mode_dict_norm_train[m] = pd.DataFrame()
            mode_dict_norm_test[m] = pd.DataFrame()
            for col in range(train_dict[m].shape[1]):
                dfmax = train_dict[m][:, col].max()
                dfmin = train_dict[m][:, col].min()
                norm_col_train = pd.DataFrame((train_dict[m][:, col] - 
                                                    dfmin) * 1.0 / (dfmax+0.000000000000001 - dfmin))
                norm_col_test = pd.DataFrame((test_dict[m][:, col] - 
                                                    dfmin) * 1.0 / (dfmax+0.000000000000001 - dfmin))
                mode_dict_norm_train[m] = pd.concat([mode_dict_norm_train[m], pd.DataFrame(norm_col_train)], axis=1)
                mode_dict_norm_test[m] = pd.concat([mode_dict_norm_test[m], pd.DataFrame(norm_col_test)], axis=1)
            mode_dict_norm_test[m][mode_dict_norm_test[m] > 1] = 1
            mode_dict_norm_test[m][mode_dict_norm_test[m] < 0] = 1
            mode_dict_norm_train[m] = np.array(mode_dict_norm_train[m])
            mode_dict_norm_test[m] = np.array(mode_dict_norm_test[m])
        return(mode_dict_norm_train, mode_dict_norm_test)
    
from cycler import cycler
import matplotlib.pyplot as plt

def pretty_fig(title, ylabel, xlabel, xfont=15, yfont=15):

    fig = plt.figure(facecolor='white', figsize = (10,5))
    colors = cycler('color',
                    ['#EE6666', '#3388BB', '#9988DD',
                    '#EECC55', '#88BB44', '#FFBBBB'])
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
        axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('xtick', direction='out', color='black', labelsize=xfont)
    plt.rc('ytick', direction='out', color='black', labelsize=yfont)
    plt.rc('patch', edgecolor='#E6E6E6')
    plt.rc('axes', titlesize=15)     # fontsize of the axes title
    plt.rc('axes', labelsize=15)
    plt.rc('lines', linewidth=2)
    plt.rcParams['figure.dpi'] = 600
    #plt.rcParams.update({'font.size': 22})
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    return fig

def encode_exclude_nan(mode):

    from sklearn import preprocessing
    from sklearn.preprocessing import LabelEncoder 
    le = preprocessing.LabelEncoder()
    nanmap = mode.isnull()
    for c in mode.keys():
        le.fit(mode[c])
        mode[c] = le.transform(mode[c])
    mode[nanmap == True] = np.nan

    return mode

from sklearn.metrics import roc_auc_score, precision_recall_curve, \
    matthews_corrcoef, precision_recall_fscore_support, make_scorer

def auprc(y_true, y_scores):

    return sklearn.metrics.average_precision_score(y_true, y_scores)

#auprc_sklearn = make_scorer(auprc, greater_is_better=True, needs_proba=True)

def f_minority_score(y_true, y_pred):

    if np.bincount(y_true)[0] < np.bincount(y_true)[1]:
        minor_class = 0
    else:
        minor_class = 1

    return fmeasure_score(y_true, y_pred, pos_label=minor_class)['F']

def fmeasure_score(labels, predictions, thres=None, 
                    beta = 1.0, pos_label = 1, thres_same_cls = False):
    
    """
        Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
        Manning, C. D. et al. (2008). Evaluation in Information Retrieval. In Introduction to Information Retrieval. Cambridge University Press.
    """
    np.seterr(divide='ignore', invalid='ignore')
    if pos_label == 0:
        labels = 1-np.array(labels)
        predictions = 1-np.array(predictions)
        # if not(thres is None):
        #     thres = 1-thres
    # else:


    if thres is None:  # calculate fmax here
        np.seterr(divide='ignore', invalid='ignore')
        precision, recall, threshold = sklearn.metrics.precision_recall_curve(labels, predictions,
                                                                            #   pos_label=pos_label
                                                                              )

        fs = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
        fmax_point = np.where(fs==np.nanmax(fs))[0]
        p_maxes = precision[fmax_point]
        r_maxes = recall[fmax_point]
        pr_diff = np.abs(p_maxes - r_maxes)
        balance_fmax_point = np.where(pr_diff == min(pr_diff))[0]
        p_max = p_maxes[balance_fmax_point[0]]
        r_max = r_maxes[balance_fmax_point[0]]
        opt_threshold = threshold[fmax_point][balance_fmax_point[0]]

        return {'F':np.nanmax(fs), 'thres':opt_threshold, 'P':p_max, 'R':r_max, 'PR-curve': [precision, recall]}

    else:  # calculate fmeasure for specific threshold
        binary_predictions = np.array(predictions)
        if thres_same_cls:
            binary_predictions[binary_predictions >= thres] = 1.0
            binary_predictions[binary_predictions < thres] = 0.0
        else:
            binary_predictions[binary_predictions > thres] = 1.0
            binary_predictions[binary_predictions <= thres] = 0.0
        precision, recall, fmeasure, _ = precision_recall_fscore_support(labels,
                                                                        binary_predictions, 
                                                                        average='binary',
                                                                        # pos_label=pos_label
                                                                        )
        return {'P':precision, 'R':recall, 'F':fmeasure}  

def get_all_metrics(ei_model, base=False, print_thresh = False):
    if base == True:
        concat_results = pd.DataFrame()
        for i in range(5):
            concat_results = pd.concat([concat_results, ei_model.meta_test_data[i]], axis=0) 
        preds = concat_results.keys()
        y_true = concat_results['labels']
    else:
        preds = ei_model.meta_predictions.keys()  
        y_true = ei_model.meta_predictions['labels']  
        
    metrics = {'AUC':[], 'fmax (minority)':[], 'f (majority)':[], 'fmax (majority)':[], 'f (minority)':[]}
    
    for pred in preds:
        if pred != 'labels':
            if base == True:
                y_pred = concat_results[pred] 
            else:
                y_pred = ei_model.meta_predictions[pred] 
            fmax_minor_thresh = fmeasure_score(y_true, y_pred, pos_label=1)['thres']
            fmax_major_thresh = fmeasure_score(y_true, y_pred, pos_label=0)['thres']
            
            metrics['AUC'] += [roc_auc_score(y_true, y_pred)]
            metrics['fmax (minority)']+= [fmeasure_score(y_true, y_pred, pos_label=1)['F']]
            metrics['f (majority)']+= [fmeasure_score(y_true, y_pred, pos_label=0, thres=1-fmax_minor_thresh)['F']]
            metrics['fmax (majority)']+= [fmeasure_score(y_true, y_pred, pos_label=0)['F']]
            metrics['f (minority)']+= [fmeasure_score(y_true, y_pred, pos_label=1, thres=1-fmax_major_thresh)['F']]
        if print_thresh == True:
                    print(pred)
                    print(f'Fmax minor threshold = {fmax_minor_thresh}')
                    print(f'Fmax major threshold = {fmax_major_thresh}')
    return metrics

def get_labels_for_int_feature_rank(EI_int_obj, ei_model, mode_dict_orig, specify_model = None):
    #best_model = list(set(list(dict(sorted(ei_model.meta_summary['metrics'].iloc[0].items(), key=lambda item: item[1], reverse=True)).keys()))-set(['Mean', 'Median']))[0]
    models = list(ei_model.meta_summary['metrics'].iloc[0].items())
    best_model = ['none', float(0)]
    if specify_model == None:
        for model in models:
            #print(float(list(model)[1]))
            print(model)
            if (float(list(model)[1]) > best_model[1]) and (model[0] != 'Median'):

                best_model[0] = list(model)[0]
                best_model[1] = float(list(model)[1])
        best_model = best_model[0]
        print(best_model)
        print(EI_int_obj.ensemble_feature_ranking.keys())
        if best_model not in ['Mean', 'Median', 'S.CES']:
            best_model = 'S.' + best_model
    else:
        best_model = specify_model
    
    EI_int_rank = EI_int_obj.ensemble_feature_ranking[best_model].copy()
    for n, feature in enumerate(EI_int_obj.ensemble_feature_ranking[best_model]['feature']):
        for m in EI_int_obj.modalities.keys():
            if m in feature:
                feature_loc = int(feature.split(m+'_')[1])
                print(EI_int_rank['feature'].iloc[n])
                print(mode_dict_orig[m].keys()[feature_loc]  )
                EI_int_rank['feature'].iloc[n] = mode_dict_orig[m].keys()[feature_loc]    
    return EI_int_rank

def add_fdr_to_int(int_csv, fdr_csv, final_csv_name):
    int_data = pd.read_csv(int_csv)
    fdr = pd.read_csv(fdr_csv)
    #fdr = pd.read_csv(f'FDR_MCI_FINAL.csv')
    rank_col = []
    fdr_col = []
    is_sig = []
    desc_col = []
    for feature_int in int_data['feature']:
        rank_idx = 0
        check = 0
        for feature_fdr in fdr['feature']:
            if feature_int == feature_fdr:
                rank_col += [int(rank_idx) + 1]
                fdr_col += [list(fdr[fdr['feature']==feature_int]['FDR'])[0]]
                desc_col += [list(fdr[fdr['feature']==feature_int]['desc'])[0]]
                if list(fdr[fdr['feature']==feature_int]['FDR'])[0] <= 0.05:
                    is_sig += ['True']
                else:
                    is_sig += ['False']
                check = 1
            rank_idx+=1
        if check == 0:
            desc_col += [np.NaN]
            is_sig += [np.NaN]
            rank_col += [np.NaN]
            fdr_col += [np.NaN]
    
    int_data['FDR'] = fdr_col
    int_data['FDR Rank'] = rank_col
    int_data['Significant'] = is_sig
    int_data['full name'] = desc_col
    int_data = pd.DataFrame(int_data)
    int_data.to_csv(final_csv_name)

#Packages
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC

def_base_predictors = {
        'NB': GaussianNB(),
        'LR': make_pipeline(StandardScaler(), LogisticRegression()),
        "SVM": make_pipeline(Normalizer(), SVC(kernel='poly', degree=1, probability=True)),
        "Perceptron": Perceptron(),
        'AdaBoost': AdaBoostClassifier(n_estimators=10),
        "DT": DecisionTreeClassifier(),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=10),
        "RF": RandomForestClassifier(n_estimators=100),
        "XGB": XGBClassifier(), 
        'KNN': KNeighborsClassifier(n_neighbors=1),
    }

def_meta_models = {
                "RF": RandomForestClassifier(),
                "SVM": SVC(kernel='linear', probability=True, max_iter=1e7),
                "NB": GaussianNB(),
                "LR": LogisticRegression(),
                "ADA": AdaBoostClassifier(),
                "DT": DecisionTreeClassifier(),
                "GB": GradientBoostingClassifier(),
                "KNN": KNeighborsClassifier(),
                "XGB": XGBClassifier()
}

def EI_model_train_and_save(project_name, base_predictors=def_base_predictors, meta_models=def_meta_models, 
        mode_dict = None, y = None, train = False, single_mode=False, 
        mode_name='unimode', random_state=42, path=None, model_building=False, 
        k_outer=5,
        k_inner=5, 
        n_samples=1,  
        sampling_strategy="hybrid",
        sampling_aggregation = 'mean',
        meta_training = True):

    EI = EnsembleIntegration(base_predictors=base_predictors, 
                            meta_models=meta_models,  
                            k_outer=k_outer,
                            k_inner=k_inner, 
                            n_samples=n_samples,  
                            sampling_strategy=sampling_strategy,
                            sampling_aggregation=sampling_aggregation,
                            n_jobs=-1,
                            random_state=random_state,
                            parallel_backend="loky",
                            project_name=project_name,
                            #additional_ensemble_methods=["Mean", "CES"]
                            model_building = model_building
                            ) 

    if train == True:
        if single_mode == False:
            #print('Whole EI')
            for name, modality in mode_dict.items():
                EI.train_base(modality, y, base_predictors, modality=name)
                

        elif single_mode == True:
            EI.train_base(mode_dict, y, base_predictors, modality=mode_name)
                #print('Single mode')
                #print(project_name)
        #EI.save() 

        #EI = EnsembleIntegration().load(f"EI.{project_name}")  # load models from disk

        #EI.train_meta(meta_models=meta_models)  # train meta classifiers
    
        meta_models = {
            "AdaBoost": AdaBoostClassifier(),
            "DT": DecisionTreeClassifier(max_depth=5),
            "GradientBoosting": GradientBoostingClassifier(),
            "KNN": KNeighborsClassifier(n_neighbors=21),
            "LR": LogisticRegression(),
            "NB": GaussianNB(),
            "MLP": MLPClassifier(),
            "RF": RandomForestClassifier(),
            "SVM": LinearSVC(tol=1e-2, max_iter=10000),
            "XGB": XGBClassifier(use_label_encoder=False, eval_metric='error')
        }
        #EI.save() 
        #EI = EnsembleIntegration().load(f"EI.{project_name}")  # load models from disk
        if meta_training == True:
            EI.train_meta(meta_models=meta_models)  # train meta classifiers

        EI.save(path=path) 

    else:
        EI = EnsembleIntegration().load(f"EI.{project_name}")  # load models from disk
        
    return EI

#Autoencoder functions

import keras
from keras import layers
from keras.layers import Dense, Input, Dropout
from keras.losses import BinaryCrossentropy
from keras import Model
from keras.layers.core.dense import regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from keras.backend import dropout
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from keras.callbacks import LearningRateScheduler

def create_ae(layer_size, input_dim,  
              loss_fn, optimizer, dropout_rate=0, lr=.001):

  input_layer = Input(shape = (input_dim, ))
  noise = Dropout(dropout_rate)(input_layer)
  encoder = layers.Dense(layer_size, activation = 'relu')
  encoded = encoder(noise)
  #decoder = Dense(input_dim, activation = 'relu')(encoder)
  decoder = DenseTranspose(encoder, activation='relu')(encoded)
  autoencoder = Model(inputs = input_layer, outputs = decoder)
  autoencoder.compile(loss=loss_fn, 
                      optimizer=keras.optimizers.Adam(
                          learning_rate=lr))

  return autoencoder

def scheduler(epoch, lr):
   if epoch < 40:
     return lr
   else:
     return lr * tf.math.exp(-0.1)

import os 
#DANGEROUS FUNCTION

def del_checkpoints(checkpoint_filepath = '/home/opc/model_checkpoints/model_checkpoints_subfolder/'):
    for filename in os.listdir(checkpoint_filepath):
        if ('data' in filename) or ('index' in filename) or ('checkpoint' in filename):
            os.remove(checkpoint_filepath + filename)
    return 

def ae_fit_and_predict(ae, input, epochs, batch_size, checkpoint_filepath = '/home/opc/model_checkpoints/model_checkpoints_subfolder/'):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)

    callbacks = LearningRateScheduler(scheduler)

    stack = ae.fit(input, input, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[model_checkpoint_callback])
    ae.load_weights(checkpoint_filepath)
    predict = ae.predict(input, verbose=0)
    del_checkpoints()
    return predict

def classifier(classifier_input, y, input_dim, epochs, 
               activation='sigmoid'):

  classifier_input_layer = Input(shape = (input_dim, ))
  classifier_sigmoid = Dense(1, activation=activation)(classifier_input_layer)
  classifier = Model(inputs = classifier_input_layer, outputs=classifier_sigmoid)
  classifier.compile(loss='binary_crossentropy', 
                     optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                     metrics=['accuracy'])

  return classifier

'''
def fine_tuning(layer_size, dropout_rate=0.2, input_dim=input_dim, 
              loss_fn=loss_fn, optimizer=optimizer):
'''


#https://medium.com/@sahoo.puspanjali58/a-beginners-guide-to-build-stacked-autoencoder-and-tying-weights-with-it-9daee61eab2b
class DenseTranspose(keras.layers.Layer):
  def __init__(self, dense, activation=None, **kwargs):
      self.dense = dense
      self.activation = keras.activations.get(activation)
      super().__init__(**kwargs)
  def build(self, batch_input_shape):
      self.biases = self.add_weight(name="bias", 
                  initializer="zeros",shape=[self.dense.input_shape[-1]])
      super().build(batch_input_shape)
  def call(self, inputs):
      z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
      return self.activation(z + self.biases)  