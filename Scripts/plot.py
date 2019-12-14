import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#from model import split_and_standardization_dataset


def split_and_standardization_dataset(target_variables, covariates, test_size, random, type_return = 'numpy'  ):
    
    '''
    
    target_variables: pandas dataframe that contains the target variables
    covariates: pandas dataframe that contains the independant variables
    test_size: the proportion of the dataset to include in the test split
    type_return: 'numpy' if return numpy array, 'pandas' if return pandas dataframe
    '''
    target_variables_numpy = target_variables.to_numpy()
    covariates_numpy = covariates.to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(covariates_numpy, target_variables_numpy, test_size=test_size, random_state = random)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_normalized = scaler.transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    
    if type_return == 'numpy':
        
        return X_train_normalized, X_test_normalized, Y_train, Y_test
    
    elif type_return == 'pandas':
        
        X_test_normalized_df = pd.DataFrame(X_test_normalized, columns = list(covariates.columns))
        X_train_normalized_df = pd.DataFrame(X_train_normalized,columns= list(covariates.columns))
        Y_train_df = pd.DataFrame(Y_train, columns= list(target_variables.columns))
        Y_test_df = pd.DataFrame(Y_test, columns= list(target_variables.columns))
        
        return X_train_normalized_df, X_test_normalized_df, Y_train_df, Y_test_df
    
    

def plot_heatmap(list_selected_columns, target_variable, target_variable_name, cov_dataframe):
    
    '''
    list_selected_columns: a list of the columns included in dataframe for which we want to plot the heatmap
    target_variable: A serie of the target variable we want to include in the dataframe
    target_variable_name: name of the preceding serie
    cov_dataframe: dataframe of the covariates features
    
    
    '''
    
    
    plt.figure(figsize=(30,30))
    temporary_df = cov_dataframe.copy()
    temporary_df[target_variable_name] = target_variable
    cor = temporary_df[list_selected_columns].corr()
    sns.set_context("notebook", font_scale = 2.5)
    sns.heatmap(cor, annot = True, cmap = plt.cm.Reds)
    plt.show()
    
    



def heatmap(x,y,size, color, fig_size = (10,10)):
    sns.set()
    
    fig, ax = plt.subplots(figsize = fig_size)
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 500
    
    n_colors = 256 # Use 256 colors for the diverging color palette
    palette = sns.diverging_palette(20, 220, n=n_colors) # Create the palette
    color_min, color_max = [0, 1] # Range of values that will be mapped to the palette, i.e. min and max possible correlation
    
    def value_to_color(val):
        val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
        ind = int(val_position * (n_colors - 1)) # target index in the color palette
        return palette[ind]
    
    

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    
    ax.grid(False, 'major') # Turn off major gridlines
    ax.grid(True, 'minor') # Turn on minor gridlines
    ax.set_xticks([t  for t in ax.get_xticks()], minor=True) # Set gridlines to appear between integer coordinates
    ax.set_yticks([t for t in ax.get_yticks()], minor=True) # Do the same for y axis
    
    ax.set_xlim([-1, max([v for v in x_to_num.values()])+1]) 
    ax.set_ylim([-1, max([v for v in y_to_num.values()])+1])
    ax.set_facecolor('#F1F1F1')

    if color_min<color_max:

        plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x15 grid
        ax = plt.subplot(plot_grid[:,:-1]) # Use the leftmost 14 columns of the grid for the mainplot

        ax.scatter(
            x=x.map(x_to_num), # Use mapping for x
            y=y.map(y_to_num), # Use mapping for y
            s=size * size_scale, # Vector of square sizes, proportional to size parameter
            c=color.apply(value_to_color), # Vector of square colors, mapped to color palette
            marker='s' # Use square as scatterplot marker
        )
        
        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot

        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax.barh(
            y=bar_y,
            width=[5]*len(palette), # Make bars 5 units wide
            left=col_x, # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )

        ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False) # Hide grid
        ax.set_facecolor('white') # Make background white
        ax.set_xticks([]) # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right() # Show vertical ticks on the right 

        plt.show()
    

def main():

    RANDOM_SEED = 29

    reduced_df = pd.read_pickle('reduced_df_2.pkl')
    print('reduced_df shape', reduced_df.shape)
    target = pd.read_pickle('target.pkl')
    print('target shape', target.shape)
    columns = list(reduced_df.keys())
    X_train_normalized_df, X_test_normalized_df, Y_train_df, Y_test_df = split_and_standardization_dataset(target[['(GDP, million $)']], reduced_df, 0.2, random = RANDOM_SEED, type_return = 'pandas')
    print('normalized', X_train_normalized_df)
    corr = X_train_normalized_df[columns].corr()
    corr = pd.melt(corr.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    heatmap(
        x=corr['x'],
        y=corr['y'],
        size=corr['value'].abs(), 
        color = corr['value']
    )


    


    
    
    
    
    
if __name__=="__main__":

    main()