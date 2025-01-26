import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from IPython.display import display

from scipy.stats import gaussian_kde

from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.inspection import permutation_importance
#from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.manifold import TSNE

from sklearn.feature_selection import RFE, mutual_info_classif
from scipy.stats import chi2_contingency #, chi2

# Dictionary for attribute descriptions
attribute_descriptions = {
    0: 'class',
    1: 'cap_shape',
    2: 'cap_surface',
    3: 'cap_color',
    4: 'bruises',
    5: 'odor',
    6: 'gill_attachment',
    7: 'gill_spacing',
    8: 'gill_size',
    9: 'gill_color',
    10: 'stalk_shape',
    11: 'stalk_root',
    12: 'stalk_surface_above_ring',
    13: 'stalk_surface_below_ring',
    14: 'stalk_color_above_ring',
    15: 'stalk_color_below_ring',
    16: 'veil_type',
    17: 'veil_color',
    18: 'ring_number',
    19: 'ring_type',
    20: 'spore_print_color',
    21: 'population',
    22: 'habitat'
}

# Dictionary for mushroom mappings
mushroom_dict = {
    'class': {
        'e': 'edible',
        'p': 'poisonous'
    },
    'cap_shape': {
        'b': 'bell',
        'c': 'conical',
        'x': 'convex',
        'f': 'flat',
        'k': 'knobbed',
        's': 'sunken'
    },
    'cap_surface': {
        'f': 'fibrous',
        'g': 'grooves',
        'y': 'scaly',
        's': 'smooth'
    },
    'cap_color': {
        'n': 'brown',
        'b': 'buff',
        'c': 'cinnamon',
        'g': 'gray',
        'r': 'green',
        'p': 'pink',
        'u': 'purple',
        'e': 'red',
        'w': 'white',
        'y': 'yellow'
    },
    'bruises': {
        't': 'bruises',
        'f': 'no'
    },
    'odor': {
        'a': 'almond',
        'l': 'anise',
        'c': 'creosote',
        'y': 'fishy',
        'f': 'foul',
        'm': 'musty',
        'n': 'none',
        'p': 'pungent',
        's': 'spicy'
    },
    'gill_attachment': {
        'a': 'attached',
        'd': 'descending',
        'f': 'free',
        'n': 'notched'
    },
    'gill_spacing': {
        'c': 'close',
        'w': 'crowded',
        'd': 'distant'
    },
    'gill_size': {
        'b': 'broad',
        'n': 'narrow'
    },
    'gill_color': {
        'k': 'black',
        'n': 'brown',
        'b': 'buff',
        'h': 'chocolate',
        'g': 'gray',
        'r': 'green',
        'o': 'orange',
        'p': 'pink',
        'u': 'purple',
        'e': 'red',
        'w': 'white',
        'y': 'yellow'
    },
    'stalk_shape': {
        'e': 'enlarging',
        't': 'tapering'
    },
    'stalk_root': {
        'b': 'bulbous',
        'c': 'club',
        'u': 'cup',
        'e': 'equal',
        'z': 'rhizomorphs',
        'r': 'rooted',
        '?': 'missing'
    },
    'stalk_surface_above_ring': {
        'f': 'fibrous',
        'y': 'scaly',
        'k': 'silky',
        's': 'smooth'
    },
    'stalk_surface_below_ring': {
        'f': 'fibrous',
        'y': 'scaly',
        'k': 'silky',
        's': 'smooth'
    },
    'stalk_color_above_ring': {
        'n': 'brown',
        'b': 'buff',
        'c': 'cinnamon',
        'g': 'gray',
        'o': 'orange',
        'p': 'pink',
        'e': 'red',
        'w': 'white',
        'y': 'yellow'
    },
    'stalk_color_below_ring': {
        'n': 'brown',
        'b': 'buff',
        'c': 'cinnamon',
        'g': 'gray',
        'o': 'orange',
        'p': 'pink',
        'e': 'red',
        'w': 'white',
        'y': 'yellow'
    },
    'veil_type': {
        'p': 'partial',
        'u': 'universal'
    },
    'veil_color': {
        'n': 'brown',
        'o': 'orange',
        'w': 'white',
        'y': 'yellow'
    },
    'ring_number': {
        'n': 'none',
        'o': 'one',
        't': 'two'
    },
    'ring_type': {
        'c': 'cobwebby',
        'e': 'evanescent',
        'f': 'flaring',
        'l': 'large',
        'n': 'none',
        'p': 'pendant',
        's': 'sheathing',
        'z': 'zone'
    },
    'spore_print_color': {
        'k': 'black',
        'n': 'brown',
        'b': 'buff',
        'h': 'chocolate',
        'r': 'green',
        'o': 'orange',
        'u': 'purple',
        'w': 'white',
        'y': 'yellow'
    },
    'population': {
        'a': 'abundant',
        'c': 'clustered',
        'n': 'numerous',
        's': 'scattered',
        'v': 'several',
        'y': 'solitary'
    },
    'habitat': {
        'g': 'grasses',
        'l': 'leaves',
        'm': 'meadows',
        'p': 'paths',
        'u': 'urban',
        'w': 'waste',
        'd': 'woods'
    }
}

# Expected values for each attribute
expected_values = {
    'class': ['e', 'p'],
    'cap_shape': ['b', 'c', 'x', 'f', 'k', 's'],
    'cap_surface': ['f', 'g', 'y', 's'],
    'cap_color': ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
    'bruises': ['t', 'f'],
    'odor': ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],
    'gill_attachment': ['a', 'd', 'f', 'n'],
    'gill_spacing': ['c', 'w', 'd'],
    'gill_size': ['b', 'n'],
    'gill_color': ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],
    'stalk_shape': ['e', 't'],
    'stalk_root': ['b', 'c', 'u', 'e', 'z', 'r', '?'],
    'stalk_surface_above_ring': ['f', 'y', 'k', 's'],
    'stalk_surface_below_ring': ['f', 'y', 'k', 's'],
    'stalk_color_above_ring': ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
    'stalk_color_below_ring': ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
    'veil_type': ['p', 'u'],
    'veil_color': ['n', 'o', 'w', 'y'],
    'ring_number': ['n', 'o', 't'],
    'ring_type': ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'],
    'spore_print_color': ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'],
    'population': ['a', 'c', 'n', 's', 'v', 'y'],
    'habitat': ['g', 'l', 'm', 'p', 'u', 'w', 'd']
}


# Function to map values in a Series using a dictionary
def map_series_with_dict(series, key):
    """
    Maps values in a pandas Series using the mushroom_dict dictionary.

    Parameters:
    ----------
    series : pandas.Series
        The Series whose values need to be mapped.
    key : str
        The key in the mushroom_dict to use for mapping.

    Returns:
    -------
    pandas.Series
        The Series with mapped values.
    """
    return series.map(mushroom_dict[key])

# Function to rename columns of a DataFrame using a dictionary
def rename_columns(df):
    """
    Renames the columns of a pandas DataFrame using the attribute_descriptions
    dictionary.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame whose columns need to be renamed.

    Returns:
    -------
    pandas.DataFrame
        The DataFrame with renamed columns.
    """
    return df.rename(columns=dict(zip(df.columns, attribute_descriptions.values())))

    
def process_models(models, X_train, X_test, y_train, y_test,
                   p_class_weight=1, random_state=42):
    """
    Processes a list of machine learning models, evaluates their performance,
    and extracts feature importances.

    Parameters:
    ----------
    models : list of tuples
        A list of tuples, where each tuple contains:
        - model_name : str
            The name of the model (e.g., "Random Forest").
        - model_class : class
            The class of the model (e.g., RandomForestClassifier).
        - model_params : dict
            The parameters to initialize the model.

    X_train : pandas.DataFrame or numpy.ndarray
        The training data features.

    X_test : pandas.DataFrame or numpy.ndarray
        The test data features.

    y_train : pandas.Series or numpy.ndarray
        The training data labels.

    y_test : pandas.Series or numpy.ndarray
        The test data labels.

    p_class_weight: weight for the positive (poisonous mushroom) class.
        
    Returns:
    -------
    performance_metrics : dict
        A dictionary where the keys are model names and the values are dictionaries
        containing the following performance metrics:
        - accuracy : float
            The accuracy score of the model.
        - recall : float
            The recall score of the model.
        - precision : float
            The precision score of the model.
        - f1 : float
            The F1 score of the model.
        - roc_auc : float
            The ROC AUC score of the model.
        - confusion_matrix : numpy.ndarray
            The confusion matrix of the model.

    feature_importances : dict
        A dictionary where the keys are model names and the values are
        pandas.Series containing the feature importances for the model,
        sorted in descending order.
    """
    performance_metrics = {}
    feature_importances = {}
    fitted_models = {}
    
    for model_info in models:
        model_name, model_class, model_params = model_info
        print(f"Processing {model_name}...")
        
        # Initialize and train the model
        if model_name == "Naive Bayes":
            model = model_class(**model_params)
            model.fit(X_train, y_train,
                      sample_weight=[10 if label == 1 else 1 for label in y_train])
        elif model_name == "KNN":
            model = model_class(**model_params)
            model.fit(X_train, y_train)
        else:
            model_params['random_state'] = random_state
            if model_name == "XGBoost":
                model = model_class(**model_params,
                                    scale_pos_weight=p_class_weight)
            else:
                model_params['class_weight'] = {0: 1, 1: p_class_weight}
                model = model_class(**model_params)
            model.fit(X_train, y_train) 
        
        # Store the fitted model
        fitted_models[model_name] = model
        
        # Predict on test data
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='binary')
        precision = precision_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        roc_auc = roc_auc_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Store performance metrics
        performance_metrics[model_name] = {
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix
        }
        
        # Extract feature importances if applicable
        if hasattr(model, 'feature_importances_'):
            # Create a pandas Series for feature importances
            feature_importances[model_name] = pd.Series(
                model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=False)
        # elif isinstance(model, SVC) and model.kernel == 'linear':
        #     results = permutation_importance(model, X_train, y_train,
        #                                      n_repeats=10, random_state=42)
        #     feature_importances[model_name] = pd.Series(
        #         results.importances_mean,
        #         index=X_train.columns).sort_values(ascending=False)
        
        # Plot decision tree if applicable
        #if isinstance(model, tree.DecisionTreeClassifier):
        #    tree.plot_tree(model)
    
    return performance_metrics, feature_importances, fitted_models


def analyze_models(performance_metrics, feature_importances, target_encoding,
                   n_features=5, print_metrics=False, print_metrics_df=True,
                   plot_importances=False, plot_confusion_matrices=False,
                   sort_by='precision'):
    """
    Analyzes and visualizes model performance and feature importances.

    Parameters:
    -----------
    performance_metrics : dict
        A dictionary where keys are model names and values are dictionaries of
        performance metrics.
    feature_importances : dict
        A dictionary where keys are model names and values are pandas.Series of
        feature importances.
    target_encoding : dict
        A dictionary mapping class labels to their encoded values
        (e.g., {'edible': 0, 'poisonous': 1}).
    n_features : int, optional (default=5)
        The number of top features to display or plot.
    print_metrics : bool, optional (default=True)
        Whether to print detailed performance metrics for each model.
    print_metrics_df : bool, optional (default=True)
        Whether to print the performance metrics in a DataFrame format.
    plot_importances : bool, optional (default=True)
        Whether to plot the top n feature importances for each model.
    plot_confusion_matrices : bool, optional (default=True)
        Whether to plot confusion matrices for each model.
    sort_by : str, optional (default='precision')
        The column to sort the DataFrame by and highlight.

    Returns:
    --------
    pd.io.formats.style.Styler
        A styled DataFrame containing the performance metrics, sorted by the
        specified column and with the specified column highlighted in lime color.
    """
    # Extract class labels and ensure they are ordered by their encoded values
    class_labels = [k for k, v in sorted(target_encoding.items(),
                                         key=lambda item: item[1])]
    
    # Print detailed performance metrics
    if print_metrics:
        for model_name, metrics in performance_metrics.items():
            print(f"\nPerformance Metrics for {model_name}:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

    # Create performance metrics DataFrame
    performance_df = pd.DataFrame(performance_metrics).T
    performance_df.index.name = 'Model'
    performance_df = performance_df.drop(columns=['confusion_matrix'])

    # Sort the DataFrame by the specified column in descending order
    performance_df = performance_df.sort_values(by=sort_by, ascending=False)

    # Highlight the specified column in lime color
    def highlight_column(val, column_name):
        return 'background-color: lime' if column_name == sort_by else ''

    styled_df = performance_df.style.apply(lambda x: x.apply(highlight_column,
                                                             column_name=x.name))

    # Print performance metrics in a DataFrame
    if print_metrics_df:
        print(f"\nPerformance Metrics DataFrame (Sorted by {sort_by}):")
        display(styled_df)  # Use display to render the styled DataFrame

    # Plot feature importances
    if plot_importances:
        for model_name, importances in feature_importances.items():
            print(f"\nTop {n_features} Feature Importances for {model_name}:")
            print(importances.head(n_features))
            
            # Plot top n feature importances
            plt.figure(figsize=(6, 3))
            importances.head(n_features).plot(kind='barh', color='skyblue')
            plt.title(f'Top {n_features} Feature Importances for {model_name}')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.show()

    # Plot confusion matrices
    if plot_confusion_matrices:
        n_models = len(performance_metrics)
        n_cols = min(3, n_models)  # Up to 3 columns
        n_rows = (n_models + n_cols - 1) // n_cols  # Calculate number of rows

        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])  # Ensure axes is always a 2D array
        axes = axes.flatten()  # Flatten the axes array for easy iteration

        # Plot confusion matrices
        for i, (model_name, metrics) in enumerate(performance_metrics.items()):
            ax = axes[i]
            confusion_matrix = metrics['confusion_matrix']
            
            # Plot the confusion matrix
            sns.heatmap(confusion_matrix, annot=False, fmt='d',
                        cmap='Blues', cbar=False, ax=ax,
                        xticklabels=class_labels,  # Use class labels for x-axis
                        yticklabels=class_labels)  # Use class labels for y-axis
            
            # Add custom annotations
            for j in range(confusion_matrix.shape[0]):
                for k in range(confusion_matrix.shape[1]):
                    value = confusion_matrix[j, k]
                    if class_labels[j] == 'p' and class_labels[k] == 'e':
                        if value == 0:
                            ax.text(k + 0.5, j + 0.5, str(value),
                                    ha='center', va='center',
                                    fontweight='bold', color='green', fontsize=12)
                        else:
                            ax.text(k + 0.5, j + 0.5, str(value),
                                    ha='center', va='center',
                                    fontweight='bold', color='red', fontsize=12)
                    else:
                        ax.text(k + 0.5, j + 0.5, str(value),
                                ha='center', va='center',
                                fontsize=12)
            
            ax.set_title(f'Confusion Matrix for {model_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()
    
    return styled_df


def plot_roc_curves(models, X_train, X_test, y_train, y_test):
    """
    Plots ROC curves for multiple models.

    Parameters:
    - models: List of tuples, where each tuple contains
              (model_name, model_class, model_params)
    - X_train: Training data features
    - X_test: Test data features
    - y_train: Training data labels
    - y_test: Test data labels
    """
    plt.figure(figsize=(6, 4))
    
    for model_name, model_class, model_params in models:
        # Initialize the model
        model = model_class(**model_params)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Check if the model supports predict_proba
        if hasattr(model, 'predict_proba'):
            # Predict probabilities for the positive class
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Compute ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot the ROC curve
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
        else:
            print(f"Skipping {model_name} because it does not support predict_proba.")
    
    # Plot the random guess line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.title('ROC Curves for All Models')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
    

def visualize_shared_feature_importances(feature_importances, n_features=5):
    """
    Visualizes how feature importances are shared among models.

    Parameters:
    -----------
    feature_importances : dict
        A dictionary where keys are model names and values are
        pandas.Series of feature importances.
    n_features : int, optional (default=5)
        The number of top features to consider for each model.

    Returns:
    --------
    None
    """
    # Collect top n features for each model
    top_features = {}
    for model_name, importances in feature_importances.items():
        top_features[model_name] = importances.head(n_features).index.tolist()

    # Flatten the list of top features across all models
    all_top_features = [feature for features in top_features.values() for feature in features]

    # Count how many times each feature appears
    feature_counts = pd.Series(all_top_features).value_counts()

    # Plot the shared feature importances
    plt.figure(figsize=(6, 4))  # Adjust figure size for horizontal bars
    sns.barplot(x=feature_counts.values, y=feature_counts.index,
                palette='viridis', orient='h')  # Use horizontal bars
    plt.title(f"Shared Feature Importances Across Models (Top {n_features} Features)")
    plt.xlabel("Count of Models")
    plt.ylabel("Features")
    plt.tight_layout()  # Ensure labels fit properly
    plt.show()


def aggregate_onehot_feature_importances(feature_importances, onehot_prefixes,
                                         aggregation_method='sum', n=5):
    """
    Aggregates the importance scores of one-hot encoded features back to their
    original categorical features and highlights the top n features in each column
    with a gradient color.

    Parameters:
    -----------
    feature_importances : dict
        A dictionary where keys are model names and values are pandas.Series of
        feature importances.
    onehot_prefixes : list
        A list of prefixes used in one-hot encoding (e.g., ['odor', 'color']).
    aggregation_method : str, optional (default='sum')
        The method to use for aggregating the importance scores.
        Options: 'sum', 'mean', 'max'.
    n : int, optional (default=5)
        The number of top features to highlight in each column.

    Returns:
    --------
    pandas.io.formats.style.Styler
        A styled DataFrame where rows are original categorical features and columns
        are model names. Each cell contains the aggregated importance score for that
        feature and model, with the top n features in each column highlighted using
        a gradient color.
    """
    # Validate the aggregation method
    if aggregation_method not in ['sum', 'mean', 'max']:
        raise ValueError("Invalid aggregation method. "
                         "Choose from 'sum', 'mean', or 'max'.")

    # Create an empty DataFrame to store aggregated importances
    aggregated_df = pd.DataFrame()

    for model_name, importances in feature_importances.items():
        # Create a dictionary to store aggregated importances for the current model
        aggregated = {}

        # Iterate through each one-hot encoded prefix
        for prefix in onehot_prefixes:
            # Find all columns that start with the prefix
            prefix_columns = [col for col in importances.index if col.startswith(prefix)]

            # Aggregate the importance scores based on the chosen method
            if aggregation_method == 'sum':
                aggregated[prefix] = importances[prefix_columns].sum()
            elif aggregation_method == 'mean':
                aggregated[prefix] = importances[prefix_columns].mean()
            elif aggregation_method == 'max':
                aggregated[prefix] = importances[prefix_columns].max()

        # Convert the aggregated dictionary to a pandas Series and add it to the DataFrame
        aggregated_df[model_name] = pd.Series(aggregated)

    # Sort the DataFrame by the sum of importances across all models
    aggregated_df['Total'] = aggregated_df.sum(axis=1)
    aggregated_df = aggregated_df.sort_values(by='Total', ascending=False).drop(columns=['Total'])

    # Function to apply gradient coloring to the top n values in each column
    def gradient_top_n(s, n=5, cmap='YlOrRd'):
        """
        Applies gradient coloring to the top n values in a Series.
        """
        # Create a boolean mask for the top n values
        is_top_n = pd.Series(False, index=s.index)
        is_top_n[s.nlargest(n).index] = True

        # Normalize the top n values to [0, 1] for the gradient
        top_n_values = s[is_top_n]
        if len(top_n_values) > 1:  # Avoid division by zero
            normalized_values = (top_n_values - top_n_values.min()) / (top_n_values.max() - top_n_values.min())
        else:
            normalized_values = pd.Series(1.0, index=top_n_values.index)  # Single value, max gradient

        # Create a list of CSS styles for the top n values
        styles = []
        cmap_func = plt.cm.get_cmap(cmap)  # Get the color map function
        for idx, val in s.items():
            if idx in top_n_values.index:
                # Get the RGBA color from the color map
                rgba = cmap_func(normalized_values[idx])  # Use normalized value directly
                # Lighten the color by increasing the alpha value (e.g., 0.6 for 60% opacity)
                rgba = (*rgba[:3], 0.6)  # Keep RGB values, set alpha to 0.6
                # Convert RGBA to a CSS color string
                color = f"background-color: rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})"
                styles.append(color)
            else:
                styles.append("")
        return styles

    # Apply the gradient coloring to each column independently
    styled_df = aggregated_df.style.apply(lambda x: gradient_top_n(x, n=n), axis=0)

    return aggregated_df, styled_df


def styled_df_to_latex(styled_df, n=5, decimals=4):
    """
    Converts a styled DataFrame to LaTeX format with bold text and background colors for the top n values.
    The numbers are rounded to the specified number of decimal places.

    Parameters:
    -----------
    styled_df : pandas.io.formats.style.Styler
        A styled DataFrame.
    n : int, optional (default=5)
        The number of top features to highlight in each column.
    decimals : int, optional (default=4)
        The number of decimal places to round the numbers to.

    Returns:
    --------
    str
        LaTeX code for the table.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = styled_df.data.copy()

    # Round the numbers to the specified number of decimal places
    df = df.round(decimals)

    # Function to convert RGBA to LaTeX color
    def rgba_to_latex_color(rgba):
        return f"[rgb]{{{rgba[0]:.3f},{rgba[1]:.3f},{rgba[2]:.3f}}}"

    # Apply background colors and bold text to the top n values in each column
    for col in df.columns:
        top_n_indices = df[col].nlargest(n).index
        for idx in top_n_indices:
            # Get the RGBA color from the styled DataFrame
            ctx = styled_df._compute().ctx
            if (idx, col) in ctx and 'background-color' in ctx[(idx, col)]:
                rgba_str = ctx[(idx, col)]['background-color']
                rgba = tuple(float(x) for x in rgba_str.strip('rgba()').split(','))
                # Convert RGBA to LaTeX color
                latex_color = rgba_to_latex_color(rgba)
                # Apply background color and bold text
                df.loc[idx, col] = f"\\cellcolor{{{latex_color}}}\\textbf{{{df.loc[idx, col]}}}"

    # Escape underscores in column names and index
    df.columns = [col.replace("_", "\\_") for col in df.columns]
    df.index = [idx.replace("_", "\\_") for idx in df.index]

    # Convert the DataFrame to LaTeX
    latex_code = df.to_latex(escape=False, index=True)
    return latex_code


# def aggregate_onehot_feature_importances(feature_importances, onehot_prefixes,
#                                          aggregation_method='sum'):
#     """
#     Aggregates the importance scores of one-hot encoded features back to their
#     original categorical features.

#     Parameters:
#     -----------
#     feature_importances : dict
#         A dictionary where keys are model names and values are pandas.Series of
#         feature importances.
#     onehot_prefixes : list
#         A list of prefixes used in one-hot encoding (e.g., ['odor', 'color']).
#     aggregation_method : str, optional (default='sum')
#         The method to use for aggregating the importance scores.
#         Options: 'sum', 'mean', 'max'.

#     Returns:
#     --------
#     pandas.DataFrame
#         A DataFrame where rows are original categorical features and columns
#         are model names.
#         Each cell contains the aggregated importance score for that feature
#         and model.
#     """
#     # Validate the aggregation method
#     if aggregation_method not in ['sum', 'mean', 'max']:
#         raise ValueError("Invalid aggregation method. "
#                          "Choose from 'sum', 'mean', or 'max'.")

#     # Create an empty DataFrame to store aggregated importances
#     aggregated_df = pd.DataFrame()

#     for model_name, importances in feature_importances.items():
#         # Create a dictionary to store aggregated importances for the current model
#         aggregated = {}

#         # Iterate through each one-hot encoded prefix
#         for prefix in onehot_prefixes:
#             # Find all columns that start with the prefix
#             prefix_columns = [col for col in importances.index if col.startswith(prefix)]

#             # Aggregate the importance scores based on the chosen method
#             if aggregation_method == 'sum':
#                 aggregated[prefix] = importances[prefix_columns].sum()
#             elif aggregation_method == 'mean':
#                 aggregated[prefix] = importances[prefix_columns].mean()
#             elif aggregation_method == 'max':
#                 aggregated[prefix] = importances[prefix_columns].max()

#         # Convert the aggregated dictionary to a pandas Series and add it to the DataFrame
#         aggregated_df[model_name] = pd.Series(aggregated)

#     # Sort the DataFrame by the sum of importances across all models
#     aggregated_df['Total'] = aggregated_df.sum(axis=1)
#     aggregated_df = aggregated_df.sort_values(by='Total', ascending=False).drop(columns=['Total'])

#     return aggregated_df


def plot_aggregated_feature_importances_heatmap(aggregated_df,
                                                title="Aggregated Feature "
                                                "Importances Across Models"):
    """
    Plots a heatmap of the aggregated feature importances across models.

    Parameters:
    -----------
    aggregated_df : pandas.DataFrame
        A DataFrame where rows are original categorical features and
        columns are model names.
        Each cell contains the aggregated importance score for that feature
        and model.
    title : str, optional (default="Aggregated Feature Importances Across Models")
        The title of the heatmap.

    Returns:
    --------
    None
    """
    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(aggregated_df, annot=True, fmt=".2f", cmap='viridis', cbar=True)
    plt.title(title)
    plt.xlabel("Models")
    plt.ylabel("Features")
    plt.show()


    
def plot_tsne(X_onehot, csv_df, feature_to_add1=None, feature_to_add2=None,
              title="t-SNE Visualization"):
    """
    Plots a t-SNE visualization with optional additional features for
    coloring and styling.

    Parameters:
    -----------
    X_onehot : pandas.DataFrame or numpy.ndarray
        The one-hot encoded feature matrix.
    csv_df : pandas.DataFrame
        The original DataFrame containing additional features (e.g., 'odor').
    feature_to_add1 : str, optional (default=None)
        The first feature to add for coloring the plot (hue).
        If None, only 'class' will be used.
    feature_to_add2 : str, optional (default=None)
        The second feature to add for styling the plot (shape).
        If None, only 'class' will be used.
    title : str, optional (default="t-SNE Visualization")
        The title of the plot.

    Returns:
    --------
    None
    """
    class_series = map_series_with_dict(csv_df['class'], 'class')
    
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=0)
    X_onehot_tsne = tsne.fit_transform(X_onehot)

    # Create a DataFrame for t-SNE results
    df_tsne_data = np.vstack((X_onehot_tsne.T, class_series)).T
    df_tsne_columns = ['Dim1', 'Dim2', 'class']
    df_tsne = pd.DataFrame(df_tsne_data, columns=df_tsne_columns)

    # If features to add are specified, add them to the DataFrame
    if feature_to_add1 and feature_to_add2:
        df_tsne_feature_data = np.hstack((X_onehot_tsne,
                                          csv_df[['class', feature_to_add1,
                                                  feature_to_add2]].to_numpy()))
        df_tsne_feature_columns = ['Dim1', 'Dim2', 'class', feature_to_add1,
                                   feature_to_add2]
        df_tsne = pd.DataFrame(df_tsne_feature_data,
                                columns=df_tsne_feature_columns)

        # Map values in the specified feature columns
        df_tsne['class'] = map_series_with_dict(csv_df['class'], 'class')
        df_tsne[feature_to_add1] = map_series_with_dict(csv_df[feature_to_add1],
                                                        feature_to_add1)
        df_tsne[feature_to_add2] = map_series_with_dict(csv_df[feature_to_add2],
                                                        feature_to_add2)

        # Define a list of valid marker symbols for 3D scatter plots
        marker_symbols = ['circle', 'cross', 'square', 'diamond', 'x']

        # Check if feature_to_add2 has more unique values than available marker symbols
        if len(df_tsne[feature_to_add2].unique()) > len(marker_symbols):
            # Exchange feature_to_add1 and feature_to_add2
            feature_to_add1, feature_to_add2 = feature_to_add2, feature_to_add1

        # 3D Scatter Plot with Plotly
        fig = go.Figure()

        # Add traces for each combination of feature_to_add1 and feature_to_add2
        for color_val in df_tsne[feature_to_add1].unique():
            for i, shape_val in enumerate(df_tsne[feature_to_add2].unique()):
                subset = df_tsne[(df_tsne[feature_to_add1] == color_val) &
                                 (df_tsne[feature_to_add2] == shape_val)]
                fig.add_trace(go.Scatter3d(
                    x=subset['Dim1'],
                    y=subset['Dim2'],
                    z=subset['class'],
                    mode='markers',
                    marker=dict(
                        color=px.colors.qualitative.Plotly[
                            list(df_tsne[
                                feature_to_add1].unique()).index(color_val)],
                        symbol=marker_symbols[i % len(marker_symbols)],  
                        size=2,
                        opacity=0.7
                    ),
                    name=f"{feature_to_add1}: {color_val}, {feature_to_add2}: {shape_val}",
                    legendgroup=f"{feature_to_add1}: {color_val}",
                    showlegend=False  # Disable legend for these traces
                ))

        # Customize the layout
        fig.update_layout(
            title=f'3D t-SNE Visualization: {feature_to_add1} (color) and {feature_to_add2} (shape)',
            scene=dict(
                xaxis_title='Dim 1',
                yaxis_title='Dim 2',
                zaxis_title='Class'
            ),
            legend=dict(
                x=0.02, y=0.98,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='gray',
                borderwidth=1
            ),
            legend_title_text='',
            width=800,  # Set figure width
            height=600  # Set figure height
        )

        # Manually create separate legends for color and symbol
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=0),
            name=f"<b>{feature_to_add1}</b>",  # Bold title for feature_to_add1
            legendgroup="color_legend",
            showlegend=True
        ))

        for color_val in df_tsne[feature_to_add1].unique():
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(
                    color=px.colors.qualitative.Plotly[
                        list(df_tsne[feature_to_add1].unique()).index(color_val)],
                    size=10
                ),
                name=f"{color_val}",
                legendgroup="color_legend",
                showlegend=True
            ))

        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=0),
            name=f"<b>{feature_to_add2}</b>",  # Bold title for feature_to_add2
            legendgroup="symbol_legend",
            showlegend=True
        ))

        for i, shape_val in enumerate(df_tsne[feature_to_add2].unique()):
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(
                    symbol=marker_symbols[i % len(marker_symbols)],
                    size=10,
                    color='gray'
                ),
                name=f"{shape_val}",
                legendgroup="symbol_legend",
                showlegend=True
            ))

        fig.show()

    elif feature_to_add1:
        # Plot t-SNE with the first specified feature as the hue and 'class' as style
        df_tsne_feature_data = np.hstack((X_onehot_tsne,
                                          csv_df[['class',
                                                  feature_to_add1]].to_numpy()))
        df_tsne_feature_columns = ['Dim1', 'Dim2', 'class', feature_to_add1]
        df_tsne = pd.DataFrame(df_tsne_feature_data,
                                columns=df_tsne_feature_columns)

        # Map values in the specified feature column
        df_tsne['class'] = map_series_with_dict(csv_df['class'], 'class')
        df_tsne[feature_to_add1] = map_series_with_dict(csv_df[feature_to_add1],
                                                        feature_to_add1)

        # 2D Scatter Plot with Seaborn
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_tsne, x='Dim1', y='Dim2', alpha=0.7,
                        hue=feature_to_add1, style='class')
        plt.title(f'Link between {feature_to_add1} and edibility')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()
    else:
        # Plot t-SNE with 'class' as the hue
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_tsne, x='Dim1', y='Dim2', hue='class', alpha=0.7)
        plt.title(title)
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()
        
 
# Function to perform Recursive Feature Elimination (RFE)
def perform_rfe(model, X_train, y_train, n_features_to_select):
    """
    Performs Recursive Feature Elimination (RFE) to select the most 
    mportant features.

    Parameters:
    ----------
    model : sklearn estimator
        The model to use for feature selection.
    X_train : pandas.DataFrame or numpy.ndarray
        The training data features.
    y_train : pandas.Series or numpy.ndarray
        The training data labels.
    n_features_to_select : int
        The number of features to select.

    Returns:
    -------
    selected_features : list
        A list of selected feature names.
    """
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    rfe.fit(X_train, y_train)
    selected_features = X_train.columns[rfe.support_]
    return selected_features

# Function to calculate Mutual Information scores
def calculate_mutual_info(X_train, y_train):
    """
    Calculates Mutual Information scores between features and the
    target variable.

    Parameters:
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        The training data features.
    y_train : pandas.Series or numpy.ndarray
        The training data labels.

    Returns:
    -------
    mi_scores : pandas.Series
        A Series containing Mutual Information scores for each feature.
    """
    mi_scores = mutual_info_classif(X_train, y_train)
    mi_scores = pd.Series(mi_scores, index=X_train.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def calculate_chi2_for_features(X_train, y_train, min_expected_freq=5):
    """
    Perform Chi-Square Test for each categorical feature in the dataset.
    Filters out features that do not satisfy the expected frequencies
    assumption.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        The dataset containing the features.
    y_train : pandas.Series
        The target variable.
    min_expected_freq : int, optional (default=5)
        The minimum expected frequency required for the Chi-Square Test to be
        valid.
    
    Returns:
    --------
    chi2_scores : pandas.Series
        A Series containing Chi-Square scores for features that satisfy the
        expected frequencies assumption.
    chi2_results : dict
        A dictionary where keys are feature names and values are dictionaries
        containing Chi-Square statistic, p-value, degrees of freedom, expected
        frequencies, and the contingency table as a DataFrame.
    """
    # Initialize the dictionary to store results
    chi2_results = {}
    chi2_scores = {}

    # Get the feature columns
    feature_columns = X_train.columns
    
    # Perform Chi-Square Test for each feature
    for feature in feature_columns:
        # Create a contingency table
        contingency_table = pd.crosstab(X_train[feature], y_train)
        
        # Perform Chi-Square Test
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Check if all expected frequencies are at least min_expected_freq
        if (expected >= min_expected_freq).all():
            # Store the results, including the contingency table as a DataFrame
            chi2_results[feature] = {
                'Chi-Square Statistic': chi2_stat,
                'P-value': p_value,
                'Degrees of Freedom': dof,
                'Expected Frequencies': pd.DataFrame(expected,
                                                     index=contingency_table.index,
                                                     columns=contingency_table.columns),
                'Contingency Table': contingency_table
            }
            # Store the Chi-Square score
            chi2_scores[feature] = chi2_stat

    # Convert chi2_scores to a pandas Series and sort it
    chi2_scores = pd.Series(chi2_scores).sort_values(ascending=False)

    return chi2_scores, chi2_results


def get_styled_results(combined_results, top_n=10):
    """
    Returns a styled DataFrame with the top N values in each column highlighted.
    Automatically assigns unique colors to each column.
    
    Parameters:
    -----------
    combined_results : pandas.DataFrame
        The DataFrame containing the combined results.
    top_n : int, optional (default=7)
        The number of top values to highlight in each column.
    
    Returns:
    --------
    pandas.io.formats.style.Styler
        A styled DataFrame with the top N values in each column highlighted.
    """
    def highlight_top_values(column, color='lightgreen'):
        """
        Highlights the top N values in a column.
        :param column: A pandas Series (column of the DataFrame).
        :param color: The color to use for highlighting.
        :return: A list of CSS styles for the column.
        """
        # Find the top N values in the column
        unique_values = column.unique()
        n = min(top_n, len(unique_values))
        top_values = column.nlargest(n)
        # Return a list of CSS styles for the column
        return ['background-color: {}'.format(color)
                if value in top_values.values else '' for value in column]

    # Define a list of colors to use for highlighting
    colors = [
        'lightgreen', 'lightblue', 'lightcoral', 'lightyellow', 'lightpink',
        'lightskyblue', 'lightseagreen', 'lightsteelblue', 'lightcyan', 'lightgray'
    ]
    
    # If there are more columns than colors, cycle through the colors
    if len(combined_results.columns) > len(colors):
        colors = colors * (len(combined_results.columns) // len(colors) + 1)

    # Apply the highlighting function to each column with a unique color
    styled_results = combined_results.style
    for i, column in enumerate(combined_results.columns):
        styled_results = styled_results.apply(highlight_top_values,
                                              color=colors[i], subset=[column])

    return styled_results


def perform_feature_selection_analysis(X_train, y_train, models,
                                       onehot_prefixes,
                                       n_features_to_select=10,
                                       features_to_analyze=None,
                                       sort_by=None):
    """
    Perform RFE, Mutual Information, and Chi-Square analysis for feature selection.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        The training data features.
    y_train : pandas.Series
        The training data labels.
    models : list of tuples
        A list of tuples where each tuple contains (model_name, model_instance).
        Example: [('RF', RandomForestClassifier()), ('XGB', xgb.XGBClassifier())]
    n_features_to_select : int, optional (default=10)
        The number of features to select using RFE.
    features_to_analyze : list, optional (default=None)
        A list of specific features to analyze. If None, all features are analyzed.
    sort_by : str, optional (default=None)
        The column name to sort the results by. If None, no sorting is applied.
    
    Returns:
    --------
    combined_results : pandas.DataFrame
        A DataFrame containing aggregated feature importance scores from RFE,
        Mutual Information, and Chi-Square.
    """
    # If specific features are provided, subset the data
    if features_to_analyze is not None:
        X_train = X_train[features_to_analyze]

    # Initialize a dictionary to store aggregated results
    aggregated_results = {}

    # Perform RFE for each model
    for model_name, model_instance in models:
        # Perform RFE
        selected_features_rfe = perform_rfe(model_instance, X_train, y_train,
                                            n_features_to_select)
        
        # Convert RFE results to a Series for aggregation
        rfe_importances = pd.Series(1, index=selected_features_rfe)
        
        # Aggregate RFE results for original features
        rfe_aggregated, _ = aggregate_onehot_feature_importances(
            {f'{model_name} RFE': rfe_importances}, onehot_prefixes,
            aggregation_method='sum'
        )
        
        # Store the aggregated results
        aggregated_results[f'{model_name} RFE'] = rfe_aggregated[f'{model_name} RFE']

    # Calculate Mutual Information
    mi_scores = calculate_mutual_info(X_train, y_train)
    
    # Aggregate Mutual Information results for original features
    mi_aggregated, _ = aggregate_onehot_feature_importances(
        {'Mutual Information': mi_scores}, onehot_prefixes,
        aggregation_method='sum'
    )
    aggregated_results['Mutual Information'] = mi_aggregated['Mutual Information']

    # Calculate Chi-Square scores
    chi2_scores, _ = calculate_chi2_for_features(X_train, y_train)
    
    # Aggregate Chi-Square results for original features
    chi2_aggregated, _ = aggregate_onehot_feature_importances(
        {'Chi-Square': chi2_scores}, onehot_prefixes, aggregation_method='sum'
    )
    aggregated_results['Chi-Square'] = chi2_aggregated['Chi-Square']

    # Combine the results into a single DataFrame
    combined_results = pd.concat(aggregated_results, axis=1)
    
    # Rename the columns for clarity
    combined_results.columns = [col for col in combined_results.columns]

    # Sort the results if a column is specified
    if sort_by is not None:
        if sort_by in combined_results.columns:
            combined_results = combined_results.sort_values(by=sort_by,
                                                            ascending=False)
        else:
            raise ValueError(f"Column '{sort_by}' not found in the results.")

    return combined_results


def select_top_n_indices(df, column, n):
    """
    Select the top n indices from a specific column in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the column.
    column : str
        The name of the column to select indices from.
    n : int
        The number of top indices to select.
    
    Returns:
    --------
    pandas.Index
        The top n indices from the specified column.
    
    Raises:
    -------
    ValueError
        If the column does not exist in the DataFrame or n is invalid.
    """
    # Validate the column
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")
    
    # Validate n
    if n <= 0 or n > len(df):
        raise ValueError(f"Invalid value for n: {n}. n must be between 1 and "
                         "the number of rows in the DataFrame.")
    
    # Select the top n indices
    return df[column].nlargest(n).index


def analyze_features(csv_df, features, models, target='class',
                     class_dict={'e': 0, 'p': 1}, test_size=0.4,
                     random_state=42):
    """
    Perform data analysis on a given set of features.

    Parameters:
    - csv_df: DataFrame containing the data.
    - features: List of features to be used for analysis (should not include the target).
    - models: List of models to be evaluated.
    - target: Target column name (default is 'class').
    - class_dict: Dictionary to map target classes to numerical values
                 (default is {'e': 0, 'p': 1}).
    - test_size: Proportion of the dataset to include in the test split
                (default is 0.4).
    - random_state: Seed for random number generation (default is 42).

    Returns:
    - performance_metrics: Performance metrics of the models.
    - feature_importances: Feature importances from the models.
    """
    
    # Ensure the target column is not included in the features
    if target in features:
        features = [col for col in features if col != target]
        print(f"Warning: Removed '{target}' "
              "from features list as it is the target column.")

    # Filter the features and target
    X_filtered = csv_df[features]
    Y_filtered = csv_df[target]
    
    # Map the target classes to numerical values
    Y_filtered_onehot = Y_filtered.map(class_dict)

    # creating train/test split using 70% data for training and 30% for testing
    (X_filtered_train,
     X_filtered_test,
     y_filtered_train,
     y_filtered_test) = train_test_split(X_filtered, Y_filtered_onehot,
                                         test_size = 0.3,
                                         random_state = 42)
    # Initialize the OneHotEncoder
    preprocessor = OneHotEncoder(sparse_output=False)

    # Apply preprocessing and convert back to DataFrame
    X_filtered_train = pd.DataFrame(preprocessor.fit_transform(X_filtered_train),
                                    columns=preprocessor.get_feature_names_out())
    X_filtered_test = pd.DataFrame(preprocessor.transform(X_filtered_test),
                                   columns=preprocessor.get_feature_names_out())

    # Process all models
    performance_metrics, feature_importances, fitted_models = process_models(
        models, X_filtered_train, X_filtered_test, y_filtered_train, y_filtered_test
    )

    return (fitted_models, performance_metrics, feature_importances,
            X_filtered_train, y_filtered_train, X_filtered_test, y_filtered_test)

