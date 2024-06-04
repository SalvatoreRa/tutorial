import seaborn as sns
import matplotlib.pyplot as plt
import shap
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
import lime
from sklearn.preprocessing import MinMaxScaler
import networkx as nx

def feature_importance_XGBoost(columns_name=None, _model=None, data=None, target=None):
    """
    Calculate and aggregate feature importances using different methods for an XGBoost model.

    Parameters:
    - columns_name (array-like): The names of the features.
    - _model (XGBoost model object): The trained XGBoost model.
    - data (pd.DataFrame or np.ndarray): The input data used to calculate SHAP and LIME importances.
    - target (array-like): The target values used to calculate permutation importance.

    Returns:
    - feature_importance_df (pd.DataFrame): A DataFrame containing feature importances from different methods, scaled between 0 and 1, and sorted by average importance.

    Description:
    This function calculates feature importances for an XGBoost model using multiple methods:
    - Model-based importance using `_model.feature_importances_`.
    - Different types of importance from the model's booster: weight, gain, cover, total gain, and total cover.
    - SHAP values to explain model predictions.
    - Permutation importance using scikit-learn's `permutation_importance`.
    - LIME (Local Interpretable Model-agnostic Explanations) for local feature importance.

    The function then scales all importance values between 0 and 1, calculates an average importance score, and returns a DataFrame sorted by the average importance score.
    
    example usage:
    import pandas as pd
    url = 'https://raw.githubusercontent.com/SalvatoreRa/tutorial/main/datasets/pol.csv'
    df = pd.read_csv(url, sep= ';')
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    X = df.drop(columns=['target'])
    y = df['target']

    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    num_round = 100
    model  = xgb.XGBClassifier(objective='binary:logistic', 
                                        use_label_encoder=False, eval_metric='logloss')
    model = model.fit(X_train, y_train)

    # Make predictions
    preds = model.predict(X_test)
    predictions = [round(value) for value in preds]

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy:.2f}')
    
    importance = feature_importance_XGBoost(columns_name=X.columns, _model = model, 
                           data = X_test, target=y_test )
    """
    
    # Get model-based feature importance
    feature_importance = _model.feature_importances_
    
    # Get different types of importance from the model's booster
    booster = _model.get_booster()
    weight_importance = booster.get_score(importance_type='weight')
    gain_importance = booster.get_score(importance_type='gain')
    cover_importance = booster.get_score(importance_type='cover')
    total_gain_importance = booster.get_score(importance_type='total_gain')
    total_cover_importance = booster.get_score(importance_type='total_cover')

    # Create DataFrame for all feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': columns_name,
        'XGBoost_importance': feature_importance
    })

    # Convert booster importance scores to DataFrames and merge them
    importance_dfs = []
    for importance_dict, importance_name in zip([weight_importance, gain_importance, cover_importance, total_gain_importance, total_cover_importance],
                                                ['weight_importance', 'gain_importance', 'cover_importance', 'total_gain_importance', 'total_cover_importance']):
        temp_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', importance_name])
        importance_dfs.append(temp_df)

    for df in importance_dfs:
        feature_importance_df = feature_importance_df.merge(df, on='Feature', how='left')

    # Fill NaN values with 0
    feature_importance_df = feature_importance_df.fillna(0)
    
    # SHAP feature importance
    explainer = shap.Explainer(_model)
    shap_values = explainer(data)
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df['shap_importance'] = shap_importance
    
    # Permutation importance
    perm_importance = permutation_importance(_model, data, target, 
                                             n_repeats=10, random_state=42, n_jobs=-1)
    feature_importance_df['permutation_importance'] = perm_importance.importances_mean
    
    # SHAP LIME importance
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(data.values, 
                                                   feature_names=data.columns.tolist(),
                                                   verbose=False, mode='regression')
    num_samples = 100
    lime_importance = np.zeros([len(columns_name), min(num_samples, data.shape[0])])

    # Use a subset of data for LIME explanation to reduce computation time
    for i in range(min(num_samples, data.shape[0])):
        exp = lime_explainer.explain_instance(data.values[i], _model.predict,
                                             num_features=data.shape[1])
        lime_values = np.array([importance for _, importance in exp.local_exp[1]])
        lime_importance[:,i] = lime_values

    lime_importance =  np.mean(lime_importance, axis=1).tolist()
    feature_importance_df['lime_importance'] = lime_importance
    
     # Scale each column between 0 and 1
    numeric_columns = feature_importance_df.select_dtypes(include=[np.number]).columns
    scaler = MinMaxScaler()
    feature_importance_df[numeric_columns] = scaler.fit_transform(feature_importance_df[numeric_columns])
    
    # Calculate the average importance
    feature_importance_df['average_importance'] = feature_importance_df[numeric_columns].mean(axis=1)
    
    # Sort by the average importance
    feature_importance_df = feature_importance_df.sort_values(by='average_importance', ascending=False)
    
    return feature_importance_df


importance = feature_importance_XGBoost(columns_name=X.columns, _model = model, 
                           data = X_test, target=y_test )



def plot_feature_importance_heatmap(feature_importance_df, num_features=None):
    """
    Plots a heatmap of the feature importance DataFrame.

    Parameters:
    - feature_importance_df: DataFrame containing feature importances
    - num_features: Number of top features to display in the heatmap. If None, display all features.
    example usage:
    plot_feature_importance_heatmap(importance, num_features=10)
    """
    # Sort the DataFrame by average_importance
    sorted_df = feature_importance_df.sort_values(by='average_importance', ascending=False)
    
    # Select top features if num_features is specified
    if num_features is not None:
        sorted_df = sorted_df.head(num_features)
    
    # Set the index to Feature for better readability
    sorted_df = sorted_df.set_index('Feature')
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(sorted_df, annot=True, cmap='viridis', cbar=True)
    plt.title('Feature Importance Heatmap')
    plt.show()
    
plot_feature_importance_heatmap(importance, num_features=10)


def plot_feature_importance_barplot(feature_importance_df, num_features=None):
    """
    Plots barplots of the feature importance DataFrame for each feature, showing mean and standard deviation.

    Parameters:
    - feature_importance_df: DataFrame containing feature importances.
    - num_features: Number of top features to display in the barplots. If None, display all features.

    Returns:
    - None: Displays the barplots for feature importances.
    
    example usage:
    plot_feature_importance_barplot(importance, num_features=10)
    
    """
    # Sort the DataFrame by average_importance
    sorted_df = feature_importance_df.sort_values(by='average_importance', ascending=False)
    
    # Select top features if num_features is specified
    if num_features is not None:
        sorted_df = sorted_df.head(num_features)
    
    # Calculate mean and standard deviation of the importance values
    importance_cols = sorted_df.columns.difference(['Feature', 'average_importance'])
    sorted_df['mean_importance'] = sorted_df[importance_cols].mean(axis=1)
    sorted_df['std_importance'] = sorted_df[importance_cols].std(axis=1)
    
    # Plot the barplot with error bars
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Feature', y='mean_importance', data=sorted_df, errorbar=None)
    
    # Adding error bars manually
    plt.errorbar(x=sorted_df['Feature'], y=sorted_df['mean_importance'], yerr=sorted_df['std_importance'], fmt='none', c='black', capsize=5)
    
    plt.title('Feature Importance Barplot with Standard Deviation')
    plt.xlabel('Feature')
    plt.ylabel('Importance (mean ± std)')
    plt.xticks(rotation=90)
    plt.show()

def plot_feature_importance_boxplot(feature_importance_df, num_features=None):
    """
    Plots boxplots of the feature importance DataFrame for each feature.

    Parameters:
    - feature_importance_df: DataFrame containing feature importances.
    - num_features: Number of top features to display in the boxplots. If None, display all features.

    Returns:
    - None: Displays the boxplots for feature importances.
    
    example usage:
    plot_feature_importance_boxplot(importance, num_features=10)
    """
    # Sort the DataFrame by average_importance
    sorted_df = feature_importance_df.sort_values(by='average_importance', ascending=False)
    
    # Select top features if num_features is specified
    if num_features is not None:
        sorted_df = sorted_df.head(num_features)
    
    # Melt the DataFrame for better plotting
    melted_df = sorted_df.melt(id_vars='Feature', var_name='Importance_Type', value_name='Importance_Value')
    
    # Plot the boxplot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Feature', y='Importance_Value', data=melted_df)
    plt.title('Feature Importance Boxplot')
    plt.xticks(rotation=90)
    plt.show()
    
    
def plot_feature_correlation_graph(X, feature_importance_df, num_features=10, min_corr =0.3):
    """
    Plots a graph of feature correlations and importance.

    Parameters:
    - X (pd.DataFrame): The dataset with features.
    - feature_importance_df (pd.DataFrame): DataFrame containing feature importances.
    - num_features (int): Number of top features to select for the graph.
    - min_corr: Minimum correlation between features. If 0 all edges are plotted
    
    Returns:
    - None: Displays the feature correlation graph.
    """
    # Sort the feature importance DataFrame and select the top features
    sorted_df = feature_importance_df.sort_values(by='average_importance', ascending=False).head(num_features)
    top_features = sorted_df['Feature'].tolist()
    
    # Calculate the correlation matrix for the selected features
    corr_matrix = X[top_features].corr()
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes with size proportional to the average feature importance
    for feature in top_features:
        G.add_node(feature, size=sorted_df[sorted_df['Feature'] == feature]['average_importance'].values[0] * 1000)
    
    # Add edges with width proportional to the correlation
    for i in range(len(top_features)):
        for j in range(i + 1, len(top_features)):
            feature_i = top_features[i]
            feature_j = top_features[j]
            corr_value = corr_matrix.loc[feature_i, feature_j]
            if corr_value != 0:
                if abs(corr_value) >= min_corr:
                    color = 'red' if corr_value > 0 else 'blue'
                    G.add_edge(feature_i, feature_j, weight=abs(corr_value), color=color)

    # Get node sizes and edge colors
    sizes = [nx.get_node_attributes(G, 'size')[node] for node in G.nodes()]
    edges = G.edges(data=True)
    colors = [edge[2]['color'] for edge in edges]
    weights = [edge[2]['weight'] * 10 for edge in edges]  # Multiply for better visualization
    
    # Plot the graph
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=sizes, edge_color=colors, width=weights, font_size=10, font_weight='bold')
    
    # Create legend for node sizes (upper left)
    legend_handles = []
    for size in [min(sizes), np.mean(sizes), max(sizes)]:
        legend_handles.append(plt.scatter([], [], c='black', alpha=0.5, s=size, label=f'Importance {size/1000:.2f}'))
    node_legend = plt.legend(handles=legend_handles, scatterpoints=1, frameon=False, labelspacing=1, title='Node size', loc='upper left')
    plt.gca().add_artist(node_legend)  # Add the node legend to the plot
    
    # Create legend for edge widths (lower left)
    legend_handles = []
    for weight in [min(weights), np.mean(weights), max(weights)]:
        legend_handles.append(plt.plot([], [], c='black', alpha=0.5, linewidth=weight, label=f'Correlation {weight/10:.2f}')[0])
    plt.legend(handles=legend_handles, frameon=False, labelspacing=1, title='Edge width', loc='lower left')
    
    plt.title('Feature Correlation Graph')
    plt.show()