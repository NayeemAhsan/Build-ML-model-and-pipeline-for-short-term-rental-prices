# library doc string
"""
Helper functions to run EDA.ipynb in Jupyter
Author: Nayeem Ahsan
Date: 5/162024
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def plot_boxplot(data, features, figsize=(15, 10), nrows=3, ncols=3):
  """
  Plots a seaborn boxplot with customizations.

  Args:
      data: pandas DataFrame containing the data to be plotted.
      x: String, name of the column in the DataFrame for the x-axis.

  Returns:
      A matplotlib Axes object containing the plot (optional).
  """

    # Create a figure for subplots
  fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

  axes = [axes[i][j] for i in range(nrows) for j in range(ncols)]

  for i, col in zip(range(nrows * ncols), features):
        sns.boxplot(data, x=col, ax=axes[i])
        # Customize the plot
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        plt.ylabel(f'Frequency')
        plt.xticks(rotation=45)  # Rotate x-axis labels for readability (optional)


def escape_special_chars(text):
    """Escape special characters for Matplotlib."""
    return text.replace('$', r'\$').replace('%', r'\%').replace('&', r'\&')


def plot_barplot(dataframe, column):
    '''
    Plot barplot for a column and return the plot object.

    Parameters:
        dataframe (DataFrame): Pandas DataFrame containing the data.
        column (str): Name of the column to plot.

    Returns:
        plt.figure: The plot object.
    '''
    plt.figure(figsize=(20, 10))
    dataframe[column].value_counts().plot(kind='bar')
    
    # Escape special characters in the column name
    escaped_column = escape_special_chars(column)
    
    # Use raw strings to avoid LaTeX interpretation
    plt.title(rf'Barplot of {escaped_column}', usetex=False)
    plt.xlabel(rf'{escaped_column}', usetex=False)
    plt.ylabel('Frequency', usetex=False)
    plt.legend()
    
    return plt


def plot_correlation(dataframe):
    '''
    Plots a correlation heatmap for numerical columns of a dataframe and saves as JPEG.

    Parameters:
        dataframe (DataFrame): Pandas DataFrame containing the data.
        folder_name (str): Directory to save the plot.

        Returns:
        plt.figure: The plot object.
    '''
    # plot the correlation heatmap
    numeric_dataframe = dataframe.select_dtypes(include=['number'])
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        numeric_dataframe.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.title('Correlation Heatmap')
    plt.xlabel('Numeric Columns')
    plt.ylabel('Numeric Columns')
    return plt

