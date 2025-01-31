import matplotlib.pyplot as plt
import pandas as pd 

# Global variables for customization
TITLE_FONT_SIZE = 14  # Slightly larger for better emphasis on the title
AXIS_LABEL_FONT_SIZE = 14  # Keep axis labels prominent
TICK_FONT_SIZE = 12  # Slightly smaller than labels for balance
LINE_WIDTH = 2  # Thinner lines to avoid cluttering in markdown
FIGURE_SIZE = (15, 5)  # A bit smaller to fit well in markdown


def plot_optimization_logs(log_df, save=False, file_name='optimization_plot'):
    """
    Plots optimization logs showing Best Y overall, Best Y every iteration, and Expected Improvement over iterations.
    
    Parameters:
    - log_df (pd.DataFrame): DataFrame containing the following columns:
        - 'Iteration': Iteration numbers
        - 'Best_Y': Best Y overall values
        - 'Iteration_Y': Best Y values for each iteration
        - 'Best_EI': Expected Improvement (EI) values
    """
    plt.figure(figsize=FIGURE_SIZE)

    # Best Y overall
    plt.subplot(1, 3, 1)
    plt.plot(log_df['Iteration'], log_df['Best_Y'], label="Best Y", linewidth=LINE_WIDTH)
    plt.xlabel("Iteration", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Best Y", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.title("Best Y over Iterations", fontsize=TITLE_FONT_SIZE)
    plt.grid()
    plt.yscale("log")
    plt.legend()
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    
    plt.subplot(1, 3, 2)
    plt.plot(log_df['Iteration'], log_df['Iteration_Y'], label="Iteration Y", linewidth=LINE_WIDTH)
    plt.xlabel("Iteration", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Iteration Y", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.title("Best Y over Iterations", fontsize=TITLE_FONT_SIZE)
    plt.grid()
    plt.yscale("log")
    plt.legend()
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)

    # EI over iterations
    plt.subplot(1, 3, 3)
    plt.plot(log_df['Iteration'], log_df['Best_EI'], color='orange', label="Expected Improvement (EI)", linewidth=LINE_WIDTH)
    plt.xlabel("Iteration", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("EI", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.title("Expected Improvement over Iterations", fontsize=TITLE_FONT_SIZE)
    plt.grid()
    plt.legend()
    plt.yscale('log')
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)

    plt.tight_layout()
    if save:
        plt.savefig(file_name, dpi=600, bbox_inches="tight")
        print(f"Plot saved as {file_name}")
    plt.show()
    
def plot_parameters_over_iterations(log_df, save=False, file_name='parameters_plot'):
    """
    Plots optimization logs for Sig_f, Length, and Beta over iterations.
    
    Parameters:
    - log_df (pd.DataFrame): DataFrame containing the following columns:
        - 'Iteration': Iteration numbers
        - 'Sig_f': Values of Sig_f parameter
        - 'Length': Values of Length parameter
        - 'Beta': Values of Beta parameter
    - save (bool): Whether to save the plot as a file (default: False)
    - file_name (str): File name to save the plot (if save=True)
    """
    plt.figure(figsize=FIGURE_SIZE)

    # Sig_f over iterations
    plt.subplot(1, 3, 1)
    plt.plot(log_df['Iteration'], log_df['Sig_f'], label="Sig_f", linewidth=LINE_WIDTH)
    plt.xlabel("Iteration", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Sig_f", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.title("Sig_f over Iterations", fontsize=TITLE_FONT_SIZE)
    plt.grid()
    plt.legend()
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)

    # Length over iterations
    plt.subplot(1, 3, 2)
    plt.plot(log_df['Iteration'], log_df['Length'], label="Length", color='green', linewidth=LINE_WIDTH)
    plt.xlabel("Iteration", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Length", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.title("Length over Iterations", fontsize=TITLE_FONT_SIZE)
    plt.grid()
    plt.legend()
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)

    plt.subplot(1, 3, 3)
    plt.plot(log_df['Iteration'], log_df['Beta'], label="Beta", color='blue', linewidth=LINE_WIDTH)
    plt.xlabel("Iteration", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Beta", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.title("Beta over Iterations", fontsize=TITLE_FONT_SIZE)
    plt.grid()
    plt.legend()
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    
    plt.tight_layout()
    if save:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        print(f"Plot saved as {file_name}")
    plt.show()

    
def save_log_with_explanation(log_df,name, seed, dimension, n_iterations, n_initial_points, hyperparams):
    """
    Saves a log DataFrame to a CSV file with an explanation at the top.
    
    Parameters:
    - log_df (pd.DataFrame): The DataFrame to save.
    - seed (float): The used seed in the bayesian optimization function
    - dimension (float): The number of dimensions
    - name (str): A name or identifier for the file.
    - n_iterations (float): Number of iterations of the bayesian optimization fucntion.
    - n_initial_points (float): Number of initial points starting the BOF.
    - hyperparameters (dict): Dictionarry with hyperparametters of the BOF
    """
   
    explanation = (
        f"This CSV file contains a {dimension}D problem.\n"
        f"File name: log_df_{dimension}_{name}.csv\n"
        f"Seed: {seed} \n"
        f"Iterations: {n_iterations}\n"
        f"Initial points: {n_initial_points}\n\n"
    )

    hyperparams_text = "Hyperparameters:\n" + "\n".join([f"{key}: {value}" for key, value in hyperparams.items()]) + "\n\n"

    file_name = f'log_df_{dimension}_{name}.csv'

    with open(file_name, 'w') as f:
        f.write(explanation)
        f.write(hyperparams_text)
        log_df.to_csv(f, index=False)

    print(f"File saved as {file_name}")
    