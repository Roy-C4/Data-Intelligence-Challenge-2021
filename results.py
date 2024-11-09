import os
import subprocess
import pandas as pd
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def run_experiments(grid_size, moving_obstacle: bool):
    '''
    Runs the train.py file the same number of times as we did in the experiments.
    WARNING: Takes very long.
    @grid_size: which grid to run, either 14 or 16
    @moving_obstacle: whether to run with or without moving obstacle.
    '''
    # If we want the 14x14 (small) grid
    if grid_size == 14:
        if moving_obstacle == False:

            # Run the train.py file 6 times with the specified arguments.
            for i in range(6):

                # Specify the command line arguments
                arguments = [
                    'grid_configs/small.grd',
                    '--out', 'results/',
                    '--iter', '25000',
                    '--no_gui'
                ]

                # Run the script with the specified arguments
                subprocess.call(['python', 'train.py'] + arguments)

        else:
            # Check if we are in results folder
            if os.getcwd().endswith('results'):
                os.chdir('../../')

            for i in range(6):
                arguments = [
                    'grid_configs/small.grd',
                    '--out', 'results/',
                    '--iter', '25000',
                    '--cat',
                    '--no_gui'
                ]

                # Run the script with the specified arguments
                subprocess.call(['python', 'train.py'] + arguments)


    # else if we are on the 16x16 (medium) grid
    elif grid_size == 16:
        if moving_obstacle == False:
            for i in range(6):
                arguments = [
                    'grid_configs/medium.grd',
                    '--out', 'results/',
                    '--iter', '40000',
                    '--no_gui'
                ]

                # Run the script with the specified arguments
                subprocess.call(['python', 'train.py'] + arguments)

        else:
            if os.getcwd().endswith('results'):
                os.chdir('../../')

            for i in range(6):
                arguments = [
                    'grid_configs/medium.grd',
                    '--out', 'results/',
                    '--iter', '40000',
                    '--cat',
                    '--no_gui'
                ]

                # Run the script with the specified arguments
                subprocess.call(['python', 'train.py'] + arguments)

def get_result_files():
    '''
    Gets the result .txt files from the results folder.
    '''
    # Get the current directory
    current_directory = os.getcwd()

    # Change to the results folder
    new_directory = os.path.join(current_directory, 'results')
    os.chdir(new_directory)
    updated_directory = os.getcwd()

    # Return the result files
    result_files = []
    for file in os.listdir(updated_directory):
            if file.endswith('.txt'):
                result_files.append(file)

    # Get the last 30 files
    result_files = result_files[-30:]

    return result_files

def parse_result_files(result_files):
    '''
    Parses the result files and extracts the data from them into a dataframe using regular expressions.
    @result_files: the result files returned by get_result_files()
    '''
    data_list=[]
    for result_file in result_files:
        # Open and read the text file
        with open(result_file, 'r') as file:
            content = file.read()

        data = {}

        # Extract the values using regular expressions
        agent_match = re.search(r'agent: (.+)', content)
        training_time_match = re.search(r'total training time: (.+)', content)
        dirt_cleaned_match = re.search(r'total_dirt_cleaned: (.+)', content)
        steps_match = re.search(r'total_steps: (.+)', content)
        agent_moves_match = re.search(r'total_agent_moves: (.+)', content)
        agents_at_charger_match = re.search(r'total_agents_at_charger: (.+)', content)
        failed_moves_match = re.search(r'total_failed_moves: (.+)', content)
        dirt_remaining_match = re.search(r'dirt_remaining: (.+)', content)

        # Store the extracted values in the dictionary
        data['agent'] = agent_match.group(1)
        data['total_training_time'] = float(training_time_match.group(1))
        data['total_dirt_cleaned'] = int(dirt_cleaned_match.group(1))
        data['total_steps'] = int(steps_match.group(1))
        data['total_agent_moves'] = int(agent_moves_match.group(1))
        data['total_agents_at_charger'] = int(agents_at_charger_match.group(1))
        data['total_failed_moves'] = int(failed_moves_match.group(1))
        data['dirt_remaining'] = int(dirt_remaining_match.group(1))

        # Append the data
        data_list.append(data)

    # Make the whole dataframe
    df = pd.DataFrame(data_list)

    # Extract the agent name from the text
    df['agent'] = df['agent'].apply(lambda x: x.split('.')[-2] if x.startswith('<') else x)

    return df

def calculate_score(df):
    '''
    Calculates the score metric mentioned in the report.
    @df: The dataframe outputted by parse_result_files()
    '''

    # Get the mean values per agent
    df2 = df.groupby(['agent']).mean()

    # Determine the minimum and maximum training time
    min_time = df['total_training_time'].min()
    max_time = df['total_training_time'].max()

    # Determine the minimum and maximum path length
    min_steps = 28
    max_steps = df['total_agent_moves'].max()

    # Add score column to dataframe using the formula given in the report
    df2 = df2.assign(score=0.33*((max_time - df2['total_training_time']) / (max_time - min_time)) +
                        0.33*df2['total_agents_at_charger'] +
                        0.33*((max_steps - df2['total_agent_moves']) / (max_steps - min_steps)))

    return df2

def create_plot(df, df_moving_object):
    '''
    Creates a plot showing the scores per agent with a moving obstacle, versus without moving obstacle.
    @df: Dataframe containing the data from the agents trained without the moving obstacle
    @df_moving_object: Dataframe containing the data from the agents trained with the moving obstacle
    '''
    
    # Extract the data
    agents = df.index
    score = df['score']
    score_MO = df_moving_object['score']

    # Create the bar traces
    trace = go.Bar(
        x=agents,
        y=score,
        name='Without moving obstacle',
        marker=dict(color='purple'),
        text=score,
    )

    trace_MO = go.Bar(
        x=agents,
        y=score_MO,
        name='With moving obstacle',
        marker=dict(color='orange'),
        text=score_MO,
    )

    # Create the layout
    layout = go.Layout(
        title='Agent Scores with and without moving obstacle',
        xaxis=dict(title='Agents'),
        yaxis=dict(title='Scores'),
        barmode='group',
    )

    # Create the figure and add the traces
    fig = go.Figure(data=[trace, trace_MO], layout=layout)

    return fig

def main():
    '''
    Main loop of the program. Currently only set to produce the 14x14 (small) grid to save time. 
    If you wish to see the other grid (16x16), please change the number in run_experiments(14) to run_experiments(16).
    '''
    # Run the experiments for training the agents WITHOUT moving obstacle
    run_experiments(14, moving_obstacle=False)
    result_files = get_result_files()
    df = parse_result_files(result_files)
    df_score = calculate_score(df)

    # Run the experiments for training the agents WITH moving obstacle
    run_experiments(14, moving_obstacle=True)
    result_files_MO = get_result_files()
    df_MO = parse_result_files(result_files_MO)
    df_score_MO = calculate_score(df_MO)

    # Create the plot
    fig = create_plot(df_score, df_score_MO)
    fig.show()

if __name__ == "__main__":
    main()


  