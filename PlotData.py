import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import time

# Load the data
inference_data = torch.load('./preprocessed_data/video_inference_data.pth')
outputs_combined = inference_data['outputs_combined']
states_combined = inference_data['states_combined']
combined_labels = inference_data['combined_labels']

# Video Labels
video_labels = ["City", "Forest", "Ocean", "Stock"]
label_colors = {
    'City': '#F46500',
    'Forest': '#20E0BE',
    'Ocean': '#D25EFF',
    'Stock': '#D0D919'
}

def plot_avg_hidden_states(states, labels, titles):
    for state, label, title in zip(states, labels, titles):
        hidden, _ = zip(*state)
        hidden_states = [h[0].detach().numpy() for h in hidden]
        avg_hidden_states = np.mean(hidden_states, axis=1)
        avg_hidden_state_over_time = np.mean(avg_hidden_states, axis=1)

        plt.figure(figsize=(10, 5))
        
        plt.plot(avg_hidden_state_over_time, label='Average Hidden State',
                 color='#F20D0D', linestyle='-')
        plt.xlabel('Frame Number')
        plt.ylabel('Activation')
        plt.title(f"{title} ({label}) - Avg Hidden States")
        plt.legend()
        customize_plot()
        plt.show()


def plot_diff_from_original(states, labels, titles, original_states):
    for state, label, title, original_state in zip(states, labels, titles, original_states):
        hidden, _ = zip(*state)
        hidden_states = [h[0].detach().numpy() for h in hidden]
        avg_hidden_states = np.mean(hidden_states, axis=1)
        avg_hidden_state_over_time = np.mean(avg_hidden_states, axis=1)

        original_hidden, _ = zip(*original_state)
        original_hidden_states = [h[0].detach().numpy() for h in original_hidden]
        original_avg_hidden_states = np.mean(original_hidden_states, axis=0)
        original_avg_hidden_state_over_time = np.mean(original_avg_hidden_states, axis=1)

        diff_hidden_states = avg_hidden_state_over_time - original_avg_hidden_state_over_time

        plt.figure(figsize=(10, 5))
        plt.plot(diff_hidden_states, label='Difference in Avg Hidden State', linestyle='--', color='black')
        plt.xlabel('Frame Number')
        plt.ylabel('Difference in Activation')
        plt.title(f"{title} ({label}) - Diff from Original")
        plt.legend()
        plt.show()

def plot_probabilities(outputs, labels):
    video_labels = ["City", "Forest", "Ocean", "Stock"]
    for output, label in zip(outputs, labels):
        probs = [F.softmax(o, dim=1).detach().numpy().flatten() for o in output]
        plt.figure(figsize=(10, 5))
        brops = []
        num_classes = probs[0].shape[0] 
        probs_separated = list(zip(*probs))
        plot_ready_probs = [list(zip(*probs))[i] for i in range(num_classes)]
        plt.plot(probs_separated[0], label= 'City', color=label_colors['City'])
        plt.plot(probs_separated[1], label= 'Forest', color=label_colors['Forest'])
        plt.plot(probs_separated[2], label= 'Ocean', color=label_colors['Ocean'])
        plt.plot(probs_separated[3], label= 'Stock', color=label_colors['Stock'])
        plt.xlabel('Frame Number')
        plt.ylabel('Probability')
        names = label.split('-')
        ax_num = 4
        if names[0] == names[1]:
            plt.title(f"Network Prediction: {names[0]}")
            filename = names[0]
            type = 'stagnant'
        else:
            plt.title(f"Network Prediction: {names[0]} switched to {names[1]}")
            plt.axvline(x=64, color='white', linestyle='--', label='Switch in Video')
            filename = f'{names[0]}_to_{names[1]}'
            type = 'switch'
            ax_num += 1
        customize_plot()
        ax = plt.gca()
        legend = ax.legend(facecolor='none', edgecolor='white', loc='upper center', 
                           bbox_to_anchor=(0.5, -0.13), ncol=ax_num)
        plt.gcf().subplots_adjust(bottom=0.17)
        plt.setp(legend.get_texts(), color='white')
        plt.savefig(f'./figures/probabilities/{type}/{filename}.png', dpi=400)


def plot_hidden_states(states, labels):
    for state, label in zip(states, labels):
        hidden, cell = zip(*state)
        hidden_states = [h[0].detach().numpy() for h in hidden]
        print(len(hidden))
        print(hidden.size)
        avg_hidden_states = np.mean(hidden_states, axis=0)
        plt.figure(figsize=(10, 5))
        states_separate = list(zip(*hidden_states))
        for i, hidden_state in enumerate(zip(*hidden_states)):
            plt.plot(hidden_state, color=label_colors['Stock'], linewidth=0.9)
        plt.xlabel('Frame Number')
        plt.ylabel('Activation')
        title = f'Hidden State Activations for {label}'
        plt.title(title)
        customize_plot()
        plt.savefig(f'./figures/activations/{label}_stockcolor.png', dpi=400)

def plot_diff_from_original_avg_combined(states, labels, titles, original_states):
    """
    Plot the difference in average hidden states between original and combined videos.
    """
    for original_label, original_state in zip(labels, original_states):
        original_video = original_label.split("-")[0]
        # Get the average hidden states for the original video
        hidden, _ = zip(*original_state)
        original_hidden_states = [h[0].detach().numpy() for h in hidden]
        original_avg_hidden_state_over_time = np.mean(original_hidden_states, axis=1)
        
        # Loop over the combined videos that start with the original video
        combined_labels = [label for label in labels if label.startswith(f"{original_video}-") and label != original_label]
        for combined_label in combined_labels:
            combined_state = [state for label, state in zip(labels, states) if label == combined_label][0]
            hidden, _ = zip(*combined_state)
            combined_hidden_states = [h[0].detach().numpy() for h in hidden]
            combined_avg_hidden_state_over_time = np.mean(combined_hidden_states, axis=1)
            
            # Calculate the difference
            diff_hidden_states = original_avg_hidden_state_over_time - combined_avg_hidden_state_over_time

            # Plot the difference
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(diff_hidden_states)), diff_hidden_states, label='Difference in Avg Hidden State', linestyle='--', color='black')
            plt.xlabel('Frame Number')
            plt.ylabel('Difference in Activation')
            plt.title(f"Difference between {combined_label} and {original_label}")
            plt.legend()
            plt.show()

def plot_end(states, labels, original_states):

    for original_label, original_state in zip(labels, original_states):
        original_video = original_label.split("-")[1]
        # Get the average hidden states for the original video
        hidden, _ = zip(*original_state)
        original_hidden_states = [h[0].detach().numpy() for h in hidden]
        original_avg_hidden_state_over_time = np.mean(original_hidden_states, axis=1)
        
        # Loop over the combined videos that end with the original video
        combined_labels = [label for label in labels if label.endswith(f"-{original_video}") and label != original_label]
        for combined_label in combined_labels:
            combined_state = [state for label, state in zip(labels, states) if label == combined_label][0]
            hidden, _ = zip(*combined_state)
            combined_hidden_states = [h[0].detach().numpy() for h in hidden]
            combined_avg_hidden_state_over_time = np.mean(combined_hidden_states, axis=1)
            
            # Calculate the difference
            diff_hidden_states = original_avg_hidden_state_over_time - combined_avg_hidden_state_over_time

            # Plot the difference
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(diff_hidden_states)), diff_hidden_states, linestyle='-', color='#F20D0D')
            plt.axvline(x=64, color='white', linestyle='--', label='Switch in Video')
            plt.xlabel('Frame Number')
            plt.ylabel('Difference in Activation')
            plt.title(f"{combined_label} || {original_label.split('-')[0]} as Baseline")
            plt.legend()
            customize_plot()
            plt.show()

def plot_return(states, labels):
    # Create a dictionary to map labels to their corresponding states
    label_state_dict = dict(zip(labels, states))
    
    # Find baseline labels
    baseline_labels = [label for label in labels if label.split("-")[0] == label.split("-")[1]]

    for baseline_label in baseline_labels:
        baseline_name = baseline_label.split("-")[0]
        baseline_state = label_state_dict[baseline_label]
        hidden, _ = zip(*baseline_state)
        baseline_hidden_states = [h[0].detach().numpy() for h in hidden]
        baseline_avg_hidden_state_over_time = np.mean(baseline_hidden_states, axis=1)

        # Loop over the combined videos that end with the baseline name
        combined_labels = [label for label in labels if label.endswith(f"-{baseline_name}") and label != baseline_label]
        for combined_label in combined_labels:
            combined_state = label_state_dict[combined_label]
            hidden, _ = zip(*combined_state)
            combined_hidden_states = [h[0].detach().numpy() for h in hidden]
            combined_avg_hidden_state_over_time = np.mean(combined_hidden_states, axis=1)

            # Calculate the difference
            diff_hidden_states = baseline_avg_hidden_state_over_time - combined_avg_hidden_state_over_time

            # Plot the difference
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(diff_hidden_states)), diff_hidden_states, linestyle='-', color='#73888C')
            plt.axvline(x=64, color='white', linestyle='--', label='Switch in Video')
            plt.xlabel('Frame Number')
            plt.ylabel('Difference in Activation')
            labs = combined_label.split('-')
            title = f'{labs[0]} Switched to {labs[1]}'
            plt.title(f"{title}, {baseline_name} as Baseline")
            plt.legend()
            customize_plot()
            ax = plt.gca()
            legend = ax.legend(facecolor='none', edgecolor='white', loc='upper center', 
                            bbox_to_anchor=(0.5, -0.13), ncol=1)
            plt.gcf().subplots_adjust(bottom=0.17)
            plt.setp(legend.get_texts(), color='white')
            plt.show()

def plot_return_sem(states, labels):
    """
    Plot the average absolute value of the difference in hidden states
    with SEM for original and combined videos.
    """
    # Create a dictionary to map labels to their corresponding states
    label_state_dict = dict(zip(labels, states))
    
    # Find baseline labels
    baseline_labels = [label for label in labels if label.split("-")[0] == label.split("-")[1]]

    for baseline_label in baseline_labels:
        baseline_name = baseline_label.split("-")[0]
        baseline_state = label_state_dict[baseline_label]
        hidden, _ = zip(*baseline_state)
        baseline_hidden_states = [h[0].detach().numpy() for h in hidden]
        baseline_avg_hidden_state_over_time = np.mean(baseline_hidden_states, axis=1)

        # Loop over the combined videos that end with the baseline name
        combined_labels = [label for label in labels if label.endswith(f"-{baseline_name}") and label != baseline_label]
        for combined_label in combined_labels:
            combined_state = label_state_dict[combined_label]
            hidden, _ = zip(*combined_state)
            combined_hidden_states = [h[0].detach().numpy() for h in hidden]
            combined_avg_hidden_state_over_time = np.mean(combined_hidden_states, axis=1)

            # Calculate the absolute difference and SEM
            abs_diff_hidden_states = np.abs(baseline_avg_hidden_state_over_time - combined_avg_hidden_state_over_time)
            abs_diff_mean = np.mean(abs_diff_hidden_states, axis=1)
            abs_diff_sem = np.std(abs_diff_hidden_states, axis=1) / np.sqrt(abs_diff_hidden_states.shape[1])

            # plot naming
            labs = combined_label.split('-')
            title = f'{labs[0]} Switched to {labs[1]}'

            # Plot the average absolute difference with SEM
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(abs_diff_mean)), abs_diff_mean, linestyle='-', color=label_colors[labs[0]])
            plt.fill_between(range(len(abs_diff_mean)), abs_diff_mean - abs_diff_sem, 
                             abs_diff_mean + abs_diff_sem, color=label_colors[labs[0]], edgecolor='None', alpha=0.2, label='SEM')
            plt.axvline(x=64, color='#747474', linestyle='--', label='Switch in Video')
            plt.xlabel('Frame Number')
            plt.ylabel('Average Absolute Difference in Activation')
            plt.title(f"{title}, {baseline_name} as Baseline")
            plt.legend()
            customize_plot()
            ax = plt.gca()
            legend = ax.legend(facecolor='none', edgecolor='white', loc='upper center', 
                               bbox_to_anchor=(0.5, -0.13), ncol=2)
            plt.gcf().subplots_adjust(bottom=0.17)
            plt.setp(legend.get_texts(), color='white')
            plt.savefig(f'./figures/sem/{labs[0]}_to_{labs[1]}.png', dpi=400)

def plot_vid_end(states, labels, titles, original_states):
    for original_label, original_state in zip(labels, original_states):
        original_video = original_label.split("-")[0]
        # Get the average hidden states for the original video
        hidden, _ = zip(*original_state)
        original_hidden_states = [h[0].detach().numpy() for h in hidden]
        original_avg_hidden_state_over_time = np.mean(original_hidden_states, axis=1)
        
        # Loop over the combined videos that start with the original video
        combined_labels = [label for label in labels if label.endswith(f"-{original_video}") and label != original_label]
        for combined_label in combined_labels:
            combined_state = [state for label, state in zip(labels, states) if label == combined_label][0]
            hidden, _ = zip(*combined_state)
            combined_hidden_states = [h[0].detach().numpy() for h in hidden]
            combined_avg_hidden_state_over_time = np.mean(combined_hidden_states, axis=1)
            
            # Calculate the difference
            diff_hidden_states = original_avg_hidden_state_over_time - combined_avg_hidden_state_over_time

            # Plot the difference
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(diff_hidden_states)), diff_hidden_states, linestyle='-', color='#F20D0D')
            plt.xlabel('Frame Number')
            plt.ylabel('Difference in Activation')
            plt.title(f"Difference between {combined_label} and {original_label}")
            customize_plot()
            plt.show()

def plot_diff_from_same(states, labels):
    """
    Plot the difference in average hidden states between same and combined videos,
    with the standard error of the mean (SEM) as a shaded area.
    """
    unique_labels = list(set([label.split('-')[0] for label in labels]))

    for original_label in unique_labels:
        same_label = f"{original_label}-{original_label}"
        if same_label not in labels:
            print(f"Warning: {same_label} not found.")
            continue

        same_state = [state for label, state in zip(labels, states) if label == same_label][0]
        hidden, _ = zip(*same_state)
        same_hidden_states = [h[0].detach().numpy() for h in hidden]
        same_avg_hidden_state_over_time = np.mean(same_hidden_states, axis=(1, 2))
        same_sem = np.std(same_hidden_states, axis=(1, 2)) / np.sqrt(len(same_hidden_states))

        combined_labels = [label for label in labels if label.startswith(f"{original_label}-") and label != same_label]
        for combined_label in combined_labels:
            combined_state = [state for label, state in zip(labels, states) if label == combined_label][0]
            hidden, _ = zip(*combined_state)
            combined_hidden_states = [h[0].detach().numpy() for h in hidden]
            combined_avg_hidden_state_over_time = np.mean(combined_hidden_states, axis=(1, 2))
            combined_sem = np.std(combined_hidden_states, axis=(1, 2)) / np.sqrt(len(combined_hidden_states))

            diff_hidden_states = combined_avg_hidden_state_over_time - same_avg_hidden_state_over_time
            diff_sem = np.sqrt(same_sem**2 + combined_sem**2)

            frame_numbers = np.arange(len(diff_hidden_states))
            plt.figure(figsize=(10, 5))
            plt.plot(frame_numbers, diff_hidden_states, color='#891414', label='Difference in Avg Hidden State')
            plt.fill_between(frame_numbers, diff_hidden_states - diff_sem, diff_hidden_states + diff_sem, color='#891414', alpha=0.3)
            plt.xlabel('Frame Number')
            plt.ylabel('Difference in Activation')
            plt.title(f"Difference between {combined_label} and {original_label}")
            plt.axvline(x=64, color='white', linestyle='--')
            plt.legend()
            customize_plot()
            plt.show()



def customize_plot():
    ax = plt.gca()
    ax.set_facecolor("#000424")
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.title.set_color('white')

    legend = ax.legend(facecolor='none', edgecolor='none')
    plt.setp(legend.get_texts(), color='white')
    plt.gcf().patch.set_facecolor("#000424")



# Plot average hidden states
#plot_avg_hidden_states(states_combined, combined_labels, [f'Combined Video {i+1} Avg Hidden States' for i in range(len(states_combined))])

# Plot difference in average activation
#plot_diff_from_original_avg_combined(states_combined, combined_labels, [f'Combined Video {i+1} Diff from Original' for i in range(len(states_combined))], original_states=states_combined[:4])
# plot_vid_end

# CHANGE COLOR but working
#plot_return(states_combined[:-4], combined_labels[:-4])


# plot sem with average return (WORKING)
#plot_return_sem(states_combined[:-4], combined_labels[:-4])

#plot_diff_from_same(states_combined, combined_labels)


# Plot difference from original
#plot_diff_from_original(states_combined, combined_labels, [f'Combined Video {i+1} Diff from Original' for i in range(len(states_combined))], states_combined[:4])

# Plot probabilities (WORKING)
#plot_probabilities(outputs_combined[:-4], combined_labels[:-4])

# Plot hidden states (WORKING)
#plot_hidden_states(states_combined[-4::], combined_labels[-4::])

states, labels = states_combined[-4::], combined_labels[-4::]

for state, label in zip(states, labels):
        hidden, cell = zip(*state)
        hidden_states = [h[0].detach().numpy() for h in hidden]
        print(len(hidden))
        print(hidden[0][0][0])