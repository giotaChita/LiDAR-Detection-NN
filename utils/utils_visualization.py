import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, inv


def plot_classified_human(data, bounding_box):

    # TODO : fix the model so it actually detect humans as human - now it is reversed and later correct the input as
    #  well

    # Define colors for different classes
    color_map = {'Human': 'red', 'No Human': 'blue'}

    # Create a list of colors based on the predicted labels
    colors = [color_map[label] for label in data['Predicted_Label']]

    # Create a 3D scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=data['Points_X'],
        y=data['Points_Y'],
        z=data['Points_Z'],
        mode='markers',
        marker=dict(
            size=3,
            color=colors,  # Color points based on predicted label
            opacity=0.8
        )
    ))

    # Set plot layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Points_X'),
            yaxis=dict(title='Points_Y'),
            zaxis=dict(title='Points_Z')
        ),
        title='Predicted Labels of LiDAR XYZ Points'
    )
    human_points = data[data['Predicted_Label'] == 'No Human'].copy()
    print(human_points.head())

    if bounding_box:
        # Filter human points
        human_points = data[data['Predicted_Label'] == 'No Human'][['Points_X', 'Points_Y', 'Points_Z']].values
        if human_points.size == 0:
            print("No human points detected.")
            return
        # Remove outliers using DBSCAN
        dbscan = DBSCAN(eps=0.08, min_samples=5)  # Adjust eps and min_samples as needed
        labels = dbscan.fit_predict(human_points)
        core_samples_mask = (labels != -1)
        filtered_human_points = human_points[core_samples_mask]

        if filtered_human_points.size == 0:
            print("All human points are outliers.")
            return

        data = np.vstack(filtered_human_points.T)

        # Center the data
        mean = np.mean(data, axis=1).reshape(3, 1)
        centered_data = data - mean

        # Compute covariance matrix
        cov_matrix = np.cov(centered_data)

        # Perform eigen decomposition
        eigenvalues, eigenvectors = eig(cov_matrix)

        # Order eigenvalues and eigenvectors
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Rotate data to align with principal axes
        rotated_data = inv(eigenvectors).dot(centered_data)

        # Find min and max coordinates along each axis in the rotated space
        min_coords = np.min(rotated_data, axis=1)
        max_coords = np.max(rotated_data, axis=1)

        # Create the bounding box corners in the rotated space
        corners = np.array([[min_coords[0], min_coords[1], min_coords[2]],
                            [min_coords[0], min_coords[1], max_coords[2]],
                            [min_coords[0], max_coords[1], min_coords[2]],
                            [min_coords[0], max_coords[1], max_coords[2]],
                            [max_coords[0], min_coords[1], min_coords[2]],
                            [max_coords[0], min_coords[1], max_coords[2]],
                            [max_coords[0], max_coords[1], min_coords[2]],
                            [max_coords[0], max_coords[1], max_coords[2]]])

        # Rotate the corners back to the original space
        aligned_corners = eigenvectors.dot(corners.T).T + mean.T

        # Create traces for the bounding box lines
        lines = []
        for i in range(8):
            for j in range(i + 1, 8):
                if np.sum(np.abs(corners[i] - corners[j])) == max_coords[0] - min_coords[0] or \
                        np.sum(np.abs(corners[i] - corners[j])) == max_coords[1] - min_coords[1] or \
                        np.sum(np.abs(corners[i] - corners[j])) == max_coords[2] - min_coords[2]:
                    line = go.Scatter3d(
                        x=[aligned_corners[i][0], aligned_corners[j][0]],
                        y=[aligned_corners[i][1], aligned_corners[j][1]],
                        z=[aligned_corners[i][2], aligned_corners[j][2]],
                        mode='lines',
                        line=dict(color='blue', width=2),
                        showlegend=False
                    )
                    lines.append(line)

        # Add bounding box lines to the plot
        fig.add_traces(lines)

        # Set plot layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Points_X'),
            yaxis=dict(title='Points_Y'),
            zaxis=dict(title='Points_Z')
        ),
        title='Predicted Labels of LiDAR XYZ Points with Bounding Box for Human'
    )

    # Show plot
    fig.show()

def plot_figure3D_floor(data):
    # Create a Plotly figure
    fig = go.Figure(data=[go.Scatter3d(
        x=data['Points_X'],
        y=data['Points_Y'],
        z=data['Points_Z'],
        mode='markers',
        marker=dict(
            size=1,
            color=data['Floor'].map({'No': 'green', 'Yes': 'red'})
        )
    )])

    # Add layout options for interactivity
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='cube'  # Ensures equal aspect ratio
        ),
        title='LiDAR XYZ Points with Floor'
    )

    # Show the plot
    fig.show()

def plot_figure3D(data):
    # Create a Plotly figure
    fig = go.Figure(data=[go.Scatter3d(
        x=data['Points_X'],
        y=data['Points_Y'],
        z=data['Points_Z'],
        mode='markers',
        marker=dict(
            size=1
        )
    )])

    # Add layout options for interactivity
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='cube'  # Ensures equal aspect ratio
        ),
        title='LiDAR XYZ Points'
    )

    # Show the plot
    fig.show()


# Plotting the distributions with standard deviation
def plot_distribution_with_std(data, feature, label, color):
    plt.hist(data[feature], bins=30, alpha=0.5, label=f'{label} {feature}', color=color)
    mean = data[feature].mean()
    std = data[feature].std()
    plt.axvline(mean, color=color, linestyle='dashed', linewidth=1)
    plt.axvline(mean - std, color=color, linestyle='dotted', linewidth=1)
    plt.axvline(mean + std, color=color, linestyle='dotted', linewidth=1)
    plt.legend()

 # Visualization
def plot_point_clouds(data_before, data_after, data_after_ransac):
    fig = go.Figure()

    # Original Data
    fig.add_trace(go.Scatter3d(
        x=data_before['Points_X'], y=data_before['Points_Y'], z=data_before['Points_Z'],
        mode='markers', marker=dict(size=2, color='blue', opacity=0.5),
        name='Original Data'
    ))

    # Downsampled Data
    fig.add_trace(go.Scatter3d(
        x=data_after['Points_X'], y=data_after['Points_Y'], z=data_after['Points_Z'],
        mode='markers', marker=dict(size=2, color='green', opacity=0.5),
        name='Downsampled Data'
    ))

    # RANSAC-filtered Data
    fig.add_trace(go.Scatter3d(
        x=data_after_ransac['Points_X'], y=data_after_ransac['Points_Y'], z=data_after_ransac['Points_Z'],
        mode='markers', marker=dict(size=2, color='red', opacity=0.5),
        name='RANSAC-filtered Data'
    ))

    fig.update_layout(scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
        aspectmode='cube'
    ), title='Point Clouds Visualization')

    fig.show()