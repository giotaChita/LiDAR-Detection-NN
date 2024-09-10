from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.graph_objs as go
import open3d as o3d

def preprocess_data(data):
    # Drop unnecessary columns
    data = data[['Points_X', 'Points_Y', 'Points_Z','Intensity']]
    data = data.dropna()
    return data


def create_clusters(data, eps):
    # Extract 3D points (X, Y, Z) from the DataFrame
    points = data[['Points_X', 'Points_Y', 'Points_Z']].values

    # Standardize the data
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)

    # Cluster the points
    dbscan = DBSCAN(eps, min_samples=10)
    clusters = dbscan.fit_predict(points_scaled)

    # Create a Plotly figure for the original data with different colors for each cluster
    original_fig = go.Figure()

    for cluster_label in np.unique(clusters):
        if cluster_label == -1:
            # Noise points (outliers) will be black
            cluster_color = 'black'
        else:
            # Other clusters will have random colors
            cluster_color = 'rgb({}, {}, {})'.format(np.random.randint(0, 255), np.random.randint(0, 255),
                                                     np.random.randint(0, 255))

        cluster_indices = np.where(clusters == cluster_label)[0]
        original_fig.add_trace(go.Scatter3d(
            x=points[cluster_indices, 0],
            y=points[cluster_indices, 1],
            z=points[cluster_indices, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=cluster_color
            ),
            name=f'Cluster {cluster_label}'
        ))

    original_fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='cube'  # Ensures equal aspect ratio
        ),
        title='Original LiDAR XYZ Points with Clusters'
    )

    return clusters, points


def human_only(clusters, points, data, largest_cluster_label=None):
    """
    :param clusters:
    :param points:
    :param data:
    :param largest_cluster_label:
    :return:
    """
    unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
    if largest_cluster_label is None:
        largest_cluster_label = unique_clusters[np.argmax(cluster_counts)]
    # Define the cluster label to be labeled as "human"
    human_cluster_label = largest_cluster_label

    # Create a Plotly figure for the clusters with the specified cluster labeled as "human"
    fig = go.Figure()

    labels = []

    for cluster_label in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_label)[0]
        if cluster_label == human_cluster_label:
            cluster_name = 'Human'
            cluster_color = 'red'
        #             labels.append(['Human'])
        else:
            cluster_name = f'Cluster {cluster_label}'
            cluster_color = 'blue'
        #             labels.append(['No Human'])

        fig.add_trace(go.Scatter3d(
            x=points[cluster_indices, 0],
            y=points[cluster_indices, 1],
            z=points[cluster_indices, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=cluster_color
            ),
            name=cluster_name
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='cube'  # Ensures equal aspect ratio
        ),
        title='Clustered LiDAR XYZ Points'
    )

    # Show the plot
    # fig.show()

    # Assign labels to each point based on its cluster assignment
    labels = []
    for point_index, cluster_label in enumerate(clusters):
        if cluster_label == human_cluster_label:
            labels.append('Human')
        else:
            labels.append('No Human')

    # Add the labels as a new column to the DataFrame
    data['Label'] = labels

    return data


def detect_floor(dataset):
    # Find the lowest point of Points_Z
    lowest_point_z = dataset['Points_Z'].min()

    # Define the threshold relative to the lowest point
    threshold = 0.15

    # Add 'Floor' column based on conditions
    dataset['Floor'] = dataset.apply(lambda row: 'No' if row['Points_Z'] > lowest_point_z + threshold else 'Yes',
                                     axis=1)

    # Drop rows where Floor is 'Yes' (optional step based on your requirements)
    filtered_data = dataset[dataset['Floor'] == 'No']

    # Print the shape of the original and filtered DataFrame
    print("Original DataFrame shape:", dataset.shape)
    print("Filtered DataFrame shape:", filtered_data.shape)
    return dataset

# Convert DataFrame to Open3D point cloud
def dataframe_to_pointcloud(df):
    points = df[['Points_X', 'Points_Y', 'Points_Z']].values
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd
