from sklearn.decomposition import PCA

from utils.utils import create_clusters, human_only, detect_floor, dataframe_to_pointcloud
from utils.data_processing import list_data
from utils.utils_visualization import plot_figure3D, plot_classified_human, plot_figure3D_floor, plot_point_clouds
# from model.model import NeuralNetwork, predict
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import open3d as o3d
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from comet_ml.integration.pytorch import watch
import comet_ml
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def main():
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Start main")
    plot = False
    if plot:
        for i in range(8):
            plot_figure3D(list_data[i])

    # Create Label Dataset
    clusters1, points = create_clusters(list_data[0], eps=0.1)
    data1 = human_only(clusters1, points, list_data[0])

    clusters2, points = create_clusters(list_data[1], eps=0.1)
    data2 = human_only(clusters2, points, list_data[1])

    clusters3, points = create_clusters(list_data[2], eps=0.1)
    data3 = human_only(clusters3, points, list_data[2])

    clusters4, points = create_clusters(list_data[3], eps=0.2)
    data4 = human_only(clusters4, points, list_data[3], largest_cluster_label=50)

    clusters5, points = create_clusters(list_data[4], eps=0.1)
    data5 = human_only(clusters5, points, list_data[4], largest_cluster_label=108)

    clusters6, points = create_clusters(list_data[5], eps=0.2)
    data6 = human_only(clusters6, points, list_data[5], largest_cluster_label=2)

    # initialize comet
    # comet_ml.init(project_name="Comet Try 1")
    print("Dataset end")

# #################### RANSAC & Downsampling ##########################
#     data_before = list_data[2]
#     pcd_before = dataframe_to_pointcloud(data_before)
#     # Apply voxel grid down-sampling
#     voxel_size = 0.05  # Adjust voxel size as needed
#     downsampled_pcd = pcd_before.voxel_down_sample(voxel_size)
#
#     # Convert downsampled point cloud back to DataFrame
#     downsampled_points = np.asarray(downsampled_pcd.points)
#     data_after = pd.DataFrame(downsampled_points, columns=['Points_X', 'Points_Y', 'Points_Z'])
#
#     # Visualize the original, downsampled, and RANSAC-filtered point clouds
#     o3d.visualization.draw_geometries([pcd_before], window_name="Original Point Cloud")
#     o3d.visualization.draw_geometries([downsampled_pcd], window_name="Downsampled Point Cloud")
#
#     # Apply RANSAC for plane segmentation
#     plane_model, inliers = downsampled_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
#
#     # Extract inliers and outliers
#     inlier_cloud = downsampled_pcd.select_by_index(inliers)
#     outlier_cloud = downsampled_pcd.select_by_index(inliers, invert=True)
#     # Convert outlier point cloud back to DataFrame
#     outlier_points = np.asarray(outlier_cloud.points)
#     inlier_points = np.asarray(inlier_cloud.points)
#     # outlier_points
#     data_after_ransac = pd.DataFrame(inlier_points, columns=['Points_X', 'Points_Y', 'Points_Z'])
#
#     plot_point_clouds(data_before, data_after, data_after_ransac)
#
#     #
#     # # Convert outlier point cloud back to DataFrame
#     # outlier_points = np.asarray(outlier_cloud.points)
#     # data_after_ransac = pd.DataFrame(outlier_points, columns=['Points_X', 'Points_Y', 'Points_Z'])
#     # o3d.visualization.draw_geometries([outlier_cloud], window_name="RANSAC-filtered Point Cloud")
#
# #############################################################

#################### Prepare Data For NN ####################
    # Combine datasets
    combined_data = pd.concat([data1, data2, data4, data5, data6], ignore_index=True)

    # Encode labels
    label_encoder = LabelEncoder()
    combined_data['Label'] = label_encoder.fit_transform(combined_data['Label'])

    # Split data into features and labels
    X = combined_data[['Points_X', 'Points_Y', 'Points_Z', 'Intensity']]
    y = combined_data['Label']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
###########################################################

# #################### AFTER FINE TUNING ####################
    print("load model weights")

    # Hyperparameter tuning of NN results [jupyter notebook the process]
    best_hyperparameters = {'lr': 0.01, 'num_layers': 2, 'hidden_size': 128}

    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, num_layers, hidden_size):
            super(NeuralNetwork, self).__init__()
            self.input_size = input_size
            self.num_layers = num_layers
            self.hidden_size = hidden_size

            layers = []
            layers.append(nn.Linear(self.input_size, self.hidden_size))
            layers.append(nn.ReLU())
            for _ in range(self.num_layers - 1):
                layers.append(nn.Linear(self.hidden_size, self.hidden_size))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(self.hidden_size, 1))
            layers.append(nn.Sigmoid())
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    def predict(new_data, model, scaler):
        # Preprocess new data
        new_data_scaled = scaler.transform(new_data[['Points_X', 'Points_Y', 'Points_Z', 'Intensity']])

        # Convert to tensor
        new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

        # Make predictions
        model.eval()
        with torch.no_grad():
            predictions = model(new_data_tensor).squeeze().numpy()

        predicted_labels = ['Human' if pred >= 0.5 else 'No Human' for pred in predictions]

        # Add predicted labels to new_data
        new_data['Predicted_Label'] = predicted_labels
        return new_data

    # Train the Best Model with the Best Hyperparameters
    best_model = NeuralNetwork(X_train.shape[1], best_hyperparameters['num_layers'],
                               best_hyperparameters['hidden_size'])
# ###########################################################

# #################### TRAINING PROCESS #####################
#     # Define Loss and Optimizer
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(best_model.parameters(), lr=best_hyperparameters['lr'])
#
#     train_losses = []
#     num_epochs = 7
#     # Train the Best Model
#     for epoch in range(num_epochs):
#         best_model.train()
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = best_model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         avg_loss = running_loss / len(train_loader)
#         train_losses.append(avg_loss)
#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
#
#     # Plot training loss
#     plt.figure()
#     plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
#     plt.title('Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.show()
# ###########################################################

# #################### SAVE/LOAD WEIGHTS OF MODEL #####################
#
#     # # Save models weights
    PATH = "C:\\Users\\Giota.x\\PycharmProjects\\LiDAR_Detection\\weights\\cp.pt"
#     # #
#     # # Print model's state_dict
#     # print("Model's state_dict:")
#     # for param_tensor in best_model.state_dict():
#     #     print(param_tensor, "\t", best_model.state_dict()[param_tensor].size())
#     # #
#     # torch.save(best_model.state_dict(), PATH)
    best_model.load_state_dict(torch.load(PATH))
#
# ###########################################################

# #################### EVAL MODEL #####################
    print("eval model in main")

    # Evaluate the Best Model on Test Set
    best_model.eval()
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = best_model(inputs)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f"Test Accuracy with Best Hyperparameters: {test_accuracy:.4f}")

    # Predict
    pred_data3 = predict(data3,best_model, scaler)
    plot_classified_human(pred_data3,True)

    data3f = detect_floor(data3)
    plot_figure3D_floor(data3f)


# ###########################################################

############## BBOX ######################
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import plotly.graph_objects as go
    from numpy.linalg import eig, inv
    import numpy.linalg as LA

    # # Step 1: Filter human points
    # human_points = pred_data3[pred_data3['Predicted_Label'] == 'No Human'][['Points_X', 'Points_Y', 'Points_Z']].values
    # data = np.vstack(human_points.T)  # Transform data to np.vstack([x, y, z]) format
    #
    # # Step 1: Center the data
    # mean = np.mean(data, axis=1).reshape(3, 1)
    # centered_data = data - mean
    #
    # # Step 2: Compute covariance matrix
    # cov_matrix = np.cov(centered_data)
    #
    # # Step 3: Perform eigen decomposition
    # eigenvalues, eigenvectors = eig(cov_matrix)
    #
    # # Step 4: Order eigenvalues and eigenvectors
    # order = np.argsort(eigenvalues)[::-1]
    # eigenvalues = eigenvalues[order]
    # eigenvectors = eigenvectors[:, order]
    #
    # # Step 5: Rotate data to align with principal axes
    # rotated_data = inv(eigenvectors).dot(centered_data)
    #
    # # Step 6: Find min and max coordinates along each axis in the rotated space
    # min_coords = np.min(rotated_data, axis=1)
    # max_coords = np.max(rotated_data, axis=1)
    #
    # # Step 7: Create the bounding box corners in the rotated space
    # corners = np.array([[min_coords[0], min_coords[1], min_coords[2]],
    #                     [min_coords[0], min_coords[1], max_coords[2]],
    #                     [min_coords[0], max_coords[1], min_coords[2]],
    #                     [min_coords[0], max_coords[1], max_coords[2]],
    #                     [max_coords[0], min_coords[1], min_coords[2]],
    #                     [max_coords[0], min_coords[1], max_coords[2]],
    #                     [max_coords[0], max_coords[1], min_coords[2]],
    #                     [max_coords[0], max_coords[1], max_coords[2]]])
    #
    # # Step 8: Rotate the corners back to the original space
    # aligned_corners = eigenvectors.dot(corners.T).T + mean.T
    #
    # # Create traces for the plot
    # data_trace = go.Scatter3d(
    #     x=data[0, :],
    #     y=data[1, :],
    #     z=data[2, :],
    #     mode='markers',
    #     marker=dict(size=2, color='blue'),
    #     name='Original Data'
    # )
    #
    # lines = []
    # for i in range(8):
    #     for j in range(i + 1, 8):
    #         if np.sum(np.abs(corners[i] - corners[j])) == max_coords[0] - min_coords[0] or \
    #                 np.sum(np.abs(corners[i] - corners[j])) == max_coords[1] - min_coords[1] or \
    #                 np.sum(np.abs(corners[i] - corners[j])) == max_coords[2] - min_coords[2]:
    #             line = go.Scatter3d(
    #                 x=[aligned_corners[i][0], aligned_corners[j][0]],
    #                 y=[aligned_corners[i][1], aligned_corners[j][1]],
    #                 z=[aligned_corners[i][2], aligned_corners[j][2]],
    #                 mode='lines',
    #                 line=dict(color='red', width=2),
    #                 showlegend=False
    #             )
    #             lines.append(line)
    #
    # # Combine data and lines for the plot
    # fig = go.Figure(data=[data_trace] + lines)
    #
    # fig.update_layout(
    #     scene=dict(
    #         xaxis_title='X',
    #         yaxis_title='Y',
    #         zaxis_title='Z'
    #     ),
    #     title='3D Oriented Bounding Box'
    # )
    #
    # # Show the interactive plot
    # fig.show()




    # means = np.mean(human_points, axis=0)
    # centered_data = human_points - means
    # cov_matrix = np.cov(centered_data.T)
    # evals, evecs = LA.eig(cov_matrix)
    #
    # # orthogonal eigenvectors
    # angles = [np.rad2deg(np.arccos(np.dot(evecs[:,i], evecs[:, (i+1)%3]))) for i in range(3)]
    # print("Angles between eigenvectors:", angles)
    #
    #
    # # Step 3: Visualize original and centered data
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(human_points[:,0], human_points[:,1], human_points[:,2], label="Original Data")
    # ax.scatter(centered_data[:,0], centered_data[:,1], centered_data[:,2], label="Centered Data", color='orange')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    #
    # # Cartesian basis
    # ax.plot([0, 1],  [0, 0], [0, 0], color='b', linewidth=4)
    # ax.plot([0, 0],  [0, 1], [0, 0], color='b', linewidth=4)
    # ax.plot([0, 0],  [0, 0], [0, 1], color='b', linewidth=4)
    #
    # # Eigen basis
    # colors = ['r', 'g', 'k']
    # for i in range(3):
    #     ax.plot([0, evecs[0, i]], [0, evecs[1, i]], [0, evecs[2, i]], color=colors[i], linewidth=4)
    #
    # plt.show()
    #
    # # Step 4: Compute the aligned coordinates
    # aligned_coords = np.matmul(evecs.T, centered_data.T).T
    #
    # # Visualize aligned data
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(aligned_coords[:, 0], aligned_coords[:, 1], aligned_coords[:, 2], color='g', label="Aligned Data")
    # ax.scatter(centered_data[:, 0], centered_data[:, 1], centered_data[:, 2], color='orange', label="Centered Data")
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    # plt.show()
    #
    # # Step 5: Compute the bounding box dimensions in the aligned space
    # xmin, xmax = np.min(aligned_coords[:, 0]), np.max(aligned_coords[:, 0])
    # ymin, ymax = np.min(aligned_coords[:, 1]), np.max(aligned_coords[:, 1])
    # zmin, zmax = np.min(aligned_coords[:, 2]), np.max(aligned_coords[:, 2])
    #
    # # Draw the bounding box in the aligned space
    # def draw_3d_rectangle(ax, x1, y1, z1, x2, y2, z2):
    #     # Edges parallel to x-axis
    #     ax.plot([x1, x2], [y1, y1], [z1, z1], color='b')
    #     ax.plot([x1, x2], [y2, y2], [z1, z1], color='b')
    #     ax.plot([x1, x2], [y1, y1], [z2, z2], color='b')
    #     ax.plot([x1, x2], [y2, y2], [z2, z2], color='b')
    #
    #     # Edges parallel to y-axis
    #     ax.plot([x1, x1], [y1, y2], [z1, z1], color='b')
    #     ax.plot([x2, x2], [y1, y2], [z1, z1], color='b')
    #     ax.plot([x1, x1], [y1, y2], [z2, z2], color='b')
    #     ax.plot([x2, x2], [y1, y2], [z2, z2], color='b')
    #
    #     # Edges parallel to z-axis
    #     ax.plot([x1, x1], [y1, y1], [z1, z2], color='b')
    #     ax.plot([x2, x2], [y1, y1], [z1, z2], color='b')
    #     ax.plot([x1, x1], [y2, y2], [z1, z2], color='b')
    #     ax.plot([x2, x2], [y2, y2], [z1, z2], color='b')
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(aligned_coords[:, 0], aligned_coords[:, 1], aligned_coords[:, 2], color='g', label="Aligned Data")
    # draw_3d_rectangle(ax, xmin, ymin, zmin, xmax, ymax, zmax)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    # plt.show()
    #
    # # Step 6: Undo the rotation and translation
    # rect_coords = lambda x1, y1, z1, x2, y2, z2: np.array([[x1, x1, x2, x2, x1, x1, x2, x2],
    #                                                        [y1, y2, y2, y1, y1, y2, y2, y1],
    #                                                        [z1, z1, z1, z1, z2, z2, z2, z2]])
    #
    # realigned_coords = np.matmul(evecs, aligned_coords.T).T + means
    # rrc = np.matmul(evecs, rect_coords(xmin, ymin, zmin, xmax, ymax, zmax)).T + means
    #
    # # Plot the final bounding box in the original space
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(realigned_coords[:, 0], realigned_coords[:, 1], realigned_coords[:, 2], label="Realigned Data")
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    #
    # # Draw bounding box
    # for i in range(4):
    #     ax.plot(rrc[[i, (i + 1) % 4], 0], rrc[[i, (i + 1) % 4], 1], rrc[[i, (i + 1) % 4], 2], color='b')
    #     ax.plot(rrc[[i + 4, (i + 1) % 4 + 4], 0], rrc[[i + 4, (i + 1) % 4 + 4], 1], rrc[[i + 4, (i + 1) % 4 + 4], 2],
    #             color='b')
    #     ax.plot(rrc[[i, i + 4], 0], rrc[[i, i + 4], 1], rrc[[i, i + 4], 2], color='b')
    #
    # plt.show()


if __name__ == "__main__":
    main()






