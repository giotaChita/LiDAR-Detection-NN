import pandas as pd
from .utils import preprocess_data


filepath1 ="C:\\Users\\Giota.x\\Desktop\\LiDAR\\C32_v4.0.0_2024-04-29-13-32_2Frameso.csv"
data1_init = pd.read_csv(filepath1)
data1 = preprocess_data(data1_init)

filepath3 ="C:\\Users\\Giota.x\\Desktop\\LiDAR\\C32_v4.0.0_2024-04-29-13-54_2FrameChristos.csv"
data3_init = pd.read_csv(filepath3)
data3 = preprocess_data(data3_init)

filepath2 ="C:\\Users\\Giota.x\\Desktop\\LiDAR\\C32_v4.0.0_2024-04-29-13-29_2FrameMaria.csv"
data2_init = pd.read_csv(filepath2)
data2 = preprocess_data(data2_init)

filepath4 = "C:\\Users\\Giota.x\\Desktop\\LiDAR\\C32_v4.0.0_2024-04-30-08-27_1FrameDesp.csv"
data4_init = pd.read_csv(filepath4)
data4 = preprocess_data(data4_init)

filepath5 ="C:\\Users\\Giota.x\\Desktop\\LiDAR\\C32_v4.0.0_2024-04-30-09-41_1FrameNata.csv"
data5_init = pd.read_csv(filepath5)
data5 = preprocess_data(data5_init)

filepath6 ="C:\\Users\\Giota.x\\Desktop\\LiDAR\\C32_v4.0.0_2024-05-09-11-41_1Framemar2.csv"
data6_init = pd.read_csv(filepath6)
data6 = preprocess_data(data6_init)

filepath7 ="C:\\Users\\Giota.x\\Desktop\\LiDAR\\lidar_data.csv"
data7_init = pd.read_csv(filepath7)
data7 = preprocess_data(data7_init)

filepath8 ="C:\\Users\\Giota.x\\Desktop\\LiDAR\\C32_v4.0.0_2024-07-01-15-01_1Floor.csv"
data8_init = pd.read_csv(filepath8)
data8 = preprocess_data(data8_init)

list_data = [data1,data2,data3,data4,data5,data6,data7,data8]