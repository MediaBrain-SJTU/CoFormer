import numpy as np
from tqdm import tqdm

Path_to_PTdict_list = ''
Save_path = ''

data = np.load(Path_to_PTdict_list,allow_pickle=True)
all_num = len(data)
sensor_num = data[0]['arr'].shape[1]
time_length = data[0]['time'].shape[0]
print(all_num,sensor_num,time_length)

result_data = np.ones((all_num,sensor_num,time_length))* 1e10
result_time = np.ones((all_num,sensor_num,time_length))* -1e10
encoder_mask = np.zeros((all_num,sensor_num))
static_data = np.zeros((all_num,9))

for i in tqdm(range(all_num)):
    curr_data = data[i]
    curr_series = curr_data['arr']
    curr_time = curr_data['time']
    sensor_time = curr_time[:,0]
    all_record = np.where(curr_series!=0)
    static_data[i,:] = curr_data['extended_static']
    # print(all_record)
    # print(curr_series)
    # if len(all_record[0])<=0:
    #     print(i)
    #     count += 1
        # print(curr_time)
    for j in range(sensor_num):
        sensor_series = curr_series[:,j]
        records = np.where(sensor_series>0)
        if len(records[0]) > 0:
            # print(records[0].shape[0])
            result_data[i,j,:records[0].shape[0]] = sensor_series[records]
            result_time[i,j,:records[0].shape[0]] = sensor_time[records]
            encoder_mask[i,j] = records[0].shape[0]
print(np.max(np.max(encoder_mask)))
result_data = result_data[:,:,:189]
result_time = result_time[:,:,:189]
np.save(Save_path+'array.npy',result_data)
np.save(Save_path+'time.npy',result_time)
np.save(Save_path+'mask.npy',encoder_mask)
np.save(Save_path+'static.npy',static_data)
