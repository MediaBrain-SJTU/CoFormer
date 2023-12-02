import numpy as np
from tqdm import tqdm

Path_to_PTdict_list = ''
Save_path = ''
arr_outcome_path = ''
gt = np.load(arr_outcome_path)
np.save(Save_path+'gt.npy',gt)


data = np.load(Path_to_PTdict_list,allow_pickle=True)
all_num = len(data)
sensor_num = data[0]['arr'].shape[1]
time_length = data[0]['time'].shape[0]
print(all_num,sensor_num,time_length)
result_data = np.ones((all_num,sensor_num,time_length))* 1e10
result_time = np.ones((all_num,sensor_num,time_length))* -1e10
encoder_mask = np.zeros((all_num,sensor_num,time_length,time_length))
mask = np.zeros((all_num,sensor_num))
count = 0
for i in tqdm(range(all_num)):
    curr_data = data[i]
    curr_series = curr_data['arr']
    curr_time = curr_data['time']
    sensor_time = curr_time[:,0]
    all_record = np.where(curr_series!=0)
    # print(all_record)
    # print(curr_series)
    if len(all_record[0])<=0:
        print(i)
        count += 1
        # print(curr_time)
    for j in range(sensor_num):
        sensor_series = curr_series[:,j]
        records = np.where(sensor_series>0)
        if len(records) > 0:
            result_data[i,j,:records[0].shape[0]] = sensor_series[records]
            result_time[i,j,:records[0].shape[0]] = sensor_time[records]
            encoder_mask[i,j,:records[0].shape[0],:records[0].shape[0]] = 1


for i in range(all_num):
    for j in range(sensor_num):
        mask[i,j] = np.sum(encoder_mask[i,j,:,0])
print('count',count)
np.save(Save_path+'array.npy',result_data)
np.save(Save_path+'time.npy',result_time)
np.save(Save_path+'mask.npy',mask)
# data = np.load('/home/ubuntu/P19data/processed_data/PT_dict_list_6.npy',allow_pickle=True)
# all_num = len(data)
static_data = np.zeros((all_num,6))
for i in tqdm(range(all_num)):
    static_data[i,:] = data[i]['extended_static']
np.save(Save_path+'static.npy',static_data)
# print(data[0]['extended_static'])
