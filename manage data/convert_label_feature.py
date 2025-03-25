import pandas as pd
import os

def process_csv_file(file_path):
    # อ่านไฟล์ CSV
    df = pd.read_csv(file_path)
    
    # 1. เปลี่ยนค่า 'both hands' เป็น 1 ในคอลัม Label
    df['Label'] = df['Label'].replace('both hands', 1)
    
    # 2. เพิ่มคอลัม feature ก่อนคอลัม Label
    label_index = df.columns.get_loc('Label')
    
    # แทรกคอลัม feature ก่อน Label โดยใส่ค่า 1
    df.insert(label_index, 'feature', 1)
    
    # บันทึกไฟล์
    df.to_csv(file_path, index=False)
    print(f'Processed: {file_path}')

def process_all_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                process_csv_file(file_path)

# ใช้งานฟังก์ชัน
root_directory = './Both_hand'
process_all_files(root_directory)

# import pandas as pd
# import os

# def process_csv_file(file_path, output_root_dir):
#     # อ่านไฟล์ CSV
#     df = pd.read_csv(file_path)
    
#     # กำหนดค่า feature
#     def assign_feature(label):
#         if label == 'both hand':
#             return 1
#         elif label == 'left hand':
#             return 2
#         elif label == 'right hand':
#             return 3
#         else:  # non-request
#             return 0
    
#     # เพิ่มคอลัมน์ feature ก่อน Label
#     df.insert(df.columns.get_loc('Label'), 'feature', df['Label'].apply(assign_feature))
    
#     # แปลง Label เป็นตัวเลข
#     label_mapping = {
#         'both hand': 1,
#         'left hand': 1, 
#         'right hand': 1,
#         'non-request': 0
#     }
#     df['Label'] = df['Label'].map(label_mapping)
    
#     # สร้างพาธใหม่สำหรับบันทึกไฟล์
#     relative_path = os.path.relpath(file_path, './Type_abnormal_data')
#     new_file_path = os.path.join(output_root_dir, relative_path)
    
#     # สร้างไดเร็กทอรีถ้ายังไม่มี
#     os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    
#     # บันทึกไฟล์
#     df.to_csv(new_file_path, index=False)
#     print(f'Processed: {file_path} -> {new_file_path}')

# def process_all_files(input_root_dir, output_root_dir):
#     for dirpath, dirnames, filenames in os.walk(input_root_dir):
#         for filename in filenames:
#             if filename.endswith('.csv'):
#                 file_path = os.path.join(dirpath, filename)
#                 process_csv_file(file_path, output_root_dir)

# # ใช้งานฟังก์ชัน
# input_directory = './Type_abnormal_data'
# output_directory = './convert_Type_abnormal_data'
# process_all_files(input_directory, output_directory)