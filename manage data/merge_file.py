# -----------รวมไฟล์ที่แยกจากที่split-------------------------
# import os
# import pandas as pd

# def merge_csv_files(input_root_dir, output_file):
#     all_dataframes = []

#     for root, dirs, files in os.walk(input_root_dir):
#         for file in files:
#             if file.endswith('.csv'):
#                 file_path = os.path.join(root, file)
#                 df = pd.read_csv(file_path)
#                 all_dataframes.append(df)

#     merged_df = pd.concat(all_dataframes, ignore_index=True)
#     merged_df.to_csv(output_file, index=False)
#     print(f"Merged CSV saved to {output_file}")

# input_root = 'split_abnormal_data'
# output_file = 'all_abnormal_data.csv'
# merge_csv_files(input_root, output_file)



# ---------รวม normal-ab------------------------
import pandas as pd

# อ่านไฟล์ CSV
df1 = pd.read_csv('./all_data2.csv')
df2 = pd.read_csv('./all_abnormal_data.csv')

# รวมข้อมูล
merged_df = pd.concat([df1, df2], ignore_index=True)

# เรียงลำดับตาม feature และ Label
merged_df_sorted = merged_df.sort_values(by=['feature', 'Label'])

# บันทึกไฟล์ที่รวมและเรียงลำดับแล้ว
merged_df_sorted.to_csv('./data_normal_abnormal.csv', index=False)

print('Merge and sort completed successfully!')


# data_normal_abnormal












