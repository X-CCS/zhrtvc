#coding=utf-8
import os
import pandas as pd
import glob

# csv_list = glob.glob("/home/project/zhrtvc/data/samples_new/csv_file/*.csv")
# # print(u"共发现%s个CSV文件" % len(csv_list))
# # print(u"正在处理…")

# for i in csv_list: #循环读取同文件夹下的csv文件
#     fr = open(i,"rb").read()
# with open("/home/project/zhrtvc/data/samples_new/csv_file/result1.csv","ab") as f: #将结果保存为result.csv
#     f.write(fr)
# print(u"合并完毕！")
def hebing():
    csv_list = glob.glob('/home/project/zhrtvc/data/samples_new/csv_file/*.csv')
    print(u'共发现%s个CSV文件'% len(csv_list))
    print(u'正在处理............')
    for i in csv_list:
        fr = open(i,'r').read()
        with open('/home/project/zhrtvc/data/samples_new/csv_file/metadata.csv','a') as f:
            f.write(fr)
    print(u'合并完毕！')

def quchong(file):
    df = pd.read_csv(file,header=0)
    datalist = df.drop_duplicates()
    datalist.to_csv(file)
    print(u'去重完毕！')

if __name__ == '__main__':
    hebing()
    # quchong("/home/project/zhrtvc/data/samples_new/csv_file/metadata.csv")