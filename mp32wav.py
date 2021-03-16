import os
import glob
import os
# import threading
from tqdm import tqdm
from pydub import AudioSegment


def mp3_wav(s_path, d_path):
    files = []

    for f in os.listdir(s_path):
        if not f.startswith('.') and f.endswith('.mp3'):
            files.append(f)
    for i in tqdm(range(0, len(files))):  # tqmd模块百分比模块
        sound = AudioSegment.from_mp3(s_path+"/"+files[i])
        # sound.export(d_path+"/"+files[i], format='wav')
        sound.export(d_path+'/'+files[i].split('.')[0]+'.wav', format='wav')
    return 0

def processing_thch30(file_sign):
    destin_path = "/home/project/zhrtvc/data/samples/zhthchs30/{}".format(file_sign)
    if not os.path.exists(destin_path):
        os.makedirs(destin_path)
    mp3_wav(destin_path, destin_path)
    for infile in glob.glob(os.path.join(destin_path, '*.mp3')):
        os.remove(infile)
    print('done {} file'.format(file_sign))


folder_name_list=["zhspeechocean","zhmagicdata","zhprimewords","zhaishell","zhaidatatang"]
def processing_zhstcmds():
    for folder_name in folder_name_list:
        filePath = "/home/project/zhrtvc/data/samples/{}/".format(folder_name)
        for i,j,k in os.walk(filePath):
            file_name = i.split("/")[-1]
            destin_path = filePath+"{}".format(file_name)
            if not os.path.exists(destin_path):
                os.makedirs(destin_path)
            mp3_wav(destin_path, destin_path)
            for infile in glob.glob(os.path.join(destin_path, '*.mp3')):
                os.remove(infile)
            print('done {} file'.format(file_name))
        print('done {} folder'.format(folder_name))



if __name__ == "__main__":
    pass

    # # zhthchs30
    # processing_thch30("C12")
    # processing_thch30("C13")
    # processing_thch30("C14")
    # processing_thch30("C17")
    # processing_thch30("C18")
    # processing_thch30("C19")
    # processing_thch30("C20")
    # processing_thch30("C21")
    # processing_thch30("C22")
    # processing_thch30("C23")
    # processing_thch30("C31")
    # processing_thch30("C32")
    # processing_thch30("D4")
    # processing_thch30("D6")
    # processing_thch30("D7")
    # processing_thch30("D8")
    # processing_thch30("D11")
    # processing_thch30("D12")
    # processing_thch30("D13")
    # processing_thch30("D21")
    # processing_thch30("D31")
    # processing_thch30("D32")


    # zhstcmds P00406I
    # processing_zhstcmds()
  
    # 测试
    # import os
    # filePath = '/home/project/zhrtvc/data/samples/zhstcmds/'
    # for i,j,k in os.walk(filePath):
    # #     # print(i,j,k)
    #     print(i) # 文件名
    #     file_name = i.split("/")[-1]
    #     print(file_name)
    #     print()
        # print(j)
        # print(k)
    # filePath = '路径名\'
    # a = os.listdir(filePath)
    # print("start")
    # print(a)
    # # source_file_path = "/home/project/zhrtvc/data/samples/zhthchs30/A11" # 资源本身
    # destin_path = "/home/project/zhrtvc/data/samples/zhthchs30/C8" # 输出的文件 23 32-36
    # # destin_path = "/home/project/zhrtvc/data/zhvoice/zhthchs30/A12_wav" # 输出的文件
    # if not os.path.exists(destin_path):
    #     os.makedirs(destin_path)
    # # mp3_wav(source_file_path, destin_path)
    # mp3_wav(destin_path, destin_path)
    # for infile in glob.glob(os.path.join(destin_path, '*.mp3')):
    #     os.remove(infile)
    print('done!')
