import os, random, shutil
def moveFile(fileDir):
        pathDir = os.listdir(fileDir)    #取图片的原始路径
        for id,file in enumerate(pathDir):
            if file.endswith('txt'):
                del pathDir[id]
        filenumber=len(pathDir)
        rate=0.3    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
        picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
        sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
        print (sample)
        for name in sample:
            shutil.move(fileDir+name, tarDir+name)
            shutil.move(fileDir + name.replace('png', 'txt'), tarDir + name.replace('png', 'txt'))
        return 0

def move_img():
    for img in os.listdir(fileDir):
        if img.endswith('png'):
            shutil.move(fileDir+img,train_path+img)
    for img in os.listdir(tarDir):
        if img.endswith('png'):
            shutil.move(tarDir+img,val_path+img)

if __name__ == '__main__':
    fileDir = "D:/workspace/crnet/data/train/"    #源图片文件夹路径
    tarDir = "D:/workspace/crnet/data/val/"    #移动到新的文件夹路径
    #moveFile(fileDir)
    # train_path = 'D:/workspace/crnet/dataset/train_img/'
    # val_path = 'D:/workspace/crnet/dataset/val_img/'
    # move_img()
    train_path = 'D:/workspace/crnet/data/coco/images/train/'
    val_path = 'D:/workspace/crnet/data/coco/images/val/'
