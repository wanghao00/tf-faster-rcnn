为方便使用此项目而创建的文本，记录常用命令
git clone -b r1.0 https://github.com/endernewton/tf-faster-rcnn.git
## 训练数据的软连接创建
ln -s '/home/wanghao/data/faster_np' /home/wanghao/tf-faster-rcnn/data/VOCdevkit2007

## 清空每次模型训练产生的cache文件 
./data/clean_cache.sh 1 1 1
最后一位的1 （可以时任何数字）表示 使用哪一组数据
文件增加执行权限
sudo chmod 777 ./data/clean_cache.sh

bak文件夹
1. 第一次备份output文件夹，里面是第一次合成图像模型训练，660张图像+3300 负样本扣件 avg=5，准确率mAP=99%
评价： 定位很准确，但是类别判断不好
