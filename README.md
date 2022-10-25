# Project
# Target-speed-measurement

本文档记录开发的过程和教程的相关记录，并在每个工作节点结束后进行内容的更新

## 一、前期准备

1. 环境准备
   1. 虚拟环境搭建
      1. git，包括githua项目准备、本地文件夹准备
      2. Yolov5、deepsort、opencv环境需要
      3. CUDA、CUDNN的版本匹配

## 二、新建项目环境

1. 搭建Git&Github项目 ---- Target-speed-measurement

   1. ```python
      # 1.在Github创建一个新项目
      # 2.用Git克隆项目
      git clone git@github.com:SedateAiot/Target-speed-measurement.git
      # 3.用anaconda创建基于Git克隆项目的环境
      conda create -p=E:\Luoy_project_git\aiot\Target-speed-measurement\env python=3.8.1
      # 4.激活虚拟环境
      conda activate  E:\Luoy_project_git\aiot\Target-speed-measurement\env\TSM
      
      # 更新git内容
      git add "xxxx.md"
      git commit -m "introdution"
      git push origin main/master
      
      # 5.修改环境换成清华源
      conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
      conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
      conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
      conda config --set show_channel_urls yes
      ```

2. 确定项目所需环境

   1. 我的电脑当前的驱动版本：517.48，CUDA版本：11.7
   2. 安装pytorch-gup版本`conda install pytorch torchvision torchaudio cudatoolkit=11.3`
   3. 安装paddle`conda install paddlepaddle-gpu==2.3.2 cudatoolkit=11.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge `
   4. 安装yolov5-strongsort项目文件，安装v6.0的版本
      1. 内部文件有个requirement文件，cd到这个路径下，用虚拟环境的pip安装`pip install -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple` 
      2. 由于这个安装文件没有将Yolov5和REID的文件都储存，所以需要再根据链接里的蓝色路径进入，clone这些文件到本地
      3. CMD进去两个模型文件，执行第一步

## 三、项目分解：

### 1.首先，需要搭建一个基于Yolov5的目标检测框架

1. 下载Yolov5和预训练权重文件

2. 训练自己的模型

   1. 准备需要的数据集，用labelimg标记数据集

   2. 对标记好的数据集进行划分--训练集和验证集，将数据放在Yolov5的主级目录下

   3. 修改数据配置文件，

      1. data的yaml文件修改
         1. 找到data-scripts-voc.yaml文件，复制到这个路径下，重命名为xxx.yaml
         2. 打开该文件，将download注释掉，train和val需要i需改自己的数据集路径
         3. nc：主要为需要目标检测的几种分类
         4. names：表示种类的名称
      2. models权重文件修改
         1. 基于需要用到怎样的pt权重文件，就拿对应的yaml文件进行复制，命名为xxx.yaml文件
         2. 打开xxx文件，修改nc值，为上述data的文件中的nc值

3. 开始训练

   1. 打开train.py文件

      1. ```python
         # 模型的主要参数解析
         if __name__ == '__main__':
         """
             opt模型主要参数解析：
             --weights：初始化的权重文件的路径地址
             --cfg：模型yaml文件的路径地址
             --data：数据yaml文件的路径地址
             --hyp：超参数文件路径地址
             --epochs：训练轮次
             --batch-size：喂入批次文件的多少
             --img-size：输入图片尺寸
             --rect:是否采用矩形训练，默认False
             --resume:接着打断训练上次的结果接着训练
             --nosave:不保存模型，默认False
             --notest:不进行test，默认False
             --noautoanchor:不自动调整anchor，默认False
             --evolve:是否进行超参数进化，默认False
             --bucket:谷歌云盘bucket，一般不会用到
             --cache-images:是否提前缓存图片到内存，以加快训练速度，默认False
             --image-weights：使用加权图像选择进行训练
             --device:训练的设备，cpu；0(表示一个gpu设备cuda:0)；0,1,2,3(多个gpu设备)
             --multi-scale:是否进行多尺度训练，默认False
             --single-cls:数据集是否只有一个类别，默认False
             --adam:是否使用adam优化器
             --sync-bn:是否使用跨卡同步BN,在DDP模式使用
             --local_rank：DDP参数，请勿修改
             --workers：最大工作核心数
             --project:训练模型的保存位置
             --name：模型保存的目录名称
             --exist-ok：模型目录是否存在，不存在就创建
         """
             parser = argparse.ArgumentParser()
             parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
             parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
             parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
             parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
             parser.add_argument('--epochs', type=int, default=300)
             parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
             parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
             parser.add_argument('--rect', action='store_true', help='rectangular training')
             parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
             parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
             parser.add_argument('--notest', action='store_true', help='only test final epoch')
             parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
             parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
             parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
             parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
             parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
             parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
             parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
             parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
             parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
             parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
             parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
             parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
             parser.add_argument('--project', default='runs/train', help='save to project/name')
             parser.add_argument('--entity', default=None, help='W&B entity')
             parser.add_argument('--name', default='exp', help='save to project/name')
             parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
             parser.add_argument('--quad', action='store_true', help='quad dataloader')
             parser.add_argument('--linear-lr', action='store_true', help='linear LR')
             parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
             parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
             parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
             parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
             parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
             opt = parser.parse_args()
         ```

      2. 开始调整参数

         1. 首先将weights权重的路径填写到对应的参数里面   

         2. 修改models模型的路径

         3. 修改data数据模型的路径

         4. ```python
                parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='initial weights path')
                parser.add_argument('--cfg', type=str, default='models/yolov5s_hat.yaml', help='model.yaml path')
                parser.add_argument('--data', type=str, default='data/hat.yaml', help='data.yaml path')
            ```

         5. 调整训练轮数`parser.add_argument('--epochs', type=int, default=300)`

         6. 根据电脑性能调整相关参数  `parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')`and`parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')`

      3. 虚拟内存调整

         1. 找到utils->datasets.py文件
         2. 将81行的 num_worker改为0

      4. 运行train.py训练自己的模型

      5. 启用tensorbord

         1. 在pycharm的终端输入  `tensorboard --logdir=runs/train`
         2. 有个url做出来，就可以看训练的可视化情况
         3. 查看训练结果  `tensorboard --logdir=runs`

      6. 结果

         1. 运行结束后，会有一个run文件夹出现，run->train->exp->weights有两个权重文件，一个是最后一次的权重文件，一个是效果最好的权重文件

      7. 测试

         1. 找到主目录下detect.py调整参数

            1. ```python
               f __name__ == '__main__':
               """
               --weights:权重的路径地址
               --source:测试数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头),也可以是rtsp等视频流
               --output:网络预测之后的图片/视频的保存路径
               --img-size:网络输入图片大小
               --conf-thres:置信度阈值
               --iou-thres:做nms的iou阈值
               --device:是用GPU还是CPU做推理
               --view-img:是否展示预测之后的图片/视频，默认False
               --save-txt:是否将预测的框坐标以txt文件形式保存，默认False
               --classes:设置只保留某一部分类别，形如0或者0 2 3
               --agnostic-nms:进行nms是否也去除不同类别之间的框，默认False
               --augment:推理的时候进行多尺度，翻转等操作(TTA)推理
               --update:如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
               --project：推理的结果保存在runs/detect目录下
               --name：结果保存的文件夹名称
               """
                   parser = argparse.ArgumentParser()
                   parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
                   parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
                   parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
                   parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
                   parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
                   parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
                   parser.add_argument('--view-img', action='store_true', help='display results')
                   parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
                   parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
                   parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
                   parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
                   parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
                   parser.add_argument('--augment', action='store_true', help='augmented inference')
                   parser.add_argument('--update', action='store_true', help='update all models')
                   parser.add_argument('--project', default='runs/detect', help='save results to project/name')
                   parser.add_argument('--name', default='exp', help='save results to project/name')
                   parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
                   opt = parser.parse_args()
               ```

               2. 传权重文件    `parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp/weights/best.pt', help='model.pt path(s)')`

            2. 如果要对图片进行推理，可以修改如下参数    ` parser.add_argument('--source', type=str, default='000295.jpg', help='source') `

               2. 推理结会保存到runs->detect->exp->xxx
               3. 跑测试的可以是图片、视频

            3. 摄像头存在Bug

               1. 找到datasets.py文件
               2. 找到279行代码，给两个url加上str就可以了

      8. 后续可以把标签变成中文，及制作数据集

### 2.进行目标追踪算法：StrongSORT_OSNet

1. 首先，准备环境和工具

   1. 进入Yolov5_StrongSORT_OSNet的github库，clone到本地

      1. 库中的两个文件需要点击链接到相对应的github库里进行clone到本地，一个是yolov5，一个是Strong_sort/deep/reid

   2. 解压到相应的文件夹后，激活conda环境`conda activate ...`

      1. 我目前使用的环境是python3.8.5

   3. cd 进入Yolov5_StrongSORT_OSNet

      1. 执行命令 `pip install -r requirements.txt -i https://https://pypi.tuna.tsinghua.edu.cn/simple`
      2. 安装成功后，执行`conda install pytorch torchvision torchaudio cudatoolkit=11.3`
      3. 安装成功后，执行`cd yolov5`，后执行`pip install -r requirements.txt -i https://https://pypi.tuna.tsinghua.edu.cn/simple`
      4. 安装成功后，退出yolov5文件夹，进入strong_sort，执行`pip install -r requirements.txt -i https://https://pypi.tuna.tsinghua.edu.cn/simple`
      5. 安装成功后，进入/deep/reid文件夹，执行`pip install -r requirements.txt -i https://https://pypi.tuna.tsinghua.edu.cn/simple`
      6. 至此，python所需要的环境安装ok了

   4. 接下来就是weights权重文件的安装，本项目需要用到两种权重文件，一种是yolov5的，一种是deepsort的，存储在项目的一级菜单weights中，需要自己创建

      1. 附上链接

         1. ```python
            # yolov5 weights：
            
            # deepsort weights：
            ```

2. 修改源码

   1. 首先，修改track.py文件的超参数部分

      1. 修改yolo-weights的文件应用，第291行，`default=WEIGHTS / 'yolov5m.pt'`
      2. 修改deepsort-weights的文件应用，第292行，`default=WEIGHTS / 'osnet_x0_25_msmt17.pt'`

   2. 其次，修改文件夹机制

      1. 第100行，源代码为`exp_name = name if name is not None else exp_name + "_" + str(strong_sort_weights).split('/')[-1].split('.')[0]`，因为微软的路径用的是\，split不了，所以把上述代码改成`exp_name = name if name is not None else exp_name + "\\" + str(strong_sort_weights).split('\\')[-1].split('.')[0]`，就可以了

      2. 第132行

         1. ```python
            # 源代码为
            for i in range(nr_sources):
                    strongsort_list.append(
                        StrongSORT(
                            strongsort_weights,
                            device,
                            max_dist=cfg.STRONGSORT.MAX_DIST,……。。。。
                            
            ```

         2. 由于查找之后，变量`strongsort_weights`传递的时候，把路径信息传递过来，识别不从来，所以改成如下代码：

         3. ```python
            for i in range(nr_sources):
                    strongsort_list.append(
                        StrongSORT(
                            'E:/strongsort/strong6.0/weights/osnet_x0_25_msmt17.pt',
                            device,
                            max_dist=cfg.STRONGSORT.MAX_DIST,
                       
            ```

3. 这样，就可以在虚拟环境下运行track.py文件了

   1. 当然可以传递一些值，让track.py文件自定义生成内容；

   2. ```python
      # 在cmd中输入相关的参数可以生成很多东西
      --source 1.mp4 # 这个地方用图片、视频、摄像头流等；
      --save-vid # 保存处理后的视频
      --save-txt # 保存相应的信息点
      --yolo-weights "xxxx.pt" # 修改yolo的权重文件
      --strongsort-weights "xxxx.pt" # 修改sort的权重文件
      
      # 。。。。待补充
      ```

4. 需要传递的内容：

   1. 生成的txt文件，内容和含义，能否读到相关的内容

### 3.速度测算：

1. 难点：

   1. 方法一，利用画面中的矩形进行透视变换，来计算位移距离：
      1. 找到画面中世界坐标的矩形（边缘检测工具）
      2. 根据矩形与世界坐标进行透视变换
      3. 得到透视变换矩阵，根据这个矩阵进行其他点位的换算
   2. 方法二，手动圈选画面中的矩形，进行透视变换(省去自动检测边缘的功能测试)
      1. 其余的功能实现与方法一一致
   3. 方法三，利用每一帧检测出来的车辆宽度和真实世界的车辆平均宽度与识别框中心点距离和真实距离的关系，可以得到真是距离

2. 开发选择：

   了解完技术方式和背景之后，先采用了方法三的方式实现视频的测速

3. 开发流程：

   1. 方法三：利用每一帧的检测出来的车辆宽度和真实的车辆宽度比较，得到比值；再计算目标xy的中心点位移距离；

      1. `车辆宽度：真实车辆宽度 = xy中心点位移距离：实际位移距离`，实际位移距离就是我们要求的

      2. 车辆宽度为：car_pixel = w (近似宽度)

      3. 真实车辆宽度：因为car_pixel的实际宽度会偏大，所以对应的世界车辆car_tru需要取大一些，原本是1.6 - 1.8，现在取2.0

      4. xy中心点位移距离（有两个点）：pixel_remove =`sqrt(x2-x1)^2 + (y2-y1)^2`

         1. 这两个点就是相同ID（同个物体）的前后两帧的状态，x_center和y_center的两个时间点的两个坐标

      5. 把上述需要用到的变量，存储在带ID字典的列表里，方便我们获取和更新

      6. 得到公式：`tru_remove = car_tru * pixel_remove / car_pixel`

      7. 得到值，tru_remove，再根据ID出现的时间取间隔，可以得到速度v，并展示到画面里

         1. ```python
            # 相关代码， 基于yolov5+StrongSORT的track.py进行修改
            if int(cls) == 2:
                if id not in carid_list:  # 不存在，就存储id和存储信息
                    carid_list.append(id)
                    # 存储x的中心点、y的中心点、w宽度、出现在第几帧
                    carid_mes[str(id)] = [output[0], output[1], output[2] - output[0], seen]
                    label = f'{id} {names[c]} {conf:.2f}'
                    annotator.box_label(bboxes, label, color=colors(c, True))
                else:  # 存在，就判断位置信息
                    x_center_old, y_center_old = carid_mes.get(str(id))[0], carid_mes.get(str(id))[1]
                    x_center_new, y_center_new = output[0], output[1]
                    pixel_rem = sqrt(
                        (x_center_new - x_center_old) ** 2 + (y_center_new - y_center_old) ** 2)
                    car_pix = carid_mes.get(str(id))[2]
                    tru_rem = round(car_tru * pixel_rem / car_pix, 2)
                    # 计算时间
                    seen_old = carid_mes.get(str(id))[3]
                    t = round((int(seen) - int(seen_old)) * (1 / fps), 2)
                    # print(id, "位移的距离：", tru_rem, "所花费的时间为： ", t)
                    # 计算速度
                    v_pre = round(tru_rem / t, 2) * 3.6
                    print(id, f"当前的速度：{v_pre} Km/h!")
                    # 计算之后更新id信息
                    carid_mes[str(id)] = [x_center_new, y_center_new, car_pix, seen]
            
                    label = f'{id} {names[c]} {v_pre:.2f}km/h {conf:.2f}'
                    annotator.box_label(bboxes, label, color=colors(c, True))
                elif int(cls) == 0:
                    if id not in personid_list:  # 不存在，就存储id和存储信息
                        personid_list.append(id)
                        # 存储x的中心点、y的中心点、w宽度、出现在第几帧
                        personid_mes[str(id)] = [output[0], output[1], output[2] - output[0], seen]
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))
                    else:
                        x_center_old, y_center_old = personid_mes.get(str(id))[0], personid_mes.get(str(id))[1]
                        x_center_new, y_center_new = output[0], output[1]
                        pixel_rem_person = sqrt(
                            (x_center_new - x_center_old) ** 2 + (y_center_new - y_center_old) ** 2)
                        person_pix = personid_mes.get(str(id))[2]
                        tru_rem = round(person_tru * pixel_rem_person / person_pix, 2)
                        # 计算时间
                        seen_old = personid_mes.get(str(id))[3]
                        t = round((int(seen) - int(seen_old)) * (1 / fps), 2)
                        # print(id, "位移的距离：", tru_rem, "所花费的时间为： ", t)
                        # 计算速度
                        v_pre = round(tru_rem / t, 2) * 3.6
                        print(id, f"当前的速度：{v_pre} Km/h!")
                        # 计算之后更新id信息
                        carid_mes[str(id)] = [x_center_new, y_center_new, person_pix, seen]
                        label = f'{id} {names[c]} {v_pre:.2f}km/h {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))
            ```

         2. ![image-20221025151741829](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20221025151741829.png)
