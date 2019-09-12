# win10-python-caffe
WIN10下搭建 python2.7 +caffe神经网络识别
WIN10下搭建 python2.7 +caffe神经网络识别

第一步：caffe 源码编译
1 git:https://github.com/microsoft/caffe.git
2 下载好之后别急着打开解决方案，看看你的VS是不是2013，如果不是，请务必安装2013，因为其他版本VS只会浪费你的时间
3 先复制CommonSettings.props.example，修改复制后的文件名为===>CommonSettings.props
并删除CommonSettings.props.example  别问为什么
4 VS2013打开  caffe\windows\Caffe.sln
5 修改配置：CommonSettings.props  下面两块地方按照我的配置（下面的是仅用CPU的配置，GPU配置还没尝试过，后期会更新，毕竟GPU会更快）
  <PropertyGroup Label="UserMacros">
        <BuildDir>$(SolutionDir)..\Build</BuildDir>
        <!--NOTE: CpuOnlyBuild and UseCuDNN flags can't be set at the same time.-->
        <CpuOnlyBuild>true</CpuOnlyBuild>
        <UseCuDNN>false</UseCuDNN>
        <CudaVersion>7.5</CudaVersion>
        <!-- NOTE: If Python support is enabled, PythonDir (below) needs to be
         set to the root of your Python installation. If your Python installation
         does not contain debug libraries, debug build will not work. -->
        <PythonSupport>true</PythonSupport>
        <!-- NOTE: If Matlab support is enabled, MatlabDir (below) needs to be
         set to the root of your Matlab installation. -->
        <MatlabSupport>false</MatlabSupport>
        <CudaDependencies></CudaDependencies>

        <!-- Set CUDA architecture suitable for your GPU.
         Setting proper architecture is important to mimize your run and compile time. -->
        <CudaArchitecture>compute_35,sm_35;compute_52,sm_52</CudaArchitecture>

        <!-- CuDNN 4 and 5 are supported -->
        <CuDnnPath></CuDnnPath>
        <ScriptsDir>$(SolutionDir)\scripts</ScriptsDir>
    </PropertyGroup>
.....................................
...................................
 <PropertyGroup Condition="'$(PythonSupport)'=='true'">
        <PythonDir>你的python2.7版本安装目录</PythonDir>
        <LibraryPath>$(PythonDir)\libs;$(LibraryPath)</LibraryPath>
        <IncludePath>$(PythonDir)\include;$(IncludePath)</IncludePath>
    </PropertyGroup>
6 以上两处配置修改好之后，修改VS项目的属性，严格按照我说的做！
找到libcaffe这一个项目，右键→“属性”
修改下图红框处



 
然后，接着配置caffe项目的属性
按照下图








7 编译  （一定在release状态下编译）
生成这16个项目，右键Solution'Caffe'选择"Build"(生成)
如果成功了，恭喜你，少踩了无数个坑~！上面几个步骤花了我一天一夜时间~！
如果没成功的话欢迎微信或则公众号留言给我，帮你解决~

二、接下来就是python里面如何引用上面编译好的caffe库

1 cmd定位好你的python27目录 （注意，要提前安装好pip 也要2.7版本的，pip安装好后应该在python2.7\scripts\文件夹下面有个pip文件夹，如果没有，请先安装pip,后面运行程序需要安装其他库就需要2.7的pip，而不是你的其他版本比如3.9的pip，是不行的）
2 先尝试引用一下caffe

这个时候一般不出意外肯定会报错，找不到caffe，因为你没有配置好caffe，
3  配置caffe到python2.7
找到前面你编译好的caffe目录 （注意，要用release模式编译）我的是64位的电脑，所以我的目录在caffe\Build\x64\Release\pycaffe下的caffe文件夹，把caffe文件夹，复制到你python2.7目录的\Lib\site-packages文件夹里面
然后再重复第二步，引用caffe，如果不报错，就说嘛你的caffe没问题
4 跑程序
我这里是抄了一份别人的例子代码，可以标记图片中特征元素并且识别是什么，所以下面的命令是针对我程序起作用，其他程序大家自己去扩展学习，代码如下：
# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
 help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
 help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
 help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
 help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
 "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
 "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
 "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()



# loop over the detections
for i in np.arange(0, detections.shape[2]):
 # extract the confidence (i.e., probability) associated with the
 # prediction
    confidence = detections[0, 0, i, 2]
 # filter out weak detections by ensuring the `confidence` is
 # greater than the minimum confidence
    if confidence > args["confidence"]:
        # extract the index of the class label from the `detections`,
        # then compute the (x, y)-coordinates of the bounding box for
        # the object
        idx = int(detections[0, 0, i, 1])

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

        (startX, startY, endX, endY) = box.astype("int")

        # display the prediction
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)

        print("[INFO] {}".format(label))

        cv2.rectangle(image, (startX, startY), (endX, endY),COLORS[idx], 2)

        y = startY - 15 if startY - 15 > 15 else startY + 15

        cv2.putText(image, label, (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# show the output image
cv2.imshow("Output", image)
cv2.imwrite("output.jpg",image)
cv2.waitKey(0)

运行之前，先需要2个文件 加一张图片
deploy.prototxt
mobilenet_iter_73000.caffemodel
和一张图片

因为我这里不方便下载文件，2个文件如果你需要可以留言给我公众号，我微信加你私发给你
你也可以去我github  https://github.com/yuxiaobopp/win10-python-caffe.git 去下载模型和训练集文件


运行：




