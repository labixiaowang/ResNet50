## 使用ResNet50进行垃圾分类
* resnet50.py:ResNet50模型。
* Customdata.py:对图像进行预处理，让图像transform成规定的shape,并且返回对应的标签。
* path_and_target.py:从指定文件夹中获取图片`.jpg`的路径,将有效的写入`valid.txt`中，无效的文件写入`invalid.txt`中。
* train.py:训练代码
* test.py:测试代码


## 使用Resnet和 Squeeze-and-Excitation(SE)结合使用
* Resnet50_SE.py:Resnet50与SE的结合
