# FAQ
## 版本兼容

1. NPU内核驱动和Runtime版本兼容

Runtime向后兼容NPU内核驱动

2. RKNN-Toolkit导出模型和Runtime版本兼容

|rknn模型版本|runtime版本|
|-----|----|
|1.2.0|>=1.2.0 and <=1.5.0|
|1.3.0|>=1.3.0 and <=1.5.0|
|1.4.0|>=1.4.0 and <=1.5.0|
|1.5.0|1.5.0|
|1.5.2|>=1.5.2|


## C API使用常见问题
- **rknn_outputs_release是否会释放rknn_output数组**？

rknn_outputs_release与rknn_outputs_get配合调用，它只释放rknn_output数组里的buf。类似的情况还有rknn_destroy_mem。

- rknn_create_mem如何创建合适的大小？
对于输入而言，一般原则是：如果是量化RKNN模型，rknn_create_mem使用rknn_tensor_attr的size_with_stride分配内存；非量化模型 rknn_create_mem使用用户填充的数据类型的字节数*n_elems分配内存。

对于输出而言，rknn_create_mem使用用户填充的数据类型的字节数*n_elems分配内存。

- **输入数据如何填充**？


如果使用通用API,对于四维形状输入，fmt=NHWC, 即数据填充顺序为[batch, height, width, channel]。非四维输入形状，fmt=UNDEFINED，按照模型的原始形状填充数据。
如果使用零拷贝API,对于四维形状输入，fmt=NHWC/NC1HWC2。fmt=NC1HWC2时如何填充数据请参考《RKNN Runtime零拷贝调用》章节。非四维输入形状，fmt=UNDEFINED，按照模型的原始形状填充数据。

- **pass_through如何使用**？


输入数据格式由rknn_query查询RKNN_NATIVE_INPUT_ATTR命令获取。如果是4维形状：对于1,3,4通道，layout要求使用NHWC，其他通道要求使用NC1HWC2；如果是非4维形状，建议指定layout=UNDEFINED，pass_through mode下，量化模型通常指定输入Tensor的dtype=INT8，非量化模型通常指定输入Tensor的dtype=FLOAT16。


- **出现“failed to submit”错误如何处理**？

如果错误出现在第一层卷积并且使用零拷贝接口，可能原因是输入tensor内存分配不够导致，使用tensor属性中的size_with_stride分配内存。

如果错误出现中间的NPU层，可能原因是模型配置出错，在错误日志中提供的了最新的SDK网盘地址，建议升级最新工具链或者在转换RKNN模型时将该层指定到CPU上运行。
