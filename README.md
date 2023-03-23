# ChatGLM-MNN

## Usage
### 1. Compile MNN library
```bash
git clone https://github.com/alibaba/MNN.git
cd MNN
mkdir build
cmake ..
make -j8
cp -r include /path/to/ChatGLM-MNN/
cp libMNN.so /path/to/ChatGLM-MNN/libs
```
### 2. Download Models
Download models from `github release` to `/path/to/ChatGLM-MNN/resource/models`

### 2. Build and Run
```bash
mkdir build
cd build
cmake ..
make -j8
./chat
```

## Example

Q: 用Python帮我写一段快速排序代码

A: 当然可以!以下是使用Python实现快速排序算法的示例代码:

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        left = []
        right = []
        for i in range(1, len(arr)):
            if arr[i] < pivot:
                left.append(arr[i])
            else:
                right.append(arr[i])
        return quick_sort(left) + [pivot] + quick_sort(right)
```

这个实现中,我们首先检查传入的数组长度是否小于等于1。如果是,我们直接返回该数组,因为这种情况下快速排序是不稳定的。否则,我们选择数组的第一个元素作为基准点(pivot),并将数组分为两个部分:左边的部分的所有元素都小于基准点,右边的部分的所有元素都大于基准点。然后我们递归地对左边和右边的部分执行快速排序,并将结果拼接起来。最后,我们递归地对左右两个部分执行相同的操作,并返回结果。

你可以像这样调用这个函数:

```python
arr = [3, 6, 1, 9, 2, 5, 8, 4]
sorted_arr = quick_sort(arr)
print(sorted_arr)
```

这将输出 `[1, 2, 3, 4, 5, 6, 7, 8, 9]`,表示快速排序已成功地对数组进行了排序。

---

Q: 内存泄露如何定位

A: 内存泄漏(Memory Leak)是指程序在运行过程中,占用的内存空间不断增加,最终导致程序无法继续运行,或者运行效率降低。

以下是一些常见的内存泄漏检查方法:

1. 使用内存分析工具:例如Eclipse Memory Analyzer、VisualVM等,这些工具可以帮助检测程序中未释放的内存空间,以及内存分配的异常。

2. 检查变量的生命周期:如果一个变量在它的生命周期内没有被释放,那么它的内存空间就会被一直占用,最终导致内存泄漏。可以通过查看变量的生命周期信息来检查是否存在这个问题。

3. 检查代码中的异常:如果程序在执行过程中出现了异常,而没有释放所占用的内存空间,也会导致内存泄漏。可以通过调试程序,查找并释放这些内存空间。

4. 优化代码:如果程序中的代码存在内存泄漏问题,那么可能是因为程序的性能问题或者内存占用过高导致的。可以通过优化代码,减少内存分配和释放的次数,来解决这个问题。

5. 定期清理无用的数据:如果程序中存在大量的无用数据,也会导致内存泄漏。可以通过定期清理这些数据,释放内存空间。

内存泄漏问题的解决需要从多个方面入手,通过不断地调试和优化程序,来找到内存泄漏的根本原因,并有效地解决问题。