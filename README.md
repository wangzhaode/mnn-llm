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