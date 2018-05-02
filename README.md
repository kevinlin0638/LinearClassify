# 機器學習作業 - Linear Classify


## 一、Prepare Before Excute Program
在開始執行前,請先準備好所需的圖片.
  - 1.尚未加密的圖片  
  [![N|Solid](https://i.imgur.com/gHx1ho1.png)](https://github.com/kevinlin0638)
  - 2.圖片-Key1  
  [![N|Solid](https://i.imgur.com/ccohdiD.png)](https://github.com/kevinlin0638)
  - 3.圖片-Key2  
  [![N|Solid](https://i.imgur.com/ZUFa20k.png)](https://github.com/kevinlin0638)
  - 4.圖片-加密後的圖片  
  [![N|Solid](https://i.imgur.com/BB3Oei7.png)](https://github.com/kevinlin0638)
  - 5.圖片-需要被解密的圖片  
  [![N|Solid](https://i.imgur.com/6k7JihR.png)](https://github.com/kevinlin0638)
  
## 二、Load Picture into your program and Import Package

  - Import PIL 套件
  - Import Numpy 套件
  - 使用 PIL 的 open 函式來將圖片讀入
  - 將圖片轉為 Numpy 陣列
  ```python
        #  讀取照片
        E = Image.open("Imgs/E.png")
        
        #  將圖片轉成 numpy陣列
        np_E = np.asarray(E).copy()
  ```

## 三、宣告所需要之常數
  - MaxIterLimit : 限制 While 迴圈最多執行次數(w向量長度最多改變次數)，若 w 遲遲不收斂，則由此離開 While。
  - α : the learning rate : 為一個很小的數，為每次的 w 作微小修正。
  - 𝜖 : 判斷是否收斂的依據，若前後二次的加權向量變化（差向量） > 0，則繼續While迴圈，反之則收斂。 
  - rows & cols : 圖片的大小 300 * 400
  
  ```python
        # 宣告常數
        Epoch = 1
        apha = 0.00001
        eptho = 0
    
        # 圖片大小
        rows = np_E.shape[0]
        cols = np_E.shape[1]
    
        # w 向量長度最多改變次數
        MaxLimit = 20
  ```


## 四、主要程式
[//]:U2FsdGVkX1+S0wU/4R6RatUoEm8KT+cRx05NtMHy2bq49ne9ep9nY985c6WcJAdo
先附上程式碼:

  ```python
            # 初始化 w 陣列
    wEpoch = np.array([0.0, 0.0, 0.0])

    pre_wEpoch = 0
    while Epoch == 1 or (Epoch < MaxLimit and abs(wEpoch.flatten().dot(wEpoch.flatten()) - pre_wEpoch) > eptho):
        pre_wEpoch = wEpoch.flatten().dot(wEpoch.flatten())
        for k in range(rows * cols):
            # 得到 𝑎(𝑘)
            a_k = np.array([wEpoch[0] * np_Key1.flatten()[k], 
            wEpoch[1] * np_Key2.flatten()[k], wEpoch[2] * np_I.flatten()[k]])

            # 得到 x(k)
            x_k = np.array([np_Key1.flatten()[k], np_Key2.flatten()[k], np_I.flatten()[k]])

            # 𝑒(𝑘) = 𝐸(𝑘) − 𝑎(𝑘)
            e_k = np_E.flatten()[k] - (a_k[0] + a_k[1] + a_k[2])

            # 𝒘𝑬𝒑𝒐𝒄𝒉(𝑘 + 1) = 𝐰𝑬𝒑𝒐𝒄𝒉(𝑘) + 𝛼 ⋅ 𝑒(𝑘) ⋅ 𝐱(𝑘)
            wEpoch[0] = wEpoch[0] + (apha * e_k * x_k[0])
            wEpoch[1] = wEpoch[1] + (apha * e_k * x_k[1])
            wEpoch[2] = wEpoch[2] + (apha * e_k * x_k[2])
        Epoch += 1
  ```
  
  ### 詳細說明 :
1. 首先 先初始化 W 陣列，並初始化 pre_wEpoch 為 0 
2. while 中如果 Epoch 等於 1 就進入，其餘條件 如果超過 MaxLimit 則跳出，若在 MaxLimit 限制之內，則判斷是否收斂
3. 一開始先記錄當前的加權向量
4. 進入for迴圈，執行圖片大小(300 * 400)的次數
5. 獲得 a_k = [w(0) * Key1(k), w(1) * Key2(k), w(2) * I(k)]
6. 獲得 x_k = [Key1(k), Key2(k), I(k)]
7. 獲得 e_k = E(k) - (a_k[0] + a_k[1] + a_k[2])
8. 修正 w 的值 wEpoch = wEpoch + (apha * e_k * x_k)
9. 最後跳出 for 迴圈 Epoch += 1

#### #備註 : 所有陣列皆已使用 .flatten() 函數打平處理

## 五、獲得之3W
  - W1 = 0.24914331
  - W2 = 0.6613819
  - W3 = 0.08923953
  
[![N|Solid](https://i.imgur.com/wbCnqkb.jpg)](https://github.com/kevinlin0638)  
  執行結束示意圖  ↑

## 六、解碼圖片
一樣，先附上完整程式碼
```python
    # 讀取需要解密之圖片
    Encrypt = Image.open("Imgs/Eprime.png")
    Encrypt = Encrypt.convert('L')
    np_Encrypt = np.asarray(Encrypt).copy()

    # 運算
    L = np.zeros(rows * cols)
    for i in range(rows * cols):
        L[i] = (np_Encrypt.flatten()[i] - wEpoch[0] * np_Key1.flatten()[i] - wEpoch[1] * np_Key2.flatten()[i]) / wEpoch[2]

    # 存圖
    L = L.reshape((rows, cols))
    img_l = Image.fromarray(np.uint8(L))
    img_l.show()
```  

[![N|Solid](https://i.imgur.com/RyAHWDb.jpg)](https://github.com/kevinlin0638)  
解碼成功示意圖示意圖，帥氣導師 ↑  

## 七、Extra Work
拿到3個 W 後，也可以來做個加解密的 funtion 了，於是做了加密與解密的函式

## 八、心得

> 學習到了如何使用 python 來進行 Linear Classify 的解題
> 在撰寫的過程中遇到不少小問題，例如:型態的轉換、灰階圖片轉換等
> 不知道是不是不太了解 numpy 的陣列運算，好像不可以直接以陣列運算，要以 index 的方式一個一個算
> 最後找到老師大頭滿有成久感的，因為 De 一個灰階的儲存小bug de了兩天.....


