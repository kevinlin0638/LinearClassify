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
