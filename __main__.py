from PIL import Image
import numpy as np


def linear_classify():
    global np_E
    try:
        E = Image.open("Imgs/E.png")
        E = E.convert('L')
        I = Image.open("Imgs/I.png")
        I = I.convert('L')
        Key1 = Image.open("Imgs/key1.png")
        Key1 = Key1.convert('L')
        Key2 = Image.open("Imgs/key2.png")
        Key2 = Key2.convert('L')

        np_E = np.asarray(E).copy()
        np_I = np.asarray(I).copy()
        np_Key1 = np.asarray(Key1).copy()
        np_Key2 = np.asarray(Key2).copy()
    except FileNotFoundError:
        print("找不到此檔案請重新輸入!")
        exit(0)

    Epoch = 1
    apha = 0.00001
    eptho = 0

    rows = np_E.shape[0]
    cols = np_E.shape[1]

    MaxLimit = 20

    wEpoch = np.array([0.0, 0.0, 0.0])

    pre_wEpoch = 0
    while Epoch == 1 or (Epoch < MaxLimit and abs(wEpoch.flatten().dot(wEpoch.flatten()) - pre_wEpoch) > eptho):
        pre_wEpoch = wEpoch.flatten().dot(wEpoch.flatten())
        for k in range(rows * cols):
            # 得到 𝑎(𝑘)
            a_k = np.array([wEpoch[0] * np_Key1.flatten()[k], wEpoch[1] * np_Key2.flatten()[k], wEpoch[2] * np_I.flatten()[k]])

            # 得到 x(k)
            x_k = np.array([np_Key1.flatten()[k], np_Key2.flatten()[k], np_I.flatten()[k]])

            # 𝑒(𝑘) = 𝐸(𝑘) − 𝑎(𝑘)
            e_k = np_E.flatten()[k] - (a_k[0] + a_k[1] + a_k[2])

            # 𝒘𝑬𝒑𝒐𝒄𝒉(𝑘 + 1) = 𝐰𝑬𝒑𝒐𝒄𝒉(𝑘) + 𝛼 ⋅ 𝑒(𝑘) ⋅ 𝐱(𝑘)
            wEpoch[0] = wEpoch[0] + (apha * e_k * x_k[0])
            wEpoch[1] = wEpoch[1] + (apha * e_k * x_k[1])
            wEpoch[2] = wEpoch[2] + (apha * e_k * x_k[2])
        Epoch += 1

        print(Epoch, pre_wEpoch, wEpoch)

    Encrypt = Image.open("Imgs/Eprime.png")
    Encrypt = Encrypt.convert('L')
    np_Encrypt = np.asarray(Encrypt).copy()

    L = np.zeros(rows * cols)
    for i in range(rows * cols):
        L[i] = (np_Encrypt.flatten()[i] - wEpoch[0] * np_Key1.flatten()[i] - wEpoch[1] * np_Key2.flatten()[i]) / wEpoch[2]

    L = L.reshape((rows, cols))
    img_l = Image.fromarray(L.astype(int))
    img_l.show()
    img_l.save("out_E.png")


if __name__ == "__main__":
    linear_classify()
