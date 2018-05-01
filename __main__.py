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
    # print(rows, cols)

    MaxLimit = 20

    # Randomize 𝐰𝑬𝒑𝒐𝒄𝒉(0) = [𝑤1𝐸𝑝𝑜𝑐ℎ(0), 𝑤2𝐸𝑝𝑜𝑐ℎ(0), 𝑤3𝐸𝑝𝑜𝑐ℎ(0)]

    wEpoch = np.array([0.0, 0.0, 0.0])
    # wEpoch[0][0] = np.random.rand()
    # wEpoch[1][0] = np.random.rand()
    # wEpoch[2][0] = np.random.rand()

    pre_wEpoch = 0
    while Epoch == 1 or (Epoch < MaxLimit and abs(wEpoch.flatten().dot(wEpoch.flatten()) - pre_wEpoch) > 0):
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
        # np.savetxt('outputW1-new', wEpoch[0])
        # np.savetxt('outputW2-new', wEpoch[1])
        # np.savetxt('outputW3-new', wEpoch[2])
        print(Epoch, pre_wEpoch, wEpoch)

    Encrypt = Image.open("Imgs/Eprime.png")
    Encrypt = Encrypt.convert('L')
    np_Encrypt = np.asarray(Encrypt).copy()

    # w1 = np.loadtxt('outputW1-new', dtype=float)
    # w2 = np.loadtxt('outputW2-new', dtype=float)
    # w3 = np.loadtxt('outputW3-new', dtype=float)

    L = (np_Encrypt - wEpoch[0] * np_Key1 - wEpoch[1] * np_Key2) / wEpoch[2]

    # np.savetxt('DecryptPic', L.astype(int))
    img_l = Image.fromarray(L.astype(int), mode='L')
    img_l.save("out_E.jpg")


def violence_method():
    Key1 = Image.open("Imgs/key1.png")
    Key1 = Key1.convert('L')
    Key2 = Image.open("Imgs/key2.png")
    Key2 = Key2.convert('L')

    np_Key1 = np.asarray(Key1, dtype=float).copy()
    np_Key2 = np.asarray(Key2, dtype=float).copy()

    Encrypt = Image.open("Imgs/Eprime.png")
    Encrypt = Encrypt.convert('L')
    np_Encrypt = np.asarray(Encrypt, dtype=float).copy()

    for i in range(1, 99):
        for j in range(80, 99):
            for k in range(1, 40):
                L = (np_Encrypt - (float(i) * 0.01) * np_Key1 - (float(j) * 0.01) * np_Key2) / (float(k) * 0.01)
                img_l = Image.fromarray(L.astype(int), mode='L')
                img_l.save("output/" + str(i) + str(j) + str(k) + '.jpg')


if __name__ == "__main__":
    linear_classify()
