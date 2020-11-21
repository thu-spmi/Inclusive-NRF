import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os




def produce_train_data():
    data = np.zeros((1600,2),'float32')
    ind_poi1=np.zeros((8,2))
    ind_poi2=np.zeros((8,2))
    for i in range(8):
        ind_poi1[i]=np.c_[np.cos(i*2*np.pi/8),np.sin(i*2*np.pi/8)]
        ind_poi2[i]=np.c_[np.cos((i*2+1)*np.pi/8),np.sin((i*2+1)*np.pi/8)]
        data[i*50:i*50+50]= np.tile(ind_poi2[i]*3,(50,1))
        data[i * 50 + 400:i * 50 + 450] = np.tile(ind_poi1[i]*2,(50,1))
        data[i*50+800:i*50+850]= np.tile(ind_poi2[i]*5,(50,1))
        data[i * 50 + 1200:i * 50 + 1250] = np.tile(ind_poi1[i]*4,(50,1))

    idx = list(range(1600))
    np.random.shuffle(idx)
    data = np.cast['float32'](data[idx] + 0.1*np.random.normal(size=(1600,2)))
    return data


def get_modes(gen_data,mode_arr=None,thre_dis=0.3*0.3):
    # threshold=3*sigma, to regrad as covering a mode
    mode=np.zeros((32,2))
    for i in range(8):
        mode[i*4]=np.c_[np.cos(i*2*np.pi/8),np.sin(i*2*np.pi/8)]*2
        mode[i*4+1]=np.c_[np.cos((i*2+1)*np.pi/8),np.sin((i*2+1)*np.pi/8)]*3
        mode[i*4+2]=np.c_[np.cos(i*2*np.pi/8),np.sin(i*2*np.pi/8)]*4
        mode[i*4+3]=np.c_[np.cos((i*2+1)*np.pi/8),np.sin((i*2+1)*np.pi/8)]*5
    if mode_arr==None:
        mode_arr=np.zeros(32,'int32')
    mode_number=np.zeros(32,'int32')

    for x in range(len(gen_data)):
        dis=np.zeros(32,'float32')
        for i in range(32):
            dis[i]=(gen_data[x,0]-mode[i,0])**2+(gen_data[x,1]-mode[i,1])**2
        mn=np.argmin(dis)
        if dis[mn]<thre_dis:
            mode_number[mn]+=1
        if mode_arr[mn]==0 and dis[mn]<thre_dis:
            mode_arr[mn]=1
    return mode_arr,mode_number

def plot_ux(func,name):
    if not os.path.exists('image'):
        os.mkdir('image')
    plt.clf()
    plt.cla()
    plt.figure(figsize=(10, 8))
    theta = np.linspace(0, 2 * np.pi, 800)
    x, y = np.cos(theta) * 2, np.sin(theta) * 2
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = np.cos(theta) * 3, np.sin(theta) * 3
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = np.cos(theta) * 4, np.sin(theta) * 4
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = np.cos(theta) * 5, np.sin(theta) * 5
    plt.plot(x, y, color='gray', linewidth=0.5)

    h=np.linspace(-6,6, 800)
    x, y = h ,  np.tan(np.pi/8)*h
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = h ,  np.tan(np.pi/8*2)*h
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = h ,  np.tan(np.pi/8*3)*h
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = h ,  0.*h
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = h ,  -np.tan(np.pi/8)*h
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = h ,  -np.tan(np.pi/8*2)*h
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = h ,  -np.tan(np.pi/8*3)*h
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = 0.*h ,  h
    plt.plot(x, y, color='gray', linewidth=0.5)

    mode=np.zeros((32,2))
    for i in range(8):
        mode[i*4]=np.c_[np.cos(i*2*np.pi/8),np.sin(i*2*np.pi/8)]*2
        mode[i*4+1]=np.c_[np.cos((i*2+1)*np.pi/8),np.sin((i*2+1)*np.pi/8)]*3
        mode[i*4+2]=np.c_[np.cos(i*2*np.pi/8),np.sin(i*2*np.pi/8)]*4
        mode[i*4+3]=np.c_[np.cos((i*2+1)*np.pi/8),np.sin((i*2+1)*np.pi/8)]*5

    plt.plot(mode[:, 0], mode[:, 1], 'o', markerfacecolor='r', markersize=5, markeredgecolor='r',markeredgewidth=2)

    x = np.linspace(-6, 6, 50)
    y = np.linspace(-6, 6, 50)
    X, Y = np.meshgrid(x, y)
    XX = X.reshape(-1)
    YY = Y.reshape(-1)
    D = np.cast['float32'](np.vstack((XX, YY)).T)
    z=func(D)
    Z = z.reshape((50, 50))
    m = plt.contourf(X, Y, Z, 40,cmap=plt.cm.bone)
    plt.colorbar(m)

    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.savefig('image/'+name+'.png', bbox_inches='tight')

def plot_data(gen,name=''):
    if not os.path.exists('image'):
        os.mkdir('image')
    plt.clf()
    plt.cla()
    plt.figure(figsize=(10, 10))
    theta = np.linspace(0, 2 * np.pi, 800)
    x, y = np.cos(theta) * 2, np.sin(theta) * 2
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = np.cos(theta) * 3, np.sin(theta) * 3
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = np.cos(theta) * 4, np.sin(theta) * 4
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = np.cos(theta) * 5, np.sin(theta) * 5
    plt.plot(x, y, color='gray', linewidth=0.5)
    h=np.linspace(-6,6, 800)
    x, y = h ,  np.tan(np.pi/8)*h
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = h ,  np.tan(np.pi/8*2)*h
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = h ,  np.tan(np.pi/8*3)*h
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = h ,  0.*h
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = h ,  -np.tan(np.pi/8)*h
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = h ,  -np.tan(np.pi/8*2)*h
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = h ,  -np.tan(np.pi/8*3)*h
    plt.plot(x, y, color='gray', linewidth=0.5)
    x, y = 0.*h ,  h
    plt.plot(x, y, color='gray', linewidth=0.5)


    plt.scatter(gen[:, 0], gen[:, 1], c='b', marker='.',s=32)


    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.savefig('image/'+name+'.png', bbox_inches='tight')