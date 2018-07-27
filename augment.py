"""
Prepare training data, randomly merge bottles with



"""
import cv2, numpy as np, os, matplotlib.pyplot as plt, json
import time

def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou1 = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou1

PATH = '/home/alex/xplain'

bottles = [i for i in os.listdir(os.path.join(PATH, 'bottles')) if i.find('.jpg')!=-1]
logos = [i for i in os.listdir(os.path.join(PATH, 'logo')) if i.find('.jpg')!=-1]
bottle_dict = {}

counter = 0

for enum in range(14000):
    flag = -1
    while flag == -1:
        bottle_num = np.random.randint(low=2, high=3, size=1)[0]

        bottle_dict[0] = {'is_logo': True}
        i = 0
        bottle_dict[i]['bottles'] = os.path.join('bottles', bottles[np.random.randint(0, len(bottles), 1)[0]])
        if bottle_dict[i]['is_logo'] == True:
            bottle_dict[i]['logo'] = os.path.join('logo', logos[np.random.randint(0, len(logos), 1)[0]])
        for i in range(1, bottle_num):
            bottle_dict[i] = {'is_logo': True if np.random.rand() > 1.4 else False}#np.random.randint(-10, 4, 1)[0] > 0}
            bottle_dict[i]['bottles'] = os.path.join('bottles', bottles[np.random.randint(0, len(bottles), 1)[0]])
            if bottle_dict[i]['is_logo'] == True:
                bottle_dict[i]['logo'] = os.path.join('logo', logos[np.random.randint(0, len(logos), 1)[0]])

        img = (np.zeros((300, 300, 3)) + np.random.randint(240, 255, 3)).astype(int)
        for i, i0 in enumerate(list(bottle_dict.keys())):
            aux_img = cv2.imread(os.path.join(PATH, bottle_dict[i]['bottles']))
            #aux_img = cv2.cvtColor(aux_img, cv2.COLOR_BGR2RGB)

            new_shape = [int(z * 100. / np.min(aux_img.shape[:2])) for z in aux_img.shape]
            aux_img = cv2.resize(aux_img, (min(new_shape[1], 250), min(new_shape[0], 250)),
                                 interpolation=cv2.INTER_NEAREST)
            aux_img = np.maximum(np.minimum(aux_img - np.random.randint(-30, 30, 3), 255), 0)
            if bottle_dict[i].get('logo', -1) != -1:
                aux_logo = cv2.imread(os.path.join(PATH, bottle_dict[i]['logo']))
                #aux_logo = cv2.cvtColor(aux_logo, cv2.COLOR_BGR2RGB)
                new_shape = [int(z * 80. / float(np.max(aux_logo.shape[:2]))) for z in aux_logo.shape]
                aux_logo = cv2.resize(aux_logo, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_NEAREST)

                if (aux_img.shape[1] - aux_logo.shape[1]) % 2 == 0:

                    try:
                        aux_img[int(aux_img.shape[0] / 2.):int(aux_img.shape[0] / 2.) + int(aux_logo.shape[0]),
                        int((aux_img.shape[1] - aux_logo.shape[1]) / 2.):
                        int((aux_img.shape[1] - aux_logo.shape[1]) / 2.) + aux_logo.shape[1], :] = aux_logo
                    except:
                        break
                    # xc, yc, dx, dy
                    bottle_dict[i0]['logo_coords'] = [int(float(aux_img.shape[0] + aux_logo.shape[0]) / 2.),
                                                      int(aux_img.shape[1]) - 50,
                                                      aux_logo.shape[0], aux_logo.shape[1]]
                else:

                    try:
                        aux_img[int(aux_img.shape[0] / 2.):int(aux_img.shape[0] / 2.) + int(aux_logo.shape[0]),
                        int((aux_img.shape[1] - aux_logo.shape[1]) / 2.):
                        int((aux_img.shape[1] - aux_logo.shape[1]) / 2.) + aux_logo.shape[1], :] = aux_logo
                    except:
                        break
                    bottle_dict[i0]['logo_coords'] = [int(float(aux_img.shape[0] + aux_logo.shape[0]) / 2.),
                                                      int(aux_img.shape[1]) - 1 -50,
                                                      aux_logo.shape[0], aux_logo.shape[1] + 2]
            xrand, yrand = np.random.randint(0, 299 - aux_img.shape[0], 1)[0], \
                           np.random.randint(0, 299 - aux_img.shape[1], 1)[0]
            bottle_dict[i0]['bottle_coords'] = [xrand, yrand, xrand + aux_img.shape[0], yrand + aux_img.shape[1]]

            img[xrand:xrand + aux_img.shape[0], yrand:yrand + aux_img.shape[1], :] = aux_img
            """
            c = bottle_dict[i0]['logo_coords']
            s = img.copy()/255.
            a = (xrand + c[0] - int(c[2]/2), yrand + c[1] - int(c[3]/2))
            b = (xrand + c[0] + int(c[2]/2), yrand + c[1] + int(c[3]/2))
            print(a,b)
            cv2.rectangle(s, (a[1], a[0]), (b[1], b[0]), (255,0,0))
            cv2.imshow('img', s)
            cv2.waitKey()
            1
            """
        if bottle_dict[0].get('bottle_coords', -1) != -1 and bottle_dict[1].get('bottle_coords', -1) != -1:
            if iou(bottle_dict[1]['bottle_coords'], bottle_dict[0]['bottle_coords']) < 0.1:
                flag = 0
            if bottle_dict[i]['is_logo'] == False:
                del bottle_dict[i]
                time.sleep(0.01)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if len(list(bottle_dict.keys())) == 0:
        continue
    try:
        json.dump(bottle_dict, open(os.path.join(PATH, 'markup', str(counter) + '.json'), 'w'))
    except:
        continue
    cv2.imwrite(os.path.join(PATH, 'imgs', str(counter) + '.jpg'), img)
    counter += 1
    #plt.imshow(img)
    #plt.show()
    print(enum, bottle_dict)
    #dict_keys = list(bottle_dict.keys())