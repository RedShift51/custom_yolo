import mxnet as mx, os, numpy as np, pandas as pd, cv2
import json
import xml.etree.ElementTree as ET

""" Useful functions: iou, loader class, nms, learning rate scheduler class """

def onehot(x, num_cls):
    a = np.zeros((len(x), num_cls))
    a[np.arange(len(x)), x] = 1.
    return a

""" Firstly let's load data without augmentation, since we have big amount of data """
""" Pascal format """
class SimpleLoader():
    def __init__(self, path, imgs_root, mark_root, mode, out_shape, num_anc, shuffle=False):
        self.path = path
        self.imgs_root, self.mark_root = imgs_root, mark_root
        self.mode = mode
        self.shuffle = shuffle
        self.out_shape, self.num_anc = out_shape, num_anc
        if os.path.exists(os.path.join(self.path, 'path_dict.json')):
            self.paths_dict = json.load(open(os.path.join(self.path, 'path_dict.json'), 'r'))
        else:
            self.paths_dict = {}
            imgs = [i[:i.find('.')] for i in os.listdir(os.path.join(self.path, self.imgs_root))]
            self.ext = os.listdir(os.path.join(self.path, self.imgs_root))[0]
            self.ext = self.ext[self.ext.find('.'):]
            marks = [i[:i.find('.')] for i in os.listdir(os.path.join(self.path, self.mark_root))]

            objs = list(set(imgs).intersection(set(marks)))
            for i0,i in enumerate(objs):
                self.paths_dict[str(i0)] = {'img': os.path.join(self.path, self.imgs_root, i + self.ext),
                                            'mar': os.path.join(self.path, self.mark_root, i + '.json')}
            json.dump(self.paths_dict, open(os.path.join(self.path, 'path_dict.json'), 'w'))
            self.paths_dict = json.load(open(os.path.join(self.path, 'path_dict.json')))
        #self.paths_dict = json.load(open(os.path.join(self.path, 'path_dict.json')

        self.objs = list(self.paths_dict.keys())
        #if self.shuffle == True:
        self.objs = np.array(self.objs)
        np.random.shuffle(self.objs)
        self.objs = list(self.objs)

    def __len__(self):
        return len(self.objs)

    def __getitem__(self, idx):

        img = cv2.imread(self.paths_dict[str(self.objs[idx])]['img']) / 255.
        #try:
        #    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #except:
        #    1
        img = img.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, [2, 0, 1])

        markup = []
        #print(self.paths_dict[str(self.objs[idx])]['mar'])
        flag = -1
        while flag == -1:
            #try:
            cur_obj = json.load(open(self.paths_dict[str(self.objs[idx])]['mar'], 'r'))
            flag = 1
            #except:
            #    idx = np.random.randint(low=0,high=len(self.objs),size=1)[0]

        ans_mat = np.zeros(self.out_shape)
        #print(1111111111111111)
        for i0,i in enumerate(list(cur_obj.keys())):
            logo, bottle = cur_obj[i]['logo_coords'], cur_obj[i]['bottle_coords']
            # x,y,dx,dy
            markup.append({'coords': [logo[0] + bottle[0], logo[1] + bottle[1], logo[2], logo[3]]})

            channels, cellx, celly = self.out_shape
            """
            img=np.transpose(img, [1,2,0])
            a = (logo[0]+bottle[0], logo[1]+bottle[1])
            b = (logo[0]+bottle[0]+logo[2],logo[1]+bottle[1]+logo[3])
            cv2.rectangle(img, (a[1], a[0]-50), (b[1]-90, b[0]-50), (255,0,0))
            cv2.imshow('img', img)
            cv2.waitKey()
            """
            cellsize = 300. / float(cellx)
            estm_cellx = markup[-1]['coords'][0] // cellsize
            estm_celly = markup[-1]['coords'][1] // cellsize
            estm_dx = float(markup[-1]['coords'][2]) / float(cellsize)
            estm_dy = float(markup[-1]['coords'][3]) / float(cellsize)

            estm_cellposx = (float(markup[-1]['coords'][0] - cellsize * float(estm_cellx))) / cellsize
            estm_cellposy = (float(markup[-1]['coords'][1] - cellsize * float(estm_celly))) / cellsize

            """ Here we will choose anchor for the current object """
            """
            if estm_dx / estm_dy < 0.84:
                anc = 0
            else:
                anc = 1
            """
            if estm_dx / estm_dy < 0.58:
                anc = 0
            elif estm_dx / estm_dy < 0.9:
                anc = 1
            else:
                anc = 2


            #try:
            ans_mat[:self.num_anc, int(estm_cellx), int(estm_celly)] = onehot(np.array([anc]), self.num_anc)[0, :]
            #print(onehot(np.array([anc]), self.num_anc)[0, :])
            #except:
            #    1
            ans_mat[self.num_anc + anc * 4 : self.num_anc + (anc + 1) * 4, int(estm_cellx), int(estm_celly)] = \
                np.array([estm_cellposx, estm_cellposy, estm_dx, estm_dy]) #-50/(300./19.)

        #print(np.mean(ans_mat.ravel()), np.std(ans_mat.ravel()))
        return img.astype(np.float32), ans_mat.astype(np.float32)


        """
        img = cv2.imread(self.paths_dict[str(self.objs[idx])]['img'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = (cv2.resize(img, (300, 300)) / 255.).astype(np.float32)

        xml = self.paths_dict[str(self.objs[idx])]['xml']
        tree = ET.parse(xml)
        objs = tree.findall('object')
        boxes, gt_classes = np.zeros((len(objs), 4)), np.zeros((len(objs), len(list(self.classes.keys()))))
        for ix,obj in enumerate(objs):
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text) / img.shape[0] * 300
            y1 = float(bbox.find('ymin').text) / img.shape[1] * 300
            x2 = float(bbox.find('xmax').text) / img.shape[0] * 300
            y2 = float(bbox.find('ymax').text) / img.shape[1] * 300
            cls = self.classes[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix, cls] = 1.

            #cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0))
        #cv2.imshow('img', img)
        #cv2.waitKey()

        img = (cv2.resize(img, (300, 300)) / 255.).astype(np.float32)
        img = np.transpose(img, [2, 0, 1])

        #return img, gt_classes, boxes

        ans_mat = np.zeros((self.cells, self.cells, self.num_anc * (len(list(self.classes.keys())) + 5) ))
        step = 300. / self.cells
        for i in range(len(boxes[:, 0])):
            x_center = ((boxes[i, 2] + boxes[i, 0]) / 2.) % step
            y_center = ((boxes[i, 3] + boxes[i, 1]) / 2.) % step
            dx = float(boxes[i, 2] - boxes[i, 0]) / float(step)
            dy = float(boxes[i, 3] - boxes[i, 1]) / float(step)
            cellx = ((boxes[i, 2] + boxes[i, 0]) / 2.) // step
            celly = ((boxes[i, 2] + boxes[i, 0]) / 2.) // step
            if dx / dy < 0.85:
                anc = 0
            elif dy / dx < 0.85:
                anc = 1
            else:
                anc = 2
            # totally self.classes = 0, 3rd dimension = self.num_anc *
            ans_mat[cellx, celly, :self.num_anc] = onehot(np.array([anc]), 3)
            ans_mat[cellx, celly, self.num_anc + (len(list(self.classes.keys())) + 4)*anc:
                                self.num_anc + (len(list(self.classes.keys())) + 4)*(anc + 1)] = np.array([x_center, y_center, dx, dy])

        return img, ans_mat
        """



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

class PolyScheduler(mx.lr_scheduler.LRScheduler):
    def __init__(self, base_lr, lr_power, total_steps):
        super(PolyScheduler, self).__init__(base_lr=base_lr)
        self.lr_power = lr_power
        self.total_steps = total_steps

    def __call__(self, num_update):
        lr = self.base_lr * ((1 - float(num_update)/self.total_steps) ** self.lr_power)
        return lr

def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
