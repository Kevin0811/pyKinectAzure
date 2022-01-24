import cv2
import numpy as np

from area import poly_area

# 手部關鍵點估計
def mediapipe_detection(frame, model):
    # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
    # 左右翻轉影像，並將格式從 BGR 轉換成 RGB
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    
    # To improve performance, optionally mark the image as not writeable to pass by reference.
    # 影像設定成唯讀，增加執行速度
    image.flags.writeable = False
    
    # 將影像送入模型，手部關鍵點辨識
    results = model.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    
    # 將格式從 BGR 轉換成 RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# 畫出手部關鍵點

dis_min = 0.0008
dis_max = 0.05

def draw_landmarks(image, results, mp_hands, mp_drawing_styles, mp_drawing):
    no_hand = False
    inRange = False
    if results.multi_hand_landmarks:
        
        pose = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark])
        poly = [pose[0], pose[5], pose[17]]
        area = poly_area(poly)
        # print(area)

        if area > dis_max:
            cv2.putText(image ,'Too Close',(250, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
        elif area < dis_min:
            cv2.putText(image ,'Too Far',(250, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
        else:
            cv2.putText(image ,"In Range",(250, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
            inRange = True

        if inRange:
            # 依序讀取關鍵點像素座標
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        else:
            no_hand = True
    else:
        cv2.putText(image ,"Can't Find Hand",(210, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
        no_hand = True
                     
    return image, no_hand

# 歸一化 [0~1]
def normalization(data):
    _range = np.max(abs(data))
    return data / _range

# 標準化
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

# 將關鍵點轉為一維陣列
def extract_keypoints(results):
    pose = np.array([[res.x, res.y] for res in results.multi_hand_landmarks[0].landmark]).flatten()
    depth = np.array([[res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten()
    return np.append(standardization(pose), depth)


# 長條圖顏色
colors = [(245,117,16), (117,245,16), (16,117,245)]

# 顯示預測數值
def prob_viz(res, actions, input_frame, colors = colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,55+num*30), (int(prob*50), 85+num*30), colors[num%3], -1)
        cv2.putText(output_frame, actions[num], (0, 75+num*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

# 顯示混淆矩陣
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()