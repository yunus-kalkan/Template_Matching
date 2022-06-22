import cv2
import numpy as np
import os
from tqdm import tqdm



tmp0 = cv2.imread("0.jpg")
tmp1 = cv2.imread("1.jpg")
tmp2 = cv2.imread("2.jpg")
tmp3 = cv2.imread("3.jpg")
tmp4 = cv2.imread("4.jpg")
tmp5 = cv2.imread("5.jpg")
tmp6 = cv2.imread("6.jpg")
tmp7 = cv2.imread("7.jpg")
tmp8 = cv2.imread("8.jpg")
tmp9 = cv2.imread("9.jpg")
tmpb = cv2.imread("blank.jpg")

tmp_list = [tmp0, tmp1, tmp2, tmp3, tmp4, tmp5,
            tmp6, tmp7, tmp8, tmp9, tmpb]


first_digit_l_upper = (178,6)
first_digit_r_lower = (198, 33)

second_digit_l_upper = (199,6)
secon_digit_r_lower = (219, 33)

third_digit_l_upper = (220,6)
third_digit_r_lower = (240, 33)

fourth_digit_l_upper = (241,6)
fourth_digit_r_lower = (261, 33)

coord_list = [(first_digit_l_upper, first_digit_r_lower),
              (second_digit_l_upper, secon_digit_r_lower),
              (third_digit_l_upper, third_digit_r_lower),
              (fourth_digit_l_upper, fourth_digit_r_lower)]




def find_hsv(img):

    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    result = image.copy()

    lower1 = np.array([50, 100, 100])
    upper1 = np.array([70, 255, 255])

    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])
    
    lower_mask = cv2.inRange(image, lower1, upper1)
    upper_mask = cv2.inRange(image, lower2, upper2)
    
    full_mask = lower_mask + upper_mask;
    
    result = cv2.bitwise_and(result, result, mask=full_mask)
    return result
 




def match_img(im, templ, method=cv2.TM_SQDIFF_NORMED):
    h,w = templ.shape[:2]
    res = cv2.matchTemplate(im, templ, method)
    # for some scoring functions a higher score denotes a better match, for others - lower score
    if method in [cv2.TM_SQDIFF_NORMED, cv2.TM_SQDIFF]: res = -res
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # thanks to this function we don't need to do the unravel_index() thing
    #print(f'max score = {max_val:.3}')
    return max_loc,res, max_val



def show_matches(im, max_loc, res):
    im_copy = im.copy()
    top_left, btm_right = max_loc, (max_loc[0] + w, max_loc[1] + h)
    cv2.rectangle(im_copy,top_left,btm_right,(0,255,0), 3)
    cv2.imshow("result", im_copy)
    cv2.waitKey(0)



def find_distance(img):
    distance = []
    for coord in coord_list:
        cropped_img = img[coord[0][1]:coord[1][1], coord[0][0]:coord[1][0]]
        score_list = []
        for i in range(11):
            result = find_hsv(cropped_img)
            _, _, val = match_img(result, tmp_list[i])     
            score_list.append(val)
        max_val = max(score_list)
        max_index = score_list.index(max_val)
        if max_index != 10:
            distance.append(max_index)
    distance_str = "(" +"".join(map(str, distance)) + ")"
    return distance_str




path_of_the_directory= "C://Users//Kalkany//Desktop//tesseract//dataset"
for filename in tqdm(os.listdir(path_of_the_directory)):
    f = os.path.join(path_of_the_directory,filename)
    if os.path.isfile(f):
        img = cv2.imread(f)
        
        dist_str = find_distance(img)
        name = f[:-4] + dist_str + ".jpg"
        cv2.imwrite(name, img)