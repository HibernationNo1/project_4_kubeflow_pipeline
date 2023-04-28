import os, os.path as osp
import random
import cv2
import glob
from tqdm import tqdm

IMAGE_NAME = 'img_test.jpg'
INPUT_IMG_DIR = 'input'
OUTPUT_IMG_DIR = "output"

NUM_BOARD = 4		# n*n 개의 board를 만든다
GET_BOARD = 2		# NUM_BOARD*NUM_BOARD개수 만큼의 board중에서 select할 board의 최대 개수

BOARD_TEXT = ["A", "B", "C", "D", "E"]
COLOR_LIST = [	
                dict(
                    background = (128, 128, 128),
                    letter = (0, 0, 0)
                ),
                dict(
                    background = (211, 211, 211),
                    letter = (0, 0, 0)
                ),
                # dict(
                #     background = (255, 255, 255),
                #     letter = (0, 0, 0)
                # ),
                dict(
                    background = (100, 212, 255),
                    letter = (0, 0, 0)
                ),
                dict(
                    background = (255, 212, 100),
                    letter = (0, 0, 0)
                )
                # dict(
                #     background = (255, 255, 255),
                #     letter = (0, 0, 0)
                # ),
                # dict(
                #     background = (0, 212, 255),
                #     letter = (0, 0, 0)
                # ),
                # dict(
                #     background = (255, 212, 0),
                #     letter = (0, 0, 0)
                # )

             ]
RECT = [
    dict(
        type="regular",
        info = [
            dict(
                type = "s",
                width = 200, 
                height = 100,
                fontScale = 2.5, 
                thickness = 3,
                fontFace = 2
            ),
            dict(
                type = "r",
                width = 300, 
                height = 150,
                fontScale = 4, 
                thickness = 5,
                fontFace = 2
            )
            # dict(
            #     type = "l",
            #     width = 500, 
            #     height = 250,
            #     fontScale = 8, 
            #     thickness = 10,
            #     fontFace = 2
            # ),
            # dict(
            #     type = "xl",
            #     width = 700, 
            #     height = 350,
            #     fontScale = 12, 
            #     thickness = 10,
            #     fontFace = 2
            # )
        ]
        
    ),
    dict(
        type="long",
        info = [
            dict(
                type = "s",
                width = 210, 
                height = 40,
                space_text_num = 2,     # text사이 공간 크기
                fontScale = 2.0, 
                thickness = 3,
                fontFace = 2
            ),
            dict(
                type = "r",
                width = 300, 
                height = 60,
                space_text_num = 2,     
                fontScale = 2.9, 
                thickness = 5,
                fontFace = 2
            )
            # dict(
            #     type = "l",
            #     width = 450, 
            #     height = 80,
            #     space_text_num = 2,     
            #     fontScale = 4.5, 
            #     thickness = 7,
            #     fontFace = 2
            # ),
            # dict(
            #     type = "xl",
            #     width = 700, 
            #     height = 120,
            #     space_text_num = 2,     
            #     fontScale = 7.2, 
            #     thickness = 10,
            #     fontFace = 2
            # )
        ]
        
    )
]


img_list = glob.glob(osp.join(os.getcwd(), INPUT_IMG_DIR)+"/*.jpg")

for img_path in tqdm(img_list):

    # create rectangle board
    image = cv2.imread(img_path)
    height, width, channel = image.shape
    virt_divi, hori_divi = int(width/(NUM_BOARD+1)), int(height/(NUM_BOARD+1))
    rectangle_info_list = []
    for i in range(NUM_BOARD):
        for j in range(NUM_BOARD):
            type_random = random.randrange(len(RECT))

            rec_info_idx = random.randrange(len(RECT[type_random]['info']))

            rect_width = RECT[type_random]['info'][rec_info_idx]['width']
            rect_height = RECT[type_random]['info'][rec_info_idx]['height']
            rectangle_info_list.append([[int(virt_divi*(i+1) - rect_width), 
                                        int(hori_divi*(j+1) - rect_height),
                                        int(virt_divi*(i+1) + rect_width),
                                        int(hori_divi*(j+1) + rect_height)
                                        ],
                                        type_random,
                                        rec_info_idx
                                        ])  

    num_selected_board = random.randrange(1, GET_BOARD+1)
    selected_board = []
    for _ in range(num_selected_board):
        selected_board.append(rectangle_info_list[random.randrange(len(rectangle_info_list))])
        

    for baord_info in selected_board:
        points, board_type_idx, size_type_idx = baord_info

        x_min, y_min, x_max, y_max = points 

        color_random = random.randrange(len(COLOR_LIST))
        color_back = COLOR_LIST[color_random]['background']       # board의 background color
        color_text = COLOR_LIST[color_random]['letter']           # board의 text color

        left_top = (x_min, y_min)
        right_bottom = (x_max, y_max)
        cv2.rectangle(image, 
                    left_top, 
                    right_bottom, 
                    color = color_back, 
                    thickness = -1, 
                    lineType = None, 
                    shift = None)
        

        board_first_num_list = []
        board_last_num_list = []
        for num in range(2):
            board_first_num_list.append(random.randrange(9))
        for num in range(4):
            board_last_num_list.append(random.randrange(9))

        alphabet = BOARD_TEXT[random.randrange(len(BOARD_TEXT))]		# alphabet

        board_dict = RECT[board_type_idx]
        board_type = board_dict['type']
        board_info = board_dict['info'][size_type_idx]
        board_size_type = board_info['type']
        if board_type == "long":
            text = " "
            for num in board_first_num_list:
                text +=f"{num}"
            text += f" {alphabet}"
            for _ in range(board_info['space_text_num']):
                text +=f" "
            for num in board_last_num_list:
                text +=f"{num}"

            if board_size_type == "s":
                text_loc = (x_min-10, y_max-20)            
            elif board_size_type == "r":
                text_loc = (x_min-20, y_max-33)
            elif board_size_type == "l":
                text_loc = (x_min-39, y_max-35)
            elif board_size_type == "xl":
                text_loc = (x_min-90, y_max-45)


            

            cv2.putText(image, 
                        text, 
                        text_loc, 
                        color = color_text, 
                        fontFace = board_info['fontFace'],
                        fontScale = board_info['fontScale'],
                        thickness = board_info['thickness'])
        elif board_type == "regular":
            top_text = ""
            for num in board_first_num_list:
                top_text +=f"{num}"
            top_text += f" {alphabet}"
            
            board_height = y_max - y_min
            board_width = x_max - x_min

            if board_size_type == "s":
                top_texr_loc = (x_min + 105, y_max-130)
                bottom_texr_loc = (int(x_min + 63), y_max-20)
            elif board_size_type == "r":
                top_texr_loc = (int(x_min + 140), int(y_max-(board_height*3/5)))
                bottom_texr_loc = (int(x_min + 95), y_max-30)
            elif board_size_type == "l":
                top_texr_loc = (x_min + 200, int(y_max-(board_height*3/5)))
                bottom_texr_loc = (x_min + 150, y_max-40)
            elif board_size_type == "xl":
                top_texr_loc = (x_min + 240, y_max-400)
                bottom_texr_loc = (x_min + 190, y_max-50)
            
            bottom_text = ""
            for num in board_last_num_list:
                bottom_text +=f"{num}"

            # 상단 text
            cv2.putText(image, 
                    top_text, 
                    top_texr_loc, 
                    color = color_text, 
                    fontFace = board_info['fontFace'],
                    fontScale = board_info['fontScale'],
                    thickness = board_info['thickness'])
            
            # 하단 text
            cv2.putText(image, 
                        bottom_text, 
                        bottom_texr_loc, 
                        color = color_text, 
                        fontFace = board_info['fontFace'],
                        fontScale = board_info['fontScale']+1,
                        thickness = board_info['thickness']|1)
        
        



    # cv2.imshow("img", image)
    # while True:
    #     if cv2.waitKey() == 27: break

    img_name = osp.basename(img_path)
    img_name = img_name.split(".")[0] + "_3" + "." + img_name.split(".")[1]
   
    output_img_path = osp.join(os.getcwd(), OUTPUT_IMG_DIR, img_name)
    cv2.imwrite(output_img_path, image)