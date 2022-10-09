from symbol import arglist
from turtle import distance
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time
from PIL import Image

class LaneDetection:
    '''
    Lane detection module using edge detection and b-spline fitting

    args: 
        cut_size (cut_size=65) cut the image at the front of the car
        spline_smoothness (default=10)
        gradient_threshold (default=14)
        distance_maxima_gradient (default=3)

    '''

    def __init__(self, cut_size=65, spline_smoothness=10, gradient_threshold=14, distance_maxima_gradient=3):
        self.car_position = np.array([48,0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old = 0
        self.lane_boundary2_old = 0


    def cut_gray(self, state_image_full):
        '''
        ##### TODO #####
        This function should cut the image at the front end of the car (e.g. pixel row 65) 
        and translate to gray scale

        input:
            state_image_full 96x96x3

        output:
            gray_state_image 65x96x1

        '''
        # Extract cutting limits
        cut_size = self.cut_size
        # Transform image to greyscale
        img = np.array(state_image_full)
        height = img.shape[0]
        width = img.shape[1]

        gray_state_image = np.zeros((height, width, 1), dtype=np.uint8)

        for h in range(height):
            for w in range(width):
                b = img[h, w, 0].astype(np.float32)
                g = img[h, w, 1].astype(np.float32)
                r = img[h, w, 2].astype(np.float32)

                intensity = (r + g + b) / 3
                gray_state_image[h, w, 0] = intensity

        # Crop Information bar from image
        gray_state_image = gray_state_image[:cut_size, :]
        
        return gray_state_image[::-1] 


    def edge_detection(self, gray_image):
        '''
        ##### TODO #####
        In order to find edges in the gray state image, 
        this function should derive the absolute gradients of the gray state image.
        Derive the absolute gradients using numpy for each pixel. 
        To ignore small gradients, set all gradients below a threshold (self.gradient_threshold) to zero. 
        
        input:
            gray_state_image 65x96x1

        output:
            gradient_sum 65x96x1
        '''

        gray_image = np.reshape(gray_image,(65,96))
        img_shape = gray_image.shape
        threshold = self.gradient_threshold

        result = []
        for i in range (img_shape[0]) :
            gs = np.gradient(np.array(gray_image[i]))
            result.append(gs)

        gradient_sum = result

        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                if gradient_sum[i][j] < 0:
                    gradient_sum[i][j] *= -1

                if gradient_sum[i][j] < threshold:
                    gradient_sum[i][j] = 0
    
        # print(gradient_sum[0])
        return gradient_sum

    # def edge_detection(self, gray_image):
    #     '''
    #     ##### TODO #####
    #     In order to find edges in the gray state image, 
    #     this function should derive the absolute gradients of the gray state image.
    #     Derive the absolute gradients using numpy for each pixel. 
    #     To ignore small gradients, set all gradients below a threshold (self.gradient_threshold) to zero. 

    #     input:
    #         gray_state_image 65x96x1

    #     output:
    #         gradient_sum 65x96x1

    #     '''
    #     threshold = self.gradient_threshold

    #     # Sobel edge kernels
    #     G_x = np.array([
    #         [-1, 0, 1],
    #         [-2, 0, 2],
    #         [-1, 0, 1]
    #     ])
    #     G_y = np.array([
    #         [1, 2, 1],
    #         [0, 0, 0],
    #         [-1, -2, -1]
    #     ])

    #     gray_image = np.reshape(gray_image, (65, 96))
    #     img_shape = gray_image.shape
    #     filter_size = G_x.shape
    #     result = []

    #     for i in range(img_shape[0]):
    #         temp = np.gradient(gray_image[i])
    #         result.append(temp)

    #     gradient_sum = np.array(result)
    #     # print(gradient_sum)

    #     for i in range(img_shape[0]):
    #         for j in range(img_shape[1]):
    #             g = gradient_sum[i][j]
    #             if g < 0:
    #                 gradient_sum[i][j] = g * -1
    #             if g < threshold:
    #                 gradient_sum[i][j] = 0

        # print(gradient_sum)

        # result1 = np.zeros(img_shape)
        # result2 = np.zeros(img_shape)
        # # result = []

        # for h in range(0, img_shape[0]):  # 행 개수
        #     for w in range(0, img_shape[1]):  # 열 개수
        #         # list[s:e]: s~e-1까지 잘라서 list로 반환
        #         hb = h >= (img_shape[0] - filter_size[0])
        #         wb = w >= (img_shape[1] - filter_size[1])

        #         if hb and wb:
        #             tmp = gray_image[h-filter_size[0]:h, w-filter_size[1]:w]
        #         elif not hb and wb:
        #             tmp = gray_image[h:h+filter_size[0], w-filter_size[1]:w]
        #         elif hb and not wb:
        #             tmp = gray_image[h-filter_size[0]:h, w:w+filter_size[1]]
        #         else:    
        #             tmp = gray_image[h:h+filter_size[0], w:w+filter_size[1]]

        #         result1[h, w] = np.abs(np.sum(tmp * G_x))
        #         result2[h, w] = np.abs(np.sum(tmp * G_y))

        # result = result1 + result2
        # print(result)
        # gradient_sum = np.zeros(img_shape)
        # gradient_sum[result>threshold] = 1
        # print(gradient_sum)
        
        return gradient_sum


    def find_maxima_gradient_rowwise(self, gradient_sum):
        '''
        ##### TODO #####
        This function should output arguments of local maxima for each row of the gradient image.
        You can use scipy.signal.find_peaks to detect maxima. 
        Hint: Use distance argument for a better robustness.

        input:
            gradient_sum 65x96x1

        output:
            maxima (np.array) shape : (Number_maxima, 2)

        '''
        # print(gradient_sum)
        distance = self.distance_maxima_gradient
        gradient_sum = np.array(gradient_sum)
        # print(gradient_sum)
        g_shape = gradient_sum.shape
        gsum = np.reshape(gradient_sum, (g_shape[0], g_shape[1]))
        argmaxima = np.empty((0, 2))
        
        for i in range(g_shape[0]):
            peaks, props = find_peaks(gradient_sum[i], distance=distance)
            for j in peaks:
                coord = np.array([[j, i]])
                argmaxima = np.append(argmaxima, coord, axis=0)
        # print(argmaxima)
        return argmaxima


    def find_first_lane_point(self, gradient_sum):
        '''
        Find the first lane_boundaries points above the car.
        Special cases like just detecting one lane_boundary or more than two are considered. 
        Even though there is space for improvement ;) 

        input:
            gradient_sum 65x96x1

        output: 
            lane_boundary1_startpoint
            lane_boundary2_startpoint
            lanes_found  true if lane_boundaries were found
        '''
        
        # Variable if lanes were found or not
        lanes_found = False
        row = 0

        # loop through the rows
        while not lanes_found:
            
            # Find peaks with min distance of at least 3 pixel 
            argmaxima = find_peaks(gradient_sum[row],distance=3)[0]

            # if one lane_boundary is found
            if argmaxima.shape[0] == 1:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])

                if argmaxima[0] < 48:
                    lane_boundary2_startpoint = np.array([[0,  row]])
                else: 
                    lane_boundary2_startpoint = np.array([[96,  row]])

                lanes_found = True
            
            # if 2 lane_boundaries are found
            elif argmaxima.shape[0] == 2:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])
                lane_boundary2_startpoint = np.array([[argmaxima[1],  row]])
                lanes_found = True

            # if more than 2 lane_boundaries are found
            elif argmaxima.shape[0] > 2:
                # if more than two maxima then take the two lanes next to the car, regarding least square
                A = np.argsort((argmaxima - self.car_position[0])**2)
                lane_boundary1_startpoint = np.array([[argmaxima[A[0]],  0]])
                lane_boundary2_startpoint = np.array([[argmaxima[A[1]],  0]])
                lanes_found = True

            row += 1
            
            # if no lane_boundaries are found
            if row == self.cut_size:
                lane_boundary1_startpoint = np.array([[0,  0]])
                lane_boundary2_startpoint = np.array([[0,  0]])
                break

        return lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found


    def lane_detection(self, state_image_full):
        '''
        ##### TODO #####
        This function should perform the road detection 

        args:
            state_image_full [96, 96, 3]

        out:
            lane_boundary1 spline
            lane_boundary2 spline
        '''

        # to gray
        gray_state = self.cut_gray(state_image_full)

        # edge detection via gradient sum and thresholding
        gradient_sum = self.edge_detection(gray_state)
        maxima = self.find_maxima_gradient_rowwise(gradient_sum)
       

        # first lane_boundary points
        lane_boundary1_points, lane_boundary2_points, lane_found = self.find_first_lane_point(gradient_sum)
        
        prev1 = lane_boundary1_points
        prev2 = lane_boundary2_points
        up = True
        # if no lane was found,use lane_boundaries of the preceding step
        if lane_found:
            
            ##### TODO #####
            #  in every iteration: 
            # 1- find maximum/edge with the lowest distance to the last lane boundary point 
            # 2- append maximum to lane_boundary1_points or lane_boundary2_points
            # 3- delete maximum from maxima
            # 4- stop loop if there is no maximum left 
            #    or if the distance to the next one is too big (>=100)

            # 1- 마지막 lane boundary point까지 가장 낮은 거리의 maximum/edge 찾기
            # 2- maximum을 lane_boundary1_points 또는 lane_boundary2_points에 append
            # 3- maxima에서 maximum 삭제
            # 4- maximum이 없거나 다음까지의 거리가 너무 크기 전까지(>=100) loop
            # lane_boundary 1
            
            # lane_boundary 2

            ################
            for row in range(0, self.cut_size):
                # maxima[row].shape[0] : 2
                if row < 0 or row >= self.cut_size:
                    break

                if maxima[row].shape[0] != 0:
                    points = []

                    for edge in maxima[row]:
                        points.append(np.array([edge, row]))

                    points = np.asarray(points)
                    dists = np.sum((points - prev1) ** 2, axis=1)
                    min_idx = np.argmin(dists)

                    closet_point_1 = points[min_idx: min_idx + 1, :]
                    dist_1 = dists[min_idx]

                    dists = np.sum((points - prev2) ** 2, axis=1)
                    min_idx = np.argmin(dists)

                    closet_point_2 = points[min_idx: min_idx + 1, :]
                    dist_2 = dists[min_idx]

                    if (100 >= dist_1 > 0) and (100 >= dist_2 > 0):
                        prev1 = closet_point_1
                        prev2 = closet_point_2
                        lane_boundary1_points = np.concatenate((lane_boundary1_points, prev1))
                        lane_boundary2_points = np.concatenate((lane_boundary2_points, prev2))
                        maxima[row] = np.delete(maxima[row], np.where(maxima[row] == closet_point_1))
                        maxima[row] = np.delete(maxima[row], np.where(maxima[row] == closet_point_2))
                    elif (100 >= dist_1 > 0) and (100 <= dist_2):
                        prev1 = closet_point_1
                        lane_boundary1_points = np.concatenate((lane_boundary1_points, prev1))
                        maxima[row] = np.delete(maxima[row], np.where(maxima[row] == closet_point_1))
                    elif (100 >= dist_2 > 0) and (100 <= dist_1):
                        prev2 = closet_point_2
                        lane_boundary2_points = np.concatenate((lane_boundary2_points, prev2))
                        maxima[row] = np.delete(maxima[row], np.where(maxima[row] == closet_point_2))
                        
                if up:
                    row += 1
                    if row == self.cut_size:
                        up = False
                if not up:
                    row -= 1
                    
                    

            ##### TODO #####
            # spline fitting using scipy.interpolate.splprep 
            # and the arguments self.spline_smoothness
            # 
            # if there are more lane_boundary points points than spline parameters 
            # else use perceding spline
            if lane_boundary1_points.shape[0] > 4 and lane_boundary2_points.shape[0] > 4:

                # Pay attention: the first lane_boundary point might occur twice
                # lane_boundary 1
                x1 = np.float64(lane_boundary1_points[:, 0])
                y1 = np.float64(lane_boundary1_points[:, 1])
                lane_boundary1, _ = splprep([x1, y1], s=self.spline_smoothness)
                # lane_boundary 2
                x2 = np.float64(lane_boundary2_points[:, 0])
                y2 = np.float64(lane_boundary2_points[:, 1])
                lane_boundary2, _ = splprep([x2, y2], s=self.spline_smoothness)
                
            else:
                lane_boundary1 = self.lane_boundary1_old
                lane_boundary2 = self.lane_boundary2_old
            ################

        else:
            lane_boundary1 = self.lane_boundary1_old
            lane_boundary2 = self.lane_boundary2_old

        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2

        # output the spline
        return lane_boundary1, lane_boundary2


    def plot_state_lane(self, state_image_full, steps, fig, waypoints=[]):
        '''
        Plot lanes and way points
        '''
        # evaluate spline for 6 different spline parameters.
        t = np.linspace(0, 1, 6)
        lane_boundary1_points_points = np.array(splev(t, self.lane_boundary1_old))
        lane_boundary2_points_points = np.array(splev(t, self.lane_boundary2_old))
        
        plt.gcf().clear()
        plt.imshow(state_image_full[::-1])
        plt.plot(lane_boundary1_points_points[0], lane_boundary1_points_points[1]+96-self.cut_size, linewidth=5, color='orange')
        plt.plot(lane_boundary2_points_points[0], lane_boundary2_points_points[1]+96-self.cut_size, linewidth=5, color='orange')
        if len(waypoints):
            plt.scatter(waypoints[0], waypoints[1]+96-self.cut_size, color='white')

        plt.axis('off')
        plt.xlim((-0.5,95.5))
        plt.ylim((-0.5,95.5))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        fig.canvas.flush_events()
