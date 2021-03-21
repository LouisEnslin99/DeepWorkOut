import pandas as pd
from pathlib import Path
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.spatial import distance
import warnings
from scipy import signal
from scipy.signal import filtfilt
from scipy.stats import iqr
from scipy.stats import spearmanr



class Analyzer:
    ''' Class to describe an analysed video. When initialising the object it gets a dataframe with coordinates of all bodypoints.
    This class provides different methods and functions for cleaning data, normalize it and then get features out of it.
    '''

    # With path and information of Scorer and FPS it generates several attributes where information about the video is stored
    def __init__(self, path, DLCscorer, fps):
        # loading output of DLC
        self.df_orig = pd.read_hdf(path)[DLCscorer]
        self.df = self.df_orig
        self.fps = fps
        self.pixel_to_centi = 1.0
        # Get name and number of exercise from path
        self.name = path[path.find('pandas\\')+len('pandas\\'):path.rfind('DLC')]
        self.exercise = self.name.split('_')[0]

    @staticmethod
    def interpolate_outliers(df_orig):
        '''
        This function interpolates outliers of every given array or dataframe by using the IQR
        '''
        df = np.copy(df_orig)
        Q1 = np.nanpercentile(df, 25)
        Q3 = np.nanpercentile(df, 75)
        IQR = Q3 - Q1
        for i in range(len(df)):
            if (df[i] < (Q1 - 1.5 * IQR)) | (df[i] > (Q3 + 1.5 * IQR)):
                df[i] = np.median(df)
        return df

    @staticmethod
    def reduce_noise(array_orig):
        '''
        This function reduces noise of a given array by applying a butterworth filter on the data
        '''
        b, a = signal.butter(3, 0.15)
        reduced_noise = filtfilt(b, a, array_orig, padlen=20)
        return reduced_noise

    def clean_data(self, pcutoff=0.9, min_rate=0.95):
        '''
        This method cleans the videos datapoints by removing whole bodypoints or frames by checking the likelyhood of the coordinates (likelyhood given from Deeplabcut)
        '''
        bpts = self.df.columns.get_level_values(0)[1::3]
        # Delete bodyparts
        for bpindex, bp in enumerate(bpts):
            Index = self.df[bp]['likelihood'].values > pcutoff
            if(np.count_nonzero(Index == True) / len(Index)) < min_rate:
                self.df = self.df.drop(bp, 1)
        bpts = self.df.columns.get_level_values(0)[1::3]
        # Delete frames
        for bpindex, bp in enumerate(bpts):
            self.df.drop(self.df.index[self.df[bp]
                         ['likelihood'] < pcutoff], inplace=True)
            
    def calibrate_metric(self, bp1, bp2, length):
        '''
        This methods calculates how to get the metric from pixels to centimeter. This is import to compare later the features of the videos with the same metric. 
        '''
        x1 = self.df[bp1]['x']
        y1 = self.df[bp1]['y']
        x2 = self.df[bp2]['x']
        y2 = self.df[bp2]['y']
        x_diff = np.abs(x2 - x1)
        y_diff = np.abs(y2 - y1)
        dist = np.sqrt(x_diff**2 + y_diff**2)
        self.pixel_to_centi = length/np.mean(dist)

    
    def get_velocity(self, bp, interpolate=True, reduce_noise=True, in_centi_sec=True):
        '''
        This methods return an array of velocitys (from each frame) from the video from a given bodypoint
        thereby the values get: interpolated, noise gets reduced, metric is changed to centimeter
        '''
        x_values = self.df[bp]['x']
        y_values = self.df[bp]['y']
        v1 = np.vstack([x_values, y_values]).T
        # loop over each pair of points and extract distances
        dist = []
        for n, pos in enumerate(v1):
            # Get a pair of points
            if n == 0:  # get the position at time 0, velocity is 0
                p0 = pos
                dist.append(0)
            else:
                p1 = pos  # get position at current frame

                # Calc distance
                dist.append(np.abs(distance.euclidean(p0, p1)))

                # Prepare for next iteration, current position becomes the old one and repeat
                p0 = p1
        vel = np.array(dist)
        # Delete outliers and interpolate them
        if interpolate == True:
            vel = self.interpolate_outliers(vel)
        # Reduce noise of the signal
        if reduce_noise == True:
            vel = self.reduce_noise(vel)
        # Change metric from pixek per frame to centimeter per second
        if in_centi_sec == True:
            vel = vel * self.fps
            vel = vel * self.pixel_to_centi
        return vel

    def get_acceleration(self, bp, interpolate=True, reduce_noise=True, in_centi_sec=True):
        '''
        This methods returns the accelaration of the given bodypart. It gets returned in an array which representates the acc in each frame
        '''
        # in Pixel per frame
        vel = self.get_velocity(bp, interpolate=True, reduce_noise=True, in_centi_sec=False)
        # In frames
        x = np.arange(len(vel))
        acc = np.diff(vel)/np.diff(x)
         # Delete outliers and interpolate them
        if interpolate == True:
            acc = self.interpolate_outliers(acc)
        # Reduce noise of the signal
        if reduce_noise == True:
            acc = self.reduce_noise(acc)
        # Change metric from pixek per frame to centimeter per second
        if in_centi_sec == True:
            acc = acc * self.fps**2
            acc = acc * self.pixel_to_centi
        return acc
        
    def calculate_angle_from_bodypart(self, bp1, bp2, bp3, interpolate=True, reduce_noise=True):
        '''
        This methods retuns the angle with 3 given bodyparts
        '''
        deltay_1 = self.df[bp1]['y'].values - self.df[bp2]['y'].values
        deltax_1 = self.df[bp1]['x'].values - self.df[bp2]['x'].values
        deltay_2 = self.df[bp3]['y'].values - self.df[bp2]['y'].values
        deltax_2 = self.df[bp3]['x'].values - self.df[bp2]['x'].values

        # Calculate the two gradients
        m1 = deltay_1/deltax_1
        m2 = deltay_2/deltax_2

        # Caluculate the angle
        alpha = np.arctan(abs((m1-m2)/(1+m1*m2)))
        beta = np.pi - alpha

        # check if alpha or beta is the right angle, c: distance between bp1 and bp3
        x_diff_a = abs(self.df[bp1]['x'].values -
                    self.df[bp2]['x'].values)
        y_diff_a = abs(self.df[bp1]['y'].values -
                    self.df[bp2]['y'].values)
        x_diff_b = abs(self.df[bp2]['x'].values -
                    self.df[bp3]['x'].values)
        y_diff_b = abs(self.df[bp2]['y'].values -
                    self.df[bp3]['y'].values)
        x_diff_c = abs(self.df[bp1]['x'].values -
                    self.df[bp3]['x'].values)
        y_diff_c = abs(self.df[bp1]['y'].values -
                    self.df[bp3]['y'].values)

        # Calculate the distances between the points
        c_ist = np.sqrt(x_diff_c**2 + y_diff_c**2)
        a_ist = np.sqrt(x_diff_a**2 + y_diff_a**2)
        b_ist = np.sqrt(x_diff_b**2 + y_diff_b**2)

        c_soll = np.sqrt(a_ist**2 + b_ist**2 - 2*a_ist*b_ist*np.cos(beta))

        # Iterate over every element in the array to determine the angle
        k = 0
        angle = []
        for i in np.nditer(c_ist):
            if c_soll[k].astype('int') == c_ist[k].astype('int'):
                angle.append(beta[k])
                k += 1
            else:
                angle.append(alpha[k])
                k += 1
        # Delete outliers and interpolate them
        if interpolate == True:
            angle = self.interpolate_outliers(angle)
        # Reduce noise of the signal
        if reduce_noise == True:
            angle = self.reduce_noise(angle)

        return np.rad2deg(np.array(angle))

    def get_distance(self, bp1, bp2, interpolate=True, reduce_noise=True, in_centi=True):
        '''
        Returns the distance between 2 bodyparts
        '''
        deltay_1 = self.df[bp1]['y'].values - self.df[bp2]['y'].values
        deltax_1 = self.df[bp1]['x'].values - self.df[bp2]['x'].values

        dist = np.sqrt(deltay_1**2 + deltax_1**2)
         # Delete outliers and interpolate them
        if interpolate == True:
            dist = self.interpolate_outliers(dist)
        # Reduce noise of the signal
        if reduce_noise == True:
            dist = self.reduce_noise(dist)
        # Change metric from pixek  to centimeter
        if in_centi == True:
            dist = dist * self.pixel_to_centi
        return dist

    def get_Xcoordinates(self, bp, interpolate=True, reduce_noise=True, in_centi=True):
        '''
        Return the x-coordinates of bodypoint
        '''
        x = self.df[bp]['x']
        if interpolate == True:
            x = self.interpolate_outliers(x)
        # Reduce noise of the signal
        if reduce_noise == True:
            x = self.reduce_noise(x)
        # Change metric from pixek  to centimeter 
        if in_centi == True:
            x = x * self.pixel_to_centi
        return x

    def get_Ycoordinates(self, bp, interpolate=True, reduce_noise=True, in_centi=True):
        '''
        Return the y-coordinates of bodypoint
        '''
        y = self.df[bp]['y']
        if interpolate == True:
            y = self.interpolate_outliers(y)
        # Reduce noise of the signal
        if reduce_noise == True:
            y = self.reduce_noise(y)
        # Change metric from pixek  to centimeter 
        if in_centi == True:
            y = y * self.pixel_to_centi
        return y

    def get_Features(self, bpts):
        '''
        Returns about 235 features of the video by using mean, min, max, range, std, iqr, coorelation of acc, velo, coords, amgles, distances
        '''
        features = {}
        for bp in bpts:
            # X-Coordinates
            x_coords = self.get_Xcoordinates(bp, interpolate=True, reduce_noise=True, in_centi=True)
            features['xcoord_std_' + bp] = np.std(x_coords)
            features['xcoord_range_' + bp] = abs(np.max(x_coords) - np.min(x_coords))
            features['xcoord_iqr_' + bp] = iqr(x_coords)
            # Y-Coordinates
            y_coords = self.get_Ycoordinates(bp, interpolate=True, reduce_noise=True, in_centi=True)
            features['ycoord_std_' + bp] = np.std(y_coords)
            features['ycoord_range_' + bp] = abs(np.max(y_coords) - np.min(y_coords))
            features['ycoord_iqr_' + bp] = iqr(y_coords)
            # Velocity
            velocity = self.get_velocity(bp, interpolate=True, reduce_noise=True, in_centi_sec=True)
            features['velo_std_' + bp] = np.std(velocity)
            features['velo_range_' + bp] = abs(np.max(velocity) - np.min(velocity))
            features['velo_iqr_' + bp] = iqr(velocity)
            features['velo_mean_' + bp] = np.mean(velocity)
            features['velo_max_' + bp] = np.max(velocity)
            features['velo_min_' + bp] = np.min(velocity)
            # Acceleration
            acceleration = self.get_acceleration(bp, interpolate=True, reduce_noise=True, in_centi_sec=True)
            features['acc_std_' + bp] = np.std(acceleration)
            features['acc_range_' + bp] = abs(np.max(acceleration) - np.min(acceleration))
            features['acc_iqr_' + bp] = iqr(acceleration)
            features['acc_mean_' + bp] = np.mean(acceleration)
            features['acc_max_' + bp] = np.max(acceleration)
            features['acc_min_' + bp] = np.min(acceleration)
        # Angles
        angles = {}
        angles['kneeleft'] = self.calculate_angle_from_bodypart('ankleleft', 'kneeleft', 'hipleft', interpolate=True, reduce_noise=True)
        #angles['elbowleft'] = self.calculate_angle_from_bodypart('shoulderleft', 'elbowleft', 'wristleft', interpolate=True, reduce_noise=True)
        angles['hipleft'] = self.calculate_angle_from_bodypart('kneeleft', 'hipleft', 'shoulderleft', interpolate=True, reduce_noise=True)
        #angles['ankleleft'] = self.calculate_angle_from_bodypart('toesleft', 'ankleleft', 'kneeleft', interpolate=True, reduce_noise=True)
        for key, values in angles.items():
            features['angle_std_' + key] = np.std(values)
            features['angle_range_' + key] = abs(np.max(values) - np.min(values))
            features['angle_iqr_' + key] = iqr(values)
            features['angle_mean_' + key] = np.mean(values)
            features['angle_max_' + key] = np.max(values)
            features['angle_min_' + key] = np.min(values)
        # Distances
        distances = {}
        distances['ankle-hipleft'] = self.get_distance('ankleleft', 'hipleft', interpolate=True, reduce_noise=True, in_centi=True)
        #distances['wrist-elbowleft'] = self.get_distance('wristleft', 'elbowleft', interpolate=True, reduce_noise=True, in_centi=True)
        distances['elbow-kneeleft'] = self.get_distance('elbowleft', 'kneeleft', interpolate=True, reduce_noise=True, in_centi=True)
        distances['elbow-hipleft'] = self.get_distance('elbowleft', 'hipleft', interpolate=True, reduce_noise=True, in_centi=True)
        for key, values in distances.items():
            features['distance_std_' + key] = np.std(values)
            features['distance_range_' + key] = abs(np.max(values) - np.min(values))
            features['distance_iqr_' + key] = iqr(values)
            features['distance_mean_' + key] = np.mean(values)
            features['distance_max_' + key] = np.max(values)
            features['distance_min_' + key] = np.min(values)
        # Correlations of x and y values
        x_bpts = ['hipleft', 'elbowleft']
        y_bpts = ['hipleft', 'kneeleft', 'shoulderleft']
        for bp in bpts:
            for x_bp in x_bpts:
                _, features['x_coord_corr_' + x_bp + '_' + bp] = spearmanr(self.get_Xcoordinates(bp), self.get_Xcoordinates(x_bp))
            for y_bp in y_bpts:
                _, features['x_coord_corr_' + y_bp + '_' + bp] = spearmanr(self.get_Ycoordinates(bp), self.get_Ycoordinates(y_bp))
        # Correlations of different angles
        #_, features['angle_corr_knee-elbowleft'] = spearmanr(angles['kneeleft'], angles['elbowleft'])
        _, features['angle_corr_knee-hipleft'] = spearmanr(angles['kneeleft'], angles['hipleft'])

        return features

    def get_bodyparts(self):
        '''
        Returns all bodypoints. Normally this method is called after cleaning the data to get all boypoints that are still valid
        '''
        return list(self.df.columns.get_level_values(0)[1::3])

    
    def plotFeatures(self, values):
        '''
        plots the given Array values along time of video, asuming this is an feature of the video
        '''
        time = np.arange(len(values))*1./self.fps
        plt.plot(time, values)
        plt.show()
    
    @staticmethod
    def get_cmap(n, name='hsv'):
        '''
        return a cmap, used for {PlottingResults}
        '''
        return plt.cm.get_cmap(name, n)

    @staticmethod
    def Histogram(vector,color,bins):
        '''
        plots Histogram with given values
        '''
        dvector=np.diff(vector)
        dvector=dvector[np.isfinite(dvector)]
        plt.hist(dvector,color=color,histtype='step',bins=bins)
    
    def PlottingResults(self,bodyparts2plot, pcutoff=.5, alphavalue=.2,colormap='gist_rainbow',fs=(4,3)):
        '''
        Plots poses of video over the whole video time, if video is already calibrated its metric is in centimeter
        '''
        plt.figure(figsize=fs)
        colors = self.get_cmap(len(bodyparts2plot),name = colormap)
        # iterate over each bodypart and plot if likelihood is in range
        for bpindex, bp in enumerate(bodyparts2plot):
            Index=self.df[bp]['likelihood'].values > pcutoff
            plt.plot(self.df[bp]['x'].values[Index] * self.pixel_to_centi,self.df[bp]['y'].values[Index] * self.pixel_to_centi,'.',color=colors(bpindex),alpha=alphavalue)
        # adjust plot features
        plt.gca().invert_yaxis()
        plt.xlim(0, 640 * self.pixel_to_centi)
        plt.ylim(360 * self.pixel_to_centi, 0)
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=plt.Normalize(vmin=0, vmax=len(bodyparts2plot)-1))
        sm._A = []
        cbar = plt.colorbar(sm,ticks=range(len(bodyparts2plot)))
        cbar.set_ticklabels(bodyparts2plot)
        #plt.savefig(os.path.join(tmpfolder,"trajectory"+suffix))
        plt.figure(figsize=fs)

class MultiAnalyzer:
    '''
    This class is used to handle mutliple {#Analyzer} objects.
    '''
    def __init__(self, folderPath, DLCscorer, fps):
        '''
        creates multianalyzer object by reading from a folder path all datframe files (each is one video) and creates for each an analyzer object to later analyze it
        '''
        paths = glob.glob(folderPath + '*.h5')
        DLCscorer = DLCscorer
        self.videos = [Analyzer(paths[i], DLCscorer, fps) for i in range(len(paths))]
    
    def clean_data(self, neededPoints, pcutoff, min_rate):
        '''
        This method cleans all videos by removing not bodyparts and frames, depending on the pcutoff and min_rate. 
        After cleaning the videos it checks if all needed bodyparts are still there to use afterwards. If not the videos get removed of the multianalyzervideos and than returned 
        to handle them.
        '''
        unvalid_videos = []
        valid_videos = []
        # Cleand video
        for video in self.videos:
            video.clean_data(pcutoff, min_rate)
        # Check if needed bodyparts are still there
        for video in self.videos:
            check =  all(item in video.get_bodyparts() for item in neededPoints)
            if check == True:
                valid_videos.append(video)
            else:
                unvalid_videos.append(video)
        self.videos = valid_videos
        return unvalid_videos

    def calibrate_metric(self, bp1, bp2, length):
        '''
        calibrates the metric of all videos
        '''
        for video in self.videos:
            video.calibrate_metric(bp1, bp2, length)
           

    def getValidBodyparts(self):
        '''
        Returns a list of bodyparts that all videos have in their data
        '''
        valid_bpts = set.intersection(*[set(video.get_bodyparts()) for video in self.videos])
        return valid_bpts

    
