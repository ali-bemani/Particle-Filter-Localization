#!/usr/bin/env python
'''
**********************************************************************
* Filename    : imu.py
* Description : A module for Inertial Measurement Unit
* Author      : Ali Bemani
* Brand       : MPU5060
* E-mail      : ali.bemani2@gmail.com
* Update      : 
**********************************************************************
'''
import time
import numpy as np
import scipy as scipy
from numpy.random import uniform
import scipy.stats
import random
import smbus
import math

'''******************************************** Classes ****************************************'''
class imu(object):                        # This Class make a new opbejct from IMU devices
    
    def __init__(self,address=0x68):
        self.bus = smbus.SMBus(1)
        self.address = address
        
    def read_byte(self,adr):
        return self.bus.read_byte_data(self.address,adr)

    def read_word(self,adr):
        high = self.bus.read_byte_data(self.address, adr)
        low = self.bus.read_byte_data(self.address, adr+1)
        val = (high << 8) + low
        return val

    def read_word_2c(self,adr):
        val = self.read_word(adr)
        if (val >= 0x8000):
            return -((65535 - val) + 1)
        else:
            return val

    def dist(self,a,b):
        return math.sqrt((a*a)+(b*b))

    def get_y_rotation(self,x,y,z):
        radians = math.atan2(x, dist(y,z))
        return -math.degrees(radians)

    def get_x_rotation(self,x,y,z):
        radians = math.atan2(y, dist(x,z))
        return math.degrees(radians)


    def set_range_get_scale(self,register, range):
        # Range
        ACCEL_CONFIG_REGISTER = 0x1C
        ACCEL_2G = 0x00
        ACCEL_4G = 0x08
        ACCEL_8G = 0x10
        ACCEL_16G = 0x18
        # Scale Modifiers
        ACCEL_SCALE_MODIFIER_2G = 16384.0
        ACCEL_SCALE_MODIFIER_4G = 8192.0
        ACCEL_SCALE_MODIFIER_8G = 4096.0
        ACCEL_SCALE_MODIFIER_16G = 2048.0
        # Range
        GYRO_CONFIG_REGISTER = 0x1B
        GYRO_250DEG = 0x00
        GYRO_500DEG = 0x08
        GYRO_1000DEG = 0x10
        GYRO_2000DEG = 0x18
        # Scale Modifiers
        GYRO_SCALE_MODIFIER_250DEG = 131.0
        GYRO_SCALE_MODIFIER_500DEG = 65.5
        GYRO_SCALE_MODIFIER_1000DEG = 32.8
        GYRO_SCALE_MODIFIER_2000DEG = 16.4
        
        self.bus.write_byte_data(self.address, register, range) # set new range
        time.sleep(0.5)
        current_range = self.bus.read_byte_data(self.address, register) # read current set range
        if register == ACCEL_CONFIG_REGISTER:
            if current_range == ACCEL_2G:
                return ACCEL_SCALE_MODIFIER_2G
            elif current_range == ACCEL_4G:
                return ACCEL_SCALE_MODIFIER_4G
            elif current_range == ACCEL_8G:
                return ACCEL_SCALE_MODIFIER_8G
            else: #current_range == ACCEL_16G
                return ACCEL_SCALE_MODIFIER_16G
        if register == GYRO_CONFIG_REGISTER:
            if current_range == GYRO_250DEG:
                return GYRO_SCALE_MODIFIER_250DEG
            elif current_range == GYRO_500DEG:
                return GYRO_SCALE_MODIFIER_500DEG
            elif current_range == GYRO_1000DEG:
                return GYRO_SCALE_MODIFIER_1000DEG
            else: #current_range == GYRO_2000DEG:
                return GYRO_SCALE_MODIFIER_2000DEG  
     

    def IMU_set_par(self,acc_cfg_reg,acc_val,gyro_cfg_reg,gyro_val):
        
        # Power management registers
        power_mgmt_1 = 0x6b
        power_mgmt_2 = 0x6c
        # Now wake the 6050 up as it starts in sleep mode
        self.bus.write_byte_data(self.address, power_mgmt_1, 0)
        
        #Set configuration parameters for
        acc_scale = self.set_range_get_scale(acc_cfg_reg,acc_val)
        current_range=self.bus.read_byte_data(self.address, acc_cfg_reg)
        print("ACCEL_CONFIG_REGISTER:",current_range)
        gyro_scale = self.set_range_get_scale(gyro_cfg_reg,gyro_val)
        current_range=self.bus.read_byte_data(self.address, gyro_cfg_reg)
        print("GYRO_CONFIG_REGISTER:",current_range)



    def IMU_calibration(self,time):
        
        gyro_x_cal = 0
        gyro_y_cal = 0
        gyro_z_cal = 0

        accel_x_cal = 0
        accel_y_cal = 0
        accel_z_cal = 0
        
        accel_total_avg = 0
        
        global_gyro = np.zeros(shape=(0,3))
        global_accel = np.zeros(shape=(0,3))

        # store gyro signal in csv file and export befor to take average
#        start=time.process_time()
        for i in range(time):        
            gyro_xout = self.read_word_2c(0x43)
            gyro_yout = self.read_word_2c(0x45)
            gyro_zout = self.read_word_2c(0x47)
            
            gyro_x_cal += gyro_xout
            gyro_y_cal += gyro_yout
            gyro_z_cal += gyro_zout
            
            ##these tw0 next lines of code are used to save data on Raspberry PI
#            global_gyro = np.vstack((global_gyro,np.array([gyro_xout, gyro_yout, gyro_zout])))            
#        np.savetxt("global_gyro.csv", global_gyro, delimiter=",")

        # Divide the gyro_cal variable with 5000 to get the average offset
        gyro_x_cal /=  time
        gyro_y_cal /= time
        gyro_z_cal /= time

        print("gyro_x_cal:",gyro_x_cal)
        print("gyro_y_cal:",gyro_y_cal)
        print("gyro_z_cal:",gyro_z_cal)

#        print("process time:",time.process_time() - start)

        # average from acc
        # store acc signal in csv file and export befor to take average
#        start=time.process_time()
        for i in range(time):        
            accel_xout = self.read_word_2c(0x3b)
            accel_yout = self.read_word_2c(0x3d)
            accel_zout = self.read_word_2c(0x3f)
            
            accel_x_cal += accel_xout
            accel_y_cal += accel_yout
            accel_z_cal += accel_zout
            
            accel_total_avg += math.sqrt((accel_xout*accel_xout)+(accel_yout*accel_yout)+(accel_zout*accel_zout));  #Calculate the total accelerometer vector
            
            ##these tw0 next lines of code are used to save data on Raspberry PI
#            global_accel = np.vstack((global_accel,np.array([accel_xout, accel_yout, accel_zout])))            
#        np.savetxt("global_acc.csv", global_accel, delimiter=",")

        # Divide the accel_cal variable with 5000 to get the average offset
        accel_x_cal /=  time
        accel_y_cal /= time
        accel_z_cal /= time
        accel_z_cal = 16384 - accel_z_cal
        
        accel_total_avg /= time

        print("accel_x_cal:",accel_x_cal)
        print("accel_y_cal:",accel_y_cal)
        print("accel_z_cal:",accel_z_cal)
        print("accel_total_avg:",accel_total_avg)

        return (gyro_x_cal,gyro_y_cal,gyro_z_cal,accel_x_cal,accel_y_cal,accel_z_cal,accel_total_avg)
#        print("process time:",time.process_time() - start)
