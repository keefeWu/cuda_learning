#coding:utf-8
import os
import sys
if len(sys.argv) < 2:
    print('please input a project name!')
    exit(0)
project = sys.argv[1]
os.mkdir(project)
os.mkdir(os.path.join(project,'build'))
file = open(os.path.join(project, 'CMakeLists.txt'), 'w')
file.write('CMAKE_MINIMUM_REQUIRED(VERSION 2.8)\n')
file.write('PROJECT(%s)\n'%project)
file.write('FIND_PACKAGE(CUDA REQUIRED)\n')
file.write('CUDA_ADD_EXECUTABLE(%s main.cu)\n'%project)
file.write('TARGET_LINK_LIBRARIES(%s)'%project)
file.close()