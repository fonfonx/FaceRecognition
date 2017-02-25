# RSC_python

This is the source code of my article

Fontaine, X., Achanta, R., & Süsstrunk, S. (2017). Face Recognition in Real-world Images. In *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* (No. EPFL-CONF-224338).

It is inspired from the RSC algorithm [[1](#rsc)].

## Installation

+ Install some Python packages:
```
pip install numpy scipy Pillow l1ls sklearn matplotlib
```

+ You will need OpenCV for Python. One installation solution is explained in this [link](https://junise.wordpress.com/2015/05/18/how-to-install-opencv-2-4-10-in-ubuntu-14-04-lts/) with the following `cmake` command:
```
cmake -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON  -D WITH_OPENGL=ON -D WITH_VTK=ON -D WITH_GTK=ON -D WITH_CUDA=OFF ..
```

+ The following dependencies may be missing:
```
sudo apt-get install libopenblas-dev
```

+ Download Boost from the [Boost official website](http://boost.org) and go to the boost directory and run the following commands:
```
./bootstrap.sh --with-libraries=python
./b2
sudo ./b2 install
```

+ Then download DLIB:
```
git clone https://github.com/davisking/dlib.git
git checkout tags/v19.0 # current version did not work for me
sudo python setup.py install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```

+ Change the path to `haarcascade_frontalface_default.xml` and to the `shape_predictor_68_face_landmarks.dat` file in the `config.py` file.

## Run the Face Recognizer

Run `python recognizer.py` to run the code of the paper. You can change the parameters (number of training images, etc) in the file `config.py`. It should be the only file you have to modify.

## References

[<a name="#rsc">1</a>] Yang, M., Zhang, L., Yang, J., & Zhang, D. (2011, June). Robust sparse coding for face recognition. In *Computer Vision and Pattern Recognition (CVPR), 2011 IEEE Conference on* (pp. 625-632). IEEE.
