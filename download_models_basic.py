#!/usr/bin/env python

from __future__ import print_function
import hashlib
import os
import sys
import tarfile
import requests

if sys.version_info[0] < 3:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen


class Model:
    MB = 1024*1024
    BUFSIZE = 10*MB

    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.url = kwargs.pop('url', None)
        self.downloader = kwargs.pop('downloader', None)
        self.filename = kwargs.pop('filename')
        self.sha = kwargs.pop('sha', None)
        self.archive = kwargs.pop('archive', None)
        self.member = kwargs.pop('member', None)

    def __str__(self):
        return 'Model <{}>'.format(self.name)

    def printRequest(self, r):
        def getMB(r):
            d = dict(r.info())
            for c in ['content-length', 'Content-Length']:
                if c in d:
                    return int(d[c]) / self.MB
            return '<unknown>'
        print('  {} {} [{} Mb]'.format(r.getcode(), r.msg, getMB(r)))

    def verify(self):
        if not self.sha:
            return False
        print('  expect {}'.format(self.sha))
        sha = hashlib.sha1()
        try:
            with open(self.filename, 'rb') as f:
                while True:
                    buf = f.read(self.BUFSIZE)
                    if not buf:
                        break
                    sha.update(buf)
            print('  actual {}'.format(sha.hexdigest()))
            self.sha_actual = sha.hexdigest()
            return self.sha == self.sha_actual
        except Exception as e:
            print('  catch {}'.format(e))

    def get(self):
        if self.verify():
            print('  hash match - skipping')
            return True

        basedir = os.path.dirname(self.filename)
        if basedir and not os.path.exists(basedir):
            print('  creating directory: ' + basedir)
            os.makedirs(basedir, exist_ok=True)

        if self.archive or self.member:
            assert(self.archive and self.member)
            print('  hash check failed - extracting')
            print('  get {}'.format(self.member))
            self.extract()
        elif self.url:
            print('  hash check failed - downloading')
            print('  get {}'.format(self.url))
            self.download()
        else:
            assert self.downloader
            print('  hash check failed - downloading')
            sz = self.downloader(self.filename)
            print('  size = %.2f Mb' % (sz / (1024.0 * 1024)))

        print(' done')
        print(' file {}'.format(self.filename))
        candidate_verify = self.verify()
        if not candidate_verify:
            self.handle_bad_download()
        return candidate_verify

    def download(self):
        try:
            r = urlopen(self.url, timeout=60)
            self.printRequest(r)
            self.save(r)
        except Exception as e:
            print('  catch {}'.format(e))

    def extract(self):
        try:
            with tarfile.open(self.archive) as f:
                assert self.member in f.getnames()
                self.save(f.extractfile(self.member))
        except Exception as e:
            print('  catch {}'.format(e))

    def save(self, r):
        with open(self.filename, 'wb') as f:
            print('  progress ', end='')
            sys.stdout.flush()
            while True:
                buf = r.read(self.BUFSIZE)
                if not buf:
                    break
                f.write(buf)
                print('>', end='')
                sys.stdout.flush()

    def handle_bad_download(self):
        if os.path.exists(self.filename):
            # rename file for further investigation
            try:
                # NB: using `self.sha_actual` may create unbounded number of files
                rename_target = self.filename + '.invalid'
                # TODO: use os.replace (Python 3.3+)
                try:
                    if os.path.exists(rename_target):  # avoid FileExistsError on Windows from os.rename()
                        os.remove(rename_target)
                finally:
                    os.rename(self.filename, rename_target)
                    print('  renaming invalid file to ' + rename_target)
            except:
                import traceback
                traceback.print_exc()
            finally:
                if os.path.exists(self.filename):
                    print('  deleting invalid file')
                    os.remove(self.filename)


def GDrive(gid):
    def download_gdrive(dst):
        session = requests.Session()  # re-use cookies

        URL = "https://docs.google.com/uc?export=download"
        response = session.get(URL, params = { 'id' : gid }, stream = True)

        def get_confirm_token(response):  # in case of large files
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value
            return None
        token = get_confirm_token(response)

        if token:
            params = { 'id' : gid, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)

        BUFSIZE = 1024 * 1024
        PROGRESS_SIZE = 10 * 1024 * 1024

        sz = 0
        progress_sz = PROGRESS_SIZE
        with open(dst, "wb") as f:
            for chunk in response.iter_content(BUFSIZE):
                if not chunk:
                    continue  # keep-alive

                f.write(chunk)
                sz += len(chunk)
                if sz >= progress_sz:
                    progress_sz += PROGRESS_SIZE
                    print('>', end='')
                    sys.stdout.flush()
        print('')
        return sz
    return download_gdrive


models = [
    Model(
        name='TinyYolov2 (ONNX)',
        url='https://www.cntk.ai/OnnxModels/tiny_yolov2/opset_1/tiny_yolov2.tar.gz',
        sha='b9102abb8fa6f51368119b52146c30189353164a',
        filename='tiny_yolov2.tar.gz'),
    Model(
        name='TinyYolov2 (ONNX)',
        archive='tiny_yolov2.tar.gz',
        member='tiny_yolov2/model.onnx',
        sha='433fecbd32ac8b9be6f5ee10c39dcecf9dc5c151',
        filename='onnx/models/tiny_yolo2.onnx'),
    Model(
        name='TinyYolov2 (ONNX)',
        archive='tiny_yolov2.tar.gz',
        member='tiny_yolov2/test_data_set_0/input_0.pb',
        sha='a0412fde98ca21d726c0c86ef007c11aa4678e3c',
        filename='onnx/data/input_tiny_yolo2.pb'),
    Model(
        name='TinyYolov2 (ONNX)',
        archive='tiny_yolov2.tar.gz',
        member='tiny_yolov2/test_data_set_0/output_0.pb',
        sha='f9be0446cac76fe38bb23cb09ed23c317907f505',
        filename='onnx/data/output_tiny_yolo2.pb'),
    Model(
        name='Emotion FERPlus (ONNX)',
        url='https://www.cntk.ai/OnnxModels/emotion_ferplus/opset_7/emotion_ferplus.tar.gz',
        sha='9ff80899c0cd468999db5d8ffde98780ef85455e',
        filename='emotion_ferplus.tar.gz'),
    Model(
        name='Emotion FERPlus (ONNX)',
        archive='emotion_ferplus.tar.gz',
        member='emotion_ferplus/model.onnx',
        sha='2ef5b3a6404a5feb8cc396d66c86838c4c750a7e',
        filename='onnx/models/emotion_ferplus.onnx'),
    Model(
        name='Emotion FERPlus (ONNX)',
        archive='emotion_ferplus.tar.gz',
        member='emotion_ferplus/test_data_set_0/input_0.pb',
        sha='29621536528116fc12f02bc81c7265f7ffe7c8bb',
        filename='onnx/data/input_emotion_ferplus.pb'),
    Model(
        name='Emotion FERPlus (ONNX)',
        archive='emotion_ferplus.tar.gz',
        member='emotion_ferplus/test_data_set_0/output_0.pb',
        sha='54f7892240d2d9298f5a8064a46fc3a8987015a5',
        filename='onnx/data/output_emotion_ferplus.pb'),
    Model(
        name='Squeezenet (ONNX)',
        url='https://s3.amazonaws.com/download.onnx/models/opset_8/squeezenet.tar.gz',
        sha='57348321d4d460c07c41af814def3abe728b3a03',
        filename='squeezenet.tar.gz'),
    Model(
        name='Squeezenet (ONNX)',
        archive='squeezenet.tar.gz',
        member='squeezenet/model.onnx',
        sha='c3f272e672fa64a75fb4a2e48dd2ca25fcc76c49',
        filename='onnx/models/squeezenet.onnx'),
    Model(
        name='YOLOv4',  # https://github.com/opencv/opencv/issues/17148
        downloader=GDrive('1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT'),
        sha='0143deb6c46fcc7f74dd35bf3c14edc3784e99ee',
        filename='yolov4.weights'),
    Model(
        name='YOLOv4-tiny',  # https://github.com/opencv/opencv/issues/17148
        url='https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights',
        sha='d110379b7b86899226b591ad4affc7115f707157',
        filename='yolov4-tiny.weights'),
    Model(
        name='YOLOv4x-mish',  # https://github.com/opencv/opencv/issues/18975
        url='https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4x-mish.weights',
        sha='a6f2879af2241de2e9730d317a55db6afd0af00b',
        filename='yolov4x-mish.weights'),
    Model(
        name='YuNet',
        url='https://github.com/ShiqiYu/libfacedetection.train/raw/7a9738d6ca7bc4a3216578b06a739126435d40ef/tasks/task1/onnx/yunet.onnx',
        sha='49c52f484b1895e8298dc59e37f262ba7841a601',
        filename='onnx/models/yunet-202109.onnx'),
    Model(
        name='face_recognizer_fast',
        url='https://drive.google.com/uc?export=dowload&id=1ClK9WiB492c5OZFKveF3XiHCejoOxINW',
        sha='12ff8b1f5c8bff62e8dd91eabdacdfc998be255e',
        filename='onnx/models/face_recognizer_fast.onnx'),
]

# Note: models will be downloaded to current working directory
#       expected working directory is <testdata>/dnn
if __name__ == '__main__':

    selected_model_name = None
    if len(sys.argv) > 1:
        selected_model_name = sys.argv[1]
        print('Model: ' + selected_model_name)

    failedModels = []
    for m in models:
        print(m)
        if selected_model_name is not None and not m.name.startswith(selected_model_name):
            continue
        if not m.get():
            failedModels.append(m.filename)

    if failedModels:
        print("Following models have not been downloaded:")
        for f in failedModels:
            print("* {}".format(f))
        exit(15)
