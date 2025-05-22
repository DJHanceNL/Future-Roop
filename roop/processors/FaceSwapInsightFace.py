import roop.globals
import cv2
import numpy as np
import onnx
import onnxruntime

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path

class FaceSwapInsightFace():
    plugin_options:dict = None
    model_swap_insightface = None

    processorname = 'faceswap'
    type = 'swap'

    # Static detection cache
    detection_cache = {
        'last_frame': None,
        'last_detection': None
    }

    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_swap_insightface is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            graph = onnx.load(model_path).graph
            self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            self.input_mean = 0.0
            self.input_std = 255.0
            sess_options = onnxruntime.SessionOptions()
            sess_options.enable_cpu_mem_arena = False
            self.model_swap_insightface = onnxruntime.InferenceSession(model_path, sess_options, providers=roop.globals.execution_providers)

    def detect_faces_with_cache(self, frame, real_detect_func):
        # Convert to grayscale for quick compare
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        reuse_detection = False
        if self.detection_cache['last_frame'] is not None:
            diff = np.abs(gray.astype(np.float32) - self.detection_cache['last_frame'].astype(np.float32))
            if np.mean(diff) < 5.0:
                reuse_detection = True
        if reuse_detection:
            return self.detection_cache['last_detection']
        else:
            faces = real_detect_func(frame)
            self.detection_cache['last_frame'] = gray
            self.detection_cache['last_detection'] = faces
            return faces

    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        latent = source_face.normed_embedding.reshape((1,-1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        io_binding = self.model_swap_insightface.io_binding()           
        io_binding.bind_cpu_input("target", temp_frame)
        io_binding.bind_cpu_input("source", latent)
        io_binding.bind_output("output", self.devicename)
        self.model_swap_insightface.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()[0]
        return ort_outs[0]

    def Release(self):
        del self.model_swap_insightface
        self.model_swap_insightface = None
