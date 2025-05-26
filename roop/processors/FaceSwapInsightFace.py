import roop.globals
import cv2
import numpy as np
import onnx
import onnxruntime
import hashlib
from collections import OrderedDict
import os

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path

class FaceSwapInsightFace():
    plugin_options:dict = None
    model_swap_insightface = None

    processorname = 'faceswap'
    type = 'swap'

    # Aggressive LRU cache for face detection
    detection_cache = OrderedDict()
    CACHE_SIZE = 10
    last_face_result = None
    last_frame_hash = None
    SIMILARITY_THRESHOLD = 2.0  # Lower = more aggressive reuse

    # --- PERFORMANCE: Preload and pin model weights for CUDA ---
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
            # CUDA: Pin memory and optimize graph for inference
            if 'CUDAExecutionProvider' in roop.globals.execution_providers:
                sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.enable_mem_pattern = True
                sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
            self.model_swap_insightface = onnxruntime.InferenceSession(model_path, sess_options, providers=roop.globals.execution_providers)
        # --- END PERFORMANCE ---

    # --- PERFORMANCE: Async disk I/O for saving frames (video) ---
    def save_frame_async(self, frame, path):
        import threading, cv2
        def _save():
            cv2.imwrite(path, frame)
        t = threading.Thread(target=_save)
        t.daemon = True
        t.start()
    # --- END PERFORMANCE ---

    def detect_faces_with_cache(self, frame, real_detect_func):
        try:
            # Use hash for robust caching
            frame_bytes = frame.tobytes()
            frame_hash = hashlib.md5(frame_bytes).hexdigest()
            # Frame similarity skipping: if very similar to last, reuse
            if self.last_frame_hash is not None:
                diff = np.mean(np.abs(np.frombuffer(frame_bytes, dtype=np.uint8) - np.frombuffer(self.last_frame_hash, dtype=np.uint8)))
                if diff < self.SIMILARITY_THRESHOLD and self.last_face_result is not None:
                    return self.last_face_result
            if frame_hash in self.detection_cache:
                result = self.detection_cache[frame_hash]
            else:
                result = real_detect_func(frame)
                self.detection_cache[frame_hash] = result
                if len(self.detection_cache) > self.CACHE_SIZE:
                    self.detection_cache.popitem(last=False)
            self.last_frame_hash = frame_bytes
            self.last_face_result = result
            return result
        except Exception:
            # Fallback to original logic
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            reuse_detection = False
            if hasattr(self, 'detection_cache') and getattr(self, 'detection_cache', None) and 'last_frame' in self.detection_cache:
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

    # --- PERFORMANCE: Add batch size tuning for CUDA ---
    def get_optimal_batch_size(self):
        # You can tune this based on your GPU, or make it configurable
        if 'CUDAExecutionProvider' in roop.globals.execution_providers:
            return 8  # Try 8 for modern GPUs, or make this user-configurable
        return 2

    # --- PERFORMANCE: Memory reuse for temp arrays (buffer pooling) ---
    _latents_buffer = None
    _frames_buffer = None
    def get_latents_buffer(self, batch_size, latent_dim):
        if self._latents_buffer is None or self._latents_buffer.shape != (batch_size, latent_dim):
            self._latents_buffer = np.empty((batch_size, latent_dim), dtype=np.float32)
        return self._latents_buffer
    def get_frames_buffer(self, batch_size, h, w, c):
        if self._frames_buffer is None or self._frames_buffer.shape != (batch_size, h, w, c):
            self._frames_buffer = np.empty((batch_size, h, w, c), dtype=np.float32)
        return self._frames_buffer

    def RunBatch(self, source_faces, target_faces, temp_frames):
        try:
            batch_size = self.get_optimal_batch_size()
            latent_dim = source_faces[0].normed_embedding.size
            h, w, c = temp_frames[0].shape
            results = []
            for i in range(0, len(source_faces), batch_size):
                batch_sf = source_faces[i:i+batch_size]
                batch_tf = target_faces[i:i+batch_size]
                batch_frames = temp_frames[i:i+batch_size]
                latents = self.get_latents_buffer(len(batch_sf), latent_dim)
                for j, sf in enumerate(batch_sf):
                    latents[j] = np.dot(sf.normed_embedding.reshape((1, -1)), self.emap)
                latents = np.ascontiguousarray(latents)
                latents /= np.linalg.norm(latents, axis=1, keepdims=True)
                frames_buf = self.get_frames_buffer(len(batch_frames), h, w, c)
                for j, frame in enumerate(batch_frames):
                    frames_buf[j] = frame
                temp_frames_batch = np.ascontiguousarray(frames_buf)
                io_binding = self.model_swap_insightface.io_binding()
                io_binding.bind_cpu_input("target", temp_frames_batch)
                io_binding.bind_cpu_input("source", latents)
                io_binding.bind_output("output", self.devicename)
                self.model_swap_insightface.run_with_iobinding(io_binding)
                ort_outs = io_binding.copy_outputs_to_cpu()[0]
                results.extend([out for out in ort_outs])
            return results
        except Exception:
            # Fallback to sequential if batch fails
            results = []
            for sf, tf, frame in zip(source_faces, target_faces, temp_frames):
                try:
                    results.append(self.Run(sf, tf, frame))
                except Exception:
                    # Fallback to original logic if even single frame fails
                    latent = sf.normed_embedding.reshape((1,-1))
                    latent = np.dot(latent, self.emap)
                    latent /= np.linalg.norm(latent)
                    io_binding = self.model_swap_insightface.io_binding()
                    io_binding.bind_cpu_input("target", frame)
                    io_binding.bind_cpu_input("source", latent)
                    io_binding.bind_output("output", self.devicename)
                    self.model_swap_insightface.run_with_iobinding(io_binding)
                    ort_outs = io_binding.copy_outputs_to_cpu()[0]
                    results.append(ort_outs[0])
            return results

    # --- PERFORMANCE: Skip unchanged frames in video (frame similarity) ---
    def should_skip_frame(self, frame, prev_frame, threshold=1.0):
        if prev_frame is None:
            return False
        diff = np.mean(np.abs(frame.astype(np.float32) - prev_frame.astype(np.float32)))
        return diff < threshold

    def Release(self):
        del self.model_swap_insightface
        self.model_swap_insightface = None
        # Aggressive resource cleanup
        import gc
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()
