class ProcessOptions:

    def __init__(self, swap_model, processordefines:dict, face_distance,  blend_ratio, swap_mode, selected_index, masking_text, imagemask, num_steps, subsample_size, show_face_area, restore_original_mouth, show_mask=False):
        if swap_model is not None: 
            self.swap_modelname = swap_model
            self.swap_output_size = int(swap_model.split()[-1])
            # Model metadata for hyperswap support
            if swap_model == "Hyperswap 256":
                self.model_type = 'hyperswap'
                self.model_mean = [0.5, 0.5, 0.5]
                self.model_std = [0.5, 0.5, 0.5]
                self.model_precision = 'fp16'
            elif swap_model == "inswapper 128 fp16":
                self.model_type = 'inswapper'
                self.model_mean = [0.0, 0.0, 0.0]
                self.model_std = [1.0, 1.0, 1.0]
                self.model_precision = 'fp16'
            else:
                self.model_type = 'inswapper'
                self.model_mean = [0.0, 0.0, 0.0]
                self.model_std = [1.0, 1.0, 1.0]
                self.model_precision = 'fp32'
        else:
            self.swap_output_size = 128
            self.model_type = 'inswapper'
            self.model_mean = [0.0, 0.0, 0.0]
            self.model_std = [1.0, 1.0, 1.0]
            self.model_precision = 'fp32'
        self.processors = processordefines
        self.face_distance_threshold = face_distance
        self.blend_ratio = blend_ratio
        self.swap_mode = swap_mode
        self.selected_index = selected_index
        self.masking_text = masking_text
        self.imagemask = imagemask
        self.num_swap_steps = num_steps
        self.show_face_area_overlay = show_face_area
        self.show_face_masking = show_mask
        self.subsample_size = subsample_size
        self.restore_original_mouth = restore_original_mouth
        self.max_num_reuse_frame = 15