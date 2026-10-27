import os
import shutil
import pathlib
import gradio as gr
import roop.utilities as util
import roop.globals
import ui.globals
from roop.face_util import extract_face_images, create_blank_image
from roop.capturer import get_video_frame, get_video_frame_total, get_image_frame
from roop.ProcessEntry import ProcessEntry
from roop.ProcessOptions import ProcessOptions
from roop.FaceSet import FaceSet
from roop.utilities import clean_dir

last_image = None


IS_INPUT = True
SELECTED_FACE_INDEX = 0

SELECTED_INPUT_FACE_INDEX = 0
SELECTED_TARGET_FACE_INDEX = 0

input_faces = None
target_faces = None
face_selection = None
previewimage = None

selected_preview_index = 0

is_processing = False            

list_files_process : list[ProcessEntry] = []
model_swap_choices = ["InSwapper 128", "ReSwapper 128", "ReSwapper 256" , "inswapper_128_fp16" , "Hyperswap 256"]

no_face_choices = ["Use untouched original frame","Retry rotated", "Skip Frame", "Skip Frame if no similar face", "Use last swapped"]
swap_choices = ["First found", "All input faces", "All input faces (random)", "All female", "All male", "All faces", "Selected face"]

current_video_fps = 50

manual_masking = False


def faceswap_tab():
    global no_face_choices, previewimage

    with gr.Tab("🎭 Face Swap"):
        with gr.Row(variant='panel'):
            with gr.Column(scale=2):
                with gr.Row():
                    input_faces = gr.Gallery(label="Input faces gallery", allow_preview=False, preview=False, height=138, columns=64, object_fit="scale-down", interactive=False)
                    target_faces = gr.Gallery(label="Target faces gallery", allow_preview=False, preview=False, height=138, columns=64, object_fit="scale-down", interactive=False)
                with gr.Row():
                    bt_move_left_input = gr.Button("⬅ Move left", size='sm')
                    bt_move_right_input = gr.Button("➡ Move right", size='sm')
                    bt_move_left_target = gr.Button("⬅ Move left", size='sm')
                    bt_move_right_target = gr.Button("➡ Move right", size='sm')
                with gr.Row():
                    bt_remove_selected_input_face = gr.Button("❌ Remove selected", size='sm')
                    bt_clear_input_faces = gr.Button("💥 Clear all", variant='stop', size='sm')
                    bt_remove_selected_target_face = gr.Button("❌ Remove selected", size='sm')
                    bt_add_local = gr.Button('Add local files from', size='sm')

                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Accordion(label="Advanced Masking", open=False):
                            chk_showmaskoffsets = gr.Checkbox(
                                label="Show mask overlay in preview",
                                value=False,
                                interactive=True,
                            )
                            chk_restoreoriginalmouth = gr.Checkbox(
                                label="Restore original mouth area",
                                value=False,
                                interactive=True,
                            )
                            mask_top = gr.Slider(
                                0,
                                1.0,
                                value=0,
                                label="Offset Face Top",
                                step=0.01,
                                interactive=True,
                            )
                            mask_bottom = gr.Slider(
                                0,
                                1.0,
                                value=0,
                                label="Offset Face Bottom",
                                step=0.01,
                                interactive=True,
                            )
                            mask_left = gr.Slider(
                                0,
                                1.0,
                                value=0,
                                label="Offset Face Left",
                                step=0.01,
                                interactive=True,
                            )
                            mask_right = gr.Slider(
                                0,
                                1.0,
                                value=0,
                                label="Offset Face Right",
                                step=0.01,
                                interactive=True,
                            )
                            mask_erosion = gr.Slider(
                                1.0,
                                3.0,
                                value=1.0,
                                label="Erosion Iterations",
                                step=1.00,
                                interactive=True,
                            )
                            mask_blur = gr.Slider(
                                10.0,
                                50.0,
                                value=20.0,
                                label="Blur size",
                                step=1.00,
                                interactive=True,
                            )
                            bt_toggle_masking = gr.Button(
                                "Toggle manual masking", variant="secondary", size="sm"
                            )
                            selected_mask_engine = gr.Dropdown(
                                ["None", "Clip2Seg", "DFL XSeg"],
                                value="None",
                                label="Face masking engine",
                            )
                            clip_text = gr.Textbox(
                                label="List of objects to mask and restore back on fake face",
                                value="cup,hands,hair,banana",
                                interactive=False,
                            )
                            bt_preview_mask = gr.Button(
                                "👥 Show Mask Preview", variant="secondary"
                            )
                    with gr.Column(scale=2):
                        local_folder = gr.Textbox(show_label=False, placeholder="/content/", interactive=True)
                with gr.Row(variant='panel'):
                    bt_srcfiles = gr.Files(label='Source Images or Facesets', file_count="multiple", file_types=["image", ".fsz", ".webp"], elem_id='filelist', height=233)
                    bt_destfiles = gr.Files(label='Target File(s)', file_count="multiple", file_types=["image", "video", ".webp"], elem_id='filelist', height=233)
                with gr.Row(variant='panel'):
                    ui.globals.ui_selected_swap_model = gr.Dropdown(model_swap_choices, value=model_swap_choices[0], label="Specify Face Swap Model")
                    forced_fps = gr.Slider(minimum=0, maximum=120, value=0, label="Video FPS", info='Overrides detected fps if not 0', step=1.0, interactive=True, container=True)

            with gr.Column(scale=2):
                previewimage = gr.Image(label="Preview Image", height=576, interactive=False, visible=True, format=get_gradio_output_format())
                maskimage = gr.ImageEditor(label="Manual mask Image", sources=["clipboard"], transforms="", type="numpy",
                                             brush=gr.Brush(color_mode="fixed", colors=["rgba(255, 255, 255, 1"]), interactive=True, visible=False)
                with gr.Row(variant='panel'):
                    fake_preview = gr.Checkbox(label="Face swap frames", value=False)
                    bt_refresh_preview = gr.Button("🔄 Refresh", variant='secondary', size='sm')
                    bt_use_face_from_preview = gr.Button("Use Face from this Frame", variant='primary', size='sm')
                with gr.Row():
                    preview_frame_num = gr.Slider(1, 1, value=1, label="Frame Number", info='0:00:00', step=1.0, interactive=True)
                with gr.Row():
                    text_frame_clip = gr.Markdown('Processing frame range [0 - 0]')
                    set_frame_start = gr.Button("⬅ Set as Start", size='sm')
                    set_frame_end = gr.Button("➡ Set as End", size='sm')
        with gr.Row(visible=False) as dynamic_face_selection:
            with gr.Column(scale=2):
                face_selection = gr.Gallery(label="Detected faces", allow_preview=False, preview=False, height=138, object_fit="cover", columns=32)
            with gr.Column():
                bt_faceselect = gr.Button("☑ Use selected face", size='sm')
                bt_cancelfaceselect = gr.Button("Done", size='sm')
            with gr.Column():
                gr.Markdown(' ') 

        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                selected_face_detection = gr.Dropdown(swap_choices, value="First found", label="Specify face selection for swapping")
            with gr.Column(scale=1):
                num_swap_steps = gr.Slider(1, 5, value=1, step=1.0, label="Number of swapping steps", info="More steps may increase likeness")
            with gr.Column(scale=2):
                ui.globals.ui_selected_enhancer = gr.Dropdown(["None", "Codeformer", "DMDNet", "GFPGAN", "GPEN", "Restoreformer++"], value="None", label="Select post-processing")

        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                max_face_distance = gr.Slider(0.01, 1.0, value=0.65, label="Max Face Similarity Threshold", info="0.0 = identical 1.0 = no similarity")
            with gr.Column(scale=1):
                ui.globals.ui_upscale = gr.Dropdown(["128px", "256px", "512px"], value="128px", label="Subsample upscale to", interactive=True)
            with gr.Column(scale=2):
                ui.globals.ui_blend_ratio = gr.Slider(0.0, 1.0, value=0.65, label="Original/Enhanced image blend ratio", info="Only used with active post-processing")

        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                video_swapping_method = gr.Dropdown(["Extract Frames to media","In-Memory processing"], value="In-Memory processing", label="Select video processing method", interactive=True)
                no_face_action = gr.Dropdown(choices=no_face_choices, value=no_face_choices[0], label="Action on no face detected", interactive=True)
                vr_mode = gr.Checkbox(label="VR Mode", value=False)
            with gr.Column(scale=1):
                with gr.Group():
                    autorotate = gr.Checkbox(label="Auto rotate horizontal Faces", value=True)
                    roop.globals.skip_audio = gr.Checkbox(label="Skip audio", value=False)
                    roop.globals.keep_frames = gr.Checkbox(label="Keep Frames (relevant only when extracting frames)", value=False)
                    roop.globals.wait_after_extraction = gr.Checkbox(label="Wait for user key press before creating video ", value=False)

        with gr.Row(variant='panel'):
            with gr.Column():
                bt_start = gr.Button("▶ Start", variant='primary')
            with gr.Column():
                bt_stop = gr.Button("⏹ Stop", variant='secondary', interactive=False)
                gr.Button("👀 Open Output Folder", size='sm').click(fn=lambda: util.open_folder(roop.globals.output_path))
            with gr.Column(scale=2):
                output_method = gr.Dropdown(["File","Virtual Camera", "Both"], value="File", label="Select Output Method", interactive=True)
        with gr.Row(variant='panel'):
            with gr.Column():
                resultfiles = gr.Files(label='Processed File(s)', interactive=False)
            with gr.Column():
                resultimage = gr.Image(type='filepath', label='Final Image', interactive=False )
                resultvideo = gr.Video(label='Final Video', interactive=False, visible=False)

    previewinputs = [ui.globals.ui_selected_swap_model, preview_frame_num, bt_destfiles, fake_preview, ui.globals.ui_selected_enhancer, selected_face_detection,
                        max_face_distance, ui.globals.ui_blend_ratio, selected_mask_engine, clip_text, no_face_action, vr_mode, autorotate, maskimage, chk_showmaskoffsets, chk_restoreoriginalmouth, num_swap_steps, ui.globals.ui_upscale]
    previewoutputs = [previewimage, maskimage, preview_frame_num] 
    input_faces.select(on_select_input_face, None, None).success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs)
    
    bt_move_left_input.click(fn=move_selected_input, inputs=[bt_move_left_input], outputs=[input_faces])
    bt_move_right_input.click(fn=move_selected_input, inputs=[bt_move_right_input], outputs=[input_faces])
    bt_move_left_target.click(fn=move_selected_target, inputs=[bt_move_left_target], outputs=[target_faces])
    bt_move_right_target.click(fn=move_selected_target, inputs=[bt_move_right_target], outputs=[target_faces])

    bt_remove_selected_input_face.click(fn=remove_selected_input_face, outputs=[input_faces])
    bt_srcfiles.change(fn=on_srcfile_changed, show_progress='full', inputs=bt_srcfiles, outputs=[dynamic_face_selection, face_selection, input_faces, bt_srcfiles])

    mask_top.release(fn=on_mask_top_changed, inputs=[mask_top], show_progress='hidden')
    mask_bottom.release(fn=on_mask_bottom_changed, inputs=[mask_bottom], show_progress='hidden')
    mask_left.release(fn=on_mask_left_changed, inputs=[mask_left], show_progress='hidden')
    mask_right.release(fn=on_mask_right_changed, inputs=[mask_right], show_progress='hidden')
    mask_erosion.release(fn=on_mask_erosion_changed, inputs=[mask_erosion], show_progress='hidden')
    mask_blur.release(fn=on_mask_blur_changed, inputs=[mask_blur], show_progress='hidden')
    selected_mask_engine.change(fn=on_mask_engine_changed, inputs=[selected_mask_engine], outputs=[clip_text], show_progress='hidden')

    target_faces.select(on_select_target_face, None, None)
    bt_remove_selected_target_face.click(fn=remove_selected_target_face, outputs=[target_faces])

    forced_fps.change(fn=on_fps_changed, inputs=[forced_fps], show_progress='hidden')
    bt_destfiles.change(fn=on_destfiles_changed, inputs=[bt_destfiles], outputs=[preview_frame_num, text_frame_clip], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    bt_destfiles.select(fn=on_destfiles_selected, outputs=[preview_frame_num, text_frame_clip, forced_fps], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    bt_destfiles.clear(fn=on_clear_destfiles, outputs=[target_faces, selected_face_detection])
    resultfiles.select(fn=on_resultfiles_selected, inputs=[resultfiles], outputs=[resultimage, resultvideo])

    face_selection.select(on_select_face, None, None)
    bt_faceselect.click(fn=on_selected_face, outputs=[input_faces, target_faces, selected_face_detection])
    bt_cancelfaceselect.click(fn=on_end_face_selection, outputs=[dynamic_face_selection, face_selection])

    bt_clear_input_faces.click(fn=on_clear_input_faces, outputs=[input_faces])

    bt_add_local.click(fn=on_add_local_folder, inputs=[local_folder], outputs=[bt_destfiles])
    bt_preview_mask.click(fn=on_preview_mask, inputs=[ui.globals.ui_selected_swap_model, preview_frame_num, bt_destfiles, clip_text, selected_mask_engine], outputs=[previewimage]) 

    start_event = bt_start.click(fn=start_swap, 
        inputs=[ui.globals.ui_selected_swap_model, output_method, ui.globals.ui_selected_enhancer, selected_face_detection, roop.globals.keep_frames, roop.globals.wait_after_extraction,
                    roop.globals.skip_audio, max_face_distance, ui.globals.ui_blend_ratio, selected_mask_engine, clip_text,video_swapping_method, no_face_action, vr_mode, autorotate, chk_restoreoriginalmouth, num_swap_steps, ui.globals.ui_upscale, maskimage],
        outputs=[bt_start, bt_stop, resultfiles], show_progress='full')
    after_swap_event = start_event.success(fn=on_resultfiles_finished, inputs=[resultfiles], outputs=[resultimage, resultvideo])

    bt_stop.click(fn=stop_swap, cancels=[start_event, after_swap_event], outputs=[bt_start, bt_stop], queue=False)

    bt_refresh_preview.click(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs)            
    bt_toggle_masking.click(fn=on_toggle_masking, inputs=[previewimage, maskimage], outputs=[previewimage, maskimage])            
    fake_preview.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs)
    preview_frame_num.release(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden', )
    bt_use_face_from_preview.click(fn=on_use_face_from_selected, show_progress='full', inputs=[bt_destfiles, preview_frame_num], outputs=[dynamic_face_selection, face_selection, target_faces, selected_face_detection])
    set_frame_start.click(fn=on_set_frame, inputs=[set_frame_start, preview_frame_num], outputs=[text_frame_clip])
    set_frame_end.click(fn=on_set_frame, inputs=[set_frame_end, preview_frame_num], outputs=[text_frame_clip])


def on_mask_top_changed(mask_offset):
    set_mask_offset(0, mask_offset)

def on_mask_bottom_changed(mask_offset):
    set_mask_offset(1, mask_offset)

def on_mask_left_changed(mask_offset):
    set_mask_offset(2, mask_offset)

def on_mask_right_changed(mask_offset):
    set_mask_offset(3, mask_offset)

def on_mask_erosion_changed(mask_offset):
    set_mask_offset(4, mask_offset)
def on_mask_blur_changed(mask_offset):
    set_mask_offset(5, mask_offset)


def set_mask_offset(index, mask_offset):
    global SELECTED_INPUT_FACE_INDEX

    if len(roop.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        offs = roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets
        offs[index] = mask_offset
        if offs[0] + offs[1] > 0.99:
            offs[0] = 0.99
            offs[1] = 0.0
        if offs[2] + offs[3] > 0.99:
            offs[2] = 0.99
            offs[3] = 0.0
        roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets = offs

def on_mask_engine_changed(mask_engine):
    if mask_engine == "Clip2Seg":
        return gr.Textbox(interactive=True)
    return gr.Textbox(interactive=False)


def on_add_local_folder(folder):
    files = util.get_local_files_from_folder(folder)
    if files is None:
        gr.Warning("Empty folder or folder not found!")
    return files


def on_srcfile_changed(srcfiles, progress=gr.Progress()):
    global SELECTION_FACES_DATA, IS_INPUT, input_faces, face_selection, last_image
    
    IS_INPUT = True

    if srcfiles is None or len(srcfiles) < 1:
        return gr.Column(visible=False), None, ui.globals.ui_input_thumbs, None

    for f in srcfiles:    
        source_path = f.name
        if source_path.lower().endswith('fsz'):
            progress(0, desc="Retrieving faces from Faceset File")      
            unzipfolder = os.path.join(os.environ["TEMP"], 'faceset')
            if os.path.isdir(unzipfolder):
                files = os.listdir(unzipfolder)
                for file in files:
                    os.remove(os.path.join(unzipfolder, file))
            else:
                os.makedirs(unzipfolder)
            util.mkdir_with_umask(unzipfolder)
            util.unzip(source_path, unzipfolder)
            is_first = True
            face_set = FaceSet()
            for file in os.listdir(unzipfolder):
                if file.endswith(".png"):
                    filename = os.path.join(unzipfolder,file)
                    progress(0, desc="Extracting faceset")      
                    SELECTION_FACES_DATA = extract_face_images(filename,  (False, 0))
                    for f in SELECTION_FACES_DATA:
                        face = f[0]
                        face.mask_offsets = (0,0,0,0,1,20)
                        face_set.faces.append(face)
                        if is_first: 
                            image = util.convert_to_gradio(f[1])
                            ui.globals.ui_input_thumbs.append(image)
                            is_first = False
                        face_set.ref_images.append(get_image_frame(filename))
            if len(face_set.faces) > 0:
                if len(face_set.faces) > 1:
                    face_set.AverageEmbeddings()
                roop.globals.INPUT_FACESETS.append(face_set)
                                        
        elif util.has_image_extension(source_path):
            progress(0, desc="Retrieving faces from image")      
            roop.globals.source_path = source_path
            SELECTION_FACES_DATA = extract_face_images(roop.globals.source_path,  (False, 0))
            progress(0.5, desc="Retrieving faces from image")
            for f in SELECTION_FACES_DATA:
                face_set = FaceSet()
                face = f[0]
                face.mask_offsets = (0,0,0,0,1,20)
                face_set.faces.append(face)
                image = util.convert_to_gradio(f[1])
                ui.globals.ui_input_thumbs.append(image)
                roop.globals.INPUT_FACESETS.append(face_set)
                
    progress(1.0)
    return gr.Column(visible=False), None, ui.globals.ui_input_thumbs,None


def on_select_input_face(evt: gr.SelectData):
    global SELECTED_INPUT_FACE_INDEX

    SELECTED_INPUT_FACE_INDEX = evt.index


def remove_selected_input_face():
    global SELECTED_INPUT_FACE_INDEX

    if len(roop.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        f = roop.globals.INPUT_FACESETS.pop(SELECTED_INPUT_FACE_INDEX)
        del f
    if len(ui.globals.ui_input_thumbs) > SELECTED_INPUT_FACE_INDEX:
        f = ui.globals.ui_input_thumbs.pop(SELECTED_INPUT_FACE_INDEX)
        del f

    return ui.globals.ui_input_thumbs

def move_selected_input(button_text):
    global SELECTED_INPUT_FACE_INDEX

    if button_text == "⬅ Move left":
        if SELECTED_INPUT_FACE_INDEX <= 0:
            return ui.globals.ui_input_thumbs
        offset = -1
    else:
        if len(ui.globals.ui_input_thumbs) <= SELECTED_INPUT_FACE_INDEX:
            return ui.globals.ui_input_thumbs
        offset = 1
    
    f = roop.globals.INPUT_FACESETS.pop(SELECTED_INPUT_FACE_INDEX)
    roop.globals.INPUT_FACESETS.insert(SELECTED_INPUT_FACE_INDEX + offset, f)
    f = ui.globals.ui_input_thumbs.pop(SELECTED_INPUT_FACE_INDEX)
    ui.globals.ui_input_thumbs.insert(SELECTED_INPUT_FACE_INDEX + offset, f)
    return ui.globals.ui_input_thumbs
        

def move_selected_target(button_text):
    global SELECTED_TARGET_FACE_INDEX

    if button_text == "⬅ Move left":
        if SELECTED_TARGET_FACE_INDEX <= 0:
            return ui.globals.ui_target_thumbs
        offset = -1
    else:
        if len(ui.globals.ui_target_thumbs) <= SELECTED_TARGET_FACE_INDEX:
            return ui.globals.ui_target_thumbs
        offset = 1
    
    f = roop.globals.TARGET_FACES.pop(SELECTED_TARGET_FACE_INDEX)
    roop.globals.TARGET_FACES.insert(SELECTED_TARGET_FACE_INDEX + offset, f)
    f = ui.globals.ui_target_thumbs.pop(SELECTED_TARGET_FACE_INDEX)
    ui.globals.ui_target_thumbs.insert(SELECTED_TARGET_FACE_INDEX + offset, f)
    return ui.globals.ui_target_thumbs




def on_select_target_face(evt: gr.SelectData):
    global SELECTED_TARGET_FACE_INDEX

    SELECTED_TARGET_FACE_INDEX = evt.index

def remove_selected_target_face():
    if len(ui.globals.ui_target_thumbs) > SELECTED_TARGET_FACE_INDEX:
        f = roop.globals.TARGET_FACES.pop(SELECTED_TARGET_FACE_INDEX)
        del f
    if len(ui.globals.ui_target_thumbs) > SELECTED_TARGET_FACE_INDEX:
        f = ui.globals.ui_target_thumbs.pop(SELECTED_TARGET_FACE_INDEX)
        del f
    return ui.globals.ui_target_thumbs


def on_use_face_from_selected(files, frame_num):
    global IS_INPUT, SELECTION_FACES_DATA

    IS_INPUT = False
    thumbs = []
    
    roop.globals.target_path = files[selected_preview_index].name
    if util.is_image(roop.globals.target_path) and not roop.globals.target_path.lower().endswith(('gif')):
        SELECTION_FACES_DATA = extract_face_images(roop.globals.target_path,  (False, 0))
        if len(SELECTION_FACES_DATA) > 0:
            for f in SELECTION_FACES_DATA:
                image = util.convert_to_gradio(f[1])
                thumbs.append(image)
        else:
            gr.Info('No faces detected!')
            roop.globals.target_path = None
                
    elif util.is_video(roop.globals.target_path) or roop.globals.target_path.lower().endswith(('gif')):
        selected_frame = frame_num
        SELECTION_FACES_DATA = extract_face_images(roop.globals.target_path, (True, selected_frame))
        if len(SELECTION_FACES_DATA) > 0:
            for f in SELECTION_FACES_DATA:
                image = util.convert_to_gradio(f[1])
                thumbs.append(image)
        else:
            gr.Info('No faces detected!')
            roop.globals.target_path = None
    else:
        gr.Info('Unknown image/video type!')
        roop.globals.target_path = None

    if len(thumbs) == 1:
        roop.globals.TARGET_FACES.append(SELECTION_FACES_DATA[0][0])
        ui.globals.ui_target_thumbs.append(thumbs[0])
        return gr.Row(visible=False), None, ui.globals.ui_target_thumbs, gr.Dropdown(value='Selected face')

    return gr.Row(visible=True), thumbs, gr.Gallery(visible=True), gr.Dropdown(visible=True)


def on_select_face(evt: gr.SelectData):  # SelectData is a subclass of EventData
    global SELECTED_FACE_INDEX
    SELECTED_FACE_INDEX = evt.index


def on_selected_face():
    global IS_INPUT, SELECTED_FACE_INDEX, SELECTION_FACES_DATA
    
    fd = SELECTION_FACES_DATA[SELECTED_FACE_INDEX]
    image = util.convert_to_gradio(fd[1])
    if IS_INPUT:
        face_set = FaceSet()
        fd[0].mask_offsets = (0,0,0,0,1,20)
        face_set.faces.append(fd[0])
        roop.globals.INPUT_FACESETS.append(face_set)
        ui.globals.ui_input_thumbs.append(image)
        return ui.globals.ui_input_thumbs, gr.Gallery(visible=True), gr.Dropdown(visible=True)
    else:
        roop.globals.TARGET_FACES.append(fd[0])
        ui.globals.ui_target_thumbs.append(image)
        return gr.Gallery(visible=True), ui.globals.ui_target_thumbs, gr.Dropdown(value='Selected face')

#        bt_faceselect.click(fn=on_selected_face, outputs=[dynamic_face_selection, face_selection, input_faces, target_faces])

def on_end_face_selection():
    return gr.Column(visible=False), None


def on_preview_frame_changed(swap_model, frame_num, files, fake_preview, enhancer, detection, face_distance, blend_ratio,
                              selected_mask_engine, clip_text, no_face_action, vr_mode, auto_rotate, maskimage, show_face_area, restore_original_mouth, num_steps, upsample):
    global SELECTED_INPUT_FACE_INDEX, manual_masking, current_video_fps

    from roop.core import live_swap, get_processing_plugins

    manual_masking = False
    mask_offsets = (0,0,0,0)
    if len(roop.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        if not hasattr(roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0], 'mask_offsets'):
            roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets = mask_offsets
        mask_offsets = roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets

    timeinfo = '0:00:00'
    if files is None or selected_preview_index >= len(files) or frame_num is None:
        return None,None, gr.Slider(info=timeinfo)

    filename = files[selected_preview_index].name
    if util.is_video(filename) or filename.lower().endswith('gif'):
        current_frame = get_video_frame(filename, frame_num)
        if current_video_fps == 0:
            current_video_fps = 1
        secs = (frame_num - 1) / current_video_fps
        minutes = secs / 60
        secs = secs % 60
        hours = minutes / 60
        minutes = minutes % 60
        milliseconds = (secs - int(secs)) * 1000
        timeinfo = f"{int(hours):0>2}:{int(minutes):0>2}:{int(secs):0>2}.{int(milliseconds):0>3}"  
    else:
        current_frame = get_image_frame(filename)
    if current_frame is None:
        return None, None, gr.Slider(info=timeinfo)
    
    layers = None
    if maskimage is not None:
        layers = maskimage["layers"]

    if not fake_preview or len(roop.globals.INPUT_FACESETS) < 1:
        return gr.Image(value=util.convert_to_gradio(current_frame), visible=True), gr.ImageEditor(visible=False), gr.Slider(info=timeinfo)

    roop.globals.face_swap_mode = translate_swap_mode(detection)
    roop.globals.selected_enhancer = enhancer
    roop.globals.distance_threshold = face_distance
    roop.globals.blend_ratio = blend_ratio
    roop.globals.no_face_action = index_of_no_face_action(no_face_action)
    roop.globals.vr_mode = vr_mode
    roop.globals.autorotate_faces = auto_rotate
    roop.globals.subsample_size = int(upsample[:3])


    mask_engine = map_mask_engine(selected_mask_engine, clip_text)

    roop.globals.execution_threads = roop.globals.CFG.max_threads
    mask = layers[0] if layers is not None else None
    face_index = SELECTED_INPUT_FACE_INDEX
    if len(roop.globals.INPUT_FACESETS) <= face_index:
        face_index = 0
   
    options = ProcessOptions(swap_model, get_processing_plugins(mask_engine), roop.globals.distance_threshold, roop.globals.blend_ratio,
                              roop.globals.face_swap_mode, face_index, clip_text, maskimage, num_steps, roop.globals.subsample_size, show_face_area, restore_original_mouth)

    current_frame = live_swap(current_frame, options)
    if current_frame is None:
        return gr.Image(visible=True), None, gr.Slider(info=timeinfo)
    return gr.Image(value=util.convert_to_gradio(current_frame), visible=True), gr.ImageEditor(visible=False), gr.Slider(info=timeinfo)

def map_mask_engine(selected_mask_engine, clip_text):
    if selected_mask_engine == "Clip2Seg":
        mask_engine = "mask_clip2seg"
        if clip_text is None or len(clip_text) < 1:
          mask_engine = None
    elif selected_mask_engine == "DFL XSeg":
        mask_engine = "mask_xseg"
    else:
        mask_engine = None
    return mask_engine


def on_toggle_masking(previewimage, mask):
    global manual_masking

    manual_masking = not manual_masking
    if manual_masking:
        layers = mask["layers"]
        if len(layers) == 1:
            layers = [create_blank_image(previewimage.shape[1],previewimage.shape[0])]
        return gr.Image(visible=False), gr.ImageEditor(value={"background": previewimage, "layers": layers, "composite": None}, visible=True)
    return gr.Image(visible=True), gr.ImageEditor(visible=False)

def gen_processing_text(start, end):
    return f'Processing frame range [{start} - {end}]'

def on_set_frame(sender:str, frame_num):
    global selected_preview_index, list_files_process
    
    idx = selected_preview_index
    if list_files_process[idx].endframe == 0:
        return gen_processing_text(0,0)
    
    start = list_files_process[idx].startframe
    end = list_files_process[idx].endframe
    if sender.lower().endswith('start'):
        list_files_process[idx].startframe = min(frame_num, end)
    else:
        list_files_process[idx].endframe = max(frame_num, start)
    
    return gen_processing_text(list_files_process[idx].startframe,list_files_process[idx].endframe)


def on_preview_mask(swap_model, frame_num, files, clip_text, mask_engine):
    from roop.core import live_swap, get_processing_plugins
    global is_processing

    if is_processing or files is None or selected_preview_index >= len(files) or clip_text is None or frame_num is None:
        return None
        
    filename = files[selected_preview_index].name
    if util.is_video(filename) or filename.lower().endswith('gif'):
        current_frame = get_video_frame(filename, frame_num
                                        )
    else:
        current_frame = get_image_frame(filename)
    if current_frame is None or mask_engine is None:
        return None
    if mask_engine == "Clip2Seg":
        mask_engine = "mask_clip2seg"
        if clip_text is None or len(clip_text) < 1:
          mask_engine = None
    elif mask_engine == "DFL XSeg":
        mask_engine = "mask_xseg"
    options = ProcessOptions(swap_model, get_processing_plugins(mask_engine), roop.globals.distance_threshold, roop.globals.blend_ratio,
                              "all", 0, clip_text, None, 0, 128, False, False, True)

    current_frame = live_swap(current_frame, options)
    return util.convert_to_gradio(current_frame)


def on_clear_input_faces():
    ui.globals.ui_input_thumbs.clear()
    roop.globals.INPUT_FACESETS.clear()
    return ui.globals.ui_input_thumbs

def on_clear_destfiles():
    roop.globals.TARGET_FACES.clear()
    ui.globals.ui_target_thumbs.clear()
    return ui.globals.ui_target_thumbs, gr.Dropdown(value="First found")    


def index_of_no_face_action(dropdown_text):
    global no_face_choices

    return no_face_choices.index(dropdown_text) 

def translate_swap_mode(dropdown_text):
    if dropdown_text == "Selected face":
        return "selected"
    elif dropdown_text == "First found":
        return "first"
    elif dropdown_text == "All input faces":
        return "all_input"
    elif dropdown_text == "All input faces (random)":
        return "all_random"
    elif dropdown_text == "All female":
        return "all_female"
    elif dropdown_text == "All male":
        return "all_male"
    
    return "all"


def start_swap( swap_model, output_method, enhancer, detection, keep_frames, wait_after_extraction, skip_audio, face_distance, blend_ratio,
                selected_mask_engine, clip_text, processing_method, no_face_action, vr_mode, autorotate, restore_original_mouth, num_swap_steps, upsample, imagemask, progress=gr.Progress()):
    from ui.main import prepare_environment
    from roop.core import batch_process_regular
    global is_processing, list_files_process

    if list_files_process is None or len(list_files_process) <= 0:
        return gr.Button(variant="primary"), None, None
    
    if roop.globals.CFG.clear_output:
        clean_dir(roop.globals.output_path)

    if not util.is_installed("ffmpeg"):
        msg = "ffmpeg is not installed! No video processing possible."
        gr.Warning(msg)

    prepare_environment()

    roop.globals.selected_enhancer = enhancer
    roop.globals.target_path = None
    roop.globals.distance_threshold = face_distance
    roop.globals.blend_ratio = blend_ratio
    roop.globals.keep_frames = keep_frames
    roop.globals.wait_after_extraction = wait_after_extraction
    roop.globals.skip_audio = skip_audio
    roop.globals.face_swap_mode = translate_swap_mode(detection)
    roop.globals.no_face_action = index_of_no_face_action(no_face_action)
    roop.globals.vr_mode = vr_mode
    roop.globals.autorotate_faces = autorotate
    roop.globals.subsample_size = int(upsample[:3])
    mask_engine = map_mask_engine(selected_mask_engine, clip_text)

    if roop.globals.face_swap_mode == 'selected':
        if len(roop.globals.TARGET_FACES) < 1:
            gr.Error('No Target Face selected!')
            return gr.Button(variant="primary"), None, None

    is_processing = True            
    yield gr.Button(variant="secondary", interactive=False), gr.Button(variant="primary", interactive=True), None
    roop.globals.execution_threads = roop.globals.CFG.max_threads
    roop.globals.video_encoder = roop.globals.CFG.output_video_codec
    roop.globals.video_quality = roop.globals.CFG.video_quality
    roop.globals.max_memory = roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None

    batch_process_regular(swap_model, output_method, list_files_process, mask_engine, clip_text, processing_method == "In-Memory processing", imagemask, restore_original_mouth, num_swap_steps, progress, SELECTED_INPUT_FACE_INDEX)
    is_processing = False
    outdir = pathlib.Path(roop.globals.output_path)
    outfiles = [str(item) for item in outdir.rglob("*") if item.is_file()]
    if len(outfiles) > 0:
        yield gr.Button(variant="primary", interactive=True),gr.Button(variant="secondary", interactive=False),gr.Files(value=outfiles)
    else:
        yield gr.Button(variant="primary", interactive=True),gr.Button(variant="secondary", interactive=False),None


def stop_swap():
    roop.globals.processing = False
    gr.Info('Aborting processing - please wait for the remaining threads to be stopped')
    return gr.Button(variant="primary", interactive=True),gr.Button(variant="secondary", interactive=False),None


def on_fps_changed(fps):
    global selected_preview_index, list_files_process

    if len(list_files_process) < 1 or list_files_process[selected_preview_index].endframe < 1:
        return
    list_files_process[selected_preview_index].fps = fps


def on_destfiles_changed(destfiles):
    global selected_preview_index, list_files_process, current_video_fps

    if destfiles is None or len(destfiles) < 1:
        list_files_process.clear()
        return gr.Slider(value=1, maximum=1, info='0:00:00'), ''
    
    for f in destfiles:
        list_files_process.append(ProcessEntry(f.name, 0,0, 0))

    selected_preview_index = 0
    idx = selected_preview_index    
    
    filename = list_files_process[idx].filename
    
    if util.is_video(filename) or filename.lower().endswith('gif'):
        total_frames = get_video_frame_total(filename)
        if total_frames is None or total_frames < 1:
            total_frames = 1
            gr.Warning(f"Corrupted video {filename}, can't detect number of frames!")
        else:
            current_video_fps = util.detect_fps(filename)
    else:
        total_frames = 1
    list_files_process[idx].endframe = total_frames
    if total_frames > 1:
        return gr.Slider(value=1, maximum=total_frames, info='0:00:00'), gen_processing_text(list_files_process[idx].startframe,list_files_process[idx].endframe)
    return gr.Slider(value=1, maximum=total_frames, info='0:00:00'), ''


def on_destfiles_selected(evt: gr.SelectData):
    global selected_preview_index, list_files_process, current_video_fps

    if evt is not None:
        selected_preview_index = evt.index
    idx = selected_preview_index    
    filename = list_files_process[idx].filename
    fps = list_files_process[idx].fps
    if util.is_video(filename) or filename.lower().endswith('gif'):
        total_frames = get_video_frame_total(filename)
        current_video_fps = util.detect_fps(filename)
        if list_files_process[idx].endframe == 0:
            list_files_process[idx].endframe = total_frames 
    else:
        total_frames = 1
    
    if total_frames > 1:
        return gr.Slider(value=list_files_process[idx].startframe, maximum=total_frames, info='0:00:00'), gen_processing_text(list_files_process[idx].startframe,list_files_process[idx].endframe), fps
    return gr.Slider(value=1, maximum=total_frames, info='0:00:00'), gen_processing_text(0,0), fps


def on_resultfiles_selected(evt: gr.SelectData, files):
    selected_index = evt.index
    filename = files[selected_index].name
    return display_output(filename)

def on_resultfiles_finished(files):
    selected_index = 0
    if files is None or len(files) < 1:
        return None, None
    
    filename = files[selected_index].name
    return display_output(filename)


def get_gradio_output_format():
    if roop.globals.CFG.output_image_format == "jpg":
        return "jpeg"
    return roop.globals.CFG.output_image_format


def display_output(filename):
    if util.is_video(filename) and roop.globals.CFG.output_show_video:
        return gr.Image(visible=False), gr.Video(visible=True, value=filename)
    else:
        if util.is_video(filename) or filename.lower().endswith('gif'):
            current_frame = get_video_frame(filename)
        else:
            current_frame = get_image_frame(filename)
        return gr.Image(visible=True, value=util.convert_to_gradio(current_frame)), gr.Video(visible=False)
